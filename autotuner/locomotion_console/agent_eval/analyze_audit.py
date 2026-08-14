"""Offline response-latency baseline from the LLM gateway audit logs.

Reads the JSON call records written by autotuner/llm_gateway/client.py and
reports how the agent actually behaves per operator question: per-call latency,
how many sequential model round-trips a single question costs, wall-clock per
question, and failure/empty rates. No network, no cost.

Usage:
    python -m autotuner.locomotion_console.agent_eval.analyze_audit [AUDIT_DIR]

If AUDIT_DIR is omitted it resolves the same default the client uses:
    $LOCOMOTION_CONSOLE_LLM_AUDIT_DIR
    | $LOCOMOTION_CONSOLE_STATE_ROOT/llm_gateway_audit
    | <tempdir>/locomotion_console_state/llm_gateway_audit
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
import tempfile


def _default_audit_dir() -> str:
    override = os.environ.get("LOCOMOTION_CONSOLE_LLM_AUDIT_DIR")
    if override:
        return override
    root = os.environ.get(
        "LOCOMOTION_CONSOLE_STATE_ROOT",
        os.path.join(tempfile.gettempdir(), "locomotion_console_state"),
    )
    return os.path.join(root, "llm_gateway_audit")


def _classify(rec: dict):
    """Classify one audit record into (kind, tool).

    IMPORTANT: the final-synthesis step logs with schema_name "text:*" and returns free-form
    prose ON PURPOSE (see llm_gateway.client.call_llm_text). That prose is NOT JSON, so a naive
    json.loads() mislabels the agent's best, complete answers as "unparsable" and treats them as
    failures. Trust schema_name first: a non-empty text:* record is a real terminal reply.
    """
    raw = (rec.get("raw") or "").strip()
    schema = str(rec.get("schema_name") or "")
    if schema.startswith("text:"):
        return ("reply" if raw else "empty"), None
    if not raw:
        return "empty", None
    try:
        j = json.loads(raw)
    except Exception:
        return "unparsable", None
    if isinstance(j, dict):
        if j.get("tool"):
            return "tool", j.get("tool")
        if "reply" in j:
            return "reply", None
    return "other", None


def _pct(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * p))
    return ordered[idx]


def analyze(audit_dir: str, since: str = "", model: str = "") -> dict:
    """Summarize the audit logs.

    since  = "YYYYMMDD" keeps only calls on/after that day (the audit dir accumulates across
             sessions, so old pre-improvement logs otherwise pollute the aggregate).
    model  = keep only calls made with this model id (e.g. isolate the current fast-model era).
    """
    files = sorted(glob.glob(os.path.join(audit_dir, "*.json")))
    calls = []
    for fn in files:
        # filename: llm_call_YYYYMMDD_HHMMSS_...
        day = os.path.basename(fn).split("_")[2] if os.path.basename(fn).count("_") >= 2 else ""
        if since and day and day < since:
            continue
        try:
            d = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        if model and d.get("model") != model:
            continue
        kind, tool = _classify(d)
        calls.append({
            "kind": kind,
            "tool": tool,
            "model": d.get("model") or "",
            "day": day,
            "elapsed": float(d.get("elapsed_s") or 0.0),
            "error": d.get("error"),
        })

    # A question = the run of calls up to and including a terminal call (reply / empty / failure).
    questions, cur = [], []
    for c in calls:
        cur.append(c)
        if c["kind"] in ("reply", "empty", "unparsable") or c["error"]:
            questions.append(cur)
            cur = []
    if cur:
        questions.append(cur)

    el = [c["elapsed"] for c in calls]
    hops = [len(q) for q in questions]
    wall = [sum(c["elapsed"] for c in q) for q in questions]
    from collections import Counter
    kinds = Counter(c["kind"] for c in calls)
    tools = Counter(c["tool"] for c in calls if c["tool"])

    def _lat(rows):
        e = [c["elapsed"] for c in rows]
        return {"n": len(rows),
                "p50": round(statistics.median(e), 1) if e else 0.0,
                "p90": round(_pct(e, 0.9), 1),
                "max": round(max(e), 1) if e else 0.0}

    models = sorted({c["model"] for c in calls})
    per_model = {m: _lat([c for c in calls if c["model"] == m]) for m in models}
    days = Counter(c["day"] for c in calls)

    return {
        "audit_dir": audit_dir,
        "days": dict(sorted(days.items())),
        "per_model_latency_s": per_model,
        "calls": len(calls),
        "questions": len(questions),
        "errors": sum(1 for c in calls if c["error"]),
        "empty_answers": kinds.get("empty", 0),
        "per_call_latency_s": {
            "p50": round(statistics.median(el), 1) if el else 0.0,
            "p90": round(_pct(el, 0.9), 1),
            "max": round(max(el), 1) if el else 0.0,
        },
        "hops_per_question": {
            "min": min(hops) if hops else 0,
            "median": int(statistics.median(hops)) if hops else 0,
            "max": max(hops) if hops else 0,
        },
        "wallclock_per_question_s": {
            "median": round(statistics.median(wall)) if wall else 0,
            "max": round(max(wall)) if wall else 0,
        },
        "call_kinds": dict(kinds),
        "top_tools": tools.most_common(8),
    }


def main() -> None:
    args = [a for a in sys.argv[1:]]
    since = ""
    model = ""
    positional = []
    for a in args:
        if a.startswith("--since="):
            since = a.split("=", 1)[1]
        elif a.startswith("--model="):
            model = a.split("=", 1)[1]
        else:
            positional.append(a)
    audit_dir = positional[0] if positional else _default_audit_dir()
    if not os.path.isdir(audit_dir):
        print(f"audit dir not found: {audit_dir}")
        raise SystemExit(1)
    report = analyze(audit_dir, since=since, model=model)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
