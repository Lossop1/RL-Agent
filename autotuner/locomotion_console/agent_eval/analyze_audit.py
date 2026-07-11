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


def _classify(raw: str):
    raw = (raw or "").strip()
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


def analyze(audit_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(audit_dir, "*.json")))
    calls = []
    for fn in files:
        try:
            d = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        kind, tool = _classify(d.get("raw"))
        calls.append({
            "kind": kind,
            "tool": tool,
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

    return {
        "audit_dir": audit_dir,
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
    audit_dir = sys.argv[1] if len(sys.argv) > 1 else _default_audit_dir()
    if not os.path.isdir(audit_dir):
        print(f"audit dir not found: {audit_dir}")
        raise SystemExit(1)
    report = analyze(audit_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
