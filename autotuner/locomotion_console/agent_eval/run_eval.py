"""Direct-model accuracy/latency probe for the console agent's brain.

This does NOT run the full agent tool-loop (that needs the live backend and
remote evidence). It isolates the *model* layer: it asks the configured DeepSeek
model each eval question directly, with thinking enabled, and reads BOTH fields
the API returns:

  - content            (the ONLY field the current app keeps)
  - reasoning_content  (the model's actual thinking — the app discards this)

Scoring is a lenient tripwire, not a grader:
  A-class: fraction of code-derived gold facts that appear. Scored twice —
           once over content-only (what the app sees) and once over
           content+reasoning (what the model actually produced) — so the gap
           quantifies how much answer quality is thrown away.
  B-class: fails if the answer commits the rejected pattern (e.g. judging
           progress by raw step fraction); passes if it references the real
           progress signal (phase / gate).

Usage:
    python -m autotuner.locomotion_console.agent_eval.run_eval [--dry]

--dry prints the plan without calling the API.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

from autotuner.locomotion_console.curriculum_facts import build_facts_card

HERE = Path(__file__).resolve().parent
CASES_PATH = HERE / "cases.json"
CONFIG_PATH = Path("config/llm.json")


def _load_key_and_endpoint():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw = cfg.get("api_key_env_var", "")
    key = raw if raw.startswith(("sk-", "key-")) else os.environ.get(raw, "")
    base = (cfg.get("base_url") or "https://api.deepseek.com").rstrip("/")
    model = cfg.get("model", "deepseek-v4-pro")
    return key, base, model


def _call(base, key, model, question, timeout=180):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"},
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=timeout)
    j = json.loads(r.read())
    msg = j["choices"][0]["message"]
    return {
        "elapsed_s": round(time.time() - t0, 1),
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
    }


def _score_a(case, content, reasoning):
    facts = case.get("gold_facts", [])
    both = content + "\n" + reasoning
    app_hit = sum(1 for f in facts if f in content)
    full_hit = sum(1 for f in facts if f in both)
    n = max(1, len(facts))
    return {
        "app_score": round(app_hit / n, 2),      # what the app actually shows
        "model_score": round(full_hit / n, 2),   # what the model produced
        "facts": len(facts),
    }


def _score_b(case, content):
    rejected = [p for p in case.get("reject_if_any", []) if p in content]
    expected = [p for p in case.get("expect_any", []) if p in content]
    return {
        "passed": (not rejected) and bool(expected),
        "committed_fallacy": rejected,
        "referenced_real_signal": expected,
    }


def main() -> None:
    import sys
    dry = "--dry" in sys.argv
    with_facts = "--with-facts" in sys.argv
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    key, base, model = _load_key_and_endpoint()
    for a in sys.argv:
        if a.startswith("--model="):
            model = a.split("=", 1)[1]
    mode = "WITH facts card" if with_facts else "baseline (no facts)"
    results_path = HERE / ("results_with_facts.json" if with_facts else "baseline_results.json")
    facts = build_facts_card() if with_facts else ""
    print(f"model={model}  endpoint={base}  cases={len(cases)}  key={'ok' if key else 'MISSING'}  mode={mode}")
    if dry or not key:
        for c in cases:
            print(f"  [{c['id']}] {c['class']}  {c['question']}")
        return

    results = []
    for c in cases:
        question = c["question"]
        if with_facts:
            question = facts + "\n\n请依据上面的事实卡回答（可直接引用其中数字）：\n" + question
        try:
            out = _call(base, key, model, question)
        except Exception as e:
            print(f"  [{c['id']}] ERROR {type(e).__name__}: {str(e)[:100]}")
            results.append({"id": c["id"], "error": str(e)})
            continue
        row = {
            "id": c["id"], "class": c["class"],
            "elapsed_s": out["elapsed_s"],
            "content_chars": len(out["content"]),
            "reasoning_chars": len(out["reasoning"]),
        }
        if c["class"] == "A":
            row["score"] = _score_a(c, out["content"], out["reasoning"])
        else:
            row["score"] = _score_b(c, out["content"])
        results.append(row)
        print(f"  [{c['id']}] {out['elapsed_s']}s  content={len(out['content'])}ch "
              f"reasoning={len(out['reasoning'])}ch  score={row['score']}")

    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    # aggregate
    a = [r for r in results if r.get("class") == "A" and "score" in r]
    b = [r for r in results if r.get("class") == "B" and "score" in r]
    if a:
        app = sum(r["score"]["app_score"] for r in a) / len(a)
        mod = sum(r["score"]["model_score"] for r in a) / len(a)
        print(f"\nA-class: app-visible accuracy={app:.0%}  model-produced accuracy={mod:.0%}  "
              f"(gap = reasoning the app discards)")
    if b:
        print(f"B-class: {sum(1 for r in b if r['score']['passed'])}/{len(b)} avoided the fallacy")
    print(f"written: {results_path}")


if __name__ == "__main__":
    main()
