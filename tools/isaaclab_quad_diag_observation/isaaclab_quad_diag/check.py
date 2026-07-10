from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .util import read_json, load_yaml, write_json


def get_path(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except Exception:
                return None
        else:
            return None
    return cur


def compare(value: Any, op: str, expected: Any) -> bool | None:
    if value is None:
        return None
    try:
        v = float(value)
        e = float(expected)
    except Exception:
        if op in ["==", "eq"]:
            return value == expected
        if op in ["!=", "ne"]:
            return value != expected
        return None
    if op in ["<=", "le"]:
        return v <= e
    if op in ["<", "lt"]:
        return v < e
    if op in [">=", "ge"]:
        return v >= e
    if op in [">", "gt"]:
        return v > e
    if op in ["==", "eq"]:
        return v == e
    if op in ["!=", "ne"]:
        return v != e
    return None


def apply_target(metrics: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    rules = target.get("rules", {}) if isinstance(target, dict) else {}
    results = []
    for name, rule in rules.items():
        metric_path = rule.get("metric")
        op = rule.get("op", "<=")
        value = rule.get("value")
        actual = get_path(metrics, metric_path) if metric_path else None
        result = compare(actual, op, value)
        results.append({
            "rule": name,
            "metric": metric_path,
            "actual": actual,
            "op": op,
            "value": value,
            "result": result,
        })
    return {
        "schema_version": "ilqd_external_check_v0.5.1",
        "note": "This is optional external target evaluation. The diagnostics package itself does not define targets.",
        "rules_total": len(results),
        "rules_with_missing_metric": sum(1 for r in results if r["result"] is None),
        "rules_matching_condition": sum(1 for r in results if r["result"] is True),
        "rules_not_matching_condition": sum(1 for r in results if r["result"] is False),
        "results": results,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Apply an external target file to metrics. Not used by default diagnostics.")
    p.add_argument("--metrics", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    metrics = read_json(args.metrics)
    target = load_yaml(args.target)
    report = apply_target(metrics, target)
    out = Path(args.out)
    if out.suffix:
        write_json(out, report)
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "external_check_report.json", report)
        md = ["# External Target Check", "", "Diagnostics are observation-only; this report is produced only because a target file was provided.", ""]
        for r in report["results"]:
            md.append(f"- {r['rule']}: actual={r['actual']} {r['op']} {r['value']} -> {r['result']}")
        (out / "external_check_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"external target report written to: {args.out}")


if __name__ == "__main__":
    main()
