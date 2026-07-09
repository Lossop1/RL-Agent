"""Safe, reversible editing of the tuning STRATEGY (reward weights + curriculum gates) — the "tune"
primitive the copilot needs to autonomously complete a tuning task, with an undo stack.

Every apply() allowlists the keys, validates the values against per-family bounds, edits the YAML
in place (preserving comments/structure), and PUSHES a reversible entry onto a rollback stack, so a
multi-field tune can be undone field-for-field (the design's "keyed rollback stack"). Pure + file-
based; no remote or GPU. The applied deltas are what deploy+launch then ship to the box.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_STRATEGY_YAML = _REPO / "autotuner" / "blind_locomotion" / "taili_blind_config.yaml"
_ROLLBACK_STACK = _REPO / "autotuner" / "blind_locomotion" / ".strategy_rollback.jsonl"

# Allowlisted tunable keys (bare yaml key -> (min, max)). Reward weights + slip/clearance shaping +
# curriculum advancement gates. NOTHING else is editable through this path (no arbitrary keys).
_BOUNDS: dict[str, tuple[float, float]] = {
    **{f"w_{n}": (0.0, 30.0) for n in (
        "tracking_lin", "track_far", "tracking_yaw", "yaw_far", "stand", "stand_far",
        "stand_contact", "terrain_progress", "gait_anchor", "feet_air_time", "off_axis",
        "wrong_dir", "action_rate", "stance_slip", "stance_slip_late", "landing_impact",
        "landing_impact_late", "orient", "base_vz", "base_wxy", "clearance_under",
        "clearance_over", "diagonal_contact", "duty_balance", "torque_margin", "torque_saturation")},
    "slip_free_speed": (0.02, 0.5), "slip_speed_scale": (0.1, 3.0),
    "flat_clearance_target": (0.03, 0.20), "terrain_clearance_margin": (0.0, 0.2),
    **{f"phase_gate_{n}": (0.0, 10.0) for n in (
        "prog_0", "prog_1", "prog_2", "prog_3", "diag_0", "diag_1",
        "terrain_2", "fall_2", "air_0", "air_1")},
}


def _key_bounds(key: str) -> tuple[float, float] | None:
    return _BOUNDS.get(key.split(".")[-1])


def _line_re(bare_key: str) -> re.Pattern:
    # match "  <key>: <number>" capturing indent+key and the numeric value, keeping any trailing comment
    return re.compile(rf"^(\s*{re.escape(bare_key)}:\s*)(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(\s*(?:#.*)?)$")


def validate_changes(changes: dict[str, Any]) -> list[str]:
    """Return a list of human-readable rejection reasons (empty = all valid)."""
    errs: list[str] = []
    for key, val in changes.items():
        bare = key.split(".")[-1]
        b = _key_bounds(bare)
        if b is None:
            errs.append(f"{key}: not an allowlisted tunable key")
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            errs.append(f"{key}: value {val!r} is not numeric")
            continue
        if not (b[0] <= f <= b[1]):
            errs.append(f"{key}: {f} out of bounds [{b[0]}, {b[1]}]")
    return errs


def apply_weight_changes(changes: dict[str, Any], *, yaml_path: Path | None = None,
                         note: str = "", stamp: str = "") -> dict[str, Any]:
    """Apply allowlisted numeric changes to the strategy YAML and push a rollback entry.

    changes: {key: new_value} — key may be dotted (env.curriculum.phase_gate_prog_0) or bare
    (w_stance_slip); only the final segment is matched against the allowlist + the YAML line.
    Returns {ok, applied:[{key, old, new}], errors, rollback_id}. Idempotent-safe: a key already at
    the target value is a no-op entry. Never partially applies past a validation error.
    """
    path = yaml_path or _STRATEGY_YAML
    errs = validate_changes(changes)
    if errs:
        return {"ok": False, "applied": [], "errors": errs, "rollback_id": None}

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    applied: list[dict[str, Any]] = []
    not_found: list[str] = []
    for key, val in changes.items():
        bare = key.split(".")[-1]
        rx = _line_re(bare)
        new_val = float(val)
        new_str = repr(int(new_val)) if new_val == int(new_val) and abs(new_val) < 1e6 else repr(new_val)
        hit = False
        for i, line in enumerate(lines):
            m = rx.match(line.rstrip("\n"))
            if m:
                old = m.group(2)
                lines[i] = f"{m.group(1)}{new_str}{m.group(3)}\n"
                applied.append({"key": bare, "old": old, "new": new_str})
                hit = True
                break
        if not hit:
            not_found.append(key)
    if not_found:
        return {"ok": False, "applied": [], "errors": [f"key(s) not found in YAML: {not_found}"],
                "rollback_id": None}

    path.write_text("".join(lines), encoding="utf-8")
    rb_id = f"rb_{stamp}" if stamp else f"rb_{len(applied)}_{'_'.join(a['key'] for a in applied)[:40]}"
    entry = {"rollback_id": rb_id, "note": note, "path": str(path), "applied": applied}
    with _ROLLBACK_STACK.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"ok": True, "applied": applied, "errors": [], "rollback_id": rb_id}


def rollback_stack() -> list[dict[str, Any]]:
    if not _ROLLBACK_STACK.exists():
        return []
    return [json.loads(l) for l in _ROLLBACK_STACK.read_text(encoding="utf-8").splitlines() if l.strip()]


def rollback_last(*, yaml_path: Path | None = None) -> dict[str, Any]:
    """Pop the most recent apply and restore its prior values, field-for-field."""
    stack = rollback_stack()
    if not stack:
        return {"ok": False, "errors": ["rollback stack is empty"]}
    entry = stack[-1]
    path = yaml_path or Path(entry.get("path") or _STRATEGY_YAML)
    restore = {a["key"]: float(a["old"]) for a in entry.get("applied", [])}
    # apply the OLD values back (validation-safe: they were valid when set) without pushing a new entry
    res = apply_weight_changes(restore, yaml_path=path, note=f"rollback of {entry['rollback_id']}")
    if res["ok"]:
        _ROLLBACK_STACK.write_text(
            "".join(json.dumps(e) + "\n" for e in stack[:-1]), encoding="utf-8")
        return {"ok": True, "restored": res["applied"], "rolled_back": entry["rollback_id"]}
    return {"ok": False, "errors": res["errors"]}
