"""Structured, read-only view of the detailed TUNING STRATEGY (reward weights, curriculum phases +
advancement gates, AMP hyperparameters) from taili_blind_config.yaml — so the operator can SEE the
detailed settings in the panel (not just monitor telemetry) and the copilot can annotate them.

Pure + defensive: never raises into the request path; on any failure returns {available: False}.
"""
from __future__ import annotations

from typing import Any


def _find_all(obj: Any, pred, path: str = "") -> dict[str, Any]:
    """Collect {dotted_path: value} for every leaf key matching pred(key)."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{path}{k}"
            if not isinstance(v, (dict, list)) and pred(str(k)):
                out[kp] = v
            out.update(_find_all(v, pred, kp + "."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_find_all(v, pred, f"{path}{i}."))
    return out


def build_strategy_view() -> dict[str, Any]:
    try:
        from autotuner.blind_locomotion.taili_blind_config import load_taili_blind_config
        cfg = load_taili_blind_config()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"could not load strategy config: {exc}"}

    reward = cfg.get("reward", {}) if isinstance(cfg.get("reward"), dict) else {}
    recipe = cfg.get("training_recipe", {}) if isinstance(cfg.get("training_recipe"), dict) else {}
    skrl_agent = (((cfg.get("skrl") or {}).get("agent")) or {}) if isinstance(cfg.get("skrl"), dict) else {}

    # reward weights grouped: what the operator most wants to reason about while tuning
    weights = {k: v for k, v in reward.items() if str(k).startswith("w_")}
    tracking = {k: v for k, v in weights.items() if "track" in k or "yaw" in k or "stand" in k}
    gait_quality = {k: v for k, v in weights.items()
                    if any(t in k for t in ("slip", "clearance", "landing", "diagonal", "duty",
                                            "air", "gait", "off_axis", "wrong_dir", "torque"))}
    other_weights = {k: v for k, v in weights.items() if k not in tracking and k not in gait_quality}
    thresholds = {k: v for k, v in reward.items()
                  if not str(k).startswith("w_") and isinstance(v, (int, float))}

    # curriculum: per-phase settings + the advancement gates (wherever they live in the tree)
    phases_raw = recipe.get("phases") if isinstance(recipe.get("phases"), dict) else {}
    phases = {str(k): v for k, v in sorted(phases_raw.items(), key=lambda kv: str(kv[0]))}
    phase_gates = _find_all(cfg, lambda k: "phase_gate" in k or k.startswith("gate_"))

    amp = {k: skrl_agent.get(k) for k in (
        "style_reward_weight", "task_reward_weight", "discriminator_reward_scale",
        "discriminator_loss_scale", "learning_rate", "entropy_loss_scale", "value_loss_scale",
        "rollouts", "learning_epochs", "mini_batches", "amp_batch_size") if k in skrl_agent}

    return {
        "available": True,
        "profile": cfg.get("profile"),
        "reward": {
            "tracking": tracking,
            "gait_quality": gait_quality,
            "other": other_weights,
            "thresholds": thresholds,
        },
        "curriculum": {
            "init_phase": recipe.get("init_phase"),
            "command_mode": recipe.get("command_mode"),
            "phases": phases,
            "advancement_gates": phase_gates,
        },
        "amp": amp,
        "counts": {"reward_weights": len(weights), "phases": len(phases),
                   "advancement_gates": len(phase_gates)},
        "note": ("Read-only view of the editable strategy contract (taili_blind_config.yaml). Changes "
                 "are applied via the edit_config action (rollback available), not here."),
    }
