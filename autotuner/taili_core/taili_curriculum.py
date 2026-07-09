"""Curriculum schedule config — strict per taili_strategy_decisions.md
(Curriculum 日程细则 C: #2 采样 / #5 terrain / #6 DR / #7 push + 3b φ0-φ4).

Structure + starting values are the design; absolute numbers are physeval/materialize-tuned.
Pure data the training env consumes. Spec targets (A5 envelope, E5 DR ranges, D terrain) are
sourced from taili_spec.md. CPU-testable for internal consistency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── #2 command sampling (φ0–φ1) ──────────────────────────────────────────────
COMMAND_PROPORTIONS_PHI0 = {        # single-axis + stand; Σ = 1.0
    "stand": 0.20, "forward": 0.30, "backward": 0.15, "lateral": 0.15, "yaw": 0.20,
}
T_CMD_RANGE_S = (2.0, 4.0)
CMD_SMOOTH_ALPHA = 0.93
# per-direction ceilings ramp from start band -> A5 envelope target (spec A5)
SPEED_ENVELOPE_TARGET = {"forward": 1.5, "backward": 0.8, "lateral": 0.7, "yaw": 1.5}

# ── #5 terrain families (φ2) — up/down separate, per-env adaptive level ───────
TERRAIN_FAMILIES = ["slope", "rough", "boxes", "stairs_up", "stairs_down"]
TERRAIN_DIFFICULTY_TARGET = {       # spec D
    "slope": {"angle_deg": 20.0},
    "rough": {"noise_m": 0.12},
    "boxes": {"height_m": 0.25},
    "stairs_up": {"step_m": 0.25},
    "stairs_down": {"step_m": 0.25},
}

# ── #6 DR (φ3) — deploy-faithful always-on vs difficulty ramp ─────────────────
DEPLOY_FAITHFUL_ALWAYS_ON = ["obs_noise", "known_latency"]   # from φ0; fidelity not difficulty
DR_DIMENSIONS_ORDER = ["friction", "mass", "com", "imu_bias", "kp_kd", "latency", "actuator_mismatch"]
DR_RANGES = {                       # spec E5
    "friction": (0.4, 1.4),
    "mass_kg": (-5.0, 20.0),
    "com_m": (-0.05, 0.05),
    "kp_kd_frac": (-0.40, 0.40),
}
KP_KD_DR = {"prereq": "actuator_kp_kd_sanity_passed", "start": "small_around_nominal",
            "target_frac": 0.40, "can_disable": True,
            "note": "disable is fallback only; E5 acceptance still requires +-40%"}

# ── #7 push (φ4) ─────────────────────────────────────────────────────────────
PUSH = {"form": "lateral_delta_v_impulse", "frame": "body", "directions": ["+x", "-x", "+y", "-y", "yaw"],
        "magnitude_target_mps": 1.0, "recovery_window_s": 1.0, "keep_command_in_recovery": True}


@dataclass
class Phase:
    name: str
    opens: str
    commands: str
    advance_gate: str


PHASES = [
    Phase("phi0", "flat; deploy-faithful obs-noise + known latency; no heavy DR/push/terrain",
          "single-axis + stand (incl backward)",
          "per-direction A1/A2 in-band + A3/C clean stop + no fall; backward gated INDEPENDENTLY"),
    Phase("phi1", "flat", "+ mixed; raise speed ceilings",
          "A4 each component in-band (no sacrificing vy/wz for vx) + clean stop"),
    Phase("phi2", "terrain families ramp (per-family, per-env level)", "forward-dominant; tracking relaxed",
          "controlled D1-D4 per difficulty before raising; stairs_down treated as more dangerous"),
    Phase("phi3", "heavy DR ramp (friction->mass->com->imu->kp_kd->latency->actuator_mismatch)", "as before",
          "hold phi0-phi2 acceptance under rising DR (E3/E4); per-dim gate; Kp-Kd DR needs actuator sanity"),
    Phase("phi4a", "+ standing push", "stand", "<=1.0s back to clean stand; no fall (E2)"),
    Phase("phi4b", "+ walking push (>=1.0 m/s equiv)", "walking",
          "<=1.0s recover tracking; no fall (E1)"),
]


def phase_order():
    return [p.name for p in PHASES]
