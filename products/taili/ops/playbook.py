"""Taili 产品的运行经验档案。

系统层只定义通用工作流与取证纪律；这里保存 Taili 的门控、杠杆和运行约束。
新增机器人时应新增自己的产品插件，而不是修改系统 playbook。
"""
from __future__ import annotations

from typing import Any

from autotuner.locomotion_console.playbook import build_playbook


PROFILE: dict[str, Any] = {
    "taili": {
        "label": "Taili 39kg blind quadruped (AMP+PPO, IsaacLab)",
        "regime": {
            "peak_steps": "~18k resume-steps (longer overtrains B4/A2)",
            "iter_cost": "~1.5h to 18k",
            "mem_limit": "50GB cgroup (2 Isaac procs OOM)",
            "resume": "TAILI_INIT_PHASE=3 (checkpoint doesn't store curriculum phase)",
            "envs": "1024 (2048 OOMs ~16k)",
        },
        "benchmark": {
            "families": 19,
            "gate_instances": 26,
            "best": "10/26 (dirclr agent_18000)",
            "gate_defs": {
                "D[stairs]": "DESCENDING (pyramid, spawn-on-top; fwd_speed+fall_rate)",
                "D[stairs_up]": "ASCENDING (inverted-pit, spawn-bottom; climbed>=0.15m+no fall)",
            },
        },
        "gate_levers": {
            "A3": {
                "symptom": "duty_min<0.95 (foot lifts during stand); settle>1.0s",
                "lever": "TELEMETRY (0706): stand_gate~0.16 = the robot practises standing only ~16% of training, so w_stand_contact 0.8->1.8 did NOT move A3 (duty_min 0.88->0.89). The real lever is EXPOSURE: raise stand_prob (0.10->0.18) so it practises standing. Testing isolated.",
                "caution": "one-lever only (a combined A3+stairs run confounded + regressed 10->9). +2 candidate.",
            },
            "C": {"symptom": "A3=False and/or stand_sym>0.05", "lever": "same as A3 (coupled)", "caution": "fix A3 first."},
            "A2": {
                "symptom": "yaw p90>0.15 (median often passes)",
                "lever": "rew_imitate turn-gated + pure-yaw sampling (x_axis_prob<1)",
                "caution": "w_tracking_yaw is WEIGHT-IMMUNE (proven). Moved 0.7->0.47; 0.15 still hard.",
            },
            "B4": {"symptom": "L/R duty asym grows with speed", "lever": "likely coupled to yaw-drift -> fix A2", "caution": "equivariance already ON; not an actor-symmetry gap."},
            "D[stairs]": {"symptom": "descending fall_rate>0", "lever": "keep clearance moderate", "caution": "aggressive lift regresses descent (9.4%->28%)."},
            "D[stairs_up]": {"symptom": "climbed<0.15m", "lever": "foot_h4 + w_climb + stairs_up curriculum", "caution": "HARD (SOTA edge). climbs 8cm not 25cm."},
            "F2": {"symptom": "peak torque/limit~1.0", "lever": "w_torque_margin GENTLE (<=1.8)", "caution": "2.2 hangs the sim (deterministic physics explosion)."},
            "B2": {"symptom": "stance slip p90>0.05", "lever": "SKIP (historic artifact; near floor)", "caution": "low ROI."},
        },
        "ops_signatures": {
            "step frozen + process alive + idle GPU + no traceback": {
                "diagnosis": "GPU kernel stall (rented-box transparent fault). NOT code/OOM.",
                "action": "self-heal driver recovers via fresh-LOG check (process-alive alone is fooled by the husk).",
            },
            "run overshot past peak_steps (e.g. 52k)": {
                "diagnosis": "driver stopped monitoring but didn't KILL at target -> overtraining regressed 10->9.",
                "action": "driver now kills at target; measure a peak-steps checkpoint, not the overshoot.",
            },
            "train + diagnostic both running": {
                "diagnosis": "two Isaac procs exceed the 50GB cgroup -> OOM.",
                "action": "never concurrent; driver HOLDs relaunch while a diagnostic runs.",
            },
            "AttributeError acceptance_score": {
                "diagnosis": "payload has TWO acceptance_score.py (root + package); physeval imports the ROOT.",
                "action": "deploy to BOTH locations.",
            },
        },
        "harness_gotchas": [
            "diagnose_taili multi-case records only case 0 -> use physeval for per-condition gates",
            "judge climb over FULL per-case trajectory range, not the leading 1/3 (this caused a 3-iteration false 'can't climb')",
        ],
    },
}


def get_playbook(task: str = "", gate: str = "", robot: str = "") -> dict[str, Any]:
    """返回 Taili 档案与系统通用工作流的组合。"""
    return build_playbook(task=task, gate=gate, robot=robot, profiles=PROFILE, default_robot="taili")

