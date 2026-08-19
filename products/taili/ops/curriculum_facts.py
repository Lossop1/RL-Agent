"""Deterministic curriculum & gate facts card for the console agent.

Why this exists: the agent kept spending 6-10 tool hops and STILL answering "I
don't have the curriculum/gate definition" — because those facts live in code
(taili_amp_env_cfg.py / taili_amp_env.py / terrain_curriculum.py / blind_tp_env.py)
and were never surfaced to it in a labelled form. This module returns one compact,
code-cited card so the agent can answer factual questions correctly WITHOUT a hop.

The semantic facts (formulas, rules) are stable and always true. Only the specific
gate *numbers* can be overridden by a run's effective_config; those are labelled as
code defaults, with a pointer to the live source (telemetry `phase_gate`, shown with
provenance in the console gate panel). Pass `live_gate` to overlay live values.
"""
from __future__ import annotations

from typing import Any

# Code defaults — source of truth in the repo. Cited so the agent can attribute them.
GATE_DEFAULTS = {
    "prog_0": 0.60,   # phi0->phi1  taili_amp_env_cfg.py:226
    "prog_1": 0.70,   # phi1->phi2  taili_amp_env_cfg.py:228
    "prog_2": 0.50,   # phi2->phi3  taili_amp_env_cfg.py:233
    "terrain_2": 6.0,  # phi2->phi3 mean terrain level  taili_amp_env_cfg.py:230
    "fall_2": 0.05,   # phi2->phi3 fall rate            taili_amp_env_cfg.py:232
    "penalty_ramp_intervals": 25,   # taili_amp_env_cfg.py:225
    "terrain_levels": 10,           # num_rows            taili_amp_env_cfg.py:18
}


def _num(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_facts_card(live_gate: dict | None = None) -> str:
    """Return the compact facts card. `live_gate` may carry parsed per-run gate
    thresholds (e.g. from telemetry `phase_gate.conditions`) to overlay defaults;
    the card marks whichever source is used."""
    g = dict(GATE_DEFAULTS)
    src_tag = "代码默认值"
    if live_gate:
        src_tag = "本次 run 配置"
        g["prog_0"] = _num(live_gate.get("progress_min"), g["prog_0"])
        g["prog_1"] = _num(live_gate.get("progress_min"), g["prog_1"])
        g["prog_2"] = _num(live_gate.get("progress_min"), g["prog_2"])
        g["terrain_2"] = _num(live_gate.get("terrain_min"), g["terrain_2"])
        g["fall_2"] = _num(live_gate.get("fall_max"), g["fall_2"])
    ramp = int(g["penalty_ramp_intervals"])
    step = round(1.0 / ramp, 3) if ramp else 0.0
    levels = int(g["terrain_levels"])
    last = max(1, levels - 1)

    return f"""[课程与门控事实卡 — 代码为准，回答时直接引用、勿臆测]

门控 / 阶段推进阈值（phase_gate，数字为{src_tag}；本次 run 的实际门槛以遥测 phase_gate / 控制台门控面板为准）:
- phi0→phi1: 四方向 progress 的最小值 ≥ {g['prog_0']:.2f}
- phi1→phi2: progress ≥ {g['prog_1']:.2f}
- phi2→phi3: progress ≥ {g['prog_2']:.2f}, 平均地形等级 ≥ {g['terrain_2']:.1f}, 摔倒率 ≤ {g['fall_2']:.2f}
- 每一阶段还要求 penalty_gate 已到 1.0, 且连续 phase_intervals 次满足才推进
- 判断“推进快慢”看当前阶段与门控差距, 不看绝对步数占比(如 step/total)

惩罚渐入 penalty_gate（taili_amp_env.py:214,878-884）:
- 0→1 线性渐入, 每次课程更新 step = 1/penalty_ramp_intervals (当前 penalty_ramp_intervals={ramp}, 即每步 ±{step})
- phase≥1 自举为 1.0; 按比例混合质量惩罚强度, 避免早期 critic 冲击

actual_vx（blind_tp_env.py:1427）:
- = 机身坐标系下基座线速度前向(x)分量, 对所有并行环境取均值 float(vb[:,0].mean()); 实测前进速度, 用于对比 cmd_vx

地形推进 / 升降级（taili_core/terrain_curriculum.py）:
- 逐环境。升级需一次“稳定通过”: 走够距离(dist>terrain_move_up_dist) 或 受控高度上下变化, 且结尾稳定(直立/基座高度够/足接触够/机身角速度低)并速度受控
- 摔倒/终止/失稳 = 立即降级; 走得比预期少也降级。仅在 terrain_start_phase 之后启用。单纯高度变化不算成功, 冲下楼梯判失败

地形难度等级（taili_amp_env_cfg.py:17-37）:
- 共 num_rows={levels} 个难度等级(0..{last}); 台阶高度随难度在各自区间内线性插值: 等级 k → 难度 k/{last}
- 两个楼梯子地形 step_height 区间: 矮楼梯 (0.05,0.15), 高楼梯 (0.08,0.30)
- 例: 等级3 → 难度 3/{last}≈{3/last:.3f} → 矮楼梯≈{0.05 + (3/last)*(0.15-0.05):.3f}m, 高楼梯≈{0.08 + (3/last)*(0.30-0.08):.3f}m (取决于是哪种楼梯)"""


if __name__ == "__main__":
    print(build_facts_card())
