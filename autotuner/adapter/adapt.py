"""Adapter 顶层入口(§6 + §9 AdaptedConfig)。

`URDF (+mass +task) → 完整适配配置`,把三块派生合一:
  · derive            执行器(effort/vel/Kp/Kd)+ 维度 + 几何阈值        [§6 派生]
  · reward_scale      reward 阈值 ∝ 腿长/质量                          [§6 缩放,留口②]
  · regenerate_reference  AMP 参考 clip(URDF→IK→相位,可选实际生成)   [§6 重生成,留口①]

框架(reward 结构/课程/A-C/对角模式)不变;这里只产出机器人专属数值 + 参考。
身份测试:Taili URDF → 应复现现有 proven 配置(零 GPU)。
"""
from __future__ import annotations

from typing import Optional

from autotuner.adapter.derive import derive
from autotuner.adapter.ref_geom import extract as extract_ref_geom
from autotuner.adapter.reward_scale import scale_rewards

_DEFAULT_URDF = "assets/robots/taili-dog/robot.urdf"


def adapt(urdf_path: str, mass: Optional[float] = None,
          regenerate_clips_to: Optional[str] = None) -> dict:
    """URDF → AdaptedConfig(§9)。mass 不给则用 URDF 总质量;regenerate_clips_to 给则实际生成参考。"""
    d = derive(urdf_path)
    m = mass if mass is not None else d["mass_kg"]
    leg = d["leg_length_m"]
    geom = extract_ref_geom(urdf_path)
    rewards = scale_rewards(leg, m)

    clips = None
    if regenerate_clips_to:
        from autotuner.adapter.regenerate_reference import regenerate
        clips, _ = regenerate(urdf_path, regenerate_clips_to, which="all")

    return {
        "provenance": {"urdf": urdf_path, "mass_kg": m, "leg_length_m": leg,
                       "framework": "taili_amp (M1-M9, §7)"},
        # —— 派生:执行器 + 维度 ——
        "actuator": {"effort": d["effort_limit"], "velocity": d["velocity_limit"],
                     "Kp": d["Kp_per_joint"], "Kd": d["Kd_per_joint"],
                     "joint_range": d["joint_range"]},
        "dims": {"n_actuated_joints": d["n_actuated_joints"]},
        # —— 派生:参考几何(注入 gen_taili_gaits 用)——
        "reference_geom": geom.as_globals() if hasattr(geom, "as_globals") else None,
        "reference_clips": clips,            # None=未生成;路径列表=已生成
        # —— 缩放:reward 阈值 ——
        "reward_thresholds": rewards,
        # —— 派生:几何健康带(评估用)——
        "health_band": {"base_h_target": d["base_h_target_m"], "base_h_band": d["base_h_band_m"],
                        "swing_healthy_cm": d["swing_healthy_cm"], "drag_threshold_cm": d["drag_threshold_cm"]},
        "note": d["note"],
    }


if __name__ == "__main__":
    import sys, json
    urdf = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_URDF
    cfg = adapt(urdf)
    # 身份测试:Taili → 复现 proven 关键值
    a, r, h = cfg["actuator"], cfg["reward_thresholds"], cfg["health_band"]
    checks = {
        "effort hip/thigh/calf": (a["effort"] == {"hip": 320.0, "thigh": 110.0, "calf": 220.0}),
        "Kp 349/120/240": (a["Kp"] == {"hip": 349, "thigh": 120, "calf": 240}),
        "12 joints": (cfg["dims"]["n_actuated_joints"] == 12),
        "stand_height 0.52": (r["stand_height"] == 0.52),
        "base_clearance 0.07": (r["base_clearance"] == 0.07),
        "cmd_fwd (0.3,0.7)": (r["cmd_fwd_range"] == (0.3, 0.7)),
        "dr_mass_3 (-3,10)": (r["dr_mass_range_3"] == (-3.0, 10.0)),
        "base_h 0.552": (h["base_h_target"] == 0.552),
        "ref_geom L1 0.36385": (abs(cfg["reference_geom"]["L1"] - 0.36385) < 1e-4),
    }
    print(json.dumps({k: v for k, v in cfg.items() if k != "reference_geom"},
                     ensure_ascii=False, indent=2, default=str)[:1400], "...\n")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'XX'}] {k}")
    print(f"\nAdapter 全配置身份测试: {'PASS' if all(checks.values()) else 'FAIL'}  "
          f"(Taili URDF → 复现 proven 配置,零 GPU)")
