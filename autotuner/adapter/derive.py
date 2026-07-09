"""Adapter(§15.2):URDF + 规格 + 任务 → 机器人专属配置(从不变框架派生)。

系统的新核心:框架结构不变,这里派生机器人专属的数值/参考。本模块先做"派生 + 身份测试"
(喂 Taili 自己的 URDF 应≈现有 proven 配置,证明 Adapter 正确捕获了框架的机器人派生)。

派生项(§15.2):执行器(effort/vel=URDF;Kp∝effort 锚定;Kd=临界阻尼)· 几何(腿长/base_h/swing 带)
· 维度(关节数)· 命令范围(运动学)。**参考重生成(AMP)单列**(gen_taili_gaits,形态相关),此处先给派生数值。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict

from autotuner.adapter._safe_xml import safe_parse_urdf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import robot_kinematics as rk  # noqa: E402


def _link_inertials(root) -> Dict[str, dict]:
    out = {}
    for ln in root.iter("link"):
        ine = ln.find("inertial")
        if ine is None:
            continue
        m = float(ine.find("mass").attrib.get("value", 0))
        I = ine.find("inertia")
        iyy = float(I.attrib.get("iyy", 0)) if I is not None else 0.0
        o = ine.find("origin")
        com = [float(x) for x in (o.attrib.get("xyz", "0 0 0").split())] if o is not None else [0, 0, 0]
        out[ln.attrib.get("name", "")] = {"m": m, "iyy": iyy, "com": com}
    return out


def derive(urdf_path: str, ref_kp_per_effort: float = 120.0 / 110.0,
           max_pos_error_rad: float = 0.92) -> dict:
    """URDF → 派生配置。ref_kp_per_effort 锚定已验证 thigh(Kp120/effort110)。"""
    root = safe_parse_urdf(urdf_path)
    km = rk._parse_urdf(root)
    inert = _link_inertials(root)

    # 每关节 effort/velocity(URDF limit = 可靠权威)
    eff, vel, lo, hi = {}, {}, {}, {}
    for j in root.iter("joint"):
        if j.attrib.get("type") == "fixed":
            continue
        lim = j.find("limit")
        if lim is None:
            continue
        role = j.attrib.get("name", "").split("_")[-2] if "_" in j.attrib.get("name", "") else j.attrib.get("name")
        if role in eff:
            continue
        eff[role] = float(lim.attrib.get("effort", 0))
        vel[role] = float(lim.attrib.get("velocity", 0))
        lo[role] = float(lim.attrib.get("lower", 0))
        hi[role] = float(lim.attrib.get("upper", 0))

    leg = km["leg_segments"]["total_leg_length"]
    mass = km["total_mass"]

    # 执行器派生:Kp ∝ effort(锚定);Kd = 临界阻尼 2√(Kp·I_link)
    # I_link 近似:thigh 驱动 thigh+calf+foot;calf 驱动 calf+foot;hip 整腿挂落
    def leg_mass_sum(prefix):
        return sum(v["m"] for k, v in inert.items() if k.startswith(prefix) and any(s in k for s in ["thigh", "calf", "foot"]))
    # 取一条腿(FR)估算
    thigh_iyy = next((v["iyy"] for k, v in inert.items() if "FR_thigh" in k), 0.047)
    calf = next((v for k, v in inert.items() if "FR_calf" in k), {"m": 0.73, "iyy": 0.012})
    foot = next((v for k, v in inert.items() if "FR_foot" in k), {"m": 0.094, "iyy": 0.0})
    I_thigh = thigh_iyy + (calf["iyy"] + calf["m"] * 0.366 ** 2) + (foot.get("iyy", 0) + foot["m"] * 0.417 ** 2)
    I_calf = calf["iyy"] + calf["m"] * 0.166 ** 2 + foot["m"] * 0.217 ** 2
    I_hip = leg_mass_sum("FR") * 0.30 ** 2
    I_link = {"hip": I_hip, "thigh": I_thigh, "calf": I_calf}

    Kp = {r: round(ref_kp_per_effort * eff.get(r, 0)) for r in eff}
    Kd = {r: round(2 * math.sqrt(max(Kp.get(r, 0), 1) * I_link.get(r, 0.05)), 1) for r in eff}

    njoints = len([j for j in root.iter("joint") if j.attrib.get("type") != "fixed"
                   and j.find("limit") is not None])

    return {
        "mass_kg": mass,
        "leg_length_m": leg,
        "n_actuated_joints": njoints,
        "effort_limit": eff,
        "velocity_limit": vel,
        "joint_range": {r: [lo[r], hi[r]] for r in eff},
        "Kp_per_joint": Kp,
        "Kd_per_joint": Kd,
        "base_h_target_m": round(0.81 * leg, 3),
        "base_h_band_m": [round(0.78 * leg, 3), round(0.90 * leg, 3)],
        "swing_healthy_cm": [round(0.18 * leg * 100, 1), round(0.25 * leg * 100, 1)],
        "drag_threshold_cm": round(0.08 * leg * 100, 1),
        "note": "派生:执行器 effort/vel=URDF;Kp∝effort 锚thigh;Kd=临界;几何阈值∝腿长。AMP参考需 gen_taili_gaits 重生成(未含)。",
    }


if __name__ == "__main__":
    import json
    urdf = sys.argv[1] if len(sys.argv) > 1 else "assets/robots/taili-dog/robot.urdf"
    d = derive(urdf)
    print(json.dumps(d, ensure_ascii=False, indent=2))
