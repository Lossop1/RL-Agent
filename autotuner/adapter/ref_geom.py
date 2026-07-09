"""Adapter — AMP 参考几何派生(§6 "AMP 参考=重生成")。

`gen_taili_gaits.py` 把腿几何(L1/FOOT_OFF/HIPX/HIPY/THIGHY/CALFZ/BASE_Z/clearance)写成模块级
硬编码常量,只对 Taili 对。换 URDF 要"重生成"参考 → 这里从 URDF 关节原点把这套几何**派生出来**,
注入生成器,再跑生成。框架(IK/相位/对角模式)不变,变的只是这套机器人几何。

派生来源(都是 URDF `<joint><origin xyz>`,可靠权威):
  HIPX,HIPY,hip_z = base_link→{leg}_hip_joint
  THIGHY          = {leg}_hip→{leg}_thigh_joint  (y)
  L1,CALFZ        = {leg}_thigh→{leg}_calf_joint  (z)
  FOOT_OFF        = {leg}_calf→{leg}_foot_joint   (xyz)
风格比例(框架证明有效,∝腿长,缩放型):
  BASE_Z   = BASE_Z_RATIO * leg   (站立参考姿态;Taili single-height nominal 0.52)
  clearance= CLEAR_RATIO  * leg   (摆动抬脚;Taili proven 0.09)
"""
from __future__ import annotations

from dataclasses import dataclass

from autotuner.adapter._safe_xml import safe_parse_urdf
from typing import List

# 框架证明有效的参考风格(站立更挺的 base 高度 + 摆动抬脚)。这两个不是纯几何,是框架 style——
# 用户可调("腿抬高"→clearance↑)。默认 = proven 绝对值 按腿长等比缩放到新机器人:
#   v = PROVEN * total_leg / _LEG_PROVEN   (Taili total_leg==_LEG_PROVEN → 精确复现 0.52/0.09)
_LEG_PROVEN = 0.36385 + 0.3179635      # Taili total_leg = L1 + |foot_off_z|
_BASE_Z_PROVEN = 0.52
# 参考抬腿高度:用户领域修正(2026-06-23)——平地 3-5cm(高效低抬),不是早先的 9cm(过高/prancing)。
# 坡/台阶上 = 离面 3-5cm。取 4cm 为框架默认。注:Taili 已部署 clip 用的是旧 9cm。
_CLEAR_PROVEN = 0.04


@dataclass
class RefGeom:
    L1: float            # thigh length
    foot_off: List[float]  # calf->foot xyz
    hipx: float
    hipy: float
    hip_z: float
    thighy: float
    calfz: float         # = -L1
    base_z: float
    clearance: float
    total_leg: float

    def as_globals(self) -> dict:
        """要注入 gen_taili_gaits 模块全局的名字→值。"""
        return {
            "L1": self.L1,
            "FOOT_OFF": __import__("numpy").array(self.foot_off, dtype=float),
            "HIPX": self.hipx, "HIPY": self.hipy,
            "THIGHY": self.thighy, "CALFZ": self.calfz,
            "BASE_Z": self.base_z,
        }


def _origin(root, jname) -> List[float]:
    for j in root.iter("joint"):
        if j.attrib.get("name") == jname:
            o = j.find("origin")
            return [float(x) for x in (o.attrib.get("xyz", "0 0 0").split())] if o is not None else [0, 0, 0]
    raise KeyError(f"joint {jname} not in URDF")


def extract(urdf_path: str, leg: str = "FL") -> RefGeom:
    """从 URDF 一条腿(默认 FL)的关节链派生参考几何。其余腿由 SGN 对称镜像(框架处理)。"""
    root = safe_parse_urdf(urdf_path)
    hip = _origin(root, f"{leg}_hip_joint")
    thigh = _origin(root, f"{leg}_thigh_joint")
    calf = _origin(root, f"{leg}_calf_joint")
    foot = _origin(root, f"{leg}_foot_joint")

    L1 = abs(calf[2])
    total_leg = L1 + abs(foot[2])  # 粗略矢状腿长(thigh + calf→foot 竖直分量)
    return RefGeom(
        L1=L1,
        foot_off=foot,
        hipx=abs(hip[0]), hipy=abs(hip[1]), hip_z=hip[2],
        thighy=abs(thigh[1]),
        calfz=calf[2],
        base_z=round(_BASE_Z_PROVEN * total_leg / _LEG_PROVEN, 4),
        clearance=round(_CLEAR_PROVEN * total_leg / _LEG_PROVEN, 4),
        total_leg=total_leg,
    )


if __name__ == "__main__":
    import sys
    u = sys.argv[1] if len(sys.argv) > 1 else "assets/robots/taili-dog/robot.urdf"
    g = extract(u)
    # 身份测试:派生值 vs gen_taili_gaits 硬编码常量
    expect = dict(L1=0.36385, hipx=0.30414, hipy=0.065, thighy=0.1432, calfz=-0.36385,
                  foot_off=[0.0252765, 0.0, -0.3179635], base_z=0.52, clearance=0.09)
    print(f"{'field':10} {'derived':>14} {'hardcoded':>14} {'match':>6}")
    ok = True
    for k, ev in expect.items():
        dv = getattr(g, k)
        if isinstance(ev, list):
            m = all(abs(a - b) < 2e-3 for a, b in zip(dv, ev))
        else:
            m = abs(dv - ev) < 2e-3
        ok &= m
        print(f"{k:10} {str(round(dv,5) if not isinstance(dv,list) else [round(x,5) for x in dv]):>14} "
              f"{str(ev):>14} {'OK' if m else 'XX':>6}")
    print(f"\n参考几何身份测试: {'PASS' if ok else 'FAIL'}  (total_leg={g.total_leg:.4f})")
