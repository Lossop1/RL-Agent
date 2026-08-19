"""从四足 URDF 关节链提取解析式参考生成器所需的几何。

该算法是可选机制，不是所有产品的默认假设。产品插件选择使用它并声明腿标识、
参考腿长和风格锚点；人形或其他形态可以提供自己的参考几何插件。

派生来源(都是 URDF `<joint><origin xyz>`,可靠权威):
  HIPX,HIPY,hip_z = base_link→{leg}_hip_joint
  THIGHY          = {leg}_hip→{leg}_thigh_joint  (y)
  L1,CALFZ        = {leg}_thigh→{leg}_calf_joint  (z)
  FOOT_OFF        = {leg}_calf→{leg}_foot_joint   (xyz)
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from autotuner.adapter._safe_xml import safe_parse_urdf
from typing import List

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
        """返回解析式四足参考生成器使用的几何字段。"""
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


def extract(
    urdf_path: str,
    leg: str = "FL",
    *,
    style: Mapping[str, float] | None = None,
) -> RefGeom:
    """从 URDF 派生参考几何，并应用产品声明的参考风格。"""
    root = safe_parse_urdf(urdf_path)
    hip = _origin(root, f"{leg}_hip_joint")
    thigh = _origin(root, f"{leg}_thigh_joint")
    calf = _origin(root, f"{leg}_calf_joint")
    foot = _origin(root, f"{leg}_foot_joint")

    L1 = abs(calf[2])
    total_leg = L1 + abs(foot[2])  # 粗略矢状腿长(thigh + calf→foot 竖直分量)
    style_data = style or {}
    reference_leg = float(style_data.get("reference_leg_length", total_leg))
    if reference_leg <= 0:
        raise ValueError("reference_leg_length must be positive")
    base_height = float(style_data.get("base_height", 0.0))
    clearance_value = float(style_data.get("swing_clearance", 0.0))
    if base_height <= 0 or clearance_value < 0:
        raise ValueError("reference geometry base_height must be positive and swing_clearance non-negative")
    scale = total_leg / reference_leg
    return RefGeom(
        L1=L1,
        foot_off=foot,
        hipx=abs(hip[0]), hipy=abs(hip[1]), hip_z=hip[2],
        thighy=abs(thigh[1]),
        calfz=calf[2],
        base_z=round(base_height * scale, 4),
        clearance=round(clearance_value * scale, 4),
        total_leg=total_leg,
    )


__all__ = ["RefGeom", "extract"]
