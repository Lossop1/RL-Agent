"""控制台的机器人档案兼容入口。

真正的产品档案由 ``autotuner.product`` 管理。保留本模块是为了兼容旧的
控制台导入路径，但这里不再保存任何具体机器人的默认结构。
"""
from __future__ import annotations

from typing import Literal

from autotuner.product import RobotProfile, get_product


RobotProfileStatus = Literal["draft", "validated", "missing"]


def get_robot_profile(robot_id: str | None = None) -> RobotProfile:
    """返回显式或当前产品的机器人档案。"""
    return get_product(robot_id).robot_profile()


def validate_robot_profile(profile: RobotProfile | None = None) -> list[str]:
    """只校验通用本体契约，不假设腿数、DoF 或传感器类型。"""
    if profile is None:
        profile = get_robot_profile()
    issues: list[str] = []
    if not profile.id.strip():
        issues.append("robot id is required")
    if profile.dof <= 0:
        issues.append("dof must be positive")
    if profile.dof != len(profile.joint_order):
        issues.append(f"dof={profile.dof} but joint_order has {len(profile.joint_order)} joints")
    if len(set(profile.joint_order)) != len(profile.joint_order):
        issues.append("joint_order contains duplicates")
    if not profile.base_link.strip():
        issues.append("base_link is required")
    if len(set(profile.foot_links)) != len(profile.foot_links):
        issues.append("foot_links contains duplicates")
    if len(set(profile.leg_order)) != len(profile.leg_order):
        issues.append("leg_order contains duplicates")
    return issues
