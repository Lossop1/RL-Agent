"""Robot profile registry for locomotion console configuration.

V0 keeps Taili as a typed profile instead of a hardcoded paragraph. The profile
is intentionally conservative: identity and mapping are inspectable and
validatable, but not arbitrary-editable until the framework/profile contract is
stable enough to protect users from broken body mappings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


RobotProfileStatus = Literal["draft", "validated", "missing"]


@dataclass(frozen=True)
class RobotProfile:
    id: str
    label: str
    status: RobotProfileStatus
    dof: int
    base_link: str
    joint_order: tuple[str, ...]
    leg_order: tuple[str, ...]
    foot_links: tuple[str, ...]
    diagnostic_spec: str
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


TAILI_PROFILE = RobotProfile(
    id="robot.taili",
    label="Taili quadruped",
    status="draft",
    dof=12,
    base_link="base",
    joint_order=(
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ),
    leg_order=("FL", "FR", "RL", "RR"),
    foot_links=("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
    diagnostic_spec="taili.yaml",
    capabilities=("quadruped_12dof", "blind_actor", "amp_reference"),
    note="URDF/asset mapping is usable for diagnostics, but still marked draft until hardware mapping is fully audited.",
)


def get_robot_profile(robot_id: str = "robot.taili") -> RobotProfile:
    if robot_id != TAILI_PROFILE.id:
        raise ValueError(f"Unknown robot profile: {robot_id}")
    return TAILI_PROFILE


def validate_robot_profile(profile: RobotProfile = TAILI_PROFILE) -> list[str]:
    issues: list[str] = []
    if profile.dof != len(profile.joint_order):
        issues.append(f"dof={profile.dof} but joint_order has {len(profile.joint_order)} joints")
    if len(profile.leg_order) != 4:
        issues.append("quadruped profile must define four legs")
    if len(profile.foot_links) != 4:
        issues.append("quadruped profile must define four foot links")
    missing_caps = {"quadruped_12dof"} - set(profile.capabilities)
    if missing_caps:
        issues.append(f"missing capabilities: {', '.join(sorted(missing_caps))}")
    if not profile.diagnostic_spec:
        issues.append("diagnostic_spec is required")
    return issues
