"""观测契约定义 - P5.1验收标准：所有观测项有明确的物理单位和范围。

此模块定义Taili机器人观测空间的语义契约，包括：
- 每个观测项的物理单位
- 数值范围（原始/归一化前）
- 维度和数据类型
- 组装顺序和依赖关系

参考：
- taili_strategy_decisions.md Runtime IO Contract v1
- taili_obs.py 实际组装实现
- terrain_perceiver_policy.py 策略输入格式
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ObservationComponent:
    """单个观测项的语义规格。"""

    name: str
    dimension: int
    physical_unit: str
    typical_range: tuple[float, float]
    description: str
    normalization_required: bool = True


# ============================================================================
# tick54 观测项（历史tick，25帧历史）
# ============================================================================

TICK54_COMPONENTS = [
    ObservationComponent(
        name="q_rel",
        dimension=12,
        physical_unit="rad",
        typical_range=(-0.35, 0.35),
        description="关节位置相对q_default的偏移量",
        normalization_required=True,
    ),
    ObservationComponent(
        name="dq",
        dimension=12,
        physical_unit="rad/s",
        typical_range=(-10.0, 10.0),
        description="关节角速度",
        normalization_required=True,
    ),
    ObservationComponent(
        name="q_des_rel",
        dimension=12,
        physical_unit="rad",
        typical_range=(-0.35, 0.35),
        description="期望关节位置相对q_default的偏移量（action_scale * last_action）",
        normalization_required=True,
    ),
    ObservationComponent(
        name="q_error",
        dimension=12,
        physical_unit="rad",
        typical_range=(-0.35, 0.35),
        description="伺服跟踪误差（q_des - q）",
        normalization_required=True,
    ),
    ObservationComponent(
        name="gyro",
        dimension=3,
        physical_unit="rad/s",
        typical_range=(-5.0, 5.0),
        description="角速度（IMU陀螺仪）",
        normalization_required=True,
    ),
    ObservationComponent(
        name="projected_gravity",
        dimension=3,
        physical_unit="m/s^2",
        typical_range=(-9.81, 9.81),
        description="投影重力向量（body frame）",
        normalization_required=True,
    ),
]

# ============================================================================
# body57 观测项（当前状态，输入策略actor）
# ============================================================================

BODY57_COMPONENTS = [
    ObservationComponent(
        name="gyro",
        dimension=3,
        physical_unit="rad/s",
        typical_range=(-5.0, 5.0),
        description="角速度（IMU陀螺仪）",
        normalization_required=True,
    ),
    ObservationComponent(
        name="projected_gravity",
        dimension=3,
        physical_unit="m/s^2",
        typical_range=(-9.81, 9.81),
        description="投影重力向量（body frame）",
        normalization_required=True,
    ),
    ObservationComponent(
        name="command",
        dimension=3,
        physical_unit="mixed",
        typical_range=(-0.8, 1.0),
        description="速度命令（vx[m/s], vy[m/s], wz[rad/s]），vx:[-0.45,1.0], vy±0.35, wz±0.8",
        normalization_required=True,
    ),
    ObservationComponent(
        name="previous_command",
        dimension=3,
        physical_unit="mixed",
        typical_range=(-0.8, 1.0),
        description="上一命令（用于检测瞬时换向）",
        normalization_required=True,
    ),
    ObservationComponent(
        name="command_age",
        dimension=1,
        physical_unit="normalized",
        typical_range=(0.0, 1.0),
        description="命令年龄归一化到[0,1]（相对最大过渡窗口）",
        normalization_required=False,  # 已归一化
    ),
    ObservationComponent(
        name="jpos",
        dimension=12,
        physical_unit="rad",
        typical_range=(-0.35, 0.35),
        description="关节位置相对q_default（与tick54中q_rel相同）",
        normalization_required=True,
    ),
    ObservationComponent(
        name="jvel",
        dimension=12,
        physical_unit="rad/s",
        typical_range=(-10.0, 10.0),
        description="关节角速度（与tick54中dq相同）",
        normalization_required=True,
    ),
    ObservationComponent(
        name="last_action",
        dimension=12,
        physical_unit="normalized",
        typical_range=(-1.0, 1.0),
        description="上一步动作输出（策略输出范围[-1,1]）",
        normalization_required=False,  # 已归一化
    ),
    ObservationComponent(
        name="gait_clock",
        dimension=8,
        physical_unit="normalized",
        typical_range=(-1.0, 1.0),
        description="步态时钟（4腿相位的sin/cos编码）",
        normalization_required=False,  # 已归一化
    ),
]

# ============================================================================
# amp_frame51 观测项（AMP参考运动，仅训练时使用）
# ============================================================================

AMP_FRAME51_COMPONENTS = [
    ObservationComponent(
        name="root_pos_z",
        dimension=1,
        physical_unit="m",
        typical_range=(0.25, 0.45),
        description="质心高度",
        normalization_required=True,
    ),
    ObservationComponent(
        name="root_linvel",
        dimension=3,
        physical_unit="m/s",
        typical_range=(-1.0, 1.0),
        description="质心线速度（world frame）",
        normalization_required=True,
    ),
    ObservationComponent(
        name="root_angvel",
        dimension=3,
        physical_unit="rad/s",
        typical_range=(-5.0, 5.0),
        description="质心角速度（body frame）",
        normalization_required=True,
    ),
    ObservationComponent(
        name="root_quat",
        dimension=4,
        physical_unit="quat",
        typical_range=(-1.0, 1.0),
        description="质心姿态四元数（world frame）",
        normalization_required=False,  # 单位四元数
    ),
    ObservationComponent(
        name="joint_pos",
        dimension=12,
        physical_unit="rad",
        typical_range=(-0.35, 0.35),
        description="关节位置相对q_default",
        normalization_required=True,
    ),
    ObservationComponent(
        name="joint_vel",
        dimension=12,
        physical_unit="rad/s",
        typical_range=(-10.0, 10.0),
        description="关节角速度",
        normalization_required=True,
    ),
    ObservationComponent(
        name="foot_pos_rel",
        dimension=12,
        physical_unit="m",
        typical_range=(-0.5, 0.5),
        description="足端位置相对质心（body frame，4 feet * 3D）",
        normalization_required=True,
    ),
    ObservationComponent(
        name="foot_vel",
        dimension=4,
        physical_unit="m/s",
        typical_range=(-2.0, 2.0),
        description="足端速度标量（body frame，4 feet）",
        normalization_required=True,
    ),
]


# ============================================================================
# 组合观测空间
# ============================================================================

@dataclass(frozen=True)
class ObservationSpace:
    """完整观测空间定义。"""

    name: str
    components: list[ObservationComponent]
    total_dimension: int
    assembly_order: str
    privileged: bool = False

    def validate_dimension(self) -> bool:
        """验证维度一致性。"""
        computed_dim = sum(c.dimension for c in self.components)
        return computed_dim == self.total_dimension

    def components_requiring_normalization(self) -> list[ObservationComponent]:
        """返回需要归一化的观测项。"""
        return [c for c in self.components if c.normalization_required]


# 定义三种观测空间
TICK54_SPACE = ObservationSpace(
    name="tick54",
    components=TICK54_COMPONENTS,
    total_dimension=54,
    assembly_order="q_rel[12] | dq[12] | q_des_rel[12] | q_error[12] | gyro[3] | gravity[3]",
    privileged=False,
)

BODY57_SPACE = ObservationSpace(
    name="body57",
    components=BODY57_COMPONENTS,
    total_dimension=57,
    assembly_order="gyro[3] | gravity[3] | cmd[3] | prev_cmd[3] | age[1] | jpos[12] | jvel[12] | action[12] | clock[8]",
    privileged=False,
)

AMP_FRAME51_SPACE = ObservationSpace(
    name="amp_frame51",
    components=AMP_FRAME51_COMPONENTS,
    total_dimension=51,
    assembly_order="z[1] | linvel[3] | angvel[3] | quat[4] | jpos[12] | jvel[12] | foot_pos[12] | foot_vel[4]",
    privileged=True,  # 训练时特权信息
)

# 策略完整输入：body57 + history(25 * tick54) = 1407
ACTOR_OBS_SPACE = ObservationSpace(
    name="actor_obs",
    components=BODY57_COMPONENTS + [
        ObservationComponent(
            name="tick_history",
            dimension=25 * 54,
            physical_unit="mixed",
            typical_range=(-10.0, 10.0),
            description="25帧tick54历史（最老在前）",
            normalization_required=True,
        )
    ],
    total_dimension=1407,
    assembly_order="body57[57] | history[25*54]",
    privileged=False,
)


# ============================================================================
# 验证函数
# ============================================================================

def validate_all_spaces() -> dict[str, bool]:
    """验证所有观测空间定义的一致性。"""
    spaces = [TICK54_SPACE, BODY57_SPACE, AMP_FRAME51_SPACE, ACTOR_OBS_SPACE]
    return {space.name: space.validate_dimension() for space in spaces}


def print_observation_spec(space: ObservationSpace) -> None:
    """打印观测空间的完整规格（用于文档生成）。"""
    print(f"\n{'='*80}")
    print(f"观测空间: {space.name}")
    print(f"总维度: {space.total_dimension}")
    print(f"组装顺序: {space.assembly_order}")
    print(f"特权信息: {'是' if space.privileged else '否'}")
    print(f"{'='*80}\n")

    for comp in space.components:
        norm_mark = "[需归一化]" if comp.normalization_required else "[已归一化]"
        print(f"{comp.name:20s} dim={comp.dimension:2d}  {norm_mark:10s}")
        print(f"  单位: {comp.physical_unit}")
        print(f"  范围: {comp.typical_range}")
        print(f"  说明: {comp.description}")
        print()


__all__ = [
    "ObservationComponent",
    "ObservationSpace",
    "TICK54_SPACE",
    "BODY57_SPACE",
    "AMP_FRAME51_SPACE",
    "ACTOR_OBS_SPACE",
    "TICK54_COMPONENTS",
    "BODY57_COMPONENTS",
    "AMP_FRAME51_COMPONENTS",
    "validate_all_spaces",
    "print_observation_spec",
]
