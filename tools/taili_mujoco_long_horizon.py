"""Taili MuJoCo 长时闭环验证工具。

该工具从策略旁的部署契约 JSON 读取观测、控制和时序参数，避免验证脚本与训练配置
各自维护一套常量。MuJoCo 可以使用比训练更小的物理步长，但策略周期、历史更新和
动作延迟必须严格遵守部署契约。
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch


JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

DR_FACTORS = ("mechanics", "contact", "actuation", "sensing", "push")


@dataclass(frozen=True)
class DrScenario:
    """一次可复现的 MuJoCo 域随机化场景。"""

    level: int
    profile: str
    factors: tuple[str, ...]
    seed: int
    mass_delta_kg: float
    com_xy_m: np.ndarray
    friction: float
    stiffness_scale: np.ndarray
    damping_scale: np.ndarray
    action_delay_steps: int
    gyro_bias: np.ndarray
    gravity_bias: np.ndarray
    joint_position_noise: float
    joint_velocity_noise: float
    angular_velocity_noise: float
    gravity_noise: float
    push_interval_s: float
    push_velocity_m_s: float
    push_velocity_fraction: tuple[float, float]
    push_yaw_scale: float

    def as_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "profile": self.profile,
            "factors": list(self.factors),
            "seed": self.seed,
            "mass_delta_kg": self.mass_delta_kg,
            "com_xy_m": self.com_xy_m.tolist(),
            "friction": self.friction,
            "stiffness_scale": self.stiffness_scale.tolist(),
            "damping_scale": self.damping_scale.tolist(),
            "action_delay_steps": self.action_delay_steps,
            "gyro_bias": self.gyro_bias.tolist(),
            "gravity_bias": self.gravity_bias.tolist(),
            "observation_noise": {
                "joint_position_std": self.joint_position_noise,
                "joint_velocity_std": self.joint_velocity_noise,
                "angular_velocity_std": self.angular_velocity_noise,
                "gravity_std": self.gravity_noise,
            },
            "push_interval_s": self.push_interval_s,
            "push_velocity_m_s": self.push_velocity_m_s,
            "push_velocity_fraction": list(self.push_velocity_fraction),
            "push_yaw_scale": self.push_yaw_scale,
        }


def _uniform(rng: np.random.Generator, limits) -> float:
    low, high = (float(value) for value in limits)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"随机化范围无效: {limits!r}")
    return float(rng.uniform(low, high)) if low < high else low


def _parse_dr_factors(raw: str) -> tuple[str, ...]:
    factors = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    unknown = sorted(set(factors) - set(DR_FACTORS))
    if unknown:
        raise ValueError(f"未知 DR 因子: {unknown}; 可选值为 {list(DR_FACTORS)}")
    return factors


def load_dr_scenario(
    path: Path,
    *,
    level: int,
    profile: str,
    factors: tuple[str, ...],
    seed: int,
    nominal_action_delay: int,
) -> DrScenario:
    """从训练侧导出的 DR 契约中采样一个确定场景。"""
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != "taili_mujoco_dr_contract_v1":
        raise ValueError(f"不支持的 DR 契约版本: {spec.get('schema_version')!r}")
    if tuple(spec.get("factor_order", ())) != DR_FACTORS:
        raise ValueError("DR 契约的因子顺序与验证器不一致")
    if level < 0 or level > 3:
        raise ValueError("DR 等级必须位于 0 到 3")
    level_spec = spec["levels"][str(level)]
    rng = np.random.default_rng(seed)

    if profile == "none":
        selected = ()
    elif factors:
        selected = factors
    elif profile == "single":
        selected = (str(rng.choice(DR_FACTORS)),)
    elif profile == "compound":
        count = max(2, min(len(DR_FACTORS) - 1, int(spec.get("compound_factor_count", 2))))
        selected = tuple(str(value) for value in rng.choice(DR_FACTORS, size=count, replace=False))
    elif profile == "full":
        selected = DR_FACTORS
    else:
        raise ValueError(f"未知 DR profile: {profile}")

    expected_count = {"single": 1, "compound": int(spec.get("compound_factor_count", 2))}
    if profile in expected_count and len(selected) != expected_count[profile]:
        raise ValueError(f"{profile} profile 必须选择 {expected_count[profile]} 个 DR 因子")
    if profile == "full" and set(selected) != set(DR_FACTORS):
        raise ValueError("full profile 必须启用全部 DR 因子")

    mechanics = "mechanics" in selected
    contact = "contact" in selected
    actuation = "actuation" in selected
    sensing = "sensing" in selected
    push_enabled = "push" in selected

    mass_delta = _uniform(rng, level_spec["mass_delta_kg"]) if mechanics else 0.0
    com_limit = float(level_spec["com_xy_m"])
    com_xy = rng.uniform(-com_limit, com_limit, size=2) if mechanics else np.zeros(2)
    friction = _uniform(rng, level_spec["friction"]) if contact else 1.0

    if actuation:
        common_k = _uniform(rng, level_spec["stiffness_scale"])
        common_d = _uniform(rng, level_spec["damping_scale"])
        spread = float(level_spec["joint_gain_spread"])
        joint_scale = rng.uniform(1.0 - spread, 1.0 + spread, size=12)
        stiffness_scale = common_k * joint_scale
        damping_scale = common_d * joint_scale
    else:
        stiffness_scale = np.ones(12)
        damping_scale = np.ones(12)

    if sensing:
        delay_low, delay_high = (int(value) for value in level_spec["action_delay_policy_steps"])
        action_delay = int(rng.integers(delay_low, delay_high + 1))
        gyro_limit = float(level_spec["gyro_bias_rad_s"])
        gravity_limit = float(level_spec["gravity_bias"])
        gyro_bias = rng.uniform(-gyro_limit, gyro_limit, size=3)
        gravity_bias = rng.uniform(-gravity_limit, gravity_limit, size=3)
    else:
        action_delay = nominal_action_delay
        gyro_bias = np.zeros(3)
        gravity_bias = np.zeros(3)

    noise = spec["observation_noise"]
    push = spec["push"]
    return DrScenario(
        level=level,
        profile=profile,
        factors=selected,
        seed=seed,
        mass_delta_kg=mass_delta,
        com_xy_m=np.asarray(com_xy, dtype=np.float64),
        friction=friction,
        stiffness_scale=np.asarray(stiffness_scale, dtype=np.float64),
        damping_scale=np.asarray(damping_scale, dtype=np.float64),
        action_delay_steps=action_delay,
        gyro_bias=np.asarray(gyro_bias, dtype=np.float64),
        gravity_bias=np.asarray(gravity_bias, dtype=np.float64),
        joint_position_noise=float(noise["joint_position_std"]),
        joint_velocity_noise=float(noise["joint_velocity_std"]),
        angular_velocity_noise=float(noise["angular_velocity_std"]),
        gravity_noise=float(noise["gravity_std"]),
        push_interval_s=float(level_spec["push_interval_s"]) if push_enabled else 0.0,
        push_velocity_m_s=(
            float(level_spec["push_velocity_m_s"]) * float(push.get("flat_scale", 1.0))
            if push_enabled else 0.0
        ),
        push_velocity_fraction=tuple(float(value) for value in push["linear_velocity_fraction_range"]),
        push_yaw_scale=float(push["yaw_velocity_scale"]),
    )


def rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """将世界系向量按 wxyz 四元数旋转到机体系。"""
    w, xyz = quaternion[0], quaternion[1:]
    return (
        vector * (2.0 * w * w - 1.0)
        - 2.0 * w * np.cross(xyz, vector)
        + 2.0 * xyz * np.dot(xyz, vector)
    )


def rotate_forward(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """将机体系向量按 wxyz 四元数旋转到世界系。"""
    w, xyz = quaternion[0], quaternion[1:]
    return (
        vector * (2.0 * w * w - 1.0)
        + 2.0 * w * np.cross(xyz, vector)
        + 2.0 * xyz * np.dot(xyz, vector)
    )


def quaternion_to_roll_pitch(quaternion: np.ndarray) -> tuple[float, float]:
    """返回 wxyz 四元数相对世界竖直方向的 roll/pitch。"""
    w, x, y, z = (float(value) for value in quaternion)
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def apply_model_dr(model: mujoco.MjModel, scenario: DrScenario | None) -> dict[str, object]:
    """把 episode 静态 DR 写入 MuJoCo 模型，并返回审计信息。"""
    result: dict[str, object] = {"root_body": None, "root_mass_kg": None}
    if scenario is None:
        return result

    free_joints = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
    if len(free_joints) != 1:
        raise ValueError(f"预期模型只有一个 free joint，实际为 {len(free_joints)}")
    root_body = int(model.jnt_bodyid[int(free_joints[0])])
    result["root_body"] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_body)

    if "mechanics" in scenario.factors:
        next_mass = float(model.body_mass[root_body]) + scenario.mass_delta_kg
        if next_mass <= 0.1:
            raise ValueError(f"DR 后 root 质量无效: {next_mass}")
        model.body_mass[root_body] = next_mass
        model.body_ipos[root_body, :2] += scenario.com_xy_m
    if "contact" in scenario.factors:
        # Isaac Lab 使用 multiply 合并模式，地面摩擦为 1；MuJoCo 默认取较大值。
        # 同时写入双方的滑动摩擦，才能得到相同的有效接触系数。
        model.geom_friction[:, 0] = scenario.friction

    result["root_mass_kg"] = float(model.body_mass[root_body])
    result["root_com_xy_m"] = model.body_ipos[root_body, :2].tolist()
    result["sliding_friction_min"] = float(model.geom_friction[:, 0].min())
    result["sliding_friction_max"] = float(model.geom_friction[:, 0].max())
    return result


def _vector(value, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"部署契约字段 {name} 必须是长度 {size} 的有限数值")
    return array


def _mask_standing_gait_clock(
    command: np.ndarray,
    gait_clock: np.ndarray,
    mode: str = "zero",
    linear_threshold: float = 0.1,
    yaw_threshold: float = 0.05,
) -> np.ndarray:
    """Apply the selected zero-command gait-clock deployment behavior."""
    scalar_type = np.asarray(command).dtype.type
    moving = (
        np.linalg.norm(command[:2]) > scalar_type(linear_threshold)
        or np.abs(command[2]) > scalar_type(yaw_threshold)
    )
    if mode not in {"zero", "freeze", "continue"}:
        raise ValueError(f"unsupported standing gait-clock mode: {mode!r}")
    return gait_clock if moving or mode != "zero" else np.zeros_like(gait_clock)


class DeploymentContract:
    """解析并校验导出策略的运行时契约。"""

    def __init__(self, path: Path):
        self.path = path
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != "taili_deployment_contract_v2":
            raise ValueError(f"不支持的部署契约版本: {metadata.get('schema_version')!r}")

        observation = metadata["observation"]
        timing = metadata["timing"]
        command = metadata["command"]
        control = metadata["control"]
        gait = metadata["gait"]

        self.body_dim = int(observation["body_dim"])
        self.history_len = int(observation["history_len"])
        self.tick_dim = int(observation["history_tick_dim"])
        self.history_order = str(observation["history_order"])
        self.history_update_interval = int(observation["history_update_interval_policy_steps"])
        self.input_dim = self.body_dim + self.history_len * self.tick_dim
        if (self.body_dim, self.history_len, self.tick_dim, self.input_dim) != (53, 25, 54, 1403):
            raise ValueError("当前 SAR 仅支持 body53 + history25x54 的 1403 维观测")
        if self.history_order != "newest_first" or self.history_update_interval != 1:
            raise ValueError("当前策略要求历史每个策略周期更新，并按 newest_first 排列")

        self.policy_dt = float(timing["policy_dt"])
        self.action_delay_steps = int(timing["action_delay_policy_steps"])
        if self.policy_dt <= 0.0 or self.action_delay_steps < 0:
            raise ValueError("策略周期和动作延迟必须非负且有效")
        if command.get("transition_enabled") or command.get("smoothing_active"):
            raise ValueError("当前验证器只接受不改写命令的部署契约")

        self.default_q = _vector(control["q_default"], 12, "q_default")
        self.action_scale = _vector(control["action_scale"], 12, "action_scale")
        self.action_lower = _vector(control["action_lower"], 12, "action_lower")
        self.action_upper = _vector(control["action_upper"], 12, "action_upper")
        self.kp = _vector(control["stiffness"], 12, "stiffness")
        self.kd = _vector(control["damping"], 12, "damping")
        self.effort_limit = _vector(control["effort_limit"], 12, "effort_limit")
        self.velocity_limit = _vector(control["velocity_limit"], 12, "velocity_limit")
        self.saturation_effort = float(control["saturation_effort"])
        if np.any(self.action_lower > self.action_upper):
            raise ValueError("动作下界不能大于上界")

        self.gait_period = float(gait["period"])
        self.gait_period_min = float(gait["period_min"])
        self.gait_period_slope = float(gait["period_slope"])
        self.gait_yaw_equiv = float(gait["yaw_speed_equiv"])
        self.gait_offsets = _vector(gait["offsets"], 4, "gait.offsets")


class Controller:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy_path: Path,
        contract: DeploymentContract,
        scenario: DrScenario | None = None,
        seed: int = 0,
        standing_clock: str = "zero",
    ):
        self.model = model
        self.data = data
        self.contract = contract
        self.scenario = scenario
        if standing_clock not in {"zero", "freeze", "continue"}:
            raise ValueError(f"unsupported standing gait-clock mode: {standing_clock!r}")
        self.standing_clock = standing_clock
        self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        self.rng = np.random.default_rng(seed + 1)
        self.command = np.zeros(3, dtype=np.float32)
        self.gait_phase = 0.0
        self.history = np.zeros((contract.history_len, contract.tick_dim), dtype=np.float32)
        self.history_step = 0
        self.last_action = np.zeros(12, dtype=np.float32)
        self.action_delay_steps = (
            scenario.action_delay_steps if scenario is not None else contract.action_delay_steps
        )
        self.action_history = np.zeros((self.action_delay_steps + 1, 12), dtype=np.float32)
        self.applied_action = np.zeros(12, dtype=np.float32)
        self.kp = contract.kp * (
            scenario.stiffness_scale if scenario is not None else np.ones(12)
        )
        self.kd = contract.kd * (
            scenario.damping_scale if scenario is not None else np.ones(12)
        )
        self.gyro_bias = scenario.gyro_bias if scenario is not None else np.zeros(3)
        self.gravity_bias = scenario.gravity_bias if scenario is not None else np.zeros(3)
        self.joint_position_noise = scenario.joint_position_noise if scenario is not None else 0.0
        self.joint_velocity_noise = scenario.joint_velocity_noise if scenario is not None else 0.0
        self.angular_velocity_noise = scenario.angular_velocity_noise if scenario is not None else 0.0
        self.gravity_noise = scenario.gravity_noise if scenario is not None else 0.0
        self.last_torque = np.zeros(12)
        self.last_saturated = np.zeros(12, dtype=bool)
        self.last_action_delta_rms = 0.0
        self.action_clip_count = 0
        self.action_sample_count = 0
        self.qpos_adr = np.asarray([
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in JOINT_NAMES
        ])
        self.dof_adr = np.asarray([
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in JOINT_NAMES
        ])
        self.actuator_ids = np.asarray([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.removesuffix("_joint"))
            for name in JOINT_NAMES
        ])
        self.quat_adr = model.sensor_adr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
        ]
        self.gyro_adr = model.sensor_adr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        ]

    def policy_step(self, target_command: np.ndarray) -> None:
        contract = self.contract
        # 当前目标命令必须在本次推理前进入观测，不能晚一个策略周期。
        self.command[:] = target_command
        # last_action 表示刚完成控制周期中真正施加的动作。
        self.last_action[:] = self.applied_action

        q = self.data.qpos[self.qpos_adr].copy()
        dq = self.data.qvel[self.dof_adr].copy()
        quat = self.data.sensordata[self.quat_adr:self.quat_adr + 4].copy()
        gyro = self.data.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        gravity = rotate_inverse(quat, np.asarray([0.0, 0.0, -1.0]))
        q += self.rng.normal(0.0, self.joint_position_noise, size=12)
        dq += self.rng.normal(0.0, self.joint_velocity_noise, size=12)
        gyro += self.rng.normal(0.0, self.angular_velocity_noise, size=3) + self.gyro_bias
        gravity += self.rng.normal(0.0, self.gravity_noise, size=3) + self.gravity_bias
        q_rel = q - contract.default_q
        q_des_rel = contract.action_scale * self.last_action
        q_error = contract.default_q + q_des_rel - q
        tick = np.concatenate((q_rel, dq, q_des_rel, q_error, gyro, gravity)).astype(np.float32)

        if self.history_step % contract.history_update_interval == 0:
            self.history[1:] = self.history[:-1].copy()
            self.history[0] = tick
        self.history_step += 1

        phases = (self.gait_phase + contract.gait_offsets) % 1.0
        gait = np.concatenate((np.sin(2.0 * math.pi * phases), np.cos(2.0 * math.pi * phases)))
        gait = _mask_standing_gait_clock(
            self.command,
            gait,
            mode=self.standing_clock,
        )
        body = np.concatenate((
            gyro,
            gravity,
            self.command,
            q_rel,
            dq,
            self.last_action,
            gait,
        )).astype(np.float32)
        observation = np.concatenate((body, self.history.reshape(-1))).astype(np.float32)
        if observation.shape != (contract.input_dim,) or not np.isfinite(observation).all():
            raise RuntimeError("策略观测包含非有限值或维度错误")

        with torch.inference_mode():
            raw_action = self.policy(torch.from_numpy(observation).unsqueeze(0)).squeeze(0).numpy()
        if raw_action.shape != (12,) or not np.isfinite(raw_action).all():
            raise RuntimeError("策略输出包含非有限值或维度错误")
        action = np.clip(raw_action, contract.action_lower, contract.action_upper).astype(np.float32)
        self.action_clip_count += int(np.count_nonzero(np.abs(action - raw_action) > 1.0e-6))
        self.action_sample_count += 12

        previous_applied_action = self.applied_action.copy()
        self.action_history[1:] = self.action_history[:-1].copy()
        self.action_history[0] = action
        self.applied_action[:] = self.action_history[self.action_delay_steps]
        self.last_action_delta_rms = float(
            np.sqrt(np.mean(np.square(self.applied_action - previous_applied_action)))
        )

        speed = float(np.linalg.norm(self.command[:2]))
        yaw_speed = abs(float(self.command[2]))
        if speed > 0.1 or yaw_speed > 0.05:
            period = np.clip(
                contract.gait_period
                - contract.gait_period_slope * (speed + contract.gait_yaw_equiv * yaw_speed),
                contract.gait_period_min,
                contract.gait_period,
            )
            self.gait_phase = (self.gait_phase + contract.policy_dt / period) % 1.0
        elif self.standing_clock == "continue":
            self.gait_phase = (
                self.gait_phase + contract.policy_dt / contract.gait_period
            ) % 1.0

    def physics_step(self) -> None:
        contract = self.contract
        q = self.data.qpos[self.qpos_adr]
        dq = self.data.qvel[self.dof_adr]
        q_target = contract.default_q + contract.action_scale * self.applied_action
        raw_torque = self.kp * (q_target - q) - self.kd * dq
        velocity = np.clip(
            dq,
            -contract.velocity_limit * (1.0 + contract.effort_limit / contract.saturation_effort),
            contract.velocity_limit * (1.0 + contract.effort_limit / contract.saturation_effort),
        )
        max_effort = np.minimum(
            contract.saturation_effort * (1.0 - velocity / contract.velocity_limit),
            contract.effort_limit,
        )
        min_effort = np.maximum(
            contract.saturation_effort * (-1.0 - velocity / contract.velocity_limit),
            -contract.effort_limit,
        )
        torque = np.clip(raw_torque, min_effort, max_effort)
        self.last_torque = torque.copy()
        self.last_saturated = np.abs(torque - raw_torque) > 1.0e-5
        self.data.ctrl[self.actuator_ids] = torque
        mujoco.mj_step(self.model, self.data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0))
    parser.add_argument("--stand-seconds", type=float, default=3.0)
    parser.add_argument("--move-seconds", type=float, default=25.0)
    parser.add_argument(
        "--standing-clock",
        choices=("zero", "freeze", "continue"),
        default="zero",
        help="zero-command gait clock presented to the actor",
    )
    parser.add_argument(
        "--stop-seconds",
        type=float,
        default=0.0,
        help="运动段后把命令直接归零并继续仿真的时长。",
    )
    parser.add_argument("--joint-damping", type=float, default=0.0)
    parser.add_argument("--armature", type=float, default=0.0)
    parser.add_argument("--frictionloss", type=float, default=0.0)
    parser.add_argument("--physics-dt", type=float, default=None)
    parser.add_argument(
        "--integrator",
        choices=("euler", "rk4", "implicit", "implicitfast"),
        default=None,
    )
    parser.add_argument("--log-interval", type=float, default=0.25)
    parser.add_argument("--require-stable", action="store_true")
    parser.add_argument(
        "--require-quiet-stop",
        action="store_true",
        help="要求归零后连续 0.5 秒满足安静站立阈值。",
    )
    parser.add_argument("--dr-spec", type=Path, default=None)
    parser.add_argument("--dr-level", type=int, default=3)
    parser.add_argument(
        "--dr-profile",
        choices=("none", "single", "compound", "full"),
        default="none",
    )
    parser.add_argument(
        "--dr-factors",
        default="",
        help="显式指定逗号分隔的 DR 因子；single/compound 可用此参数固定组合。",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    if args.stand_seconds < 0.0 or args.move_seconds <= 0.0 or args.stop_seconds < 0.0:
        raise ValueError("stand-seconds/stop-seconds 必须非负，move-seconds 必须为正")
    if args.require_quiet_stop and args.stop_seconds <= 0.0:
        raise ValueError("require-quiet-stop 要求 stop-seconds 大于 0")
    explicit_factors = _parse_dr_factors(args.dr_factors)
    if args.dr_spec is None and (args.dr_profile != "none" or explicit_factors):
        raise ValueError("启用 DR profile 时必须提供 --dr-spec")

    contract_path = args.contract or args.policy.with_suffix(args.policy.suffix + ".json")
    contract = DeploymentContract(contract_path)
    scenario = None
    if args.dr_spec is not None:
        scenario = load_dr_scenario(
            args.dr_spec,
            level=args.dr_level,
            profile=args.dr_profile,
            factors=explicit_factors,
            seed=args.seed,
            nominal_action_delay=contract.action_delay_steps,
        )
    model = mujoco.MjModel.from_xml_path(str(args.model))
    if args.physics_dt is not None:
        model.opt.timestep = args.physics_dt
    if args.integrator is not None:
        model.opt.integrator = {
            "euler": mujoco.mjtIntegrator.mjINT_EULER,
            "rk4": mujoco.mjtIntegrator.mjINT_RK4,
            "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        }[args.integrator]
    model_dr = apply_model_dr(model, scenario)
    policy_substeps = round(contract.policy_dt / model.opt.timestep)
    if policy_substeps < 1 or not math.isclose(
        policy_substeps * model.opt.timestep,
        contract.policy_dt,
        abs_tol=1.0e-9,
    ):
        raise ValueError("MuJoCo 物理步长必须能整除策略周期")

    data = mujoco.MjData(model)
    if scenario is not None and any(
        factor in scenario.factors for factor in ("mechanics", "contact")
    ):
        mujoco.mj_setConst(model, data)
    controller = Controller(
        model,
        data,
        args.policy,
        contract,
        scenario,
        args.seed,
        standing_clock=args.standing_clock,
    )
    model.dof_damping[controller.dof_adr] = args.joint_damping
    model.dof_armature[controller.dof_adr] = args.armature
    model.dof_frictionloss[controller.dof_adr] = args.frictionloss
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    print(
        f"CONTRACT input={contract.input_dim} policy_dt={contract.policy_dt:.6f} "
        f"physics_dt={model.opt.timestep:.6f} substeps={policy_substeps} "
        f"delay_steps={controller.action_delay_steps} history={contract.history_order}/"
        f"{contract.history_update_interval} standing_clock={args.standing_clock}",
        flush=True,
    )
    if scenario is not None:
        print(
            "DR_SCENARIO " + json.dumps(scenario.as_dict(), ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
        print(
            "MODEL_DR " + json.dumps(model_dr, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
    target = np.asarray(args.command, dtype=np.float64)
    zero = np.zeros(3)
    move_start_time = args.stand_seconds
    stop_start_time = move_start_time + args.move_seconds
    end_time = stop_start_time + args.stop_seconds
    next_log = 0.0
    step = 0
    saturation_samples = 0
    sample_count = 0
    peak_dq = 0.0
    peak_torque = 0.0
    motion_samples = 0
    body_velocity_sum = np.zeros(3)
    gyro_sum = np.zeros(3)
    gyro_sq_sum = np.zeros(3)
    roll_sq_sum = 0.0
    pitch_sq_sum = 0.0
    vertical_velocity_sq_sum = 0.0
    wxy_sq_sum = 0.0
    wxy_peak = 0.0
    linear_error_sq_sum = 0.0
    yaw_error_sq_sum = 0.0
    motion_z_min = math.inf
    motion_z_max = -math.inf
    motion_upright_min = math.inf
    push_count = 0
    push_events: list[dict[str, object]] = []
    push_rng = np.random.default_rng(args.seed + 2)
    next_push = (
        args.stand_seconds + scenario.push_interval_s
        if scenario is not None and scenario.push_interval_s > 0.0
        else math.inf
    )
    stop_samples = 0
    stop_tail_samples = 0
    stop_vxy_sq_sum = 0.0
    stop_wz_sq_sum = 0.0
    stop_wxy_sq_sum = 0.0
    stop_joint_velocity_sq_sum = 0.0
    stop_action_delta_sq_sum = 0.0
    stop_tail_vxy_sq_sum = 0.0
    stop_tail_wz_sq_sum = 0.0
    stop_tail_wxy_sq_sum = 0.0
    stop_tail_joint_velocity_sq_sum = 0.0
    stop_tail_action_delta_sq_sum = 0.0
    stop_tail_action_norm_sq_sum = 0.0
    stop_tail_roll_sq_sum = 0.0
    stop_tail_pitch_sq_sum = 0.0
    stop_tail_vertical_velocity_sq_sum = 0.0
    stop_height_min = math.inf
    stop_height_max = -math.inf
    stop_upright_min = math.inf
    quiet_run_s = 0.0
    quiet_confirmed_s: float | None = None
    stop_tail_start = end_time - min(2.0, args.stop_seconds)
    while data.time < end_time:
        if data.time < move_start_time:
            command = zero
        elif data.time < stop_start_time:
            command = target
        else:
            command = zero
        if data.time < stop_start_time and data.time + 1.0e-9 >= next_push:
            angle = float(push_rng.uniform(0.0, 2.0 * math.pi))
            magnitude = scenario.push_velocity_m_s * float(
                push_rng.uniform(*scenario.push_velocity_fraction)
            )
            delta_body = np.asarray([
                magnitude * math.cos(angle),
                magnitude * math.sin(angle),
                0.0,
            ])
            delta_world = rotate_forward(data.qpos[3:7], delta_body)
            yaw_delta = float(
                push_rng.uniform(
                    -scenario.push_velocity_m_s * scenario.push_yaw_scale,
                    scenario.push_velocity_m_s * scenario.push_yaw_scale,
                )
            )
            data.qvel[:3] += delta_world
            data.qvel[5] += yaw_delta
            push_count += 1
            event = {
                "time_s": float(data.time),
                "delta_body_m_s": delta_body.tolist(),
                "delta_world_m_s": delta_world.tolist(),
                "delta_yaw_rad_s": yaw_delta,
            }
            push_events.append(event)
            print(
                f"PUSH t={data.time:.3f} delta_body={delta_body.round(4).tolist()} "
                f"delta_yaw={yaw_delta:.4f}",
                flush=True,
            )
            next_push += scenario.push_interval_s
        if step % policy_substeps == 0:
            controller.policy_step(command)
        controller.physics_step()
        step += 1
        sample_count += 12
        saturation_samples += int(controller.last_saturated.sum())
        peak_dq = max(peak_dq, float(np.max(np.abs(data.qvel[controller.dof_adr]))))
        peak_torque = max(peak_torque, float(np.max(np.abs(controller.last_torque))))
        quat = data.qpos[3:7]
        body_velocity = rotate_inverse(quat, data.qvel[:3])
        gyro = data.sensordata[controller.gyro_adr:controller.gyro_adr + 3].copy()
        roll, pitch = quaternion_to_roll_pitch(quat)
        upright = float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))
        if move_start_time <= data.time < stop_start_time:
            motion_samples += 1
            body_velocity_sum += body_velocity
            gyro_sum += gyro
            gyro_sq_sum += np.square(gyro)
            roll_sq_sum += roll * roll
            pitch_sq_sum += pitch * pitch
            vertical_velocity_sq_sum += float(body_velocity[2] ** 2)
            wxy = float(np.linalg.norm(gyro[:2]))
            wxy_sq_sum += wxy * wxy
            wxy_peak = max(wxy_peak, wxy)
            linear_error_sq_sum += float(np.dot(body_velocity[:2] - target[:2], body_velocity[:2] - target[:2]))
            yaw_error_sq_sum += float((gyro[2] - target[2]) ** 2)
            motion_z_min = min(motion_z_min, float(data.qpos[2]))
            motion_z_max = max(motion_z_max, float(data.qpos[2]))
            motion_upright_min = min(motion_upright_min, upright)
        elif args.stop_seconds > 0.0 and data.time >= stop_start_time:
            vxy = float(np.linalg.norm(body_velocity[:2]))
            wz = abs(float(gyro[2]))
            wxy = float(np.linalg.norm(gyro[:2]))
            joint_velocity_rms = float(
                np.sqrt(np.mean(np.square(data.qvel[controller.dof_adr])))
            )
            action_delta_rms = float(controller.last_action_delta_rms)
            action_norm_rms = float(np.sqrt(np.mean(np.square(controller.applied_action))))
            stop_samples += 1
            stop_vxy_sq_sum += vxy * vxy
            stop_wz_sq_sum += wz * wz
            stop_wxy_sq_sum += wxy * wxy
            stop_joint_velocity_sq_sum += joint_velocity_rms * joint_velocity_rms
            stop_action_delta_sq_sum += action_delta_rms * action_delta_rms
            stop_height_min = min(stop_height_min, float(data.qpos[2]))
            stop_height_max = max(stop_height_max, float(data.qpos[2]))
            stop_upright_min = min(stop_upright_min, upright)
            if data.time >= stop_tail_start:
                stop_tail_samples += 1
                stop_tail_vxy_sq_sum += vxy * vxy
                stop_tail_wz_sq_sum += wz * wz
                stop_tail_wxy_sq_sum += wxy * wxy
                stop_tail_joint_velocity_sq_sum += joint_velocity_rms * joint_velocity_rms
                stop_tail_action_delta_sq_sum += action_delta_rms * action_delta_rms
                stop_tail_action_norm_sq_sum += action_norm_rms * action_norm_rms
                stop_tail_roll_sq_sum += roll * roll
                stop_tail_pitch_sq_sum += pitch * pitch
                stop_tail_vertical_velocity_sq_sum += float(body_velocity[2] ** 2)

            quiet_now = (
                vxy < 0.05
                and wz < 0.05
                and wxy < 0.15
                and joint_velocity_rms < 0.35
                and upright > 0.95
                and data.qpos[2] > 0.40
            )
            quiet_run_s = quiet_run_s + model.opt.timestep if quiet_now else 0.0
            if quiet_confirmed_s is None and quiet_run_s >= 0.5:
                quiet_confirmed_s = float(data.time - stop_start_time)
        if data.time + 1.0e-9 >= next_log:
            print(
                f"t={data.time:.3f} z={data.qpos[2]:.4f} upright={upright:.4f} "
                f"vbody={body_velocity.round(4).tolist()} gyro={gyro.round(4).tolist()} "
                f"dq_max={np.max(np.abs(data.qvel[controller.dof_adr])):.3f} "
                f"tau_max={np.max(np.abs(controller.last_torque)):.2f} "
                f"sat={controller.last_saturated.mean():.3f}",
                flush=True,
            )
            next_log += args.log_interval
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            break

    quat = data.qpos[3:7]
    upright = float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    stable = finite and motion_z_min > 0.25 and motion_upright_min > 0.4
    mean_body_velocity = body_velocity_sum / max(motion_samples, 1)
    mean_gyro = gyro_sum / max(motion_samples, 1)
    gyro_rms = np.sqrt(gyro_sq_sum / max(motion_samples, 1))
    roll_rms_deg = math.degrees(math.sqrt(roll_sq_sum / max(motion_samples, 1)))
    pitch_rms_deg = math.degrees(math.sqrt(pitch_sq_sum / max(motion_samples, 1)))
    vertical_velocity_rms = math.sqrt(vertical_velocity_sq_sum / max(motion_samples, 1))
    wxy_rms = math.sqrt(wxy_sq_sum / max(motion_samples, 1))
    linear_tracking_rmse = math.sqrt(linear_error_sq_sum / max(motion_samples, 1))
    yaw_tracking_rmse = math.sqrt(yaw_error_sq_sum / max(motion_samples, 1))
    saturation_rate = saturation_samples / max(sample_count, 1)
    action_clip_rate = controller.action_clip_count / max(controller.action_sample_count, 1)
    stop_quiet = bool(
        args.stop_seconds > 0.0
        and finite
        and quiet_confirmed_s is not None
        and stop_height_min > 0.25
        and stop_upright_min > 0.4
    )

    def _rms(square_sum: float, count: int) -> float | None:
        return math.sqrt(square_sum / count) if count > 0 else None

    stop_metrics = {
        "enabled": bool(args.stop_seconds > 0.0),
        "duration_s": float(args.stop_seconds),
        "quiet": stop_quiet,
        "quiet_confirmation_time_s": quiet_confirmed_s,
        "vxy_rms_m_s": _rms(stop_vxy_sq_sum, stop_samples),
        "yaw_rate_rms_rad_s": _rms(stop_wz_sq_sum, stop_samples),
        "body_wxy_rms_rad_s": _rms(stop_wxy_sq_sum, stop_samples),
        "joint_velocity_rms_rad_s": _rms(stop_joint_velocity_sq_sum, stop_samples),
        "action_delta_rms": _rms(stop_action_delta_sq_sum, stop_samples),
        "tail_window_s": float(min(2.0, args.stop_seconds)),
        "tail_vxy_rms_m_s": _rms(stop_tail_vxy_sq_sum, stop_tail_samples),
        "tail_yaw_rate_rms_rad_s": _rms(stop_tail_wz_sq_sum, stop_tail_samples),
        "tail_body_wxy_rms_rad_s": _rms(stop_tail_wxy_sq_sum, stop_tail_samples),
        "tail_joint_velocity_rms_rad_s": _rms(
            stop_tail_joint_velocity_sq_sum, stop_tail_samples
        ),
        "tail_action_delta_rms": _rms(stop_tail_action_delta_sq_sum, stop_tail_samples),
        "tail_action_norm_rms": _rms(stop_tail_action_norm_sq_sum, stop_tail_samples),
        "tail_roll_rms_deg": (
            math.degrees(_rms(stop_tail_roll_sq_sum, stop_tail_samples))
            if stop_tail_samples else None
        ),
        "tail_pitch_rms_deg": (
            math.degrees(_rms(stop_tail_pitch_sq_sum, stop_tail_samples))
            if stop_tail_samples else None
        ),
        "tail_vertical_velocity_rms_m_s": _rms(
            stop_tail_vertical_velocity_sq_sum, stop_tail_samples
        ),
        "minimum_height_m": float(stop_height_min) if stop_samples else None,
        "height_range_m": (
            [float(stop_height_min), float(stop_height_max)] if stop_samples else None
        ),
        "minimum_upright": float(stop_upright_min) if stop_samples else None,
    }
    profile_name = scenario.profile if scenario is not None else "disabled"
    print(
        f"SUMMARY stable={stable} finite={finite} z={data.qpos[2]:.4f} upright={upright:.4f} "
        f"upright_min={motion_upright_min:.4f} "
        f"peak_dq={peak_dq:.3f} peak_tau={peak_torque:.2f} "
        f"saturation_rate={saturation_rate:.6f} action_clip_rate={action_clip_rate:.6f} "
        f"target={target.round(5).tolist()} lin_rmse={linear_tracking_rmse:.5f} "
        f"yaw_rmse={yaw_tracking_rmse:.5f} "
        f"mean_vbody={mean_body_velocity.round(5).tolist()} "
        f"mean_gyro={mean_gyro.round(5).tolist()} gyro_rms={gyro_rms.round(5).tolist()} "
        f"roll_rms_deg={roll_rms_deg:.4f} pitch_rms_deg={pitch_rms_deg:.4f} "
        f"vz_rms={vertical_velocity_rms:.5f} wxy_rms={wxy_rms:.5f} wxy_peak={wxy_peak:.5f} "
        f"z_range={[round(motion_z_min, 5), round(motion_z_max, 5)]} "
        f"push_count={push_count} dr_profile={profile_name} "
        f"damping={args.joint_damping} armature={args.armature} "
        f"frictionloss={args.frictionloss} physics_dt={model.opt.timestep} "
        f"integrator={int(model.opt.integrator)}",
        flush=True,
    )
    if args.stop_seconds > 0.0:
        print(
            "STOP_SUMMARY "
            + json.dumps(stop_metrics, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
    summary = {
        "stable": stable,
        "finite": finite,
        "final_z_m": float(data.qpos[2]),
        "final_upright": upright,
        "minimum_upright": motion_upright_min,
        "peak_joint_velocity_rad_s": peak_dq,
        "peak_torque_nm": peak_torque,
        "torque_saturation_rate": saturation_rate,
        "action_clip_rate": action_clip_rate,
        "target_command": target.tolist(),
        "linear_tracking_rmse_m_s": linear_tracking_rmse,
        "yaw_tracking_rmse_rad_s": yaw_tracking_rmse,
        "mean_body_velocity_m_s": mean_body_velocity.tolist(),
        "mean_body_angular_velocity_rad_s": mean_gyro.tolist(),
        "body_angular_velocity_rms_rad_s": gyro_rms.tolist(),
        "body_roll_rms_deg": roll_rms_deg,
        "body_pitch_rms_deg": pitch_rms_deg,
        "body_vertical_velocity_rms_m_s": vertical_velocity_rms,
        "body_wxy_rms_rad_s": wxy_rms,
        "body_wxy_peak_rad_s": wxy_peak,
        "motion_height_range_m": [motion_z_min, motion_z_max],
        "push_count": push_count,
        "push_events": push_events,
        "stop": stop_metrics,
        "standing_clock": args.standing_clock,
        "deployment_contract": str(contract_path),
        "dr_scenario": scenario.as_dict() if scenario is not None else None,
        "model_dr": model_dr,
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if args.require_stable and not stable:
        raise SystemExit(2)
    if args.require_quiet_stop and not stop_quiet:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
