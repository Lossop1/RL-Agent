"""盲态 TerrainPerceiver 训练环境。

该类继承 TailiAmpEnv，但不只是替换观测。当前盲态 payload 的关键训练语义在这里：
- 观测组装为 Runtime-IO 契约：body57 + tick54 历史 + critic 特权信息 + 辅助标签。
- 命令采样增加阶段课程、命令保持时间打散和过渡控制。
- 奖励路径覆盖父类，统一调用 taili_reward，并补充盲态地形、触地、质量窗口和遥测。
- 分方向 progress 在这里计算；父类 _log_training_diag 读取这些值做 phase gate、DR gate 和地形 gate。
- reset 基础流程仍调用父类，并补充历史观测、质量窗口和连续地形物理量复位。

远端 payload 以 taili_blind_runtime 包形式部署，只需要该包出现在 PYTHONPATH 中。
"""
import math
import os
import types

import numpy as np
import torch
# 旧版叫 quat_apply_inverse，这个 IsaacLab 版本（0.36.21）改名为 quat_rotate_inverse。
# 等价是数值验过的：4096 组随机单位四元数下与 quat_apply(quat_inv(q), v) 最大差 1.07e-06。
from isaaclab.utils.math import quat_rotate_inverse

try:                                            # payload 包内导入。
    from .taili_core import (taili_obs, taili_symmetry as _taili_symmetry, taili_amp_reference,
                             taili_reward, taili_terrain_labels, taili_curriculum,
                             terrain_curriculum, taili_geometry)
except ImportError:
    if __package__ == "taili_blind_runtime":
        raise
    try:                                        # 本地源码树导入。
        from products.taili.core import (taili_obs, taili_symmetry as _taili_symmetry, taili_amp_reference,
                                          taili_reward, taili_terrain_labels, taili_curriculum,
                                          terrain_curriculum, taili_geometry)
    except ImportError:
        from taili_core import (taili_obs, taili_symmetry as _taili_symmetry, taili_amp_reference,
                                taili_reward, taili_terrain_labels, taili_curriculum,
                                terrain_curriculum, taili_geometry)

# 核心对称模块随环境载入，保留其兼容性与模块初始化行为。

try:
    from .telemetry_emit import TrainingTelemetryEmitter
except Exception:  # pragma: no cover - remote deployment may copy files differently
    try:
        from telemetry_emit import TrainingTelemetryEmitter
    except Exception:
        try:
            if __package__ == "taili_blind_runtime":
                raise
            from products.taili.blind_locomotion.telemetry_emit import TrainingTelemetryEmitter
        except Exception:
            TrainingTelemetryEmitter = None

try:
    from .telemetry_payloads import (
        build_checkpoint_performance_snapshot,
        build_command_payload,
        build_curriculum_payload,
        build_health_payload,
        build_reward_payload,
    )
except Exception:  # pragma: no cover - payload 包和本地源码树的导入路径不同。
    try:
        from telemetry_payloads import (
            build_checkpoint_performance_snapshot,
            build_command_payload,
            build_curriculum_payload,
            build_health_payload,
            build_reward_payload,
        )
    except Exception:
        if __package__ == "taili_blind_runtime":
            raise
        from products.taili.blind_locomotion.telemetry_payloads import (
            build_checkpoint_performance_snapshot,
            build_command_payload,
            build_curriculum_payload,
            build_health_payload,
            build_reward_payload,
        )

try:
    from .taili_blind_config import (
        active_direction_progress,
        phase_command_spec,
        single_axis_occupancy_deficits,
    )
except Exception:  # pragma: no cover - local fallback for unusual import layouts
    try:
        from products.taili.blind_locomotion.taili_blind_config import (
            active_direction_progress,
            phase_command_spec,
            single_axis_occupancy_deficits,
        )
    except Exception:
        active_direction_progress = None
        phase_command_spec = None
        single_axis_occupancy_deficits = None

from .taili_amp_env import (
    TailiAmpEnv,
    quaternion_to_tangent_and_normal,
    _command_xy_world_from_root_yaw,
    _sample_bucketed_commands,
    _sample_core_full_mixture,
)
from .parametric_ref import flat_reference, foot_reference   # live imitation 与统一足端参考。

_ = _taili_symmetry

HIST_LEN = 25
TICK_DIM = 54


def _update_direction_progress(env, name: str, sample: float, sample_count: int) -> None:
    """保留当前批次读数，并用稳定EMA更新课程使用的方向能力。"""
    setattr(env, f"_{name}_progress_instant", float(sample))
    setattr(env, f"_{name}_progress_samples", int(sample_count))
    initialized = bool(env._progress_ema_initialized.get(name, False))
    value, initialized, effective_alpha = taili_curriculum.update_sample_weighted_ema(
        getattr(env, f"_{name}_prog", 0.0),
        sample,
        sample_count,
        alpha=float(getattr(env.cfg, "progress_ema_alpha", 0.01)),
        min_samples=int(getattr(env.cfg, "progress_ema_min_samples", 8)),
        reference_samples=int(getattr(env.cfg, "progress_ema_reference_samples", 16)),
        initialized=initialized,
    )
    setattr(env, f"_{name}_prog", value)
    setattr(env, f"_{name}_progress_ema_alpha", effective_alpha)
    env._progress_ema_initialized[name] = initialized


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _spec_float(spec, key: str, default: float) -> float:
    try:
        return float(spec.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _spec_range(spec, key: str, fallback) -> tuple[float, float]:
    value = spec.get(key, fallback)
    try:
        lo, hi = value
        return float(lo), float(hi)
    except (TypeError, ValueError):
        lo, hi = fallback
        return float(lo), float(hi)


def _sample_uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    if hi < lo:
        lo, hi = hi, lo
    if abs(hi - lo) < 1e-9:
        return torch.full((n,), float(lo), device=device)
    return torch.rand(n, device=device) * (hi - lo) + lo


class TailiBlindTPEnv(TailiAmpEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 历史顺序和采样间隔属于checkpoint语义，旧检查点必须沿用训练时配置。
        self._tick_history = torch.zeros((self.num_envs, HIST_LEN, TICK_DIM), device=self.device)
        self._history_stride = max(1, int(getattr(self.cfg, "obs_history_stride", 1)))
        self._history_order = str(getattr(self.cfg, "obs_history_order", "newest_first"))
        self._rcfg = taili_reward.reward_cfg_from_env()          # 奖励配置；可用 TAILI_RW_* 环境变量临时覆盖。
        self._reward_cfg_printed = False
        self._prev_in_contact = None                            # 用于检测摆动到支撑的触地事件。
        self._prev_support_contact = None                       # 仅承重接触；立面碰撞不能进入换层链路。
        self._td_impact = _env_flag("TAILI_TD_IMPACT", bool(getattr(self.cfg, "touchdown_impact_only", False)))
        self._telemetry = TrainingTelemetryEmitter() if TrainingTelemetryEmitter is not None else None
        # P4.2检查点管理集成
        self._checkpoint_integration = None
        self._latest_performance_snapshot = None
        self._hip_joint_ids, _ = self.robot.find_joints(".*_hip_joint")
        # 接触传感器覆盖整机。逐腿保留 hip/thigh/calf，足端仍走独立的承重与
        # 水平碰撞链；base 接触按当前平移命令映射到迎障腿。
        self._limb_contact_ids = []
        for leg in ("FL", "FR", "RL", "RR"):
            ids, _ = self._contact_sensor.find_bodies(f"{leg}_(hip|thigh|calf)")
            self._limb_contact_ids.append([int(body_id) for body_id in ids])
        base_ids, _ = self._contact_sensor.find_bodies("base_link")
        self._base_contact_ids = [int(body_id) for body_id in base_ids]
        self._stagger_pending = True     # 初始批量 reset 后做一次 episode 相位打散。
        self._cmd_hold = None            # 每个 env 独立的命令保持步数，由 _get_observations 采样。
        self._quality_duty_ema = torch.full((self.num_envs, 4), 0.5, device=self.device)
        self._quality_diag_pair_ema = torch.full((self.num_envs,), 0.5, device=self.device)
        self._quality_duty_dir_ema = torch.full((self.num_envs, 4, 4), 0.5, device=self.device)
        self._quality_diag_dir_ema = torch.full((self.num_envs, 4), 0.5, device=self.device)
        self._quality_duty_dir_valid = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self._quality_slip_speed_ema = torch.zeros(self.num_envs, device=self.device)
        # 严格质量约束从训练开始保留基础强度，并且只随能力单调增强。
        self._quality_gate_latched = 0.0
        self._refinement_gate_latched = 0.0
        self._quality_slip_excess_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_slip_high_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_slip_by_leg_ema = torch.zeros((self.num_envs, 4), device=self.device)
        self._quality_wxy_energy_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_vz_energy_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_yaw_residual_energy_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_contact_chatter_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_support_force_ema = torch.zeros((self.num_envs, 4), device=self.device)
        self._quality_height_low_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_tilt_high_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_window_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._best_lag_gait_match = 0.0
        self._gait_match_zero_lag = 0.0
        self._gait_lag_scores = torch.full((self.num_envs, 9), 0.5, device=self.device)
        self._last_touchdown_step = torch.full(
            (self.num_envs, 4), -1, dtype=torch.long, device=self.device
        )
        self._last_touchdown_step_all = torch.full(
            (self.num_envs, 4), -1, dtype=torch.long, device=self.device
        )
        self._contact_period_ema = torch.zeros(self.num_envs, device=self.device)
        self._contact_period_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._contact_period_dir_ema = torch.zeros((self.num_envs, 4), device=self.device)
        self._contact_period_dir_valid = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        # 分方向终止率按完整日志窗口累计，避免阶段门控只看到日志触发的单帧。
        self._direction_terminal_event_count = torch.zeros(4, device=self.device)
        self._direction_terminal_target_count = torch.zeros(4, device=self.device)
        self._duty_cycle_sum = torch.zeros((self.num_envs, 4), device=self.device)
        self._duty_cycle_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._duty_cycle_direction = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._duty_cycle_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._duty_cycle_valid_frac = 0.0
        self._quality_step = 0
        self._prev_foot_force_norm = torch.zeros((self.num_envs, 4), device=self.device)
        self._prev_foot_vz = torch.zeros((self.num_envs, 4), device=self.device)
        self._prev_foot_vel_w = self.robot.data.body_lin_vel_w[:, self.foot_indexes, :].detach().clone()
        self._prev_action_delta = torch.zeros((self.num_envs, 12), device=self.device)
        self._prev_base_ang_vel = self.robot.data.root_ang_vel_b.detach().clone()
        self._base_ang_vel_filtered = self.robot.data.root_ang_vel_b.detach().clone()
        self._touchdown_hold_steps = torch.zeros((self.num_envs, 4), dtype=torch.long, device=self.device)
        self._touchdown_vz_hold = torch.zeros((self.num_envs, 4), device=self.device)
        self._touchdown_xy_hold = torch.zeros((self.num_envs, 4), device=self.device)
        self._touchdown_force_hold = torch.zeros((self.num_envs, 4), device=self.device)
        self._prev_transition_fault = torch.ones(self.num_envs, device=self.device)
        self._terrain_pattern_scale_for_amp = torch.ones(self.num_envs, device=self.device)
        self._amp_style_scale_for_agent = torch.ones(self.num_envs, device=self.device)
        self._blind_amp_terrain_response = torch.zeros(self.num_envs, device=self.device)
        self._terrain_leg_response = torch.zeros((self.num_envs, 4), device=self.device)
        self._terrain_collision_trace = torch.zeros((self.num_envs, 4), device=self.device)
        self._terrain_limb_collision_response = torch.zeros((self.num_envs, 4), device=self.device)
        self._terrain_base_collision_response = torch.zeros(self.num_envs, device=self.device)
        self._terrain_support_delta = torch.zeros(self.num_envs, device=self.device)
        self._terrain_support_dispersion = torch.zeros(self.num_envs, device=self.device)
        self._terrain_support_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._terrain_contact_foot_mask = torch.zeros(
            (self.num_envs, 4), dtype=torch.bool, device=self.device
        )
        self._prev_support_z = self._terrain_height_under_base().detach().clone()
        self._terrain_prev_base_z = self.robot.data.root_pos_w[:, 2].detach().clone()
        _initial_foot_pos = self.robot.data.body_pos_w[:, self.foot_indexes, :].detach()
        _world_up = torch.zeros((self.num_envs, 3), device=self.device)
        _world_up[:, 2] = 1.0
        self._support_reference_point = (
            _initial_foot_pos.mean(dim=1) - taili_geometry.FOOT_RADIUS * _world_up
        )
        self._support_reference_normal = _world_up.clone()
        self._restore_curriculum_state()

    def _resample_commands(self, env_ids, *args, **kwargs):
        # 课程命令覆盖：用于强制早期学习命令条件，而不是学成与命令无关的爬行/站立。
        # TAILI_CUR_FIXED_VX 会把所有命令固定为 (vx, 0, 0)；未设置时使用正常采样。
        super()._resample_commands(env_ids, *args, **kwargs)
        if len(env_ids) == 0:
            return
        # 父类已经把系统诊断写入的目标命令同步到 commands；实际任务层不能
        # 随后再用分层训练分布覆盖它，否则 reset 姿态与诊断标签不一致。
        if bool(getattr(self, "use_external_commands", False)):
            return
        snap = bool(kwargs.get("snap", args[0] if args else False))
        phase = int(getattr(self, "_phase", getattr(self.cfg, "init_phase", 0)))
        spec = phase_command_spec(self.cfg, phase) if phase_command_spec is not None else {}
        mode = str(spec.get("command_mode") or getattr(self.cfg, "training_command_mode", "normal") or "normal")
        self._last_command_mode = mode
        self._last_command_spec = spec

        def _sample_command_range(name: str, n: int, lo: float, hi, device) -> torch.Tensor:
            return _sample_core_full_mixture(
                n,
                lo,
                hi,
                getattr(self.cfg, f"cmd_core_{name}_range", (lo, hi)),
                float(getattr(self.cfg, "cmd_core_sample_fraction", 0.0)),
                device,
            )

        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        stair_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        stair_local_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        try:
            self._ensure_gate_mask()
            stair_mask = self._stairs_up_terrain_mask | self._stairs_down_terrain_mask
            stair_local_mask = stair_mask[env_ids_t]
        except Exception:
            pass

        def _force_stair_forward(target: torch.Tensor) -> torch.Tensor:
            """楼梯列只采样普通前进命令，Actor 不接收地形类型。"""
            if not bool(stair_local_mask.any()):
                return target
            f_lo, f_hi = _spec_range(
                spec,
                "fwd_range",
                getattr(self.cfg, "cmd_fwd_range", (0.15, 1.0)),
            )
            f_hi = min(f_hi, float(getattr(self, "_vel_max_fwd", f_hi)))
            sampled = _sample_command_range(
                "fwd", len(env_ids), f_lo, max(f_lo, f_hi), self.device
            )
            target = target.clone()
            target[stair_local_mask] = 0.0
            target[stair_local_mask, 0] = sampled[stair_local_mask]
            return target

        try:
            target = None
            fv_override = os.environ.get("TAILI_CUR_FIXED_VX")
            lo_override, hi_override = os.environ.get("TAILI_CUR_FWD_LO"), os.environ.get("TAILI_CUR_FWD_HI")
            if fv_override is not None:
                target = torch.zeros((len(env_ids), 3), device=self.device)
                target[:, 0] = float(fv_override)
                self._last_command_mode = "env_fixed_forward"
            elif lo_override is not None and hi_override is not None:
                lo_v, hi_v = float(lo_override), float(hi_override)
                target = torch.zeros((len(env_ids), 3), device=self.device)
                target[:, 0] = torch.rand(len(env_ids), device=self.device) * (hi_v - lo_v) + lo_v
                self._last_command_mode = "env_forward_range"
            elif mode == "fixed_forward":
                target = torch.zeros((len(env_ids), 3), device=self.device)
                target[:, 0] = _spec_float(spec, "fixed_vx", float(getattr(self.cfg, "training_fixed_vx", 0.5)))
            elif mode == "forward_range":
                lo_v, hi_v = _spec_range(spec, "forward_range", getattr(self.cfg, "training_forward_range", (0.3, 0.7)))
                target = torch.zeros((len(env_ids), 3), device=self.device)
                target[:, 0] = torch.rand(len(env_ids), device=self.device) * (hi_v - lo_v) + lo_v
            elif mode == "stand_only":
                target = torch.zeros((len(env_ids), 3), device=self.device)
            elif mode == "single_axis":
                n, dev = len(env_ids), self.device
                target = torch.zeros((n, 3), device=dev)
                stand_prob = _spec_float(spec, "stand_prob", float(getattr(self.cfg, "stand_prob", 0.0)))
                weights = torch.tensor([
                    max(0.0, _spec_float(spec, "prob_fwd", float(getattr(self.cfg, "cmd_prob_fwd", 0.25)))),
                    max(0.0, _spec_float(spec, "prob_back", float(getattr(self.cfg, "cmd_prob_back", 0.25)))),
                    max(0.0, _spec_float(spec, "prob_lat", float(getattr(self.cfg, "cmd_prob_lat", 0.25)))),
                    max(0.0, _spec_float(spec, "prob_yaw", float(getattr(self.cfg, "cmd_prob_yaw", 0.25)))),
                ], device=dev)
                if float(weights.sum()) <= 1e-9:
                    weights[:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
                categories = None
                if single_axis_occupancy_deficits is not None:
                    resample_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
                    resample_mask[env_ids] = True
                    current = self._cmd_target
                    current_categories = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
                    current_categories = torch.where(current[:, 0] > 0.05, 1, current_categories)
                    current_categories = torch.where(current[:, 0] < -0.05, 2, current_categories)
                    current_categories = torch.where(current[:, 1].abs() > 0.05, 3, current_categories)
                    current_categories = torch.where(current[:, 2].abs() > 0.05, 4, current_categories)
                    # 楼梯环境必须前进，但不能因此占用平地的前进配额。先在非楼梯
                    # stratum 内维持四方向存量，再把本批楼梯环境额外固定为前进。
                    non_stair_global = ~stair_mask
                    surviving = torch.bincount(
                        current_categories[non_stair_global & ~resample_mask],
                        minlength=5,
                    )
                    non_stair_total = int(non_stair_global.sum().item())
                    deficits = list(single_axis_occupancy_deficits(
                        non_stair_total,
                        stand_prob,
                        tuple(float(value) for value in weights.tolist()),
                        tuple(int(value) for value in surviving.tolist()),
                    ))
                    pool = torch.repeat_interleave(
                        torch.arange(5, device=dev),
                        torch.tensor(deficits, dtype=torch.long, device=dev),
                    )
                    non_stair = ~stair_local_mask
                    non_stair_count = int(non_stair.sum().item())
                    if pool.numel() >= non_stair_count:
                        categories = torch.ones(n, dtype=torch.long, device=dev)
                        if non_stair_count > 0:
                            categories[non_stair] = pool[
                                torch.randperm(pool.numel(), device=dev)[:non_stair_count]
                            ]
                if categories is None:
                    active = torch.rand(n, device=dev) >= stand_prob
                    u = torch.rand(n, device=dev) * weights.sum()
                    c0 = weights[0]
                    c1 = c0 + weights[1]
                    c2 = c1 + weights[2]
                    categories = torch.zeros(n, dtype=torch.long, device=dev)
                    categories = torch.where(active & (u < c0), 1, categories)
                    categories = torch.where(active & (u >= c0) & (u < c1), 2, categories)
                    categories = torch.where(active & (u >= c1) & (u < c2), 3, categories)
                    categories = torch.where(active & (u >= c2), 4, categories)
                fwd = categories == 1
                back = categories == 2
                lat = categories == 3
                yaw = categories == 4
                f_lo, f_hi = _spec_range(spec, "fwd_range", getattr(self.cfg, "cmd_fwd_range", (0.3, 0.7)))
                b_lo, b_hi = _spec_range(spec, "back_range", getattr(self.cfg, "cmd_back_range", (0.2, 0.45)))
                l_lo, l_hi = _spec_range(spec, "lat_range", getattr(self.cfg, "cmd_lat_range", (0.15, 0.35)))
                y_lo, y_hi = _spec_range(spec, "yaw_range", getattr(self.cfg, "cmd_yaw_range", (0.25, 0.8)))
                f_hi = min(f_hi, float(getattr(self, "_vel_max_fwd", f_hi)))
                b_hi = min(b_hi, float(getattr(self, "_vel_max_back", b_hi)))
                l_hi = min(l_hi, float(getattr(self, "_vel_max_lat", l_hi)))
                y_hi = min(y_hi, float(getattr(self, "_vel_max_yaw", y_hi)))
                target[:, 0] = torch.where(
                    fwd, _sample_command_range("fwd", n, f_lo, f_hi, dev), target[:, 0]
                )
                target[:, 0] = torch.where(
                    back, -_sample_command_range("back", n, b_lo, b_hi, dev), target[:, 0]
                )
                lat_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                yaw_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                target[:, 1] = torch.where(
                    lat, lat_sign * _sample_command_range("lat", n, l_lo, l_hi, dev), target[:, 1]
                )
                target[:, 2] = torch.where(
                    yaw, yaw_sign * _sample_command_range("yaw", n, y_lo, y_hi, dev), target[:, 2]
                )
            elif mode == "bucketed" or (mode == "mixed" and hasattr(spec.get("bucket_probs"), "items")):
                n, dev = len(env_ids), self.device
                f_lo, f_hi = _spec_range(spec, "fwd_range", getattr(self.cfg, "cmd_fwd_range", (0.3, 0.7)))
                b_lo, b_hi = _spec_range(spec, "back_range", getattr(self.cfg, "cmd_back_range", (0.2, 0.45)))
                l_lo, l_hi = _spec_range(spec, "lat_range", getattr(self.cfg, "cmd_lat_range", (0.15, 0.35)))
                y_lo, y_hi = _spec_range(spec, "yaw_range", getattr(self.cfg, "cmd_yaw_range", (0.25, 0.8)))
                f_hi = min(f_hi, float(getattr(self, "_vel_max_fwd", f_hi)))
                b_hi = min(b_hi, float(getattr(self, "_vel_max_back", b_hi)))
                l_hi = min(l_hi, float(getattr(self, "_vel_max_lat", l_hi)))
                y_hi = min(y_hi, float(getattr(self, "_vel_max_yaw", y_hi)))
                target = _sample_bucketed_commands(
                    n, dev, spec,
                    ((f_lo, f_hi), (b_lo, b_hi), (l_lo, l_hi), (y_lo, y_hi)),
                    sample_value=lambda name, lo, hi: _sample_command_range(
                        name, n, lo, hi, dev
                    ),
                )
                if target is None:
                    target = torch.zeros((n, 3), device=dev)
            elif mode == "mixed":
                n, dev = len(env_ids), self.device
                target = torch.zeros((n, 3), device=dev)
                stand_prob = _spec_float(spec, "stand_prob", float(getattr(self.cfg, "stand_prob", 0.0)))
                near_zero_prob = _spec_float(spec, "near_zero_prob", 0.0)
                axis_prob = _spec_float(spec, "mixed_axis_prob", 1.0)
                active = torch.rand(n, device=dev) >= stand_prob
                near_zero = active & (torch.rand(n, device=dev) < near_zero_prob)
                moving = active & ~near_zero
                f_lo, f_hi = _spec_range(spec, "fwd_range", getattr(self.cfg, "cmd_fwd_range", (0.3, 0.7)))
                b_lo, b_hi = _spec_range(spec, "back_range", getattr(self.cfg, "cmd_back_range", (0.2, 0.45)))
                l_lo, l_hi = _spec_range(spec, "lat_range", getattr(self.cfg, "cmd_lat_range", (0.15, 0.35)))
                y_lo, y_hi = _spec_range(spec, "yaw_range", getattr(self.cfg, "cmd_yaw_range", (0.25, 0.8)))
                f_hi = min(f_hi, float(getattr(self, "_vel_max_fwd", f_hi)))
                b_hi = min(b_hi, float(getattr(self, "_vel_max_back", b_hi)))
                l_hi = min(l_hi, float(getattr(self, "_vel_max_lat", l_hi)))
                y_hi = min(y_hi, float(getattr(self, "_vel_max_yaw", y_hi)))
                p_fwd = max(0.0, _spec_float(spec, "prob_fwd", 0.5))
                p_back = max(0.0, _spec_float(spec, "prob_back", 0.5))
                p_sum = max(1e-9, p_fwd + p_back)
                x_is_fwd = torch.rand(n, device=dev) < (p_fwd / p_sum)
                # 纯 yaw 练习：mixed 采样允许 x 分量缺席，避免所有转向命令都夹带前进/后退。
                # x_axis_prob < 1 时会采样原地转向或横移；x_axis_prob=1 保持旧的全 x 分量行为。
                x_axis_prob = _spec_float(spec, "x_axis_prob", 1.0)
                x_on = torch.rand(n, device=dev) < x_axis_prob
                lat_on = torch.rand(n, device=dev) < axis_prob
                yaw_on = torch.rand(n, device=dev) < axis_prob
                # x 关闭时仍保证命令是有效横移或转向，而不是误采样成站立。
                yaw_on = yaw_on | (~x_on & ~lat_on)
                lat_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                yaw_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                target[:, 0] = torch.where(
                    x_is_fwd,
                    _sample_command_range("fwd", n, f_lo, f_hi, dev),
                    -_sample_command_range("back", n, b_lo, b_hi, dev),
                )
                target[:, 0] = torch.where(x_on, target[:, 0], torch.zeros(n, device=dev))
                target[:, 1] = torch.where(
                    lat_on,
                    lat_sign * _sample_command_range("lat", n, l_lo, l_hi, dev),
                    torch.zeros(n, device=dev),
                )
                target[:, 2] = torch.where(
                    yaw_on,
                    yaw_sign * _sample_command_range("yaw", n, y_lo, y_hi, dev),
                    torch.zeros(n, device=dev),
                )
                target = torch.where(moving[:, None], target, torch.zeros_like(target))
                nz_scale = _spec_float(spec, "near_zero_scale", 0.05)
                near_noise = (torch.rand((n, 3), device=dev) * 2.0 - 1.0) * nz_scale
                target = torch.where(near_zero[:, None], near_noise, target)
            if target is not None:
                target = _force_stair_forward(target)
                self._cmd_target[env_ids] = target
                self._begin_command_transition(env_ids, snap=snap)
                return
        except (ValueError, TypeError):
            pass
        # 显式覆盖格式错误或 mode 未知时，保留父类采样结果。
        return

    def _draw_cmd_hold(self, n: int) -> torch.Tensor:
        """按 env 采样命令保持时长，单位为控制步。"""
        step_dt = float(self.cfg.dt) * float(self.cfg.decimation)
        lo_s = float(getattr(self.cfg, "cmd_resample_s_min", getattr(self.cfg, "cmd_resample_s", 5.0)))
        hi_s = float(getattr(self.cfg, "cmd_resample_s_max", lo_s))
        lo = max(1, int(min(lo_s, hi_s) / step_dt))
        hi = max(lo + 1, int(max(lo_s, hi_s) / step_dt) + 1)
        return torch.randint(lo, hi, (n,), device=self.device, dtype=torch.long)

    def _get_observations(self) -> dict:
        if not getattr(self, "use_external_commands", False):
            # 初始批量 reset 后打散 env 相位，避免所有 env 永久同步 reset 和切换命令。
            # 否则日志窗口会周期性碰到固定 episode 相位，影响阶段门控判断。
            if getattr(self, "_stagger_pending", True):
                self.episode_length_buf[:] = torch.randint(
                    0, max(1, int(self.max_episode_length) - 1),
                    (self.num_envs,), device=self.device, dtype=self.episode_length_buf.dtype)
                self._cmd_hold = self._draw_cmd_hold(self.num_envs)
                self._stagger_pending = False
            # 每个 env 独立命令保持，替代同步的取模重采样。
            self._cmd_hold -= 1
            due = self._cmd_hold <= 0
            if due.any():
                ids = due.nonzero(as_tuple=False).flatten()
                self._resample_commands(ids)
                self._cmd_hold[ids] = self._draw_cmd_hold(len(ids))
        N, dev = self.num_envs, self.device
        lp = self._leg_phases()
        gait_obs = torch.cat([torch.sin(2 * math.pi * lp), torch.cos(2 * math.pi * lp)], dim=-1)   # (N,8)

        # 加噪本体量，贴近部署输入。
        jpos_n = self.robot.data.joint_pos + torch.randn(N, 12, device=dev) * self.cfg.obs_noise_jpos
        jvel_n = self.robot.data.joint_vel + torch.randn(N, 12, device=dev) * self.cfg.obs_noise_jvel
        angv_n = self.robot.data.root_ang_vel_b + torch.randn(N, 3, device=dev) * self.cfg.obs_noise_angvel
        grav_n = self.robot.data.projected_gravity_b + torch.randn(N, 3, device=dev) * self.cfg.obs_noise_gravity
        angv_n = angv_n + self._imu_bias[:, 0:3]
        grav_n = grav_n + self._imu_bias[:, 3:6]

        # tick54：q_rel/dq/q_des_rel/q_error/gyro/grav；q_default 使用 action_offset。
        tick = taili_obs.assemble_tick54(jpos_n, jvel_n, self.last_actions, angv_n, grav_n,
                                         action_scale=self.action_scale, q_default=self.action_offset)
        history_update = (self.episode_length_buf % self._history_stride) == 0
        self._tick_history = taili_obs.update_tick_history(
            self._tick_history,
            tick,
            history_update,
            order=self._history_order,
        )

        # 足端接触缓存，奖励计算会复用。
        forces_now = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, :].norm(dim=-1)
        contact_threshold = float(getattr(self.cfg, "contact_force_threshold", 10.0))
        self._in_contact = (forces_now > contact_threshold).float()

        # 命令历史来自控制器本身，真机可直接复现；它消除换向状态对 Actor 不可见的问题。
        max_transition_s = max(float(getattr(self.cfg, "cmd_transition_max_s", 0.35)), 1e-6)
        command_age = torch.clamp(self._cmd_age_s / max_transition_s, 0.0, 1.0)
        body = taili_obs.assemble_body57(
            angv_n, grav_n, self.commands, self._cmd_previous, command_age,
            jpos_n - self.action_offset, jvel_n, self.last_actions, gait_obs,
        )                                                                                       # (N,57)
        blind_obs = torch.cat([body, self._tick_history.reshape(N, -1)], dim=-1)                 # (N,1407)

        # 特权观测：只供 critic/AMP 使用。
        hits = self._height_scanner.data.ray_hits_w
        hscan = (self.robot.data.root_pos_w[:, 2:3] - hits[:, :, 2] - self.cfg.stand_height).clamp(-1.0, 1.0)
        hscan = torch.nan_to_num(hscan, nan=0.0, posinf=0.0, neginf=0.0)
        self._terrain_ctx = self._compute_terrain_ctx()
        priv = torch.cat([self.robot.data.root_lin_vel_b, hscan, self._terrain_ctx, self._in_contact], dim=-1)
        labels = self._compute_aux_labels()                                                       # (N,34) geom9+mask9+risk8+mask8。
        obs = torch.cat([blind_obs, priv, labels], dim=-1)                                       # (N,1638)

        # AMP 缓冲：当前复用父类 _compute_amp_obs。
        amp = torch.nan_to_num(self._compute_amp_obs(), nan=0.0, posinf=0.0, neginf=0.0)
        if getattr(self, "_amp_frame_stride", 1) > 1:
            self._push_strided_amp(amp)                 # AMP stride 窗口，导出约一个步态周期。
        else:
            for i in reversed(range(self.cfg.num_amp_observations - 1)):
                self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
            self.amp_observation_buffer[:, 0] = amp
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}
        self.extras["terrain_pattern_scale"] = getattr(
            self,
            "_terrain_pattern_scale_for_amp",
            torch.ones(self.num_envs, device=self.device),
        )
        self.extras["amp_style_scale"] = getattr(
            self,
            "_amp_style_scale_for_agent",
            self.extras["terrain_pattern_scale"],
        )
        _flat_progress_target = max(float(getattr(self.cfg, "phase_gate_prog_0", 0.65)), 1e-6)
        _flat_progress = min(
            float(getattr(self, "_fwd_prog", 0.0)),
            float(getattr(self, "_back_prog", 0.0)),
            float(getattr(self, "_lat_prog", 0.0)),
            float(getattr(self, "_yaw_prog", 0.0)),
        )
        _flat_maturity = min(max(_flat_progress / _flat_progress_target, 0.0), 1.0)
        _stair_success_target = max(
            float(getattr(self.cfg, "phase_gate_stairs_up_success_2", 0.65)),
            float(getattr(self.cfg, "phase_gate_stairs_down_success_2", 0.65)),
            1e-6,
        )
        _stair_maturity = min(max(
            min(
                float(getattr(self, "_terrain_stairs_up_success_ema", 0.0)),
                float(getattr(self, "_terrain_stairs_down_success_ema", 0.0)),
            ) / _stair_success_target,
            0.0,
        ), 1.0)
        # 平地成型只完成探索退火条件的 60%；上下楼都真实成功后才允许完全退火。
        _exploration_capability = 0.60 * _flat_maturity + 0.40 * _stair_maturity
        self.extras["exploration_capability"] = torch.full(
            (self.num_envs,),
            _exploration_capability,
            dtype=self.commands.dtype,
            device=self.device,
        )
        # 多 critic：把 _get_rewards 暂存的分组奖励向量暴露到 extras，
        # 使 agent 的 record_transition 可以把 [N,K] 写入 rollout。
        # 未启用开关时不改变 extras。
        if os.environ.get("TAILI_MULTI_CRITIC") == "1" and getattr(self, "_reward_groups", None) is not None:
            self.extras["reward_groups"] = self._reward_groups
        self.last_actions = self.actions.clone()        # FIFO 账本，父类也在 _get_observations 中维护。
        self._log_step = getattr(self, "_log_step", 0) + 1
        if self._log_step % self.cfg.log_every == 0 and not getattr(self, "use_external_commands", False):
            self._log_training_diag()
        return {"policy": obs}

    # 从特权仿真信号生成 geom/risk 辅助标签。
    def _compute_aux_labels(self) -> torch.Tensor:
        """返回 (N,34) = geom_label9 | geom_mask9 | risk_label8 | risk_mask8。

        这些标签只用于训练，附加到特权观测中，供辅助训练 hook 使用。
        """
        rd = self.robot.data
        N, dev = self.num_envs, self.device
        hits = self._height_scanner.data.ray_hits_w                                  # (N,R,3)，世界系。
        rel_xy = hits[:, :, :2] - rd.root_pos_w[:, :2].unsqueeze(1)
        z = torch.nan_to_num(hits[:, :, 2], nan=0.0)
        a, b, _ = taili_terrain_labels.fit_local_plane(rel_xy, z)                     # 世界系坡度 a,b。
        geom3 = torch.stack([taili_terrain_labels.slope_norm(a), taili_terrain_labels.slope_norm(b),
                             taili_terrain_labels.log_roughness_norm(rel_xy, z)], dim=-1)  # (N,3)
        # foot_h4：每只脚下地面相对局部基准地面的高度。
        # 该标签让感知器学习台阶高度结构，部署时仍只依赖盲本体历史推断。
        foot_pos = rd.body_pos_w[:, self.foot_indexes, :]                            # (N,4,3)
        _fdxy = foot_pos[:, :, None, :2] - hits[:, None, :, :2]                       # (N,4,R,2)
        _fnear = (_fdxy * _fdxy).sum(-1).argmin(dim=2)                                # (N,4)，每只脚最近射线。
        foot_ground = torch.gather(z, 1, _fnear)                                     # (N,4)，每只脚下地面 z。
        base_ground = z.median(dim=1).values                                        # (N,)，局部基准地面高度。
        foot_h4 = ((foot_ground - base_ground.unsqueeze(1)) / 0.40).clamp(-1.0, 1.0)  # (N,4)，归一化足下地形高度。
        edge_up = torch.clamp(foot_ground.max(dim=1).values - base_ground, min=0.0)
        edge_down = torch.clamp(base_ground - foot_ground.min(dim=1).values, min=0.0)
        edge2 = torch.stack([
            taili_terrain_labels.edge_up_norm(edge_up),
            taili_terrain_labels.edge_down_norm(edge_down),
        ], dim=-1)
        geom = torch.cat([geom3, foot_h4, edge2], dim=-1)                             # (N,9): slope3 + foot_h4 + edge2
        # 高度扫描标签只能在本体历史足以解释时监督感知器。平地零标签始终有效；
        # 连续地形在发生支撑响应后有效；离散障碍的层高必须等到新层承重触地后才有效。
        try:
            self._ensure_gate_mask()
            flat_scope = self._flat_transition_scope_mask().bool()
            discrete_scope = getattr(
                self,
                "_discrete_terrain_mask",
                torch.zeros(N, dtype=torch.bool, device=dev),
            ).bool()
        except Exception:
            flat_scope = torch.zeros(N, dtype=torch.bool, device=dev)
            discrete_scope = torch.zeros(N, dtype=torch.bool, device=dev)
        response_visible = getattr(
            self,
            "_blind_amp_terrain_response",
            torch.zeros(N, device=dev),
        ) > 0.05
        continuous_visible = (~discrete_scope) & response_visible
        coarse_visible = flat_scope | continuous_visible
        layer_visible = coarse_visible | (discrete_scope & response_visible)
        geom_mask = torch.zeros(N, 9, device=dev)
        geom_mask[:, :3] = coarse_visible[:, None].to(geom_mask.dtype)
        geom_mask[:, 3:] = layer_visible[:, None].to(geom_mask.dtype)

        forces = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, :].norm(dim=-1)  # (N,4)
        f_max = forces.max(dim=1).values
        impact_n = taili_terrain_labels.impact_score_norm(f_max, torch.tensor(self.cfg.robot_mass_kg, device=dev)
                                                          if hasattr(self.cfg, "robot_mass_kg") else torch.tensor(39.0, device=dev))
        cc = self._in_contact.sum(dim=1)
        tilt = torch.arccos(torch.clamp(-rd.projected_gravity_b[:, 2], -1.0, 1.0))
        support = torch.clamp((3.0 - cc) / 3.0, 0.0, 1.0)
        support = torch.maximum(support, torch.clamp((tilt - math.radians(10)) / math.radians(20), 0.0, 1.0))
        base_risk = torch.stack([impact_n, support], dim=-1)                         # (N,2)

        # 标签只描述当前已经发生的承重高度变化和碰撞足，不保留先导足或阶段记忆。
        support_delta = getattr(
            self,
            "_terrain_support_delta",
            torch.zeros(N, device=dev),
        )
        event_direction = torch.sign(support_delta).to(torch.int8)
        lead_foot = getattr(
            self,
            "_terrain_contact_foot_mask",
            torch.zeros((N, 4), dtype=torch.bool, device=dev),
        )
        event_visible = response_visible & (event_direction.ne(0) | lead_foot.any(dim=1))
        risk, risk_mask = taili_terrain_labels.assemble_risk8(
            base_risk[:, 0],
            base_risk[:, 1],
            event_direction,
            lead_foot,
            event_visible,
        )
        return torch.cat([geom, geom_mask, risk, risk_mask], dim=-1)                 # (N,34)

    def _reset_idx(self, env_ids):
        # 父类会在 reset 内按 AMP 地形条件选择 RSI 参考。先清除上一 episode 的
        # 碰障响应，避免新 episode 在尚未接触楼梯时继承地形步态先验。
        if hasattr(self, "_blind_amp_terrain_response") and len(env_ids) > 0:
            self._blind_amp_terrain_response[env_ids] = 0.0
        super()._reset_idx(env_ids)
        self._tick_history[env_ids] = 0.0
        if hasattr(self, "_prev_support_z") and len(env_ids) > 0:
            try:
                self._prev_support_z[env_ids] = self._terrain_height_under_base()[env_ids].detach()
            except Exception:
                self._prev_support_z[env_ids] = self._terrain.env_origins[env_ids, 2].detach()
        if hasattr(self, "_blind_amp_terrain_response") and len(env_ids) > 0:
            self._blind_amp_terrain_response[env_ids] = 0.0
            self._terrain_pattern_scale_for_amp[env_ids] = 1.0
            self._amp_style_scale_for_agent[env_ids] = 1.0
            self._terrain_leg_response[env_ids] = 0.0
            self._terrain_collision_trace[env_ids] = 0.0
            self._terrain_limb_collision_response[env_ids] = 0.0
            self._terrain_base_collision_response[env_ids] = 0.0
            self._terrain_support_delta[env_ids] = 0.0
            self._terrain_support_dispersion[env_ids] = 0.0
            self._terrain_support_valid[env_ids] = False
            self._terrain_contact_foot_mask[env_ids] = False
            self._terrain_prev_base_z[env_ids] = self.robot.data.root_pos_w[env_ids, 2].detach()
            _world_up = torch.zeros((len(env_ids), 3), device=self.device)
            _world_up[:, 2] = 1.0
            _foot_pos = self.robot.data.body_pos_w[env_ids[:, None], self.foot_indexes, :].detach()
            self._support_reference_point[env_ids] = (
                _foot_pos.mean(dim=1) - taili_geometry.FOOT_RADIUS * _world_up
            )
            self._support_reference_normal[env_ids] = _world_up
            if hasattr(self, "_episode_support_initialized"):
                self._episode_support_initialized[env_ids] = False
        # reset 后为这些 env 重新采样命令保持时长，维持队列不同步。
        if getattr(self, "_cmd_hold", None) is not None and len(env_ids) > 0:
            self._cmd_hold[env_ids] = self._draw_cmd_hold(len(env_ids))
        # RSI 会把脚放在接触附近；reset 后先标记为已接触，避免第一步误算成新触地。
        if self._prev_in_contact is not None:
            self._prev_in_contact[env_ids] = 1.0
        if self._prev_support_contact is not None:
            self._prev_support_contact[env_ids] = 1.0
        if hasattr(self, "_touchdown_hold_steps"):
            self._touchdown_hold_steps[env_ids] = 0
            self._touchdown_vz_hold[env_ids] = 0.0
            self._touchdown_xy_hold[env_ids] = 0.0
            self._touchdown_force_hold[env_ids] = 0.0
            self._prev_base_ang_vel[env_ids] = self.robot.data.root_ang_vel_b[env_ids].detach()
            self._base_ang_vel_filtered[env_ids] = self.robot.data.root_ang_vel_b[env_ids].detach()
        if hasattr(self, "_quality_window_ready"):
            self._quality_duty_ema[env_ids] = 0.5
            self._quality_diag_pair_ema[env_ids] = 0.5
            self._quality_duty_dir_ema[env_ids] = 0.5
            self._quality_diag_dir_ema[env_ids] = 0.5
            self._quality_duty_dir_valid[env_ids] = False
            self._quality_slip_speed_ema[env_ids] = 0.0
            self._quality_slip_excess_ema[env_ids] = 0.0
            self._quality_slip_high_ema[env_ids] = 0.0
            self._quality_slip_by_leg_ema[env_ids] = 0.0
            self._quality_wxy_energy_ema[env_ids] = 0.0
            self._quality_vz_energy_ema[env_ids] = 0.0
            self._quality_yaw_residual_energy_ema[env_ids] = 0.0
            self._quality_contact_chatter_ema[env_ids] = 0.0
            self._quality_support_force_ema[env_ids] = 0.0
            self._quality_height_low_ema[env_ids] = 0.0
            self._quality_tilt_high_ema[env_ids] = 0.0
            self._quality_window_ready[env_ids] = False
            self._gait_lag_scores[env_ids] = 0.5
            self._last_touchdown_step[env_ids] = -1
            self._last_touchdown_step_all[env_ids] = -1
            self._contact_period_ema[env_ids] = 0.0
            self._contact_period_valid[env_ids] = False
            self._contact_period_dir_ema[env_ids] = 0.0
            self._contact_period_dir_valid[env_ids] = False
            self._duty_cycle_sum[env_ids] = 0.0
            self._duty_cycle_steps[env_ids] = 0
            self._duty_cycle_direction[env_ids] = -1
            self._duty_cycle_valid[env_ids] = False
        if hasattr(self, "_prev_foot_force_norm"):
            self._prev_foot_force_norm[env_ids] = 0.0
            self._prev_foot_vz[env_ids] = 0.0
            self._prev_foot_vel_w[env_ids] = self.robot.data.body_lin_vel_w[
                env_ids[:, None], self.foot_indexes, :
            ]
            self._prev_action_delta[env_ids] = 0.0
            self._prev_transition_fault[env_ids] = 1.0

    def _get_dones(self):
        died, time_out = super()._get_dones()
        if not bool(getattr(self.cfg, "early_termination", True)):
            return died, time_out

        discrete = getattr(
            self,
            "_discrete_terrain_mask",
            torch.zeros_like(died, dtype=torch.bool),
        ).bool()
        if not bool(discrete.any()):
            return died, time_out

        root_pos = self.robot.data.root_pos_w
        root_lin = self.robot.data.root_lin_vel_b
        root_ang = self.robot.data.root_ang_vel_b
        finite = (
            torch.isfinite(root_pos).all(dim=1)
            & torch.isfinite(root_lin).all(dim=1)
            & torch.isfinite(root_ang).all(dim=1)
        )
        base_h_local = root_pos[:, 2] - self._terrain_height_under_base()
        recovery_height = min(
            float(getattr(self.cfg, "terrain_recovery_termination_height", 0.18)),
            float(self.cfg.termination_height),
        )
        recovery_tilt_deg = min(max(
            float(getattr(self.cfg, "terrain_recovery_termination_tilt_deg", 60.0)),
            0.0,
        ), 180.0)
        terrain_died = terrain_curriculum.terrain_recovery_termination(
            base_height_local=base_h_local,
            projected_gravity_z=self.robot.data.projected_gravity_b[:, 2],
            finite=finite,
            min_height=recovery_height,
            max_tilt_deg=recovery_tilt_deg,
        )
        # 离散地形允许低姿态碰障后的即时恢复，但侧翻、趴平和数值异常仍立即终止。
        died = torch.where(discrete & finite, terrain_died, died) | (~finite)
        return died, time_out

    def _update_quality_windows(
        self,
        in_contact,
        settled_contact,
        landing_mask,
        foot_vel_xy,
        base_h,
        tilt_rel,
        moving,
        base_lin_vel,
        base_ang_vel,
        support_force,
    ):
        """更新奖励、阶段门控和遥测共用的步态质量窗口。

        单帧代理过于宽松：四脚爬行可能看起来像对角步态，高 duty 爬行可能看起来平衡，
        单只支撑脚滑移也会被四足均值稀释。EMA 窗口更接近诊断对整段行为的评价。
        """
        f = in_contact.dtype
        self._quality_step += 1
        if not hasattr(self, "_quality_window_ready"):
            self._quality_duty_ema = torch.full_like(in_contact, 0.5)
            self._quality_diag_pair_ema = torch.full_like(base_h, 0.5)
            self._quality_slip_speed_ema = torch.zeros_like(base_h)
            self._quality_slip_excess_ema = torch.zeros_like(base_h)
            self._quality_slip_high_ema = torch.zeros_like(base_h)
            self._quality_slip_by_leg_ema = torch.zeros_like(in_contact)
            self._quality_wxy_energy_ema = torch.zeros_like(base_h)
            self._quality_vz_energy_ema = torch.zeros_like(base_h)
            self._quality_contact_chatter_ema = torch.zeros_like(base_h)
            self._quality_support_force_ema = torch.zeros_like(in_contact)
            self._quality_height_low_ema = torch.zeros_like(base_h)
            self._quality_tilt_high_ema = torch.zeros_like(base_h)
            self._quality_window_ready = torch.zeros(base_h.shape[0], dtype=torch.bool, device=base_h.device)

        beta = min(max(float(getattr(self._rcfg, "quality_window_ema_beta", 0.98)), 0.0), 0.999)
        stance = (in_contact > 0.5).to(f)
        settled = (settled_contact > 0.5).to(f)
        moving_mask = moving > 0.5

        if stance.shape[-1] >= 4:
            fl, fr, rl, rr = stance[:, 0], stance[:, 1], stance[:, 2], stance[:, 3]
            diag_pair = torch.clamp(
                fl * rr * (1.0 - fr) * (1.0 - rl)
                + fr * rl * (1.0 - fl) * (1.0 - rr),
                0.0,
                1.0,
            )
        else:
            diag_pair = torch.zeros_like(base_h)

        transition_active = getattr(self, "_cmd_transition_timer", torch.zeros_like(base_h)) > 0
        # 对角小跑指标只在适用场景更新：前进、后退或横移命令明显强于 yaw。
        # yaw 主导转向时低 diag_pair 是合理的；不应把它混进直线小跑质量 EMA。
        # 非适用帧保持 EMA 原值，避免对角步态奖励和转向步态互相冲突。
        _cmd = self.commands
        _lin_mag = torch.linalg.norm(_cmd[:, :2], dim=-1)
        _trot_applicable = (
            moving_mask
            & ~transition_active
            & (_lin_mag > 0.10)
            & (_lin_mag >= 0.6 * _cmd[:, 2].abs())
        )

        # Duty 是稳定命令窗口内的真实接触时间占比，不依赖“恰好只有一对对角腿
        # 接触”。实际周期则按同一只脚的连续 touchdown 间隔估计。这样爬行、三足
        # 支撑或尚未形成纯对角小跑时仍可被真实测量，而不会把默认 0.5 当作好步态。
        direction_template = torch.zeros_like(self._duty_cycle_direction)
        direction = torch.where(
            _cmd[:, 0].abs() >= _cmd[:, 1].abs(),
            torch.where(_cmd[:, 0] >= 0.0, direction_template, torch.ones_like(direction_template)),
            torch.where(
                _cmd[:, 1] >= 0.0,
                torch.full_like(direction_template, 2),
                torch.full_like(direction_template, 3),
            ),
        )
        direction_reset = ~_trot_applicable | (
            (self._duty_cycle_direction >= 0) & (direction != self._duty_cycle_direction)
        )
        if bool(direction_reset.any()):
            self._duty_cycle_sum[direction_reset] = 0.0
            self._duty_cycle_steps[direction_reset] = 0
            self._last_touchdown_step[direction_reset] = -1
            self._duty_cycle_valid[direction_reset] = False
            self._contact_period_valid[direction_reset] = False
            self._duty_cycle_direction[direction_reset] = -1
        self._duty_cycle_direction[_trot_applicable] = direction[_trot_applicable]
        applicable_ids = _trot_applicable.nonzero(as_tuple=False).squeeze(-1)
        if len(applicable_ids) > 0:
            applicable_dir = direction[applicable_ids]
            old_diag = self._quality_diag_dir_ema[applicable_ids, applicable_dir]
            new_diag = beta * old_diag + (1.0 - beta) * diag_pair[applicable_ids]
            self._quality_diag_dir_ema[applicable_ids, applicable_dir] = new_diag
            self._quality_diag_pair_ema[applicable_ids] = new_diag
            self._quality_duty_ema[applicable_ids] = self._quality_duty_dir_ema[applicable_ids, applicable_dir]
            self._duty_cycle_valid[applicable_ids] = self._quality_duty_dir_valid[applicable_ids, applicable_dir]
            self._contact_period_ema[applicable_ids] = self._contact_period_dir_ema[applicable_ids, applicable_dir]
            self._contact_period_valid[applicable_ids] = self._contact_period_dir_valid[applicable_ids, applicable_dir]

        dt = float(self.cfg.dt * self.cfg.decimation)
        target_period = torch.clamp(
            self.cfg.gait_period - self.cfg.gait_period_slope * _lin_mag,
            min=self.cfg.gait_period_min,
            max=self.cfg.gait_period,
        )
        target_period_all = torch.clamp(
            self.cfg.gait_period - self.cfg.gait_period_slope * (
                _lin_mag
                + float(getattr(self.cfg, "gait_yaw_speed_equiv", 0.15))
                * _cmd[:, 2].abs()
            ),
            min=self.cfg.gait_period_min,
            max=self.cfg.gait_period,
        )
        required_duty_steps = torch.ceil(target_period / max(dt, 1e-6)).long().clamp(min=1)
        self._duty_cycle_sum[_trot_applicable] += stance[_trot_applicable]
        self._duty_cycle_steps[_trot_applicable] += 1
        completed = _trot_applicable & (self._duty_cycle_steps >= required_duty_steps)
        if bool(completed.any()):
            completed_ids = completed.nonzero(as_tuple=False).squeeze(-1)
            completed_dir = direction[completed_ids]
            measured_duty = self._duty_cycle_sum[completed_ids] / self._duty_cycle_steps[
                completed_ids, None
            ].float()
            cycle_beta = min(
                max(float(getattr(self._rcfg, "duty_cycle_ema_beta", 0.60)), 0.0),
                0.95,
            )
            cold_duty = ~self._quality_duty_dir_valid[completed_ids, completed_dir]
            updated_duty = torch.where(
                cold_duty[:, None],
                measured_duty,
                cycle_beta * self._quality_duty_dir_ema[completed_ids, completed_dir]
                + (1.0 - cycle_beta) * measured_duty,
            )
            self._quality_duty_dir_ema[completed_ids, completed_dir] = updated_duty
            self._quality_duty_dir_valid[completed_ids, completed_dir] = True
            self._quality_duty_ema[completed_ids] = updated_duty
            self._duty_cycle_valid[completed_ids] = True
            self._duty_cycle_sum[completed_ids] = 0.0
            self._duty_cycle_steps[completed_ids] = 0

        chatter_scope = moving_mask & ~transition_active
        self._last_touchdown_step_all = torch.where(
            chatter_scope[:, None],
            self._last_touchdown_step_all,
            torch.full_like(self._last_touchdown_step_all, -1),
        )
        touchdown_all = (landing_mask > 0.5) & chatter_scope[:, None]
        previous_touchdown_all = self._last_touchdown_step_all
        touchdown_period_all = (
            self._quality_step - previous_touchdown_all
        ).to(f) * dt
        chatter_has_previous = touchdown_all & (previous_touchdown_all >= 0)
        chatter_period_ratio = min(max(float(getattr(
            self._rcfg, "contact_chatter_min_period_ratio", 0.65
        )), 0.25), 0.95)
        chatter_min_period = target_period_all[:, None] * chatter_period_ratio
        chatter_by_leg = torch.clamp(
            (chatter_min_period - touchdown_period_all)
            / chatter_min_period.clamp(min=1e-6),
            0.0,
            1.0,
        ) * chatter_has_previous.to(f)
        chatter_event_valid = chatter_has_previous.any(dim=1)
        chatter_event = chatter_by_leg.max(dim=1).values
        self._last_touchdown_step_all = torch.where(
            touchdown_all,
            torch.full_like(self._last_touchdown_step_all, self._quality_step),
            self._last_touchdown_step_all,
        )

        touchdown = (landing_mask > 0.5) & _trot_applicable[:, None]
        previous_touchdown = self._last_touchdown_step
        touchdown_period = (self._quality_step - previous_touchdown).to(f) * dt
        valid_touchdown = (
            touchdown
            & (previous_touchdown >= 0)
            & (touchdown_period >= 0.25)
            & (touchdown_period <= 1.50)
        )
        valid_period_env = valid_touchdown.any(dim=1)
        if bool(valid_period_env.any()):
            period_ids = valid_period_env.nonzero(as_tuple=False).squeeze(-1)
            period_dir = direction[period_ids]
            period_count = valid_touchdown[period_ids].to(f).sum(dim=1).clamp(min=1.0)
            measured_period = (
                touchdown_period[period_ids] * valid_touchdown[period_ids].to(f)
            ).sum(dim=1) / period_count
            cold_period = ~self._contact_period_dir_valid[period_ids, period_dir]
            updated_period = torch.where(
                cold_period,
                measured_period,
                0.8 * self._contact_period_dir_ema[period_ids, period_dir]
                + 0.2 * measured_period,
            )
            self._contact_period_dir_ema[period_ids, period_dir] = updated_period
            self._contact_period_dir_valid[period_ids, period_dir] = True
            self._contact_period_ema[period_ids] = updated_period
            self._contact_period_valid[period_ids] = True
        self._last_touchdown_step = torch.where(
            touchdown,
            torch.full_like(self._last_touchdown_step, self._quality_step),
            self._last_touchdown_step,
        )

        contact_denom = settled.sum(dim=-1).clamp(min=1.0)
        slip_speed_mean = (foot_vel_xy * settled).sum(dim=-1) / contact_denom
        slip_excess = torch.clamp(
            (foot_vel_xy - 0.05) / 0.05,
            min=0.0,
            max=float(getattr(self._rcfg, "slip_excess_cap", 1.5)),
        )
        slip_excess_mean = (slip_excess ** 2 * settled).sum(dim=-1) / contact_denom
        slip_high_threshold = float(getattr(self._rcfg, "slip_high_threshold", 0.20))
        slip_high_fraction = ((foot_vel_xy > slip_high_threshold).to(f) * settled).sum(dim=-1) / contact_denom
        quality_active = moving_mask & ~transition_active
        quality_col = quality_active[:, None]
        slip_speed_sample = torch.where(quality_active, slip_speed_mean, self._quality_slip_speed_ema)
        slip_excess_sample = torch.where(quality_active, slip_excess_mean, self._quality_slip_excess_ema)
        slip_high_sample = torch.where(quality_active, slip_high_fraction, self._quality_slip_high_ema)
        slip_by_leg_sample = torch.where(quality_col, foot_vel_xy * settled, self._quality_slip_by_leg_ema)
        wxy_energy_sample = torch.where(moving_mask, base_ang_vel[:, :2].pow(2).sum(dim=-1), torch.zeros_like(base_h))
        vz_energy_sample = torch.where(
            quality_active,
            base_lin_vel[:, 2].pow(2),
            self._quality_vz_energy_ema,
        )
        yaw_residual = base_ang_vel[:, 2] - self.commands[:, 2]
        yaw_energy_sample = torch.where(moving_mask, yaw_residual.pow(2), torch.zeros_like(base_h))
        support_force_sample = torch.where(quality_col, support_force * settled, self._quality_support_force_ema)

        h_ok = float(getattr(self._rcfg, "h_ok", 0.47))
        h_close = float(getattr(self._rcfg, "h_gate_close", 0.42))
        height_low = torch.clamp((h_ok - base_h) / max(h_ok - h_close, 1e-6), 0.0, 1.0)
        tilt_ok = float(getattr(self._rcfg, "tilt_ok_rad", math.radians(15)))
        tilt_close = float(getattr(self._rcfg, "tilt_gate_close_rad", math.radians(40)))
        tilt_high = torch.clamp((tilt_rel - tilt_ok) / max(tilt_close - tilt_ok, 1e-6), 0.0, 1.0)

        cold = ~self._quality_window_ready
        if bool(cold.any()):
            self._quality_slip_speed_ema[cold] = slip_speed_sample[cold]
            self._quality_slip_excess_ema[cold] = slip_excess_sample[cold]
            self._quality_slip_high_ema[cold] = slip_high_sample[cold]
            self._quality_slip_by_leg_ema[cold] = slip_by_leg_sample[cold]
            self._quality_wxy_energy_ema[cold] = wxy_energy_sample[cold]
            self._quality_vz_energy_ema[cold] = vz_energy_sample[cold]
            self._quality_yaw_residual_energy_ema[cold] = yaw_energy_sample[cold]
            self._quality_contact_chatter_ema[cold] = 0.0
            self._quality_support_force_ema[cold] = support_force_sample[cold]
            self._quality_height_low_ema[cold] = height_low[cold]
            self._quality_tilt_high_ema[cold] = tilt_high[cold]
            self._quality_window_ready[cold] = True

        self._quality_slip_speed_ema.mul_(beta).add_(slip_speed_sample * (1.0 - beta))
        self._quality_slip_excess_ema.mul_(beta).add_(slip_excess_sample * (1.0 - beta))
        self._quality_slip_high_ema.mul_(beta).add_(slip_high_sample * (1.0 - beta))
        self._quality_slip_by_leg_ema.mul_(beta).add_(slip_by_leg_sample * (1.0 - beta))
        self._quality_wxy_energy_ema.mul_(beta).add_(wxy_energy_sample * (1.0 - beta))
        self._quality_vz_energy_ema.mul_(beta).add_(vz_energy_sample * (1.0 - beta))
        self._quality_yaw_residual_energy_ema.mul_(beta).add_(yaw_energy_sample * (1.0 - beta))
        chatter_beta = min(max(
            float(getattr(self._rcfg, "duty_cycle_ema_beta", 0.60)), 0.0
        ), 0.95)
        self._quality_contact_chatter_ema.copy_(torch.where(
            chatter_event_valid,
            chatter_beta * self._quality_contact_chatter_ema
            + (1.0 - chatter_beta) * chatter_event,
            self._quality_contact_chatter_ema,
        ))
        self._quality_support_force_ema.mul_(beta).add_(support_force_sample * (1.0 - beta))
        self._quality_height_low_ema.mul_(beta).add_(height_low * (1.0 - beta))
        self._quality_tilt_high_ema.mul_(beta).add_(tilt_high * (1.0 - beta))

        duty_w = self._quality_duty_ema
        front = 0.5 * (duty_w[:, 0] + duty_w[:, 1])
        rear = 0.5 * (duty_w[:, 2] + duty_w[:, 3])
        straight_error = torch.stack(
            (
                (duty_w[:, 0] - duty_w[:, 1]).abs(),
                (duty_w[:, 2] - duty_w[:, 3]).abs(),
                (front - rear).abs(),
            ),
            dim=1,
        ).max(dim=1).values
        # 横移时左右承重可以不同，真正需要一致的是同侧前后腿。
        lateral_error = torch.stack(
            (
                (duty_w[:, 0] - duty_w[:, 2]).abs(),
                (duty_w[:, 1] - duty_w[:, 3]).abs(),
            ),
            dim=1,
        ).max(dim=1).values
        symmetry_error = torch.where(direction >= 2, lateral_error, straight_error)
        symmetry_tolerance = max(float(getattr(self._rcfg, "duty_symmetry_tolerance", 0.25)), 1e-6)
        duty_symmetry_score = taili_reward.duty_symmetry_score(symmetry_error, symmetry_tolerance)
        duty_low = float(getattr(self._rcfg, "duty_linear_min", 0.42))
        duty_high = float(getattr(self._rcfg, "duty_linear_max", 0.68))
        duty_margin = max(float(getattr(self._rcfg, "duty_linear_margin", 0.18)), 1e-6)
        duty_outside = torch.maximum(duty_low - duty_w, duty_w - duty_high).clamp(min=0.0)
        duty_target_error = duty_outside.max(dim=-1).values
        duty_target_score = 1.0 - torch.clamp(duty_target_error / duty_margin, 0.0, 1.0)
        duty_quality = torch.clamp(duty_symmetry_score * duty_target_score, 0.0, 1.0)
        period_tolerance = max(float(getattr(self._rcfg, "gait_period_tolerance", 0.22)), 1e-6)
        period_error = (self._contact_period_ema - target_period).abs()
        period_score = taili_reward.tolerance_score(period_error, period_tolerance)
        period_score = torch.where(self._contact_period_valid, period_score, torch.zeros_like(period_score))

        return {
            "duty_by_leg": duty_w,
            "duty_mean": duty_w.mean(dim=-1),
            "duty_spread": duty_w.max(dim=-1).values - duty_w.min(dim=-1).values,
            "duty_symmetry_score": duty_symmetry_score,
            "duty_target_score": duty_target_score,
            "duty_target_error": duty_target_error,
            "duty_symmetry_error": symmetry_error,
            "duty_quality": duty_quality,
            "duty_cycle_valid": self._duty_cycle_valid.to(f),
            "contact_period": self._contact_period_ema,
            "contact_period_valid": self._contact_period_valid.to(f),
            "period_error": period_error,
            "period_score": period_score,
            "diag_pair_score": self._quality_diag_pair_ema,
            "slip_speed": self._quality_slip_speed_ema,
            "slip_excess": self._quality_slip_excess_ema,
            "slip_high_fraction": self._quality_slip_high_ema,
            "slip_by_leg": self._quality_slip_by_leg_ema,
            "wxy_energy": self._quality_wxy_energy_ema,
            "vz_energy": self._quality_vz_energy_ema,
            "yaw_residual_energy": self._quality_yaw_residual_energy_ema,
            "contact_chatter": self._quality_contact_chatter_ema,
            "support_force_by_leg": self._quality_support_force_ema,
            "height_low_risk": self._quality_height_low_ema,
            "tilt_high_risk": self._quality_tilt_high_ema,
            "slip_speed_inst": slip_speed_mean,
            "slip_high_fraction_inst": slip_high_fraction,
            "diag_pair_inst": diag_pair,
        }

    # AMP：frame51 = motion43 + command3 + mode_onehot5，不直接使用地形标签。
    def _compute_amp_obs(self):
        """策略侧 AMP 观测：机器人实际 motion43 + command + mode。

        维度和顺序与 collect_reference_motions 生成的解析参考帧对齐。
        """
        jp = self.robot.data.joint_pos
        jv = self.robot.data.joint_vel
        bh = self.robot.data.root_pos_w[:, 2:3] - self._terrain_height_under_base().unsqueeze(1)
        tn = quaternion_to_tangent_and_normal(self.robot.data.root_quat_w)
        rel_b = self._feet_rel_base(self.robot.data.body_pos_w[:, self.foot_indexes],
                                    self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.n_feet)
        motion = torch.cat([jp, jv, bh, tn, rel_b], dim=-1)
        return taili_amp_reference.conditioned_frame51(motion, self.commands)

    def collect_reference_motions(self, num_samples, current_times=None):
        """用 q_default 对齐的解析生成器生成参考侧 AMP 帧。

        IsaacLab 当前 joint_names 顺序与解析参考一致，因此 motion_dof_indexes 通常是 identity。
        保留映射变量是为了以后布局变化时仍能显式对齐。
        """
        K = self.cfg.num_amp_observations
        env_sel = torch.randint(0, self.num_envs, (num_samples,), device=self.device)
        cmd_s = self.commands[env_sel]
        if current_times is None:
            current_times = np.random.uniform(0.0, 2.0, num_samples).astype(np.float32)
        _stride = getattr(self, "_amp_frame_stride", 1)                       # 与策略侧 AMP stride 间隔保持一致。
        times = (np.expand_dims(current_times, -1) - (1.0 / 50.0) * _stride * np.arange(K)).flatten()
        times_t = torch.as_tensor(times, dtype=torch.float32, device=self.device)
        cmd_rep = cmd_s.repeat_interleave(K, dim=0)
        frame = taili_amp_reference.frame51(
            cmd_rep,
            times_t,
            gait_period=self.cfg.gait_period,
            gait_period_slope=self.cfg.gait_period_slope,
            gait_period_min=self.cfg.gait_period_min,
            yaw_speed_equiv=float(getattr(self.cfg, "gait_yaw_speed_equiv", 0.15)),
            clearance_base=self.cfg.base_clearance,
            stance_dx=self.cfg.stance_dx,
        )                                                                         # (num_samples*K, 51)
        return frame.view(-1, self.amp_observation_size)

    # 奖励：使用 taili_reward 中的统一生产奖励。
    def _terrain_height_under_base(self):
        """用高度扫描器估计 base 下方局部地面高度。

        平地上等价于 env origin z；斜坡、楼梯和粗糙地形上跟随实际支撑面。
        非有限射线退回 env origin。
        """
        origins_z = self._terrain.env_origins[:, 2]
        try:
            hits_z = self._height_scanner.data.ray_hits_w[:, :, 2]
            hits_z = torch.where(torch.isfinite(hits_z), hits_z, origins_z.unsqueeze(1))
            return hits_z.median(dim=1).values
        except Exception:
            return origins_z

    def _get_rewards(self) -> torch.Tensor:
        self._evaluate_command_transition_post_physics()
        self._accumulate_curriculum_progress()
        # 1. 原始仿真量和基础状态：收集后续奖励、质量窗口和遥测共用的张量。
        rd = self.robot.data
        N, dev = self.num_envs, self.device
        cfg = self._rcfg
        _policy_managed_transition = bool(getattr(self.cfg, "cmd_transition_policy_managed", False))
        _transition_state = getattr(self, "_cmd_transition_state", torch.zeros(N, device=dev))
        _policy_transition_active = (
            _transition_state > 0
            if _policy_managed_transition
            else torch.zeros(N, dtype=torch.bool, device=dev)
        )
        # 过渡整形主要服务平地急停、反向和换轴。地形上不能撤掉推进梯度或
        # 强迫先归零；Actor 仍只看到可部署观测，地形 mask 仅用于训练奖励归因。
        _flat_transition = self._flat_transition_scope_mask()
        _flat_transition_active = _policy_transition_active & _flat_transition
        _policy_brake = _flat_transition_active & (_transition_state != 4)
        _policy_brake_f = _policy_brake.to(rd.joint_pos.dtype)
        _release = (_transition_state == 4) & _flat_transition
        _release_steps = getattr(
            self, "_cmd_transition_accel_steps", torch.ones(N, dtype=torch.long, device=dev)
        ).to(rd.joint_pos.dtype).clamp(min=1.0)
        _release_progress = torch.clamp(
            getattr(self, "_cmd_transition_stage_step", torch.zeros(N, device=dev)).to(rd.joint_pos.dtype)
            / _release_steps,
            0.0,
            1.0,
        )
        # 原始命令和任务驱动始终保持完整；只关闭与新命令立即绑定的解析风格锚点。
        # 释放阶段使用五次曲线恢复，避免 AMP/live imitation 在单帧内重新满额介入。
        _policy_steady_f = taili_reward.transition_style_weight(
            _policy_brake,
            _release,
            _release_progress,
        )
        _transition_task_weight = torch.ones_like(_policy_steady_f)
        _transition_gait_weight = torch.ones_like(_policy_steady_f)
        self._transition_task_weight_mean = float(_transition_task_weight.mean())
        self._transition_gait_weight_mean = float(_transition_gait_weight.mean())
        _flat_transition_f = _flat_transition.to(rd.joint_pos.dtype)

        terrain_h = self._terrain_height_under_base()                                 # (N,)
        base_h = rd.root_pos_w[:, 2] - terrain_h                                       # 平地使用的世界竖直高度。
        grav = rd.projected_gravity_b
        tilt_rel = torch.arccos(torch.clamp(-grav[:, 2], -1.0, 1.0))                   # 平地使用世界水平参考。
        in_contact = self._in_contact                                                 # (N,4)，由 _get_observations 缓存。
        cc = in_contact.sum(dim=1)
        spd_xy = torch.norm(self.commands[:, :2], dim=1)
        # yaw 命令门控：moving 阈值必须与 taili_reward 的 yaw_cmd_gate 对齐。
        # 否则小 yaw 命令会被当成站立，tracking_yaw 被清零，策略学成“转向时不动”。
        moving = ((spd_xy > 0.1) | (self.commands[:, 2].abs() > 0.05)).float()
        stand_gate = 1.0 - moving
        prev_contact = self._prev_in_contact
        if prev_contact is None or prev_contact.shape != in_contact.shape:
            prev_contact = torch.ones_like(in_contact)                                # 首次调用不产生虚假触地。
        self._prev_in_contact = in_contact.detach().clone()
        landing_mask = ((in_contact > 0.5) & (prev_contact < 0.5)).float()
        td_mask = None
        # 稳定支撑 = 当前帧接触且上一帧也接触。触地帧由落脚冲击负责，
        # 不计入支撑滑移，避免把同一个事件重复算进 B1 和 B2。
        foot_pos = rd.body_pos_w[:, self.foot_indexes, :]                             # (N,4,3)
        foot_vel = rd.body_lin_vel_w[:, self.foot_indexes, :]                         # (N,4,3)
        _foot_force_w = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, :]
        _support_contact_parts = terrain_curriculum.classify_support_contact(
            _foot_force_w,
            self._support_reference_normal,
            normal_force_min=float(getattr(self.cfg, "contact_force_threshold", 10.0)),
            normal_ratio_min=0.35,
        )
        _support_contact = _support_contact_parts["contact"]
        _support_contact_f = _support_contact.to(foot_pos.dtype)
        _support_force = _support_contact_parts["normal_force"]
        _prev_support_contact = self._prev_support_contact
        if _prev_support_contact is None or _prev_support_contact.shape != _support_contact_f.shape:
            _prev_support_contact = torch.ones_like(_support_contact_f)
        _support_landing_mask = (
            _support_contact & (_prev_support_contact < 0.5)
        ).to(foot_pos.dtype)
        # 轻脚奖励只接受法向承重触地；立面碰撞只形成瞬时响应，不能伪装成落脚。
        if self._td_impact:
            td_mask = _support_landing_mask
        _support_settled_contact = _support_contact_f * (_prev_support_contact > 0.5).to(foot_pos.dtype)
        self._prev_support_contact = _support_contact_f.detach().clone()
        _rel_foot_w = foot_pos - rd.root_pos_w[:, None, :]
        _qfb = rd.root_quat_w[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
        _foot_b = quat_rotate_inverse(_qfb, _rel_foot_w.reshape(-1, 3)).reshape(N, 4, 3)
        _rel_foot_vel_w = foot_vel - rd.root_lin_vel_w[:, None, :]
        _rel_foot_vel_rot_b = quat_rotate_inverse(
            _qfb, _rel_foot_vel_w.reshape(-1, 3)
        ).reshape(N, 4, 3)
        _foot_rel_vel_b = _rel_foot_vel_rot_b - torch.cross(
            rd.root_ang_vel_b[:, None, :].expand(-1, 4, -1), _foot_b, dim=-1
        )
        # 落地速度必须取触地前一帧的向下速度。触地后当前帧常已反弹向上，
        # 对其取绝对值会把反弹和接触抖动误判为重落脚。
        touchdown_down_vz = torch.clamp(-self._prev_foot_vz, min=0.0)
        self._prev_foot_vz.copy_(foot_vel[:, :, 2].detach())
        # 所有滑移相关项统一使用接触点平面速度。
        # foot link 质心速度包含脚球滚动，不等同于接触点滑移；接触点速度更接近诊断语义。
        # touchdown_vz 仍使用 foot link z 速度，以匹配落脚冲击指标。
        _foot_ang = rd.body_ang_vel_w[:, self.foot_indexes, :]                        # (N,4,3)
        _r_cp = (
            -taili_geometry.FOOT_RADIUS
            * self._support_reference_normal[:, None, :]
        )
        _foot_cp_vel = foot_vel + torch.cross(_foot_ang, _r_cp.expand_as(_foot_ang), dim=-1)
        foot_vel_xy = _foot_cp_vel[:, :, :2].norm(dim=-1)
        terrain_rise_ahead = torch.zeros(N, device=dev)
        terrain_drop_ahead = torch.zeros(N, device=dev)
        _scan_probe_zero = torch.zeros((), device=dev)
        _scan_probe_one = torch.ones((), device=dev)
        terrain_scan_probe = {
            "scan_ok": _scan_probe_zero,
            "scan_failed": _scan_probe_one,
            "scan_stage": _scan_probe_zero,
            "finite_frac": _scan_probe_zero,
            "hits_z_min": _scan_probe_zero,
            "hits_z_max": _scan_probe_zero,
            "hits_z_span_mean": _scan_probe_zero,
            "rel_x_min": _scan_probe_zero,
            "rel_x_max": _scan_probe_zero,
            "rel_y_min": _scan_probe_zero,
            "rel_y_max": _scan_probe_zero,
            "cmd_world_x_mean": _scan_probe_zero,
            "cmd_world_y_mean": _scan_probe_zero,
            "ahead_count_mean": _scan_probe_zero,
            "ahead_count_max": _scan_probe_zero,
            "ahead_has_frac": _scan_probe_zero,
            "ahead_s_min": _scan_probe_zero,
            "ahead_s_max": _scan_probe_zero,
            "support_z_mean": _scan_probe_zero,
            "support_z_min": _scan_probe_zero,
            "support_z_max": _scan_probe_zero,
            "ahead_max_z_mean": _scan_probe_zero,
            "ahead_min_z_mean": _scan_probe_zero,
            "raw_rise_mean": _scan_probe_zero,
            "raw_rise_max": _scan_probe_zero,
            "raw_drop_mean": _scan_probe_zero,
            "raw_drop_max": _scan_probe_zero,
        }
        _support_z = terrain_h
        _ground_z = terrain_h[:, None].expand(-1, 4)
        # 2. 高度扫描与 clearance：估计每只脚附近地面、前方 rise/drop 和地形探针。
        # 地形感知抬脚高度：每只脚相对其附近地面计算 clearance。
        # 平地目标保持低摆腿；粗糙/台阶地形通过 local_obstacle_h 提高目标高度。
        try:
            _hits = self._height_scanner.data.ray_hits_w                              # (N,P,3) world
            terrain_scan_probe["scan_stage"] = torch.full((), 1.0, device=dev)
            _finite_hits = torch.isfinite(_hits[:, :, 2])
            _hz_valid = torch.where(_finite_hits, _hits[:, :, 2], torch.zeros_like(_hits[:, :, 2]))
            _hz = torch.nan_to_num(_hits[:, :, 2], nan=0.0, posinf=0.0, neginf=0.0)
            _dxy = foot_pos[:, :, None, :2] - _hits[:, None, :, :2]                    # (N,4,P,2)
            _nearest = torch.argmin(torch.nan_to_num((_dxy * _dxy).sum(dim=-1), nan=1e9), dim=-1)  # (N,4)
            _ground_z = torch.gather(_hz, 1, _nearest)                                # (N,4)
            terrain_scan_probe["scan_stage"] = torch.full((), 2.0, device=dev)
            foot_clearance_terr = torch.clamp(
                taili_geometry.sole_clearance(foot_pos[:, :, 2], _ground_z),
                min=0.0,
            )                                                                         # (N,4)，足底相对局部地面。
            terrain_scan_probe["scan_stage"] = torch.full((), 3.0, device=dev)
            terrain_scan_probe.update({
                "finite_frac": _finite_hits.float().mean(),
                "hits_z_min": _hz_valid.min(),
                "hits_z_max": _hz_valid.max(),
                "hits_z_span_mean": (_hz_valid.max(dim=1).values - _hz_valid.min(dim=1).values).mean(),
                "support_z_mean": _support_z.mean(),
                "support_z_min": _support_z.min(),
                "support_z_max": _support_z.max(),
            })
            terrain_scan_probe["scan_stage"] = torch.full((), 31.0, device=dev)
            _rough = self._terrain_ctx[:, 2] if self._terrain_ctx.shape[-1] > 2 else torch.zeros(N, device=dev)
            terrain_scan_probe["scan_stage"] = torch.full((), 32.0, device=dev)
            local_obstacle_h = (torch.clamp(
                (_rough - float(getattr(self.cfg, "clr_rough_flat", 0.01)))
                / max(float(getattr(self.cfg, "clr_rough_span", 0.12)), 1e-6), 0.0, 1.0)
                * float(getattr(self.cfg, "clr_rough_bonus_max", 0.26)))              # 平地约 0，粗糙地形逐步升高。
            terrain_scan_probe["scan_stage"] = torch.full((), 41.0, device=dev)
            _relxy_tp = _hits[:, :, :2] - self.robot.data.root_pos_w[:, None, :2]
            terrain_scan_probe["scan_stage"] = torch.full((), 42.0, device=dev)
            terrain_scan_probe["scan_stage"] = torch.full((), 43.0, device=dev)
            _cmdw_tp = _command_xy_world_from_root_yaw(self.robot.data.root_quat_w, self.commands[:, :2])
            terrain_scan_probe["scan_stage"] = torch.full((), 44.0, device=dev)
            _cn_tp = _cmdw_tp / _cmdw_tp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            _ahead_s_tp = (_relxy_tp * _cn_tp[:, None, :]).sum(-1)
            _ahead_tp = _ahead_s_tp > 0.15
            _has_ahead_tp = _ahead_tp.any(dim=1)
            terrain_scan_probe["scan_stage"] = torch.full((), 45.0, device=dev)
            _hz_hi_tp = torch.nan_to_num(_hits[:, :, 2], nan=-1e3, posinf=-1e3, neginf=-1e3)
            _hz_lo_tp = torch.nan_to_num(_hits[:, :, 2], nan=1e3, posinf=1e3, neginf=1e3)
            _ahead_max_tp = torch.where(_ahead_tp, _hz_hi_tp, torch.full_like(_hz_hi_tp, -1e3)).max(dim=1).values
            _ahead_min_tp = torch.where(_ahead_tp, _hz_lo_tp, torch.full_like(_hz_lo_tp, 1e3)).min(dim=1).values
            _ahead_count_tp = _ahead_tp.float().sum(dim=1)
            _ahead_s_min_tp = torch.where(_ahead_tp, _ahead_s_tp, torch.full_like(_ahead_s_tp, 1e3)).min(dim=1).values
            _ahead_s_max_tp = torch.where(_ahead_tp, _ahead_s_tp, torch.full_like(_ahead_s_tp, -1e3)).max(dim=1).values
            _relx = _relxy_tp[:, :, 0]
            _rely = _relxy_tp[:, :, 1]
            terrain_scan_probe = {
                "scan_ok": torch.ones((), device=dev),
                "scan_failed": torch.zeros((), device=dev),
                "scan_stage": torch.full((), 5.0, device=dev),
                "finite_frac": _finite_hits.float().mean(),
                "hits_z_min": _hz_valid.min(),
                "hits_z_max": _hz_valid.max(),
                "hits_z_span_mean": (_hz_valid.max(dim=1).values - _hz_valid.min(dim=1).values).mean(),
                "rel_x_min": _relx.min(),
                "rel_x_max": _relx.max(),
                "rel_y_min": _rely.min(),
                "rel_y_max": _rely.max(),
                "cmd_world_x_mean": _cn_tp[:, 0].mean(),
                "cmd_world_y_mean": _cn_tp[:, 1].mean(),
                "ahead_count_mean": _ahead_count_tp.mean(),
                "ahead_count_max": _ahead_count_tp.max(),
                "ahead_has_frac": _has_ahead_tp.float().mean(),
                "ahead_s_min": torch.where(_has_ahead_tp, _ahead_s_min_tp, torch.zeros_like(_ahead_s_min_tp)).mean(),
                "ahead_s_max": torch.where(_has_ahead_tp, _ahead_s_max_tp, torch.zeros_like(_ahead_s_max_tp)).mean(),
                "support_z_mean": _support_z.mean(),
                "support_z_min": _support_z.min(),
                "support_z_max": _support_z.max(),
                "ahead_max_z_mean": torch.where(_has_ahead_tp, _ahead_max_tp, torch.zeros_like(_ahead_max_tp)).mean(),
                "ahead_min_z_mean": torch.where(_has_ahead_tp, _ahead_min_tp, torch.zeros_like(_ahead_min_tp)).mean(),
                "raw_rise_mean": torch.where(_has_ahead_tp, _ahead_max_tp - _support_z, torch.zeros_like(_support_z)).mean(),
                "raw_rise_max": torch.where(_has_ahead_tp, _ahead_max_tp - _support_z, torch.zeros_like(_support_z)).max(),
                "raw_drop_mean": torch.where(_has_ahead_tp, _support_z - _ahead_min_tp, torch.zeros_like(_support_z)).mean(),
                "raw_drop_max": torch.where(_has_ahead_tp, _support_z - _ahead_min_tp, torch.zeros_like(_support_z)).max(),
            }
            terrain_rise_ahead = torch.where(
                _has_ahead_tp, torch.clamp(_ahead_max_tp - _support_z, min=0.0), terrain_rise_ahead
            )
            terrain_drop_ahead = torch.where(
                _has_ahead_tp, torch.clamp(_support_z - _ahead_min_tp, min=0.0), terrain_drop_ahead
            )
            # 离散地形上根据前方升高量提高抬脚目标，避免楼梯踏面粗糙度低时误退回平地目标。
            self._ensure_gate_mask()
            _disc = getattr(self, "_discrete_terrain_mask", None)
            if _disc is not None and not bool(getattr(self.cfg, "strict_blind_terrain_reward", True)):
                # 方向感知强制抬脚：只有沿命令方向前方地面升高时才提高目标，
                # 下楼或平地不强制高抬腿。
                _hitsw = self._height_scanner.data.ray_hits_w                       # (N,P,3)
                _relxy = _hitsw[:, :, :2] - self.robot.data.root_pos_w[:, None, :2]
                _cmdw = _command_xy_world_from_root_yaw(self.robot.data.root_quat_w, self.commands[:, :2])
                _cn = _cmdw / _cmdw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                _ahead = (_relxy * _cn[:, None, :]).sum(-1) > 0.15                   # 指令方向前方的扫描点
                _hz2 = torch.nan_to_num(_hitsw[:, :, 2], nan=-1e3, posinf=-1e3, neginf=-1e3)
                _ahead_max = torch.where(_ahead, _hz2, torch.full_like(_hz2, -1e3)).max(dim=1).values
                _rise = _ahead_max - terrain_h                                       # >0 = 前方升高
                _rise = terrain_rise_ahead
                _ascending = _disc & (_rise > 0.04) & (self.commands[:, :2].norm(dim=-1) > 0.1)
                # 这里只表达障碍高度；统一余量由 terrain_clearance_targets 加一次。
                _adaptive = _rise.clamp(0.04, 0.35)
                local_obstacle_h = torch.where(_ascending, torch.maximum(local_obstacle_h, _adaptive), local_obstacle_h)
        except Exception:
            foot_clearance_terr = torch.clamp(
                taili_geometry.sole_clearance(foot_pos[:, :, 2], terrain_h[:, None]),
                min=0.0,
            )
            local_obstacle_h = torch.zeros(N, device=dev)
            _ground_z = terrain_h[:, None].expand(-1, 4)
        # 楼梯信用只使用当前真实承重足。机身跳起、摆动腿伸高和立面碰撞都不能改变该高度。
        _loaded_support = terrain_curriculum.loaded_support_height(
            ground_z=_ground_z,
            support_contact=_support_contact,
            support_force=_support_force,
            fallback_height=terrain_h,
        )
        _observed_support_z = _loaded_support["height"]
        _support_valid = _loaded_support["valid"]
        _prev_support = self._prev_support_z.to(device=dev)
        if hasattr(self, "_episode_support_initialized"):
            first_loaded = (~self._episode_support_initialized) & _support_valid
            _prev_support = torch.where(first_loaded, _observed_support_z, _prev_support)
        else:
            first_loaded = torch.zeros(N, dtype=torch.bool, device=dev)
        _support_z = terrain_curriculum.filter_loaded_support_height(
            observed_height=_observed_support_z,
            previous_height=_prev_support,
            valid=_support_valid,
            alpha=float(getattr(self.cfg, "terrain_support_height_alpha", 0.20)),
        )
        _support_delta = torch.where(
            _support_valid,
            _support_z - _prev_support,
            torch.zeros_like(_support_z),
        )

        # 水平主导碰撞是碰阶后的瞬时本体证据，但不是承重。它只释放当前碰撞足的
        # 平地轨迹约束，不创建锁存、阶段或超时状态。
        _force_norm = torch.linalg.norm(_foot_force_w, dim=-1)
        _normal_force = _support_contact_parts["normal_force"]
        _tangent_force = torch.sqrt(
            torch.clamp(_force_norm.square() - _normal_force.square(), min=0.0)
        )
        _collision_ratio = _tangent_force / _normal_force.clamp(min=1.0)
        _collision_score_by_foot = torch.clamp(
            (_collision_ratio - float(getattr(self.cfg, "terrain_collision_ratio_start", 1.25)))
            / max(float(getattr(self.cfg, "terrain_collision_ratio_span", 1.50)), 1e-6),
            0.0,
            1.0,
        ) * (_force_norm > 5.0).to(foot_pos.dtype)

        # 诊断证明低台阶首先撞到腿段或机身时，足端水平力仍为零。接触传感器覆盖
        # 整机，因此把当前 hip/thigh/calf 接触按腿聚合；机身接触只按当前平移命令
        # 分配给迎障腿。所有信号都来自已发生的接触，不读取前方扫描或保留事件状态。
        _all_contact_force_w = self._contact_sensor.data.net_forces_w
        _body_force_start = float(getattr(self.cfg, "terrain_body_collision_force_start", 5.0))
        _body_force_span = float(getattr(self.cfg, "terrain_body_collision_force_span", 45.0))
        _limb_collision_scores = []
        for _body_ids in self._limb_contact_ids:
            if _body_ids:
                _body_scores = terrain_curriculum.unexpected_body_contact_score(
                    _all_contact_force_w[:, _body_ids, :],
                    force_start=_body_force_start,
                    force_span=_body_force_span,
                )
                _limb_collision_scores.append(_body_scores.max(dim=1).values)
            else:
                _limb_collision_scores.append(torch.zeros(N, device=dev, dtype=foot_pos.dtype))
        _limb_collision_score = torch.stack(_limb_collision_scores, dim=1)
        if self._base_contact_ids:
            _base_collision_score = terrain_curriculum.unexpected_body_contact_score(
                _all_contact_force_w[:, self._base_contact_ids, :],
                force_start=_body_force_start,
                force_span=_body_force_span,
            ).max(dim=1).values
        else:
            _base_collision_score = torch.zeros(N, device=dev, dtype=foot_pos.dtype)
        _body_collision_by_leg = terrain_curriculum.map_body_collision_to_legs(
            limb_collision_score=_limb_collision_score,
            base_collision_score=_base_collision_score,
            command_xy=self.commands[:, :2],
        )
        _collision_score_by_foot = torch.maximum(
            _collision_score_by_foot,
            _body_collision_by_leg,
        )
        _up_collision_scope = (
            self._stairs_up_terrain_mask
            & (torch.linalg.norm(self.commands[:, :2], dim=-1) > 0.10)
        )
        _collision_trace = terrain_curriculum.update_terrain_collision_trace(
            current_response=_collision_score_by_foot,
            previous_trace=self._terrain_collision_trace,
            dt=float(self.step_dt),
            decay_time=float(getattr(self.cfg, "terrain_collision_trace_decay_time", 0.40)),
            active=_up_collision_scope,
        )
        self._terrain_collision_trace.copy_(_collision_trace.detach())
        _collision_full_scale = max(
            float(getattr(self.cfg, "terrain_collision_response_full_scale", 0.35)),
            1e-6,
        )
        _terrain_clearance_strength = torch.clamp(
            _collision_trace / _collision_full_scale,
            0.0,
            1.0,
        )
        _current_collision_strength = torch.clamp(
            _collision_score_by_foot / _collision_full_scale,
            0.0,
            1.0,
        )
        self._terrain_limb_collision_response.copy_(_limb_collision_score.detach())
        self._terrain_base_collision_response.copy_(_base_collision_score.detach())
        _terrain_contact = terrain_curriculum.continuous_terrain_contact_response(
            ground_z=_ground_z,
            support_contact=_support_contact,
            support_force=_support_force,
            current_height=_support_z,
            previous_height=_prev_support,
            landing_mask=_support_landing_mask,
            collision_score_by_foot=_collision_score_by_foot,
            height_scale=float(getattr(self.cfg, "terrain_response_height_scale", 0.04)),
            delta_scale=float(getattr(self.cfg, "terrain_response_delta_scale", 0.025)),
            height_deadband=float(getattr(self.cfg, "terrain_response_height_deadband", 0.008)),
            delta_deadband=float(getattr(self.cfg, "terrain_response_delta_deadband", 0.008)),
        )
        self._loaded_support_z = _support_z.detach().clone()
        self._loaded_support_valid = _support_valid.detach().clone()
        self._terrain_support_delta.copy_(_support_delta.detach())
        self._terrain_support_dispersion.copy_(_terrain_contact["dispersion"].detach())
        self._terrain_support_valid.copy_(_support_valid.detach())
        self._terrain_leg_response.copy_(_terrain_contact["leg_response"].detach())
        self._terrain_contact_foot_mask.copy_(
            (_terrain_contact["leg_response"] > 0.05).detach()
        )
        if hasattr(self, "_episode_support_initialized"):
            self._episode_start_support_z = torch.where(
                first_loaded,
                _support_z.detach(),
                self._episode_start_support_z,
            )
            self._episode_support_initialized |= first_loaded
        # 3. 质量窗口与稳定门控：这些值既影响奖励，也会被父类课程门控读取。
        # 非平地核心参考只由已经承重的足端历史形成。平地仍严格使用世界水平，
        # 地形只替换参考面，不降低姿态、角速度、角加速度和高度约束的总权重。
        _support_reference = terrain_curriculum.update_contact_support_reference(
            foot_position_w=foot_pos,
            contact=_support_settled_contact > 0.5,
            support_force=_support_force,
            previous_point_w=self._support_reference_point,
            previous_normal_w=self._support_reference_normal,
            alpha=float(getattr(self.cfg, "support_reference_alpha", 0.10)),
            foot_radius=taili_geometry.FOOT_RADIUS,
        )
        self._support_reference_point.copy_(_support_reference["point_w"].detach())
        self._support_reference_normal.copy_(_support_reference["normal_w"].detach())
        _support_normal_b = quat_rotate_inverse(rd.root_quat_w, self._support_reference_normal)
        _support_tilt_rel = torch.arccos(torch.clamp(_support_normal_b[:, 2], -1.0, 1.0))
        _support_base_h = (
            (rd.root_pos_w - self._support_reference_point) * self._support_reference_normal
        ).sum(dim=-1)
        _flat_core_scope = self._flat_transition_scope_mask().bool()
        base_h = torch.where(_flat_core_scope, base_h, _support_base_h)
        tilt_rel = torch.where(_flat_core_scope, tilt_rel, _support_tilt_rel)
        quality = self._update_quality_windows(
            _support_contact_f,
            _support_settled_contact,
            _support_landing_mask,
            foot_vel_xy,
            base_h,
            tilt_rel,
            moving,
            rd.root_lin_vel_b,
            rd.root_ang_vel_b,
            _support_force,
        )
        _foot_rel_xy = foot_pos[:, :, :2] - rd.root_pos_w[:, None, :2]
        _yaw_moment = (
            _foot_rel_xy[:, :, 0] * _foot_force_w[:, :, 1]
            - _foot_rel_xy[:, :, 1] * _foot_force_w[:, :, 0]
        ).sum(dim=-1)
        _yaw_moment_scale = (_support_force.sum(dim=-1) * 0.25).clamp(min=1.0)
        _yaw_moment_norm = torch.clamp(_yaw_moment / _yaw_moment_scale, -1.0, 1.0)
        self._base_h_min = float(base_h.min())
        self._tilt_deg_max = float(torch.rad2deg(tilt_rel).max())
        support_stand = torch.clamp((3.0 - cc) / 3.0, 0.0, 1.0)
        support_move = torch.clamp((2.0 - cc) / 2.0, 0.0, 1.0)
        support_instab = torch.where(moving > 0.5, support_move, support_stand)
        support_instab = torch.maximum(support_instab,
                                       torch.clamp((tilt_rel - math.radians(10)) / math.radians(20), 0.0, 1.0))
        support_instab = torch.maximum(support_instab, torch.clamp(quality["height_low_risk"], 0.0, 1.0))
        support_instab = torch.maximum(support_instab, torch.clamp(quality["tilt_high_risk"], 0.0, 1.0))

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.reset_terminated if hasattr(self, "reset_terminated") else torch.zeros(N, dtype=torch.bool, device=dev)
        terminal_window = (terminated & ~time_out).float()

        gate = taili_reward.stable_motion_gate(base_h, tilt_rel, support_instab,
                                               torch.zeros(N, device=dev), terminal_window, cfg)
        # 真实终止与软稳定质量必须分开。高度和倾斜在终止前由连续软门控表达；
        # 只在真实 terminal 或数值非法时清零，保留低姿态恢复所需的任务梯度。
        _hard_survival_gate = (
            torch.isfinite(base_h)
            & torch.isfinite(tilt_rel)
            & (terminal_window <= 0.5)
        ).to(rd.joint_pos.dtype)
        torque_limit = self.robot.actuators["legs"].effort_limit
        yaw_speed_equiv = float(getattr(self.cfg, "gait_yaw_speed_equiv", 0.15))

        # 地形响应只来自当前已经发生的承重高差、支撑高度变化或水平主导碰撞。
        # 平地作用域始终硬屏蔽；这里没有阶段、锁存、先导足或超时记忆。
        _terrain_response_raw = _terrain_contact["response"]
        _flat_reward_scope = self._flat_transition_scope_mask()
        _terrain_response = terrain_curriculum.scope_terrain_response(
            _terrain_response_raw, _flat_reward_scope
        )
        _discrete_style_scope = getattr(
            self,
            "_discrete_terrain_mask",
            torch.zeros(N, dtype=torch.bool, device=dev),
        )
        _discrete_event_active = _discrete_style_scope & (_terrain_response > 0.05)
        _terrain_style_relief_progress = _terrain_response
        # 通用进展不能被碰阶状态截断。地形接触时允许降速的连续比例由
        # taili_reward 的 terrain_tracking_scale 统一处理，避免重复缩放。
        _terrain_progress_gate = torch.ones_like(_terrain_response)
        _terrain_tracking_event_scale = torch.ones_like(_terrain_response)
        self._blind_amp_terrain_response = _terrain_response.detach()
        self._flat_false_terrain_response = float(
            (_terrain_response_raw * _flat_reward_scope.to(_terrain_response_raw.dtype)).mean()
        )

        # 触地承重跃迁按体重归一化；动作二阶差分只在摆动腿上使用。
        body_weight = max(float(getattr(self.cfg, "robot_mass_kg", 39.0)) * 9.81, 1e-6)
        _foot_force_norm = _support_force / body_weight
        # 只测承重上升，不惩罚卸载。接触力通常在 touchdown 后一两个控制帧达到峰值，
        # 因此下方短窗口保存最大上升量，而不是只采触地瞬间这一帧。
        _touchdown_force_rate = torch.clamp(
            _foot_force_norm - self._prev_foot_force_norm,
            min=0.0,
        )
        self._prev_foot_force_norm.copy_(_foot_force_norm.detach())
        _base_ang_accel_raw = (
            rd.root_ang_vel_b - self._prev_base_ang_vel
        ) / max(float(self.step_dt), 1e-6)
        self._prev_base_ang_vel.copy_(rd.root_ang_vel_b.detach())
        _ang_filter_beta = min(max(
            float(getattr(cfg, "base_ang_vel_filter_beta", 0.75)), 0.0
        ), 0.98)
        _base_ang_vel_filtered_next = (
            _ang_filter_beta * self._base_ang_vel_filtered
            + (1.0 - _ang_filter_beta) * rd.root_ang_vel_b
        )
        _base_ang_accel = (
            _base_ang_vel_filtered_next - self._base_ang_vel_filtered
        ) / max(float(self.step_dt), 1e-6)
        self._base_ang_vel_filtered.copy_(_base_ang_vel_filtered_next.detach())

        # touchdown 是稀疏事件。保持一个很短的代价窗口，把触地前速度、水平滑移和
        # 承重跃迁归因给导致落脚的连续动作，而不是只惩罚单个物理帧。
        _td_event_mask = td_mask if td_mask is not None else landing_mask
        _hold_steps = max(
            1,
            int(round(float(getattr(cfg, "touchdown_hold_s", 0.10)) / max(float(self.step_dt), 1e-6))),
        )
        self._touchdown_hold_steps = torch.clamp(self._touchdown_hold_steps - 1, min=0)
        _td_event = _td_event_mask > 0.5
        self._touchdown_hold_steps = torch.where(
            _td_event,
            torch.full_like(self._touchdown_hold_steps, _hold_steps),
            self._touchdown_hold_steps,
        )
        self._touchdown_vz_hold = torch.where(_td_event, touchdown_down_vz.detach(), self._touchdown_vz_hold)
        self._touchdown_xy_hold = torch.where(_td_event, foot_vel_xy.detach(), self._touchdown_xy_hold)
        _force_rate_now = _touchdown_force_rate.detach()
        self._touchdown_force_hold = torch.where(
            _td_event,
            _force_rate_now,
            self._touchdown_force_hold,
        )
        _force_window_active = self._touchdown_hold_steps > 0
        self._touchdown_force_hold = torch.where(
            _force_window_active,
            torch.maximum(self._touchdown_force_hold, _force_rate_now),
            self._touchdown_force_hold,
        )
        _touchdown_hold_mask = (self._touchdown_hold_steps > 0).to(foot_vel.dtype)
        _action_delta = self.actions - self.last_actions
        _swing_action_accel = _action_delta - self._prev_action_delta
        self._prev_action_delta.copy_(_action_delta.detach())
        _swing_foot_velocity_delta = torch.linalg.norm(foot_vel - self._prev_foot_vel_w, dim=-1)
        self._prev_foot_vel_w.copy_(foot_vel.detach())
        # 关节顺序是 hip[4] | thigh[4] | calf[4]，不能按每腿连续三个关节展开。
        _swing_foot_mask = 1.0 - in_contact
        _swing_joint_mask = taili_obs.foot_mask_to_joint12(_swing_foot_mask)

        # 统一足端轨迹：复用接触窗口估计出的全局相位偏移，保持时间顺序和四足相对时序。
        _trajectory_by_leg = torch.zeros((N, 4), device=dev)
        _terminal_swing_velocity_cost = torch.zeros(N, device=dev)
        _trajectory_weight = float(getattr(cfg, "w_foot_trajectory", 0.0))
        if _trajectory_weight != 0.0:
            _lag_offsets = torch.linspace(-0.25, 0.25, 9, device=dev, dtype=self._gait_phase.dtype)
            _lag_index = self._gait_lag_scores.argmax(dim=1)
            _linear_mag = torch.norm(self.commands[:, :2], dim=1)
            _linear_scope = (_linear_mag > 0.10) & (_linear_mag >= 0.6 * self.commands[:, 2].abs())
            _lag_valid = self._quality_window_ready & _linear_scope
            _phase_lag = torch.where(_lag_valid, _lag_offsets[_lag_index], torch.zeros(N, device=dev))
            _reference_phase = (self._gait_phase + _phase_lag) % 1.0
            _period_now = torch.clamp(
                self.cfg.gait_period - self.cfg.gait_period_slope * (
                    torch.norm(self.commands[:, :2], dim=1)
                    + yaw_speed_equiv * self.commands[:, 2].abs()
                ),
                min=self.cfg.gait_period_min,
                max=self.cfg.gait_period,
            )
            _times = _reference_phase * _period_now
            _foot_ref = foot_reference(
                self.commands,
                _times,
                gait_period=self.cfg.gait_period,
                gait_period_slope=self.cfg.gait_period_slope,
                gait_period_min=self.cfg.gait_period_min,
                yaw_speed_equiv=yaw_speed_equiv,
                clearance_base=self.cfg.base_clearance,
                stance_dx=self.cfg.stance_dx,
            )
            _foot_ref_next = foot_reference(
                self.commands,
                _times + self.step_dt,
                gait_period=self.cfg.gait_period,
                gait_period_slope=self.cfg.gait_period_slope,
                gait_period_min=self.cfg.gait_period_min,
                yaw_speed_equiv=yaw_speed_equiv,
                clearance_base=self.cfg.base_clearance,
                stance_dx=self.cfg.stance_dx,
            )
            _foot_ref_vel = (_foot_ref_next - _foot_ref) / max(float(self.step_dt), 1e-6)
            _trajectory_axis_scale = torch.tensor((0.24, 0.16, 0.18), device=dev).view(1, 1, 3)
            _trajectory_velocity_scale = torch.tensor((1.0, 0.8, 0.8), device=dev).view(1, 1, 3)
            _trajectory_position_by_leg = (
                ((_foot_b - _foot_ref) / _trajectory_axis_scale).square().mean(dim=2)
            )
            _trajectory_velocity_by_leg = (
                ((_foot_rel_vel_b - _foot_ref_vel) / _trajectory_velocity_scale).square().mean(dim=2)
            )
            # 摆动末段单独收紧竖直速度。解析参考在触地端点速度、加速度均为零，
            # 这里直接给落地前减速梯度，而不是等触地后才依赖稀疏冲击成本。
            # 末段减速必须与生成参考轨迹的 best-lag 相位完全一致。旧实现这里使用
            # 原始 gait phase，导致参考允许整体偏相、减速窗口却在另一个时刻触发。
            _trajectory_phase = (self._leg_phases() + _phase_lag[:, None]) % 1.0
            _trajectory_swing = torch.clamp(
                (_trajectory_phase - float(self.cfg.gait_duty))
                / max(1.0 - float(self.cfg.gait_duty), 1e-6),
                0.0,
                1.0,
            )
            _terminal_t = torch.clamp(
                (_trajectory_swing - 0.65) / 0.35,
                0.0,
                1.0,
            )
            _terminal_weight = _terminal_t.square() * (3.0 - 2.0 * _terminal_t)
            _terminal_vz_cost_by_leg = taili_reward.huber_excess(
                (_foot_rel_vel_b[:, :, 2] - _foot_ref_vel[:, :, 2]).abs(),
                float(getattr(cfg, "terminal_swing_vz_free", 0.15)),
                float(getattr(cfg, "terminal_swing_vz_scale", 0.35)),
            ) * _terminal_weight * _swing_foot_mask
            _terminal_denom = (_terminal_weight * _swing_foot_mask).sum(dim=1).clamp(min=1.0)
            _terminal_mean = _terminal_vz_cost_by_leg.sum(dim=1) / _terminal_denom
            _terminal_tail = _terminal_vz_cost_by_leg.max(dim=1).values
            _terminal_swing_velocity_cost = 0.5 * (_terminal_mean + _terminal_tail)
            _velocity_mix = min(max(
                float(getattr(cfg, "foot_trajectory_velocity_mix", 0.45)),
                0.0,
            ), 1.0)
            _trajectory_by_leg = (
                (1.0 - _velocity_mix) * _trajectory_position_by_leg
                + _velocity_mix * _trajectory_velocity_by_leg
            )
            # 只对当前发生高差承重或水平碰撞的足端连续退让平地轨迹，其余腿仍保持
            # 支撑参考。没有先导足锁存，响应消失后约束立即恢复。
            _terrain_leg_relief = min(max(
                float(getattr(self.cfg, "terrain_trajectory_leg_relief", 0.85)), 0.0
            ), 0.95)
            _trajectory_leg_weight = (
                1.0
                - _discrete_event_active[:, None].to(_trajectory_by_leg.dtype)
                * self._terrain_leg_response.to(_trajectory_by_leg.dtype)
                * _terrain_leg_relief
            )
            _trajectory_denom = _trajectory_leg_weight.sum(dim=1).clamp(min=1.0)
            _trajectory_mean = (
                _trajectory_by_leg * _trajectory_leg_weight
            ).sum(dim=1) / _trajectory_denom
            _trajectory_worst = (
                _trajectory_by_leg * _trajectory_leg_weight
            ).max(dim=1).values
            _foot_trajectory_error = 0.50 * _trajectory_mean + 0.50 * _trajectory_worst
        else:
            _foot_trajectory_error = torch.zeros(N, device=dev)
        # 命令过渡时保留弱轨迹连续性，但不让新方向解析轨迹压过真实制动与支撑。
        _transition_trajectory_floor = min(max(
            float(getattr(cfg, "transition_trajectory_floor", 0.15)), 0.0
        ), 1.0)
        _trajectory_style_weight = torch.where(
            _flat_transition_active,
            _transition_trajectory_floor
            + (1.0 - _transition_trajectory_floor) * _policy_steady_f,
            torch.ones_like(_policy_steady_f),
        )
        _foot_trajectory_scope = moving * _trajectory_style_weight

        # 4. RewardInput 组装：字段名是 taili_reward 的契约，后续拆函数时必须保持语义不变。
        # 阶段验收继续使用最弱方向，保证任何方向都不能被跳过。质量成熟度使用
        # 四方向几何平均：比硬最小值平滑，但单个易方向也不能独自把质量惩罚开满。
        _capability_progress = (
            float(getattr(self, "_fwd_prog", 0.0)),
            float(getattr(self, "_back_prog", 0.0)),
            float(getattr(self, "_lat_prog", 0.0)),
            float(getattr(self, "_yaw_prog", 0.0)),
        )
        _quality_start = float(getattr(cfg, "quality_reward_progress_start", 0.20))
        _quality_full = max(float(getattr(cfg, "quality_reward_progress_full", 0.65)), _quality_start + 1e-6)
        _quality_progress_balanced = taili_reward.balanced_progress_geometric(_capability_progress)
        _quality_t = min(max(
            (_quality_progress_balanced - _quality_start) / (_quality_full - _quality_start),
            0.0,
        ), 1.0)
        _quality_capability_gate = _quality_t * _quality_t * (3.0 - 2.0 * _quality_t)
        self._quality_capability_gate = _quality_capability_gate
        _quality_floor = min(max(float(getattr(cfg, "quality_reward_floor", 0.35)), 0.0), 1.0)
        _quality_candidate = _quality_floor + (1.0 - _quality_floor) * _quality_capability_gate
        # 锁存能力成熟度，避免策略通过临时降低 progress 逃避质量。质量强度不再
        # 由惩罚/正奖励比反向缩小；预算只保留为遥测，不能形成“问题越大约束越弱”。
        self._quality_gate_latched = max(float(getattr(self, "_quality_gate_latched", 0.0)), _quality_candidate)
        _quality_gate_value = self._quality_gate_latched
        _refinement_start = float(getattr(cfg, "refinement_reward_progress_start", 0.15))
        _refinement_full = max(
            float(getattr(cfg, "refinement_reward_progress_full", 0.55)),
            _refinement_start + 1e-6,
        )
        _refinement_t = min(max(
            (_quality_progress_balanced - _refinement_start) / (_refinement_full - _refinement_start),
            0.0,
        ), 1.0)
        _refinement_maturity = _refinement_t * _refinement_t * (3.0 - 2.0 * _refinement_t)
        _refinement_floor = min(max(
            float(getattr(cfg, "refinement_reward_floor", 0.08)), 0.0
        ), 1.0)
        _refinement_candidate = _refinement_floor + (1.0 - _refinement_floor) * _refinement_maturity
        self._refinement_gate_latched = max(
            float(getattr(self, "_refinement_gate_latched", 0.0)),
            _refinement_candidate,
        )
        _refinement_gate_value = self._refinement_gate_latched

        _leg_phase_reward = self._leg_phases()
        _swing_fraction = torch.clamp(
            (_leg_phase_reward - float(self.cfg.gait_duty))
            / max(1.0 - float(self.cfg.gait_duty), 1e-6),
            0.0,
            1.0,
        )
        _swing_apex_weight = torch.exp(-0.5 * ((_swing_fraction - 0.5) / 0.22) ** 2)
        _swing_apex_weight *= (_leg_phase_reward >= float(self.cfg.gait_duty)).to(_swing_apex_weight.dtype)

        self._ensure_gate_mask()
        _terrain_expected_direction = (
            self._stairs_up_terrain_mask.to(rd.joint_pos.dtype)
            - self._stairs_down_terrain_mask.to(rd.joint_pos.dtype)
        )
        # IsaacLab 在每个课程行内随机采样 difficulty。奖励取本行台阶高度上界，
        # 只在真实碰撞后作为能力目标使用；Actor 看不到地形类型、等级或该目标。
        _clearance_margin_min = float(getattr(cfg, "terrain_clearance_margin_min", 0.02))
        _clearance_margin_gain = float(getattr(cfg, "terrain_clearance_margin_gain", 0.05))
        _clearance_margin_max = float(getattr(
            cfg,
            "terrain_clearance_margin_max",
            getattr(cfg, "terrain_clearance_margin", 0.035),
        ))
        _probe_fallback = max(
            float(getattr(self.cfg, "terrain_probe_height", 0.12)) - _clearance_margin_max,
            0.01,
        )
        _terrain_step_height = torch.full(
            (N,), _probe_fallback, dtype=rd.joint_pos.dtype, device=dev
        )
        try:
            _terrain_asset_cfg = getattr(self.cfg, "terrain", None)
            if _terrain_asset_cfg is None:
                _terrain_asset_cfg = self.cfg.scene.terrain
            _generator_cfg = _terrain_asset_cfg.terrain_generator
            _terrain_levels = self._terrain.terrain_levels
            _terrain_rows = int(getattr(_generator_cfg, "num_rows", 1))
            _sub_terrains = getattr(_generator_cfg, "sub_terrains", {})
            for _terrain_name, _terrain_mask, _range_name in (
                ("stairs_up", self._stairs_up_terrain_mask, "step_height_range"),
                ("stairs", self._stairs_down_terrain_mask, "step_height_range"),
                ("boxes", self._boxes_terrain_mask, "grid_height_range"),
            ):
                _terrain_cfg = _sub_terrains.get(_terrain_name)
                _height_range = getattr(_terrain_cfg, _range_name, None)
                if _height_range is None:
                    continue
                _row_upper = terrain_curriculum.terrain_parameter_upper_bound(
                    _terrain_levels,
                    num_rows=_terrain_rows,
                    value_min=float(_height_range[0]),
                    value_max=float(_height_range[1]),
                ).to(dtype=rd.joint_pos.dtype, device=dev)
                _terrain_step_height = torch.where(
                    _terrain_mask,
                    _row_upper,
                    _terrain_step_height,
                )
        except Exception:
            pass
        _terrain_clearance_direction = torch.where(
            self._boxes_terrain_mask,
            torch.ones_like(_terrain_expected_direction),
            _terrain_expected_direction,
        )
        # 上楼净空是碰障后的无状态步态约束，不再由承重层、足序或势能状态驱动。
        # 主方向驱动由统一命令跟踪提供；这里仅告诉发生真实碰撞的腿把脚抬开。
        _terrain_clearance_foot_mask = _terrain_clearance_strength
        _terrain_clearance_event_active = (
            _up_collision_scope
            & (_terrain_clearance_foot_mask.max(dim=1).values > 0.05)
        )
        _terrain_clearance_response = torch.maximum(
            _terrain_response,
            _terrain_clearance_foot_mask.max(dim=1).values,
        )
        # 强度只在逐足 mask 中出现一次。旧实现又乘一次全局 response，导致
        # 0.55 的真实碰障只剩约 0.30 的目标，随后被 clearance 容差完全抵消。
        _terrain_clearance_ramp = _terrain_clearance_event_active.to(rd.joint_pos.dtype)
        _terrain_probe_target = None
        if bool(getattr(self.cfg, "strict_blind_terrain_reward", True)):
            _flat_probe = float(cfg.flat_clearance_target)
            _adaptive_clearance_margin = taili_reward.adaptive_terrain_clearance_margin(
                _terrain_step_height,
                minimum=_clearance_margin_min,
                gain=_clearance_margin_gain,
                maximum=_clearance_margin_max,
            )
            _terrain_probe_target = torch.maximum(
                _terrain_step_height + _adaptive_clearance_margin,
                torch.full_like(_terrain_step_height, _flat_probe),
            )

        _joint_pose_error = (rd.joint_pos - self.action_offset).abs()
        _pose_tail_count = max(1, _joint_pose_error.shape[1] // 3)
        _default_pose_error_tail = torch.topk(
            _joint_pose_error, k=_pose_tail_count, dim=1
        ).values.mean(dim=1)
        # 对角步态中前后各包含一只同相腿，前后平均足端 z 应保持一致。该量只来自
        # 当前本体运动学，不向 Actor 增加观测；真实地形响应后由奖励连续退让。
        _front_rear_extension_error = (
            _foot_b[:, 0:2, 2].mean(dim=1)
            - _foot_b[:, 2:4, 2].mean(dim=1)
        ).abs()
        # 任务正信用使用过去时间，不进入 Actor 观测。命令切换或 episode 重置都会
        # 把共同年龄拉回零，防止短时抢速度后终止仍累计完整跟踪收益。
        _episode_age_s = self.episode_length_buf.to(rd.joint_pos.dtype) * float(self.step_dt)
        _command_age_s = torch.as_tensor(
            getattr(self, "_cmd_age_s", _episode_age_s),
            dtype=rd.joint_pos.dtype,
            device=dev,
        )
        _task_credit_age_s = torch.minimum(_episode_age_s, _command_age_s)

        inp = types.SimpleNamespace(
            cmd=self.commands,
            base_lin_vel=rd.root_lin_vel_b,
            base_ang_vel=rd.root_ang_vel_b,
            base_ang_accel=_base_ang_accel,
            base_ang_accel_raw=_base_ang_accel_raw,
            base_h_above_terrain=base_h,
            tilt_rel=tilt_rel,
            stable_motion_gate=gate,
            hard_survival_gate=_hard_survival_gate,
            stand_gate=stand_gate,
            moving_gate=moving,
            task_credit_age_s=_task_credit_age_s,
            terminal_window=terminal_window,
            foot_contact=_support_contact_f,
            clearance_support_contact=_support_contact_f,
            stance_slip_contact=_support_settled_contact,
            desired_foot_contact=(self._leg_phases() < self.cfg.gait_duty).float(),
            foot_vel_xy=foot_vel_xy,                                                  # (N,4)
            duty_by_leg_window=quality["duty_by_leg"],
            duty_quality_window=quality["duty_quality"],
            duty_cycle_valid_window=quality["duty_cycle_valid"],
            contact_period_error_window=quality["period_error"],
            contact_period_valid_window=quality["contact_period_valid"],
            duty_symmetry_window=quality["duty_symmetry_score"],
            duty_target_window=quality["duty_target_score"],
            duty_spread_window=quality["duty_spread"],
            diagonal_pair_window=quality["diag_pair_score"],
            stance_slip_speed_window=quality["slip_speed"],
            stance_slip_high_fraction=quality["slip_high_fraction"],                  # 高滑移脚比例，用于补充滑移尾部风险。
            stance_slip_by_leg_window=quality["slip_by_leg"],
            contact_chatter_window=quality["contact_chatter"],
            support_force_by_leg_window=quality["support_force_by_leg"],
            foot_relative_velocity_body=_foot_rel_vel_b,
            cycle_wxy_energy=quality["wxy_energy"],
            cycle_vz_energy=quality["vz_energy"],
            cycle_yaw_residual_energy=quality["yaw_residual_energy"],
            yaw_support_moment_norm=_yaw_moment_norm,
            touchdown_vz=self._touchdown_vz_hold,
            touchdown_xy_speed=self._touchdown_xy_hold,
            touchdown_mask=_touchdown_hold_mask,
            touchdown_event_mask=td_mask,
            touchdown_persistence_scale=float(getattr(cfg, "touchdown_hold_scale", 0.30)),
            touchdown_force_rate=self._touchdown_force_hold,
            swing_action_accel=_swing_action_accel,
            swing_joint_mask=_swing_joint_mask,
            swing_foot_velocity_delta=_swing_foot_velocity_delta,
            swing_foot_mask=_swing_foot_mask,
            foot_trajectory_error=_foot_trajectory_error,
            foot_trajectory_scope=_foot_trajectory_scope,
            terminal_swing_velocity_cost=_terminal_swing_velocity_cost,
            # 使用 last_air_time 而不是 current_air_time：触地帧 current_air_time 会清零，
            # last_air_time 才保留刚完成摆动相的真实滞空时长。
            foot_air_time=self._contact_sensor.data.last_air_time[:, self._feet_contact_ids],
            # 每个 env 的滞空目标来自 gait clock 的计划摆动时长 period*(1-duty)，
            # 与当前速度自适应周期保持一致，避免反碎步奖励和时钟互相冲突。
            air_time_target=(torch.clamp(
                # yaw 感知 cadence：有效速度包含 yaw 足端杠杆项，
                # 高 yaw 命令会缩短周期，要求更快步频。
                self.cfg.gait_period - self.cfg.gait_period_slope * (
                    torch.norm(self.commands[:, :2], dim=1) + yaw_speed_equiv * self.commands[:, 2].abs()),
                min=self.cfg.gait_period_min, max=self.cfg.gait_period) * (1.0 - self.cfg.gait_duty)),
            foot_clearance=foot_clearance_terr,                                       # 每只脚相对局部地面的高度。
            swing_apex_weight=_swing_apex_weight,
            local_obstacle_h=local_obstacle_h,                                        # 由粗糙度/离散地形派生；平地约为 0。
            terrain_response=_terrain_response,                                       # 当前接触后物理量产生的被动地形响应。
            terrain_clearance_response=_terrain_clearance_response,                   # 最近碰障的短时反应资格迹。
            terrain_style_relief_progress=_terrain_style_relief_progress,             # 真实碰触时连续退让平地风格。
            terrain_lead_foot_mask=self._terrain_contact_foot_mask,
            terrain_event_active=_discrete_event_active,
            terrain_clearance_foot_mask=_terrain_clearance_foot_mask,
            terrain_collision_foot_mask=_current_collision_strength,
            terrain_event_direction=_terrain_clearance_direction,
            terrain_clearance_ramp=_terrain_clearance_ramp,
            terrain_probe_target=_terrain_probe_target,
            terrain_progress_gate=_terrain_progress_gate,
            terrain_tracking_event_scale=_terrain_tracking_event_scale,
            hip_deviation=(rd.joint_pos[:, self._hip_joint_ids]
                           - self.action_offset[:, self._hip_joint_ids]).abs().mean(dim=-1),
            hip_deviation_max=(rd.joint_pos[:, self._hip_joint_ids]
                               - self.action_offset[:, self._hip_joint_ids]).abs().max(dim=-1).values,
            hip_deviation_by_leg=(rd.joint_pos[:, self._hip_joint_ids]
                                  - self.action_offset[:, self._hip_joint_ids]).abs(),
            front_rear_extension_error=_front_rear_extension_error,
            torque=rd.applied_torque,
            torque_limit=torque_limit,
            torque_clamped=(rd.applied_torque.abs() >= torque_limit).float(),
            action=self.actions,
            last_action=self.last_actions,
            default_pose_error=_joint_pose_error.mean(dim=1),
            default_pose_error_tail=_default_pose_error_tail,
            # 制动阶段只保留弱恢复压力，释放后再平滑恢复完整站立姿态约束。
            stand_posture_scope=0.25 + 0.75 * _policy_steady_f,
            amp_reward=torch.zeros(N, device=dev),                                    # skrl AMP 侧另行加入风格奖励。
            body_collision=torch.zeros(N, device=dev),                               # 后续可接入非足端碰撞采样。
            quality_gate=torch.full((N,), _quality_gate_value, device=dev),
            refinement_gate=torch.full((N,), _refinement_gate_value, device=dev),
            heading_error=getattr(self, "_heading_error", torch.zeros(N, device=dev)),
            terminal_reason=None,                                                     # 每个 env 的 terminal 惩罚在下方处理。
        )
        # 5. 统一奖励计算：通用 tracking、步态、滑移、落脚和稳定项都在 taili_reward 中计算。
        comp = taili_reward.compute_reward_components(inp, cfg)
        self._dynamic_mechanism_metrics = dict(comp.get("_dynamic_metrics", {}))
        self._dynamic_mechanism_gates = dict(comp.get("_dynamic_gates", {}))
        self._terrain_pattern_scale_for_amp = comp["terrain_pattern_scale"].detach()
        self._amp_style_scale_for_agent = (
            comp["terrain_pattern_scale"] * _policy_steady_f
        ).detach()
        total = comp["total"] - cfg.w_terminal * terminal_window                      # 每个 env 的 terminal 惩罚。
        # 平地质量必须独立保存，不能被坡面、boxes 或楼梯样本的总体平均掩盖。
        # 这些量仅供课程 Gate 与遥测，不进入 Actor 观测。
        _flat_mask = _flat_reward_scope.bool()
        # 基础课程先验收平地稳态动作；命令切换已有独立的分类型过渡统计，不应
        # 用少量切换尖峰把已经成型的核心质量长期覆盖。
        _flat_core_mask = (
            _flat_mask
            & (_policy_steady_f > 0.5)
            & (moving > 0.5)
        )
        _flat_direction_masks = {
            "fwd": _flat_core_mask & (self.commands[:, 0] > 0.10),
            "back": _flat_core_mask & (self.commands[:, 0] < -0.10),
            "lat": _flat_core_mask & (self.commands[:, 1].abs() > 0.10),
            "yaw": _flat_core_mask & (self.commands[:, 2].abs() > 0.10),
        }
        _linear_direction_names = ("fwd", "back", "lat")
        _all_direction_names = ("fwd", "back", "lat", "yaw")

        def _masked_mean(value, mask, default=0.0):
            value = value.to(device=dev)
            return float(value[mask].mean()) if bool(mask.any()) else float(default)

        def _masked_p95(value, mask, default=0.0):
            value = value.to(device=dev)
            selected = value[mask]
            return float(torch.quantile(selected.float().reshape(-1), 0.95)) if selected.numel() else float(default)

        def _flat_mean(value, default=0.0):
            return _masked_mean(value, _flat_mask, default)

        def _flat_core_mean(value, default=0.0):
            return _masked_mean(value, _flat_core_mask, default)

        def _flat_core_p95(value, default=0.0):
            return _masked_p95(value, _flat_core_mask, default)

        def _direction_extreme(value, names, *, reduction, higher_is_worse, fallback):
            samples = []
            for name in names:
                mask = _flat_direction_masks[name]
                if not bool(mask.any()):
                    continue
                if reduction == "p95":
                    samples.append(_masked_p95(value, mask, fallback))
                else:
                    samples.append(_masked_mean(value, mask, fallback))
            if not samples:
                return float(fallback)
            return float(max(samples) if higher_is_worse else min(samples))

        def _touchdown_p95(mask, value):
            selected_values = value[mask]
            selected_active = self._touchdown_hold_steps[mask] > 0
            selected = selected_values[selected_active]
            return float(torch.quantile(selected.float(), 0.95)) if selected.numel() else 0.0

        def _direction_touchdown_p95(value, fallback):
            samples = [
                _touchdown_p95(_flat_direction_masks[name], value)
                for name in _all_direction_names
                if bool(_flat_direction_masks[name].any())
            ]
            return float(max(samples)) if samples else float(fallback)

        _flat_touchdown_values = self._touchdown_vz_hold[_flat_core_mask]
        _flat_touchdown_active = self._touchdown_hold_steps[_flat_core_mask] > 0
        _flat_touchdown_selected = _flat_touchdown_values[_flat_touchdown_active]
        _flat_touchdown_p95 = (
            float(torch.quantile(_flat_touchdown_selected.float(), 0.95))
            if _flat_touchdown_selected.numel() else 0.0
        )
        _flat_force_rate_values = self._touchdown_force_hold[_flat_core_mask]
        _flat_force_rate_selected = _flat_force_rate_values[_flat_touchdown_active]
        _flat_force_rate_p95 = (
            float(torch.quantile(_flat_force_rate_selected.float(), 0.95))
            if _flat_force_rate_selected.numel() else 0.0
        )
        _flat_ang_accel = torch.sqrt(
            _base_ang_accel_raw[:, 0].square()
            + _base_ang_accel_raw[:, 1].square()
            + _policy_steady_f * _base_ang_accel_raw[:, 2].square()
        )
        _wxy_norm = rd.root_ang_vel_b[:, :2].norm(dim=-1)
        _height_error = (
            base_h
            - float(getattr(
                cfg,
                "flat_move_height_target",
                taili_geometry.NOMINAL_BASE_HEIGHT,
            ))
        ).abs()
        _yaw_residual = torch.sqrt(torch.clamp(quality["yaw_residual_energy"], min=0.0))
        _trajectory_worst = _trajectory_by_leg.max(dim=1).values
        self._flat_quality_metrics = {
            # 核心与轻脚按最差方向验收，不能由其他方向的好样本稀释。
            "tilt_mean": _direction_extreme(
                tilt_rel, _all_direction_names, reduction="mean", higher_is_worse=True,
                fallback=_flat_core_mean(tilt_rel),
            ),
            "tilt_p95": _direction_extreme(
                tilt_rel, _all_direction_names, reduction="p95", higher_is_worse=True,
                fallback=_flat_core_p95(tilt_rel),
            ),
            "wxy_mean": _direction_extreme(
                _wxy_norm, _all_direction_names, reduction="mean", higher_is_worse=True,
                fallback=_flat_core_mean(_wxy_norm),
            ),
            "ang_accel_p95": _direction_extreme(
                _flat_ang_accel, _all_direction_names, reduction="p95", higher_is_worse=True,
                fallback=_flat_core_p95(_flat_ang_accel),
            ),
            "height_error_p95": _direction_extreme(
                _height_error, _all_direction_names, reduction="p95", higher_is_worse=True,
                fallback=_flat_core_p95(_height_error),
            ),
            "touchdown_vz_p95": _direction_touchdown_p95(
                self._touchdown_vz_hold, _flat_touchdown_p95
            ),
            "touchdown_force_rate_p95": _direction_touchdown_p95(
                self._touchdown_force_hold, _flat_force_rate_p95
            ),
            "slip_mean": _direction_extreme(
                quality["slip_speed"], _all_direction_names, reduction="mean", higher_is_worse=True,
                fallback=_flat_core_mean(quality["slip_speed"]),
            ),
            "slip_high": _direction_extreme(
                quality["slip_high_fraction"], _all_direction_names, reduction="mean", higher_is_worse=True,
                fallback=_flat_core_mean(quality["slip_high_fraction"]),
            ),
            "yaw_residual": _direction_extreme(
                _yaw_residual, _all_direction_names, reduction="mean", higher_is_worse=True,
                fallback=_flat_core_mean(_yaw_residual),
            ),
            "trajectory_worst_p95": _direction_extreme(
                _trajectory_worst, _all_direction_names, reduction="p95", higher_is_worse=True,
                fallback=_flat_core_p95(_trajectory_worst),
            ),
            # 直线对角模板不约束 yaw；duty 对称仍按四个方向中的最差方向验收。
            "diagonal_contact": _direction_extreme(
                quality["diag_pair_score"], _linear_direction_names, reduction="mean", higher_is_worse=False,
                fallback=_flat_core_mean(quality["diag_pair_score"]),
            ),
            "duty_target": _direction_extreme(
                quality["duty_target_score"], _linear_direction_names, reduction="mean", higher_is_worse=False,
                fallback=_flat_core_mean(quality["duty_target_score"]),
            ),
            "duty_symmetry": _direction_extreme(
                quality["duty_symmetry_score"], _all_direction_names, reduction="mean", higher_is_worse=False,
                fallback=_flat_core_mean(quality["duty_symmetry_score"]),
            ),
            "duty_valid": _direction_extreme(
                quality["duty_cycle_valid"], _linear_direction_names, reduction="mean", higher_is_worse=False,
                fallback=_flat_core_mean(quality["duty_cycle_valid"]),
            ),
            "period": _direction_extreme(
                quality["period_score"], _linear_direction_names, reduction="mean", higher_is_worse=False,
                fallback=_flat_core_mean(quality["period_score"]),
            ),
            "false_terrain_response": float(getattr(self, "_flat_false_terrain_response", 0.0)),
        }
        # 分方向值进入遥测，便于把“聚合值通过、单方向失败”直接定位到物理方向。
        for _name, _mask in _flat_direction_masks.items():
            if not bool(_mask.any()):
                continue
            self._flat_quality_metrics[f"{_name}_tilt_p95"] = _masked_p95(tilt_rel, _mask)
            self._flat_quality_metrics[f"{_name}_wxy_mean"] = _masked_mean(_wxy_norm, _mask)
            self._flat_quality_metrics[f"{_name}_height_error_p95"] = _masked_p95(_height_error, _mask)
            self._flat_quality_metrics[f"{_name}_touchdown_vz_p95"] = _touchdown_p95(
                _mask, self._touchdown_vz_hold
            )
            self._flat_quality_metrics[f"{_name}_duty_target"] = _masked_mean(
                quality["duty_target_score"], _mask
            )
            self._flat_quality_metrics[f"{_name}_duty_symmetry"] = _masked_mean(
                quality["duty_symmetry_score"], _mask
            )
        # 6. 盲态额外方向辅助奖励：补充需要环境几何或窄方向条件的项。
        _w_lateral_foot = float(getattr(self.cfg, "w_lateral_foot_excursion", 0.0))
        if _w_lateral_foot != 0.0:
            _nom_y = 0.2082 * torch.tensor([1.0, -1.0, 1.0, -1.0], device=dev)
            _margin_y = float(getattr(self.cfg, "lateral_foot_margin", 0.035))
            _scale_y = max(float(getattr(self.cfg, "lateral_foot_scale", 0.12)), 1e-6)
            _lat_excess = torch.clamp((_foot_b[:, :, 1] - _nom_y[None, :]).abs() - _margin_y, min=0.0) / _scale_y
            _no_lat_yaw_cmd = ((self.commands[:, 1].abs() <= 0.05) & (self.commands[:, 2].abs() <= 0.05)).float()
            _lat_cost_by_leg = _lat_excess ** 2
            _lat_cost = 0.5 * (
                _lat_cost_by_leg.mean(dim=1)
                + _lat_cost_by_leg.max(dim=1).values
            )
            comp["lateral_foot_excursion"] = (
                -_w_lateral_foot * _lat_cost
                * _no_lat_yaw_cmd * comp["terrain_pattern_scale"]
                * gate * _transition_task_weight
            )
        else:
            comp["lateral_foot_excursion"] = torch.zeros(N, device=dev)
        total = total + comp["lateral_foot_excursion"]
        # 纯后退命令的窄辅助项。YAML 一直暴露了该参数，但 TP 链路此前没有消费它；
        # 当前 phi0 的真实瓶颈是后退有效推进，不补这个驱动会只剩门控硬卡。
        _back_aux_w = float(getattr(self.cfg, "rew_backward_underspeed", 0.0))
        _back_wrong_w = float(getattr(self.cfg, "rew_backward_wrong_dir", 0.0))
        if _back_aux_w != 0.0 or _back_wrong_w != 0.0:
            _back_cmd = (
                (self.commands[:, 0] < -0.05)
                & (self.commands[:, 1].abs() < 0.05)
                & (self.commands[:, 2].abs() < 0.05)
            )
            _cmd_vx_abs = self.commands[:, 0].abs()
            _lin_tol = torch.maximum(
                torch.full_like(_cmd_vx_abs, float(getattr(self.cfg, "speed_tol_abs", 0.10))),
                float(getattr(self.cfg, "speed_tol_rel", 0.15)) * _cmd_vx_abs,
            )
            _back_signed_speed = -rd.root_lin_vel_b[:, 0]
            _back_deficit = torch.clamp((_cmd_vx_abs - _lin_tol) - _back_signed_speed, min=0.0)
            _back_wrong = torch.clamp(rd.root_lin_vel_b[:, 0], min=0.0)
            comp["backward_underspeed"] = (
                _back_aux_w * _back_deficit * _back_deficit
                + _back_wrong_w * _back_wrong * _back_wrong
            ) * _back_cmd.float() * _policy_steady_f
        else:
            comp["backward_underspeed"] = torch.zeros(N, device=dev)
        total = total + comp["backward_underspeed"]
        # 纯横移命令的窄辅助项。主目标仍由通用 tracking 负责；
        # 该项只补横移欠速，不影响前进、后退和 yaw 样本。
        _lat_aux_w = float(getattr(self.cfg, "rew_lateral_underspeed", 0.0))
        if _lat_aux_w != 0.0:
            _lat_cmd = (
                (self.commands[:, 1].abs() > 0.05)
                & (self.commands[:, 0].abs() < 0.05)
                & (self.commands[:, 2].abs() < 0.05)
            )
            _cmd_vy_abs = self.commands[:, 1].abs()
            _lin_tol = torch.maximum(
                torch.full_like(_cmd_vy_abs, float(getattr(self.cfg, "speed_tol_abs", 0.10))),
                float(getattr(self.cfg, "speed_tol_rel", 0.15)) * _cmd_vy_abs,
            )
            _lat_signed_speed = rd.root_lin_vel_b[:, 1] * torch.sign(self.commands[:, 1])
            _lat_deficit = torch.clamp((_cmd_vy_abs - _lin_tol) - _lat_signed_speed, min=0.0)
            comp["lateral_underspeed"] = (
                _lat_aux_w * _lat_deficit * _lat_deficit * _lat_cmd.float() * _policy_steady_f
            )
        else:
            comp["lateral_underspeed"] = torch.zeros(N, device=dev)
        total = total + comp["lateral_underspeed"]
        # 绊脚惩罚：足端水平接触力显著大于竖直力，通常表示脚踹在台阶沿/竖直面上。
        # 只在真实接触且水平力主导时触发，量级为绊脚足占比。
        _ff3 = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, :]
        _stumble_by_foot = ((_ff3[:, :, :2].norm(dim=-1) > 2.0 * _ff3[:, :, 2].abs().clamp(min=1.0))
                            & (_ff3.norm(dim=-1) > 5.0))
        _stumble = _stumble_by_foot.float().mean(dim=1)
        _stumble_relief = 0.75 * _terrain_response[:, None] * self._terrain_leg_response
        _stumble_penalty_by_foot = (
            _stumble_by_foot.to(foot_pos.dtype) * (1.0 - _stumble_relief)
        )
        comp["stumble"] = torch.zeros(N, device=dev)
        # 7. 楼梯约束：方向和速度任务由统一 tracking/progress 提供；这里不再按
        # 承重层或势能发任务奖励，只保留上下楼所需的直接安全约束。
        comp["terrain_contact_quality"] = torch.zeros(N, device=dev)
        comp["terrain_support_loss"] = torch.zeros(N, device=dev)
        comp["terrain_collapse"] = torch.zeros(N, device=dev)
        comp["terrain_overspeed"] = torch.zeros(N, device=dev)

        _cmd_xy = self.commands[:, :2]
        _cmd_xy_norm = torch.linalg.norm(_cmd_xy, dim=-1)
        _moving_lin_cmd = _cmd_xy_norm > 0.10
        _v_along = torch.sum(rd.root_lin_vel_b[:, :2] * _cmd_xy, dim=-1) / _cmd_xy_norm.clamp(min=1e-6)
        _stair_scope = (
            (self._stairs_up_terrain_mask | self._stairs_down_terrain_mask)
            & _moving_lin_cmd
        )
        _terrain_motion = terrain_curriculum.terrain_motion_credit(
            v_along=_v_along,
            command_speed=_cmd_xy_norm,
            progress_full_ratio=float(getattr(self.cfg, "terrain_progress_full_ratio", 0.35)),
            overspeed_start_ratio=float(getattr(self.cfg, "terrain_overspeed_start_ratio", 1.15)),
            overspeed_span_ratio=float(getattr(self.cfg, "terrain_overspeed_span_ratio", 0.35)),
        )

        _upright_quality = torch.clamp(
            (-rd.projected_gravity_b[:, 2] - 0.60) / 0.40,
            0.0,
            1.0,
        )
        _support_quality = torch.clamp((cc - 1.0) / 2.0, 0.0, 1.0)
        _stair_support_quality = torch.clamp(cc - 1.0, 0.0, 1.0)
        _wxy_quality = torch.clamp(
            1.0 - torch.linalg.norm(rd.root_ang_vel_b[:, :2], dim=-1) / 1.50,
            0.0,
            1.0,
        )
        _terrain_quality = torch.minimum(
            torch.minimum(_upright_quality, _stair_support_quality),
            _wxy_quality,
        )
        _collapse_h = float(getattr(self.cfg, "terrain_collapse_height", 0.44))
        _up_scope = self._stairs_up_terrain_mask.to(foot_pos.dtype) * _moving_lin_cmd.to(foot_pos.dtype)
        _down_scope = self._stairs_down_terrain_mask.to(foot_pos.dtype) * _moving_lin_cmd.to(foot_pos.dtype)
        _direction_ratio = _terrain_motion["ratio"]
        comp["terrain_overspeed"] = (
            -float(getattr(self.cfg, "rew_terrain_overspeed", 0.0))
            * _stair_scope.to(foot_pos.dtype)
            * _terrain_motion["overspeed"]
            * _hard_survival_gate
        )

        # 以两足支撑作为楼梯安全参考，低于该值时连续增加风险。
        _support_loss = torch.clamp((2.0 - cc) / 2.0, 0.0, 1.0)
        # 楼梯碰撞时单足或腾空比平地更容易转化为前扑，因此保留一个有界的额外
        # 安全代价。它直接读取接触数，不包含承重层、足序或势能概念。
        comp["terrain_support_loss"] = (
            -float(getattr(self.cfg, "rew_terrain_support_loss", 0.0))
            * _stair_scope.to(foot_pos.dtype)
            * _support_loss.square()
        )

        # 下楼额外约束失控下坠和重落脚；上楼不复用同一竖直速度模板。
        _down_speed = torch.clamp(-rd.root_lin_vel_b[:, 2], min=0.0)
        _down_target = max(float(getattr(self.cfg, "terrain_down_vz_target", 0.16)), 0.0)
        _down_cap = max(float(getattr(self.cfg, "terrain_down_vz_cap", 0.55)), _down_target + 1e-6)
        _down_speed_fault = torch.clamp(
            (_down_speed - _down_target) / (_down_cap - _down_target),
            0.0,
            1.0,
        )
        _touchdown_active = (_touchdown_hold_mask > 0.5).any(dim=1)
        _touchdown_tail = torch.where(
            _touchdown_active,
            self._touchdown_vz_hold.max(dim=1).values,
            torch.zeros(N, device=dev),
        )
        _down_impact_fault = torch.clamp((_touchdown_tail - 0.45) / 0.75, 0.0, 1.0)
        _down_control_fault = torch.maximum(_down_speed_fault, _down_impact_fault)
        comp["terrain_contact_quality"] = (
            -float(getattr(self.cfg, "rew_terrain_contact_quality", 0.0))
            * _down_scope
            * _down_control_fault.square()
        )

        _collapse_wxy = max(float(getattr(self.cfg, "terrain_collapse_wxy", 1.25)), 1e-6)
        _collapse_height_fault = torch.clamp(
            (_collapse_h - base_h) / max(0.25 * _collapse_h, 0.08),
            0.0,
            1.0,
        )
        _collapse_wxy_fault = torch.clamp(
            (torch.linalg.norm(rd.root_ang_vel_b[:, :2], dim=-1) - _collapse_wxy)
            / _collapse_wxy,
            0.0,
            1.0,
        )
        _collapse_fault = torch.maximum(
            terminal_window,
            torch.maximum(_collapse_height_fault, _collapse_wxy_fault),
        )
        comp["terrain_collapse"] = (
            -float(getattr(self.cfg, "rew_terrain_collapse", 0.0))
            * _stair_scope.to(foot_pos.dtype)
            * _collapse_fault.square()
        )

        terrain_probe = {
            "rise_ahead_mean": terrain_rise_ahead.mean(),
            "rise_ahead_max": terrain_rise_ahead.max(),
            "drop_ahead_mean": terrain_drop_ahead.mean(),
            "drop_ahead_max": terrain_drop_ahead.max(),
            "support_z_mean": _support_z.mean(),
            "support_z_min": _support_z.min(),
            "support_z_max": _support_z.max(),
            "support_valid_mean": _support_valid.float().mean(),
            "support_delta_mean": _support_delta.mean(),
            "support_delta_max": _support_delta.max(),
            "support_delta_min": _support_delta.min(),
            "support_dispersion_mean": _terrain_contact["dispersion"].mean(),
            "support_dispersion_max": _terrain_contact["dispersion"].max(),
            "collision_response_mean": _terrain_contact["collision_response"].mean(),
            "collision_trace_mean": _collision_trace.mean(),
            "collision_trace_max": _collision_trace.max(),
            "clearance_response_mean": _terrain_clearance_response.mean(),
            "limb_collision_response_mean": self._terrain_limb_collision_response.max(dim=1).values.mean(),
            "base_collision_response_mean": self._terrain_base_collision_response.mean(),
            "contact_response_mean": _terrain_response.mean(),
            "disc_frac": self._discrete_terrain_mask.float().mean(),
            "moving_lin_frac": _moving_lin_cmd.float().mean(),
            "upright_frac": (_upright_quality > 0.5).float().mean(),
            "slip_weight_mean": torch.clamp(
                1.0 - quality["slip_speed"] / 0.50, 0.0, 1.0
            ).mean(),
            "no_stumble_mean": (1.0 - _stumble).mean(),
            "common_mean": _terrain_quality.mean(),
            "common_max": _terrain_quality.max(),
            "up_active_mean": (_up_scope > 0.5).float().mean(),
            "up_active_max": (_up_scope > 0.5).float().max(),
            "down_active_mean": (_down_scope > 0.5).float().mean(),
            "down_active_max": (_down_scope > 0.5).float().max(),
            "capability_step_height_mean": _terrain_step_height[_stair_scope].mean() if bool(_stair_scope.any()) else torch.zeros((), device=dev),
            "capability_step_height_max": _terrain_step_height[_stair_scope].max() if bool(_stair_scope.any()) else torch.zeros((), device=dev),
            "direction_ratio_mean": torch.where(
                _stair_scope, _direction_ratio, torch.zeros_like(_direction_ratio)
            ).mean(),
            "overspeed_fault_mean": _terrain_motion["overspeed"].mean(),
            "support_loss_mean": _support_loss.mean(),
            "down_control_fault_mean": _down_control_fault.mean(),
            "collapse_fault_mean": _collapse_fault.mean(),
        }
        terrain_probe.update(terrain_scan_probe)
        terrain_probe.update({
            "support_z_mean": _support_z.mean(),
            "support_z_min": _support_z.min(),
            "support_z_max": _support_z.max(),
        })
        # 地形类型专属安全量只做诊断。上下楼仍由同一方向驱动、核心稳定、支撑、
        # 触地和 terminal 约束，避免不可观测的 terrain mask 改变训练目标。
        # 第一次足端碰障是盲态感知事件，不直接扣分；持续不前进会自然失去任务收益。
        comp["stumble"] = -_stumble_penalty_by_foot.float().mean(dim=1)
        # live joint imitation：朝干净解析参考步态的正向吸引项。
        # 它直接约束关节轨迹，使软落脚、足端回收和左右对称进入奖励，而不只依赖 AMP 判别器。
        _w_imitate = float(getattr(self.cfg, "w_imitate_live", 0.0))
        if _w_imitate != 0.0:
            # 相位对齐：flat_reference 内部用平面速度重算周期。
            # 这里传入 self._gait_phase * T_im，使解析参考相位与环境 gait clock 对齐。
            # T_im 必须保持平面速度周期，以匹配 flat_reference 内部定义。
            _yaw_speed_equiv = float(getattr(self.cfg, "gait_yaw_speed_equiv", 0.15))
            _spd_im = torch.norm(self.commands[:, :2], dim=1) + _yaw_speed_equiv * self.commands[:, 2].abs()
            _T_im = torch.clamp(self.cfg.gait_period - self.cfg.gait_period_slope * _spd_im,
                                min=self.cfg.gait_period_min, max=self.cfg.gait_period)
            _amp_terrain_ctx = self._amp_terrain_context()
            _rough_im = _amp_terrain_ctx[:, 2] if self.cfg.terrain_ctx_dim >= 3 else None
            _jp_ref = flat_reference(
                self.commands, self._gait_phase * _T_im,
                gait_period=self.cfg.gait_period, gait_period_slope=self.cfg.gait_period_slope,
                gait_period_min=self.cfg.gait_period_min, yaw_speed_equiv=_yaw_speed_equiv,
                clearance_base=self.cfg.base_clearance,
                roughness=_rough_im, clearance_rough_gain=self.cfg.ref_clearance_rough_gain,
                stance_dx=self.cfg.stance_dx, jp_only=True, iters=12)                  # 快路径：只生成关节位置，12 次 IK。
            # flat_reference 输出 clip dof 顺序；motion_dof_indexes 映射到仿真关节顺序。
            _jp_ref = _jp_ref[:, self.motion_dof_indexes]
            _jdiff = rd.joint_pos - _jp_ref                                            # (N,12)，仿真与参考关节误差。
            # 前进锚点：pure-forward 样本也需要参考轨迹约束。
            # 参考轨迹随速度缩放，只约束步态形状，不直接替代速度跟踪。
            _turn_mag = self.commands[:, 1].abs() + 0.30 * self.commands[:, 2].abs()
            _fwd_mag = self.commands[:, 0].abs()
            _turn_frac = _turn_mag / (_turn_mag + _fwd_mag + 1e-3)
            _turn_frac_floored = torch.clamp(_turn_frac, min=float(getattr(self.cfg, "imitate_fwd_floor", 0.3)))
            _sigma_im = float(getattr(self.cfg, "imitate_sigma_live", 0.12))
            # 当前碰障或换层足连续退让解析参考，其余腿保持支撑；不使用先导足记忆。
            _lead_joint_mask = taili_obs.foot_mask_to_joint12(
                self._terrain_leg_response.to(_jdiff.dtype)
            )
            _joint_relief = min(max(
                float(getattr(self.cfg, "terrain_trajectory_leg_relief", 0.85)), 0.0
            ), 0.95)
            _joint_style_weight = (
                1.0
                - _discrete_event_active[:, None].to(_jdiff.dtype)
                * _lead_joint_mask
                * _joint_relief
            )
            _joint_mse = (
                _jdiff.pow(2) * _joint_style_weight
            ).sum(dim=1) / _joint_style_weight.sum(dim=1).clamp(min=1.0)
            # 碰障后按当前物理响应退让，为大幅屈伸和前后腿换层留出空间；响应消失后
            # 自动恢复，不改变平地样本。
            _terrain_style_scale = comp.get("terrain_pattern_scale", torch.ones(N, device=dev))
            _progress_style_scale = comp.get("progress_style_gate", torch.ones(N, device=dev))
            comp["imitate"] = (_w_imitate * _turn_frac_floored
                               * torch.exp(-_joint_mse / _sigma_im) * moving
                               * _terrain_style_scale
                               * _progress_style_scale
                               * _policy_steady_f)   # 制动窗口由策略根据真实状态完成，不强拉新方向参考。
            total = total + comp["imitate"]
            # 阶段门控风格指标：运动 env 中相对参考的平均关节误差。
            _mv = moving > 0.5
            self._style_err = (float(_jdiff.abs().mean(dim=1)[_mv].mean()) if bool(_mv.any())
                               else getattr(self, "_style_err", 1.0))
        # 摆动方向吸引项：约束空中摆动脚的前后速度，避免非前进命令下仍套用前进摆腿。
        # 它只作用于空中脚，且只看机体系前后方向，不压制地形抬脚。
        _w_swing = float(getattr(self.cfg, "w_swing_dir", 0.0))
        if _w_swing != 0.0:
            _qsw = rd.root_quat_w[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
            _v_rel_w = foot_vel - rd.root_lin_vel_w[:, None, :]                           # (N,4,3)，足端相对 base 的世界速度。
            _v_fwd_b = quat_rotate_inverse(_qsw, _v_rel_w.reshape(-1, 3)).reshape(self.num_envs, 4, 3)[:, :, 0]  # 机体系前后速度。
            _foot_y0 = 0.2082 * torch.tensor([1., -1., 1., -1.], device=self.device)      # FL,FR,RL,RR 的名义足端横向位置。
            _vfx_ref = self.commands[:, 0:1] - self.commands[:, 2:3] * _foot_y0[None, :]   # (N,4)，命令对应的前后摆动速率。
            _margin_sw = float(getattr(self.cfg, "swing_dir_margin", 0.15))
            # 允许参考中段摆动峰值加固定余量，只抓非命令方向的前向漂移。
            _excess_sw = torch.clamp(_v_fwd_b - 2.0 * _vfx_ref.clamp(min=0.0) - _margin_sw, min=0.0)
            _swing_m = (in_contact < 0.5).float()                                         # (N,4)，空中脚。
            _sw_err = (_swing_m * _excess_sw.pow(2)).sum(dim=1) / _swing_m.sum(dim=1).clamp(min=1.0)
            _sig_sw = float(getattr(self.cfg, "swing_dir_sigma", 0.08))
            comp["swing_dir"] = (
                _w_swing * moving * torch.exp(-_sw_err / _sig_sw) * _policy_steady_f
            )
            total = total + comp["swing_dir"]
            self._swing_excess = float((_swing_m * _excess_sw).sum() / _swing_m.sum().clamp(min=1.0))  # 可用于遥测 EMA。
        # 策略层制动：稳态零命令继续使用总速度势差；过渡期间改为惩罚超出
        # 五次下降包络的不兼容动量。Actor 仍直接看到原始新目标。
        _transition_stability = 0.30 + 0.70 * gate
        _w_brake = float(getattr(self.cfg, "w_settle_brake", 0.0))
        if _w_brake != 0.0:
            if not hasattr(self, "_prev_settle_speed"):
                self._prev_settle_speed = torch.zeros(self.num_envs, device=self.device)
            _spd_now = (
                torch.linalg.norm(rd.root_lin_vel_b[:, :2], dim=-1)
                + 0.20 * rd.root_ang_vel_b[:, 2].abs()
            )
            _cmd_still = ((torch.norm(self.commands[:, :2], dim=1) <= 0.1)
                          & (self.commands[:, 2].abs() <= 0.05))
            _brake_signal = taili_reward.settle_brake_signal(self._prev_settle_speed, _spd_now)
            # 最不稳定的失败样本也必须保留过渡梯度；稳定 Gate 只调节强度，不得归零。
            _steady_stop_scope = _cmd_still.to(rd.joint_pos.dtype) * (1.0 - _policy_brake_f)
            _motion_excess = getattr(self, "_cmd_transition_motion_excess", torch.zeros(N, device=dev))
            _excess_sq = _motion_excess.pow(2)
            _profile_cost = -_excess_sq / (1.0 + _excess_sq)
            # 两项相加与原 settle_brake 完全等价；拆开后可以直接核验过渡成本是否进入总奖励。
            comp["settle_brake"] = _w_brake * _steady_stop_scope * gate * _brake_signal
            comp["transition_motion_profile"] = (
                _w_brake * _policy_brake_f * _transition_stability * _profile_cost
            )
            total = total + comp["settle_brake"] + comp["transition_motion_profile"]
            self._prev_settle_speed = _spd_now.detach()
        _w_transition_ready = float(getattr(cfg, "w_transition_readiness", 0.0))
        if _w_transition_ready != 0.0:
            _wxy_max = max(float(getattr(self.cfg, "cmd_transition_wxy_max", 0.25)), 1e-6)
            _tilt_limit = max(
                math.sin(math.radians(float(getattr(self.cfg, "cmd_transition_tilt_deg", 6.0)))), 1e-6
            )
            _joint_max = max(float(getattr(self.cfg, "cmd_transition_joint_speed_max", 1.5)), 1e-6)
            _motion_fault = getattr(self, "_cmd_transition_motion_fault", torch.zeros(N, device=dev))
            _require_stop = getattr(
                self, "_cmd_transition_require_stop", torch.ones(N, dtype=torch.bool, device=dev)
            )
            _joint_limit = torch.where(
                _require_stop,
                torch.full((N,), _joint_max, device=dev),
                torch.full((N,), max(2.5, 1.7 * _joint_max), device=dev),
            )
            _core_terms = torch.clamp(torch.stack((
                _motion_fault,
                torch.linalg.norm(rd.root_ang_vel_b[:, :2], dim=-1) / _wxy_max,
                torch.linalg.norm(rd.projected_gravity_b[:, :2], dim=-1) / _tilt_limit,
            ), dim=-1) / 2.0, 0.0, 1.0)
            _core_fault = taili_reward.transition_core_fault(_core_terms)
            _action_limit = max(float(getattr(self.cfg, "cmd_transition_action_rate_max", 0.12)), 1e-6)
            _action_fault = torch.clamp(
                getattr(self, "_cmd_transition_action_rate", torch.zeros(N, device=dev))
                / (2.0 * _action_limit),
                0.0,
                1.0,
            )
            _support_phase_ready = getattr(
                self, "_cmd_transition_support_phase_ready", torch.zeros(N, dtype=torch.bool, device=dev)
            ).to(rd.joint_pos.dtype)
            # 动量和机身状态主导 readiness；四足相位点与动作变化只占很小的软质量预算。
            _transition_fault = (
                0.90 * _core_fault
                + 0.05 * (1.0 - _support_phase_ready)
                + 0.05 * _action_fault
            )
            _ready_signal = taili_reward.transition_readiness_signal(
                self._prev_transition_fault, _transition_fault
            )
            comp["transition_readiness"] = (
                _w_transition_ready * _policy_brake_f * _transition_stability * _ready_signal
            )
            total = total + comp["transition_readiness"]
            self._prev_transition_fault = _transition_fault.detach()
        _w_transition_failure = float(getattr(self.cfg, "w_transition_failure", 0.0))
        comp["transition_failure"] = taili_reward.transition_failure_penalty(
            getattr(self, "_cmd_transition_failed_event", torch.zeros(N, dtype=torch.bool, device=dev)),
            _w_transition_failure,
            (0.30 + 0.70 * gate) * _flat_transition_f,
        )
        total = total + comp["transition_failure"]
        with torch.no_grad():
            cmd_xy = self.commands[:, :2]
            cmd_xy_norm_sq = torch.sum(cmd_xy * cmd_xy, dim=1)
            cmd_wz_sq = self.commands[:, 2] * self.commands[:, 2]
            progress_thr = 0.0025
            # 8. 分方向 progress：父类 _log_training_diag 会读取这些字段推进 phase / DR / 速度上限。
            lin_prog = torch.clamp(
                torch.sum(rd.root_lin_vel_b[:, :2] * cmd_xy, dim=1) / cmd_xy_norm_sq.clamp(min=progress_thr),
                0.0,
                1.0,
            )
            ang_prog = torch.clamp(
                rd.root_ang_vel_b[:, 2] * self.commands[:, 2] / cmd_wz_sq.clamp(min=progress_thr),
                0.0,
                1.0,
            )
            moving_lin = cmd_xy_norm_sq > progress_thr
            moving_ang = cmd_wz_sq > progress_thr
            # Phase progress must mean "moving while still upright", not
            # "moving while every gait-quality metric is already good".
            # Slip/diag/duty remain visible gates and rewards, but no longer
            # multiply the core phase-progress signal into a bootstrap deadlock.
            h_t = torch.clamp(
                (base_h - float(cfg.h_gate_close)) / max(float(cfg.h_ok - cfg.h_gate_close), 1e-6),
                0.0,
                1.0,
            )
            height_progress_gate = h_t * h_t * (3.0 - 2.0 * h_t)
            tilt_t = torch.clamp(
                (tilt_rel - float(cfg.tilt_ok_rad))
                / max(float(cfg.tilt_gate_close_rad - cfg.tilt_ok_rad), 1e-6),
                0.0,
                1.0,
            )
            tilt_progress_gate = 1.0 - tilt_t * tilt_t * (3.0 - 2.0 * tilt_t)
            progress_posture_gate = torch.clamp(
                height_progress_gate
                * tilt_progress_gate
                * torch.clamp(1.0 - terminal_window, 0.0, 1.0),
                0.0,
                1.0,
            )
            # 课程能力值必须排除明显的机身摇晃/弹跳，不继承奖励侧用于探索的非零 floor。
            motion_progress_gate = taili_reward.tracking_motion_quality(
                rd.root_ang_vel_b, rd.root_lin_vel_b, cfg, floor=0.0
            )
            # 物理有效性只看最低支撑结构，不看 diag/duty/slip/gait_match。
            # 这样避免回到旧的步态 validator 死锁，同时不允许趴着滑动或前脚全飞也算 progress。
            support_structure = taili_reward.support_structure_gate(in_contact, self.commands)
            if support_structure is None:
                support_structure = torch.ones(N, device=dev)
            support_progress_gate = torch.clamp(support_structure * torch.clamp(1.0 - support_instab, 0.0, 1.0), 0.0, 1.0)
            progress_physical_gate = torch.clamp(
                progress_posture_gate * support_progress_gate * motion_progress_gate,
                0.0,
                1.0,
            )
            valid_lin_prog = lin_prog * progress_physical_gate
            valid_ang_prog = ang_prog * progress_physical_gate
            # 分方向进展使用与父类相同的课程 gate mask，用于阶段推进/回退、DR 升级和速度上限调整。
            # 这样可避免硬地形 env 把平地能力进展统计拖低。
            self._ensure_gate_mask()
            # 阶段和 DR 的速度能力只在平地评价；难度地形允许减速稳定通过。
            # 地形能力由真实、离散和分类型课程等级单独约束。
            flat_mask = getattr(self, "_flat_terrain_mask", None)
            gm_e = flat_mask if flat_mask is not None and bool(flat_mask.any()) else self._gate_mask
            steady_eval = ~_policy_transition_active
            fwd_mask = moving_lin & (self.commands[:, 0] > 0.1) & gm_e & steady_eval
            back_mask = moving_lin & (self.commands[:, 0] < -0.1) & gm_e & steady_eval
            lat_mask = moving_lin & (self.commands[:, 1].abs() > 0.1) & gm_e & steady_eval
            yaw_mask = moving_ang & gm_e & steady_eval
            # 方向漏斗用于区分“没有抽到命令”“过渡/地形筛掉样本”和“物理门压低
            # progress”。它只进入详细遥测，不参与奖励或 Actor 观测。
            _target_cmd = getattr(self, "_cmd_target", self.commands)
            _target_masks = {
                "fwd": _target_cmd[:, 0] > 0.10,
                "back": _target_cmd[:, 0] < -0.10,
                "lat": _target_cmd[:, 1].abs() > 0.10,
                "yaw": _target_cmd[:, 2].abs() > 0.10,
            }
            _applied_masks = {
                "fwd": moving_lin & (self.commands[:, 0] > 0.10),
                "back": moving_lin & (self.commands[:, 0] < -0.10),
                "lat": moving_lin & (self.commands[:, 1].abs() > 0.10),
                "yaw": moving_ang,
            }
            _eval_masks = {
                "fwd": fwd_mask,
                "back": back_mask,
                "lat": lat_mask,
                "yaw": yaw_mask,
            }
            self._direction_audit = {}
            for _dir_idx, _name in enumerate(("fwd", "back", "lat", "yaw")):
                _target_mask = _target_masks[_name]
                _applied_mask = _applied_masks[_name]
                _eval_mask = _eval_masks[_name]
                _target_count = _target_mask.float().sum()
                _terminal_count = (terminal_window * _target_mask.to(terminal_window.dtype)).sum()
                self._direction_terminal_target_count[_dir_idx] += _target_count
                self._direction_terminal_event_count[_dir_idx] += _terminal_count
                _terminal_denom = self._direction_terminal_target_count[_dir_idx]
                _audit = {
                    "target_samples": float(_target_mask.float().sum()),
                    "applied_samples": float(_applied_mask.float().sum()),
                    "flat_samples": float((_applied_mask & gm_e).float().sum()),
                    "steady_samples": float((_applied_mask & steady_eval).float().sum()),
                    "eval_samples": float(_eval_mask.float().sum()),
                    "terminal_rate": (
                        float(
                            self._direction_terminal_event_count[_dir_idx]
                            / _terminal_denom.clamp(min=1.0)
                        ) if float(_terminal_denom) > 0.0 else 0.0
                    ),
                    "posture_gate": (
                        float(progress_posture_gate[_eval_mask].mean()) if bool(_eval_mask.any()) else 0.0
                    ),
                    "support_gate": (
                        float(support_progress_gate[_eval_mask].mean()) if bool(_eval_mask.any()) else 0.0
                    ),
                    "motion_gate": (
                        float(motion_progress_gate[_eval_mask].mean()) if bool(_eval_mask.any()) else 0.0
                    ),
                }
                self._direction_audit[_name] = _audit
            if bool(fwd_mask.any()):
                self._fwd_raw_prog = float(lin_prog[fwd_mask].mean())
                self._fwd_tracking_prog = float(valid_lin_prog[fwd_mask].mean())
                _update_direction_progress(self, "fwd", self._fwd_tracking_prog, int(fwd_mask.sum()))
            if bool(back_mask.any()):
                self._back_raw_prog = float(lin_prog[back_mask].mean())
                self._back_tracking_prog = float(valid_lin_prog[back_mask].mean())
                _update_direction_progress(self, "back", self._back_tracking_prog, int(back_mask.sum()))
            if bool(lat_mask.any()):
                self._lat_raw_prog = float(lin_prog[lat_mask].mean())
                self._lat_tracking_prog = float(valid_lin_prog[lat_mask].mean())
                _update_direction_progress(self, "lat", self._lat_tracking_prog, int(lat_mask.sum()))
            if bool(yaw_mask.any()):
                self._yaw_raw_prog = float(ang_prog[yaw_mask].mean())
                self._yaw_tracking_prog = float(valid_ang_prog[yaw_mask].mean())
                _update_direction_progress(self, "yaw", self._yaw_tracking_prog, int(yaw_mask.sum()))
            progress_mask = (moving_lin | moving_ang) & gm_e
            self._progress_validity = float(progress_physical_gate[progress_mask].mean()) if bool(progress_mask.any()) else 0.0
            self._support_structure_gate = float(support_structure[progress_mask].mean()) if bool(progress_mask.any()) else 0.0
            gait_match = torch.zeros(N, device=dev)
            moving_any = moving > 0.5
            try:
                leg_phase = self._leg_phases()
                desired_stance = (leg_phase < self.cfg.gait_duty).float()
                gait_match = (desired_stance * in_contact + (1.0 - desired_stance) * (1.0 - in_contact)).mean(dim=1)
                linear_mag = torch.linalg.norm(self.commands[:, :2], dim=-1)
                steady_linear = (
                    moving_any
                    & (linear_mag > 0.10)
                    & (linear_mag >= 0.6 * self.commands[:, 2].abs())
                    & (getattr(self, "_cmd_transition_timer", torch.zeros(N, device=dev)) <= 0)
                )
                gait_mean = float(gait_match[steady_linear].mean()) if bool(steady_linear.any()) else 0.0
                lag_scores = []
                for lag in torch.linspace(-0.25, 0.25, 9, device=dev):
                    lag_phase = (leg_phase + lag) % 1.0
                    lag_stance = (lag_phase < self.cfg.gait_duty).float()
                    lag_match = (lag_stance * in_contact + (1.0 - lag_stance) * (1.0 - in_contact)).mean(dim=1)
                    lag_scores.append(lag_match)
                lag_scores = torch.stack(lag_scores, dim=1)
                if bool(steady_linear.any()):
                    gait_beta = 0.95
                    self._gait_lag_scores[steady_linear] = (
                        gait_beta * self._gait_lag_scores[steady_linear]
                        + (1.0 - gait_beta) * lag_scores[steady_linear]
                    )
                    best_lag_gait_mean = float(
                        self._gait_lag_scores[steady_linear].max(dim=1).values.mean()
                    )
                else:
                    best_lag_gait_mean = 0.0
                fl, fr, rl, rr = in_contact[:, 0], in_contact[:, 1], in_contact[:, 2], in_contact[:, 3]
                diag_pair_inst_mean = float(quality["diag_pair_inst"][moving_any].mean()) if bool(moving_any.any()) else 0.0
                front = 0.5 * (fl + fr)
                rear = 0.5 * (rl + rr)
                left = 0.5 * (fl + rl)
                right = 0.5 * (fr + rr)
                duty_balance = 1.0 - torch.clamp((rear - front).abs() + (left - right).abs(), 0.0, 1.0)
                diag_mean = float(quality["diag_pair_score"][steady_linear].mean()) if bool(steady_linear.any()) else 0.0
                duty_eval = steady_linear & (quality["duty_cycle_valid"] > 0.5)
                duty_balance_product_mean = float(quality["duty_quality"][duty_eval].mean()) if bool(duty_eval.any()) else 0.0
                duty_balance_mean = min(
                    float(quality["duty_target_score"][duty_eval].mean()),
                    float(quality["duty_symmetry_score"][duty_eval].mean()),
                ) if bool(duty_eval.any()) else 0.0
                duty_balance_inst_mean = float(duty_balance[moving_any].mean()) if bool(moving_any.any()) else 0.0
            except Exception:
                gait_mean = 0.0
                diag_mean = 0.0
                diag_pair_inst_mean = 0.0
                duty_balance_mean = 0.0
                duty_balance_product_mean = 0.0
                best_lag_gait_mean = 0.0
                duty_balance_inst_mean = 0.0
                duty_eval = torch.zeros(N, dtype=torch.bool, device=dev)
            self._slip_now = float(quality["slip_speed"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._slip_inst = float(quality["slip_speed_inst"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._slip_high_fraction = float(quality["slip_high_fraction"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._diag_contact = diag_mean
            self._diag_pair_instant = diag_pair_inst_mean
            self._duty_balance = duty_balance_mean
            self._duty_balance_product = duty_balance_product_mean
            self._duty_balance_instant = duty_balance_inst_mean
            self._duty_spread = float(quality["duty_spread"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._duty_target_score = float(quality["duty_target_score"][duty_eval].mean()) if bool(duty_eval.any()) else 0.0
            self._duty_symmetry_score = float(quality["duty_symmetry_score"][duty_eval].mean()) if bool(duty_eval.any()) else 0.0
            self._duty_range_error = float(quality["duty_target_error"][duty_eval].mean()) if bool(duty_eval.any()) else 0.0
            self._duty_symmetry_error = float(quality["duty_symmetry_error"][duty_eval].mean()) if bool(duty_eval.any()) else 0.0
            self._duty_cycle_valid_frac = float(duty_eval.float().sum() / steady_linear.float().sum().clamp(min=1.0))
            self._gait_period_score = float(quality["period_score"][steady_linear].mean()) if bool(steady_linear.any()) else 0.0
            self._gait_period_error = float(quality["period_error"][steady_linear].mean()) if bool(steady_linear.any()) else 0.0
            period_valid = steady_linear & (quality["contact_period_valid"] > 0.5)
            self._actual_contact_period = float(quality["contact_period"][period_valid].mean()) if bool(period_valid.any()) else 0.0
            self._contact_period_valid_frac = float(period_valid.float().sum() / steady_linear.float().sum().clamp(min=1.0))

            cmd = self.commands
            transition_clear = getattr(self, "_cmd_transition_timer", torch.zeros(N, device=dev)) <= 0
            direction_masks = {
                "fwd": transition_clear & (cmd[:, 0] > 0.10) & (cmd[:, 1].abs() <= 0.08) & (cmd[:, 2].abs() <= 0.08),
                "back": transition_clear & (cmd[:, 0] < -0.10) & (cmd[:, 1].abs() <= 0.08) & (cmd[:, 2].abs() <= 0.08),
                "lat": transition_clear & (cmd[:, 1].abs() > 0.10) & (cmd[:, 0].abs() <= 0.08) & (cmd[:, 2].abs() <= 0.08),
                "yaw": transition_clear & (cmd[:, 2].abs() > 0.10) & (torch.linalg.norm(cmd[:, :2], dim=-1) <= 0.08),
            }
            best_lag_by_env = self._gait_lag_scores.max(dim=1).values
            self._gait_by_direction = {}
            self._duty_by_direction = {}
            self._duty_target_by_direction = {}
            self._duty_symmetry_by_direction = {}
            for name, mask in direction_masks.items():
                if bool(mask.any()):
                    if name == "yaw":
                        yaw_support = comp.get("yaw_duty_gate", torch.zeros(N, device=dev))
                        yaw_wxy = comp.get("yaw_wxy_gate", torch.zeros(N, device=dev))
                        yaw_drift = comp.get("yaw_drift_gate", torch.zeros(N, device=dev))
                        slip_score = torch.clamp(1.0 - quality["slip_speed"] / 0.30, 0.0, 1.0)
                        score = torch.minimum(
                            torch.minimum(yaw_support, yaw_wxy),
                            torch.minimum(yaw_drift, slip_score),
                        )
                    else:
                        score = torch.minimum(
                            torch.minimum(quality["diag_pair_score"], quality["duty_target_score"]),
                            torch.minimum(quality["duty_symmetry_score"], quality["period_score"]),
                        )
                    self._gait_by_direction[name] = float(score[mask].mean())
                    self._gait_by_direction[f"{name}_best_lag"] = float(best_lag_by_env[mask].mean()) if name != "yaw" else 0.0
                    duty_mask = mask & (quality["duty_cycle_valid"] > 0.5)
                    if name != "yaw" and bool(duty_mask.any()):
                        target_score = float(quality["duty_target_score"][duty_mask].mean())
                        symmetry_score = float(quality["duty_symmetry_score"][duty_mask].mean())
                        self._duty_target_by_direction[name] = target_score
                        self._duty_symmetry_by_direction[name] = symmetry_score
                        self._duty_by_direction[name] = min(target_score, symmetry_score)
                else:
                    self._gait_by_direction[name] = 0.0
                    self._gait_by_direction[f"{name}_best_lag"] = 0.0
            linear_duty = [self._duty_by_direction[name] for name in ("fwd", "back", "lat") if name in self._duty_by_direction]
            linear_target = [self._duty_target_by_direction[name] for name in ("fwd", "back", "lat") if name in self._duty_target_by_direction]
            linear_symmetry = [self._duty_symmetry_by_direction[name] for name in ("fwd", "back", "lat") if name in self._duty_symmetry_by_direction]
            if linear_duty:
                self._duty_balance = min(linear_duty)
                self._duty_target_score = min(linear_target)
                self._duty_symmetry_score = min(linear_symmetry)
            self._yaw_gait_gate = self._gait_by_direction["yaw"]
            self._height_low_risk = float(quality["height_low_risk"].mean())
            self._tilt_high_risk = float(quality["tilt_high_risk"].mean())
            duty_by_leg_mean = quality["duty_by_leg"].mean(dim=0)
            self._duty_by_leg = {
                "FL": float(duty_by_leg_mean[0]),
                "FR": float(duty_by_leg_mean[1]),
                "RL": float(duty_by_leg_mean[2]),
                "RR": float(duty_by_leg_mean[3]),
            }
            self._style_err = getattr(self, "_style_err", 1.0)
            self._dbg = (
                float(valid_lin_prog[moving_lin].mean()) if bool(moving_lin.any()) else 0.0,
                float(valid_ang_prog[moving_ang].mean()) if bool(moving_ang.any()) else 0.0,
                float(comp["tracking_lin"].mean()),
                float((comp["tracking_yaw"] + comp.get("yaw_progress", torch.zeros_like(total))).mean()),
                0.0,
                gait_mean,
            )
            self._gait_match_zero_lag = gait_mean
            self._best_lag_gait_match = best_lag_gait_mean
            self._last_gait_match = best_lag_gait_mean
            # 9. 奖励分组和调试摘要：供 multi-critic、父类文本日志和预算门控使用。
            # 预算监控：常规惩罚绝对值 / 正向任务奖励。
            # 碰撞、饱和和 terminal 这类事件惩罚不计入常规预算。
            # 超过阈值时父类会降低 penalty ramp，避免质量惩罚让行走本身变得不划算。
            _tier_s = ("torque_saturation", "terminal_penalty", "terrain_collapse")
            # 预算分母只能包含真正进入总回报的正向分量。复用奖励模块的完整 gate
            # 清单，避免把 capability/tracking 等诊断门值误算成不存在的正奖励。
            _skip = set(taili_reward.REWARD_GROUP_GATES)
            pos_sum, neg_sum = 0.0, 0.0
            for _k, _v in comp.items():
                if _k in _skip or _k in _tier_s or not torch.is_tensor(_v):
                    continue
                _m = float(_v.mean())
                if _m >= 0.0:
                    pos_sum += _m
                else:
                    neg_sum += -_m
            _ratio = neg_sum / max(pos_sum, 1e-6)
            _prev = float(getattr(self, "_budget_ratio_ema", _ratio))
            self._budget_ratio_ema = 0.995 * _prev + 0.005 * _ratio
            self._rew_dbg = {
                "lin": float(comp["tracking_lin"].mean()),
                "ang": float((comp["tracking_yaw"] + comp.get("yaw_progress", torch.zeros_like(total))).mean()),
                "gait": float(comp["gait_anchor"].mean()),
                "exchange": float(comp.get("contact_exchange", torch.zeros_like(total)).mean()),
                "traj": float(comp.get("foot_trajectory", torch.zeros(N, device=dev)).mean()),
                "diag": float(comp["diagonal_contact"].mean()),
                "duty": float(comp["duty_balance"].mean()),
                "imit": 0.0,
                "stand": float(comp["stand"].mean()),
                "height": float(comp.get("flat_move_height", torch.zeros(N, device=dev)).mean()),
                "support": float(comp.get("support_integrity", torch.zeros(N, device=dev)).mean()),
                "slip": float(comp["stance_slip"].mean()),
                "clear": float((comp["clearance_under"] + comp["clearance_over"]).mean()),
                "hip": float(comp.get("hip_deviation", torch.zeros(N, device=dev)).mean()),
                "offax": float(comp["off_axis"].mean()),
                "purity": float(comp.get("planar_purity", torch.zeros(N, device=dev)).mean()),
                "heading": float(comp.get("heading_hold", torch.zeros(N, device=dev)).mean()),
                "back_aux": float(comp["backward_underspeed"].mean()),
                "lat_aux": float(comp["lateral_underspeed"].mean()),
                "dir_progress": float(comp["supported_progress"].mean()),
                "dir_align": float(comp["direction_alignment"].mean()),
                "speed_scale": float(comp["terrain_tracking_scale"].mean()),
                "terr_support": float(comp["terrain_support_loss"].mean()),
                "terr_quality": float(comp["terrain_contact_quality"].mean()),
                "terr_collapse": float(comp["terrain_collapse"].mean()),
                "terr_over": float(comp["terrain_overspeed"].mean()),
                "terr_probe_rise": float(terrain_probe["rise_ahead_mean"]),
                "terr_probe_up_active": float(terrain_probe["up_active_mean"]),
                "terr_probe_common": float(terrain_probe["common_mean"]),
                "land": float(comp["landing_impact"].mean()),
                "torq": float((comp["torque_margin"] + comp["torque_saturation"]).mean()),
                "arate": float(comp["action_rate"].mean()),
                "vz": float(comp.get("base_vz", torch.zeros(N, device=dev)).mean()),
                "wxy": float(comp.get("base_wxy", torch.zeros(N, device=dev)).mean()),
            }

        # 周期性训练日志：奖励、跟踪、速度、门控和 collapse。
        self._rew_log_step = getattr(self, "_rew_log_step", 0) + 1
        telemetry_interval = max(1, int(os.environ.get("TAILI_TELEMETRY_INTERVAL", "10") or 10))
        total_steps = int(os.environ.get("TAILI_TOTAL_STEPS", "0") or 0)
        emit_telemetry = (
            self._rew_log_step == 1
            or self._rew_log_step % telemetry_interval == 0
            or (total_steps > 0 and self._rew_log_step >= total_steps)
        )
        if emit_telemetry and not getattr(self, "use_external_commands", False):
            checkpoint_interval = max(1, int(os.environ.get("TAILI_CHECKPOINT_INTERVAL", "2000") or 2000))
            if self._rew_log_step == 1 or self._rew_log_step % checkpoint_interval == 0:
                try:
                    self._save_curriculum_state(self._rew_log_step)
                except Exception as exc:
                    print(f"[CURRICULUM] save failed: {type(exc).__name__}: {exc}", flush=True)
            with torch.no_grad():
                # 10. 结构化遥测：前端和智能体主要读取这些 payload，字段名需要保持稳定。
                vb = rd.root_lin_vel_b
                cmd_xy = self.commands[:, :2]
                cmd_xy_norm = torch.linalg.norm(cmd_xy, dim=1)
                lin_err_vec = torch.linalg.norm(cmd_xy - vb[:, :2], dim=1)
                lin_err = lin_err_vec.mean().item()
                yaw_err = (rd.root_ang_vel_b[:, 2] - self.commands[:, 2]).abs().mean().item()
                speed = float(torch.norm(vb[:, :2], dim=1).mean())
                gait = float(gate.mean())
                v_along = torch.sum(vb[:, :2] * cmd_xy, dim=1) / torch.clamp(cmd_xy_norm, min=1e-6)
                progress_ratio = torch.clamp(v_along / torch.clamp(cmd_xy_norm, min=1e-6), 0.0, 1.0)
                torque_limit_t = torch.as_tensor(torque_limit, device=dev).float()
                if torque_limit_t.ndim == 1:
                    torque_limit_t = torque_limit_t.unsqueeze(0).expand_as(rd.applied_torque)
                torque_util = (rd.applied_torque.abs() / torque_limit_t.clamp_min(1e-6)).mean()
                terrain_mean = None
                terrain_max = None
                terrain_stats = {}
                terrain_type_payload = {}
                if hasattr(self, "_terrain") and hasattr(self._terrain, "terrain_levels"):
                    terrain_stats = self._terrain_level_stats()
                    terrain_levels = self._terrain.terrain_levels.float()
                    terrain_mean = float(terrain_stats.get("terrain_mean", float(terrain_levels.mean())))
                    terrain_max = int(terrain_stats.get("terrain_max", int(terrain_levels.max().item())))
                    if hasattr(self._terrain, "terrain_types"):
                        try:
                            self._ensure_gate_mask()
                            if hasattr(self, "_col_type") and hasattr(self, "_type_names"):
                                env_type = self._col_type[self._terrain.terrain_types]
                                for type_i, type_name in enumerate(self._type_names):
                                    mask = env_type == type_i
                                    if bool(mask.any()):
                                        key = str(type_name).replace("-", "_").replace(" ", "_")
                                        vals = terrain_levels[mask]
                                        terrain_type_payload[f"terrain_{key}_mean"] = float(vals.mean())
                                        terrain_type_payload[f"terrain_{key}_max"] = int(vals.max().item())
                        except Exception:
                            terrain_type_payload = {}
                phase = getattr(self, "_phase", None)
                dr_level = getattr(self, "_dr_level", None)
                progress_by_dir = {
                    "fwd": float(getattr(self, "_fwd_prog", 0.0)),
                    "back": float(getattr(self, "_back_prog", 0.0)),
                    "lat": float(getattr(self, "_lat_prog", 0.0)),
                    "yaw": float(getattr(self, "_yaw_prog", 0.0)),
                }
                raw_progress_by_dir = {
                    "fwd": float(getattr(self, "_fwd_raw_prog", 0.0)),
                    "back": float(getattr(self, "_back_raw_prog", 0.0)),
                    "lat": float(getattr(self, "_lat_raw_prog", 0.0)),
                    "yaw": float(getattr(self, "_yaw_raw_prog", 0.0)),
                }
                active_dirs = ()
                if active_direction_progress is not None:
                    progress_gate, active_dirs = active_direction_progress(progress_by_dir, self.cfg, getattr(self, "_phase", None))
                else:
                    progress_gate = min(progress_by_dir.values())
                gait_gate = float(getattr(self, "_last_gait_match", 0.0))
                fall_rate = None
                try:
                    # 真实跌倒需要同时满足低高度和大倾斜；低地形上的直立机器人不算跌倒。
                    # 这样可避免按绝对高度误判下坡/下台阶中的正常姿态。
                    _below = rd.root_pos_w[:, 2] < self.cfg.termination_height
                    _tilted = rd.projected_gravity_b[:, 2] >= -0.7   # 非直立。
                    fall_rate = float((_below & _tilted).float().mean())
                except Exception:
                    pass
                # 下面四个 payload 是前端、智能体和人工调参的外部契约；字段名不能随意改动。
                upright = float((-rd.projected_gravity_b[:, 2]).clamp(0.0, 1.0).mean())
                include_reward_cfg = not self._reward_cfg_printed
                reward_payload = build_reward_payload(
                    total=total,
                    lin_err=lin_err,
                    speed=speed,
                    gait=gait,
                    base_h=base_h,
                    upright=upright,
                    comp=comp,
                    terrain_probe=terrain_probe,
                    reward_cfg=cfg,
                    include_reward_cfg=include_reward_cfg,
                )
                if self._dynamic_mechanism_metrics:
                    reward_payload["dynamic_metrics"] = dict(self._dynamic_mechanism_metrics)
                if self._dynamic_mechanism_gates:
                    curriculum_payload_dynamic = dict(self._dynamic_mechanism_gates)
                if include_reward_cfg:
                    self._reward_cfg_printed = True
                command_payload = build_command_payload(
                    env=self,
                    commands=self.commands,
                    vb=vb,
                    root_ang_vel_b=rd.root_ang_vel_b,
                    v_along=v_along,
                    speed=speed,
                    lin_err=lin_err,
                    yaw_err=yaw_err,
                    progress_ratio=progress_ratio,
                    progress_gate=progress_gate,
                    progress_by_dir=progress_by_dir,
                    raw_progress_by_dir=raw_progress_by_dir,
                    active_dirs=active_dirs,
                    n=N,
                    device=dev,
                )
                curriculum_payload = build_curriculum_payload(
                    env=self,
                    phase=phase,
                    dr_level=dr_level,
                    terrain_mean=terrain_mean,
                    terrain_max=terrain_max,
                    terrain_stats=terrain_stats,
                    terrain_type_payload=terrain_type_payload,
                    progress_gate=progress_gate,
                    progress_by_dir=progress_by_dir,
                    raw_progress_by_dir=raw_progress_by_dir,
                    active_dirs=active_dirs,
                    command_payload=command_payload,
                    gait_gate=gait_gate,
                    fall_rate=fall_rate,
                )
                if self._dynamic_mechanism_gates:
                    curriculum_payload["dynamic_gates"] = curriculum_payload_dynamic
                health_payload = build_health_payload(
                    gate=gate,
                    moving=moving,
                    stand_gate=stand_gate,
                    base_h=base_h,
                    upright=upright,
                    tilt_rel=tilt_rel,
                    support_instab=support_instab,
                    cc=cc,
                    in_contact=in_contact,
                    torque_util=torque_util,
                    terminal_window=terminal_window,
                    height_low_risk=getattr(self, "_height_low_risk", 0.0),
                    tilt_high_risk=getattr(self, "_tilt_high_risk", 0.0),
                    fall_rate=fall_rate,
                )
                if self._telemetry is not None:
                    self._telemetry.emit(
                        step=self._rew_log_step,
                        total_steps=total_steps or None,
                        reward=reward_payload,
                        curriculum=curriculum_payload,
                        health=health_payload,
                        command=command_payload,
                    )

                    # P4.2集成：缓存当前步的性能快照，供检查点保存时使用。
                    # 传的是数值 phase：curriculum_payload["phase"] 是显示串（"phi0"），
                    # 曾经直接 int() 它，导致训练第一步就崩，见 build_checkpoint_performance_snapshot。
                    self._latest_performance_snapshot = build_checkpoint_performance_snapshot(
                        reward_payload=reward_payload,
                        health_payload=health_payload,
                        phase=phase,
                        # 这两个 payload 里都没有 episode_length_mean，得从 episode_length_buf 现算。
                        # 不传的话登记表里会落一个假默认值，curator 的 episode_length > 100 门就形同虚设。
                        episode_length_mean=float(self.episode_length_buf.float().mean().item()),
                    )
                else:
                    print("[TPREW] step %d rew %.3f lin_err %.3f speed %.3f gate %.2f tracking_lin %.3f stand %.3f"
                          % (self._rew_log_step, float(total.mean()), lin_err,
                             speed, gait, float(comp["tracking_lin"].mean()), float(comp["stand"].mean())), flush=True)
        # 多 critic：把同一套奖励分量拆成 [N,K] 目标向量并暂存。
        # 按 K 求和必须等于标量 total；环境侧 terminal 惩罚通过 extra_by_group 放入 stab 组。
        # _get_observations 重置 extras 后会把该向量暴露出去；未启用 TAILI_MULTI_CRITIC 时跳过。
        self._prev_support_z = torch.where(
            _support_valid,
            _support_z,
            self._prev_support_z.to(device=dev),
        ).detach().clone()
        self._terrain_prev_base_z = rd.root_pos_w[:, 2].detach().clone()
        comp["total"] = total + cfg.w_terminal * terminal_window
        if os.environ.get("TAILI_MULTI_CRITIC") == "1":
            try:
                _mc_extra = {"stab": -cfg.w_terminal * terminal_window}
                self._reward_groups = torch.nan_to_num(
                    taili_reward.group_reward_vector(comp, extra_by_group=_mc_extra),
                    nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                self._reward_groups = None
        return torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
