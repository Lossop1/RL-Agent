# Taili 四足 AMP 运动环境：任务是速度跟踪，AMP 提供小跑风格约束。
from __future__ import annotations
import math
import os
import gymnasium as gym
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from .taili_amp_env_cfg import TailiAmpEnvCfg
from .motions import MotionLoader  # noqa: F401，保留兼容导入。
from .multi_motion_loader import MultiMotionLoader
from .parametric_ref import flat_reference
try:
    from .taili_core.terrain_curriculum import compute_terrain_curriculum_moves
except ImportError:
    try:
        from autotuner.taili_core.terrain_curriculum import compute_terrain_curriculum_moves
    except ImportError:
        from taili_core.terrain_curriculum import compute_terrain_curriculum_moves
try:
    from .taili_blind_config import active_direction_progress, phase_command_spec
except ImportError:  # 本地源码布局下 env_edit 可能多嵌套一级。
    from ..taili_blind_config import active_direction_progress, phase_command_spec


def _spec_float(spec, key: str, default: float) -> float:
    try:
        return float(spec.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _spec_range(spec, key: str, fallback) -> tuple[float, float]:
    value = spec.get(key, fallback) if isinstance(spec, dict) else fallback
    try:
        lo, hi = value
        return float(lo), float(hi)
    except (TypeError, ValueError):
        lo, hi = fallback
        return float(lo), float(hi)


def _sample_uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    lo = float(lo)
    hi = max(float(hi), lo)
    return torch.rand(n, device=device) * (hi - lo) + lo


def _command_xy_world_from_root_yaw(root_quat_w: torch.Tensor, command_xy: torch.Tensor) -> torch.Tensor:
    """只使用 root yaw，把机体系 XY 命令旋转到世界系 XY。"""
    q = root_quat_w
    yaw = torch.atan2(
        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]),
    )
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    return torch.stack(
        (
            c * command_xy[:, 0] - s * command_xy[:, 1],
            s * command_xy[:, 0] + c * command_xy[:, 1],
        ),
        dim=-1,
    )


def _yaw_from_quat_w(root_quat_w: torch.Tensor) -> torch.Tensor:
    q = root_quat_w
    return torch.atan2(
        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]),
    )


def _wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class TailiAmpEnv(DirectRLEnv):
    cfg: TailiAmpEnvCfg

    def __init__(self, cfg: TailiAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_offset = self.robot.data.default_joint_pos.clone()
        self.action_scale = self.cfg.action_scale
        self._motion_loader = MultiMotionLoader(self.cfg.motion_files, device=self.device,
                                                sigma=self.cfg.cond_sigma, slope_sigma=self.cfg.slope_sigma)
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.foot_indexes = [self.robot.data.body_names.index(n) for n in self.cfg.foot_body_names]
        # 接触传感器有自己的 body 顺序，不直接等同于 robot body 顺序。
        self._feet_contact_ids, _ = self._contact_sensor.find_bodies(self.cfg.foot_body_names)
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_foot_indexes = self._motion_loader.get_body_index(self.cfg.foot_body_names)
        self.n_feet = len(self.foot_indexes)
        # AMP stride 窗口：启用 TAILI_AMP_STRIDE=1 时，让风格窗口覆盖约一个步态周期。
        # 策略侧和参考侧使用相同 amp_frame_stride 子采样，判别器输入维度随帧数自动扩展。
        # 未启用时 stride=1，不改变配置和缓冲行为。
        self._amp_frame_stride = 1
        if os.environ.get("TAILI_AMP_STRIDE") == "1":
            self._amp_frame_stride = int(getattr(self.cfg, "amp_frame_stride", 4))
            self.cfg.num_amp_observations = int(getattr(self.cfg, "amp_stride_frames", 6))
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device)
        # 原始 AMP 环形缓冲：stride>1 时保留足够相邻帧来构造导出窗口。
        if self._amp_frame_stride > 1:
            self._amp_raw_depth = (self.cfg.num_amp_observations - 1) * self._amp_frame_stride + 1
            self._amp_raw_ring = torch.zeros(
                (self.num_envs, self._amp_raw_depth, self.cfg.amp_observation_space), device=self.device)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)        # 平滑后的命令，策略实际跟踪它。
        self._cmd_target = torch.zeros((self.num_envs, 3), device=self.device)      # 原始目标命令，commands 向它低通靠近。
        self._cmd_motion_target = torch.zeros_like(self.commands)                  # 过渡整形后的目标命令。
        self._cmd_transition_start = torch.zeros_like(self.commands)
        self._cmd_transition_goal = torch.zeros_like(self.commands)
        self._cmd_transition_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cmd_transition_total = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._cmd_transition_via_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cmd_transition_zero_frac = torch.full((self.num_envs,), 0.5, device=self.device)
        self._cmd_transition_strength = torch.zeros(self.num_envs, device=self.device)
        self._cmd_heading_ref = torch.zeros(self.num_envs, device=self.device)
        self._heading_error = torch.zeros(self.num_envs, device=self.device)
        self.last_actions = torch.zeros((self.num_envs, 12), device=self.device)
        self._cmd_resample_steps = max(1, int(self.cfg.cmd_resample_s / (self.cfg.dt * self.cfg.decimation)))
        self._log_step = 0          # 周期性 stdout 训练诊断计数。
        self._terrain_ctx = torch.zeros((self.num_envs, self.cfg.terrain_ctx_dim), device=self.device)
        self._episode_start_xy = self.robot.data.root_pos_w[:, :2].detach().clone()
        self._episode_start_root_z = self.robot.data.root_pos_w[:, 2].detach().clone()
        self._episode_start_cmd_xy = torch.zeros((self.num_envs, 2), device=self.device)
        # 步态相位时钟：每个 env 一个 [0,1) 相位，只在运动命令下推进。
        # 单腿相位 = 全局相位 + 对角小跑偏置，用于接触节律奖励和策略观测。
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)
        self._trot_offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device)   # FL,FR,RL,RR 对角小跑。
        self._hip_idx = [self.robot.data.joint_names.index(f"{lg}_hip_joint") for lg in ["FL", "FR", "RL", "RR"]]
        def _joint_gain_vector(attr_name: str, defaults: dict[str, float]) -> torch.Tensor:
            role_cfg = getattr(self.cfg, attr_name, defaults)
            values = []
            for joint_name in self.robot.data.joint_names:
                if "hip_joint" in joint_name:
                    values.append(float(role_cfg.get("hip", defaults["hip"])))
                elif "thigh_joint" in joint_name:
                    values.append(float(role_cfg.get("thigh", defaults["thigh"])))
                elif "calf_joint" in joint_name:
                    values.append(float(role_cfg.get("calf", defaults["calf"])))
                else:
                    values.append(float(defaults.get("thigh", 120.0)))
            return torch.as_tensor(values, dtype=torch.float32, device=self.device)
        self._actuator_stiffness_base = _joint_gain_vector(
            "actuator_stiffness", {"hip": 120.0, "thigh": 120.0, "calf": 120.0}
        )
        self._actuator_damping_base = _joint_gain_vector(
            "actuator_damping", {"hip": 10.0, "thigh": 10.0, "calf": 10.0}
        )
        # 本体历史缓冲：最近 H 帧真实机器人可测的本体信息。
        # 不直接使用 lin_vel 或 height scanner，也能提供隐式地形/速度线索。
        H, P = self.cfg.obs_history_len, self.cfg.obs_history_dim
        self._obs_history = torch.zeros((self.num_envs, H, P), device=self.device)
        # 控制延迟缓冲：执行上一控制步动作，模拟约 20ms 计算和通信延迟。
        self._delayed_action = torch.zeros((self.num_envs, 12), device=self.device)
        # 分方向速度课程：每个方向都有独立速度上限和进展跟踪。
        self._vel_curriculum_enable = bool(getattr(self.cfg, "vel_curriculum_enable", True))
        phase0_spec = phase_command_spec(self.cfg, 0) if phase_command_spec is not None else {}
        def _phase_hi(name: str, fallback) -> float:
            try:
                return float(phase0_spec.get(name, fallback)[1])
            except Exception:
                return float(fallback[1])
        def _phase_lo(name: str, fallback) -> float:
            try:
                return float(phase0_spec.get(name, fallback)[0])
            except Exception:
                return float(fallback[0])
        self._vel_max_fwd = float(self.cfg.cmd_fwd_max) if not self._vel_curriculum_enable else _phase_hi("fwd_range", self.cfg.cmd_fwd_range)
        self._vel_max_back = float(self.cfg.cmd_back_max) if not self._vel_curriculum_enable else _phase_hi("back_range", self.cfg.cmd_back_range)
        self._vel_max_lat = float(self.cfg.cmd_lat_max) if not self._vel_curriculum_enable else _phase_hi("lat_range", self.cfg.cmd_lat_range)
        # yaw 从较容易的上限起步，再随进展提高。原地转向从满范围开始过难，
        # 会让 yaw_prog 卡住并阻塞早期阶段；先学会中等 yaw，再扩展到目标上限。
        yaw_lo = _phase_lo("yaw_range", self.cfg.cmd_yaw_range)
        yaw_hi = _phase_hi("yaw_range", self.cfg.cmd_yaw_range)
        # yaw 可达 ramp：起始上限易学，floor 保证不会退到过低范围；
        # 达到 yaw 专用进展阈值后再向 yaw_hi 扩展。
        self._vel_max_yaw = float(self.cfg.cmd_yaw_max) if not self._vel_curriculum_enable else min(yaw_hi, yaw_lo + 0.20)
        # floor 使用阶段速度范围下限；退化方向可以回到更容易的速度端重新建立能力。
        self._vel_floor_fwd = _phase_lo("fwd_range", self.cfg.cmd_fwd_range)
        self._vel_floor_back = _phase_lo("back_range", self.cfg.cmd_back_range)
        self._vel_floor_lat = _phase_lo("lat_range", self.cfg.cmd_lat_range)
        # yaw floor 设为不高于 yaw_hi 的 0.45，避免退到过低 yaw 练习。
        self._vel_floor_yaw = min(yaw_hi, max(yaw_lo, 0.45))
        self._fwd_prog = 0.0; self._back_prog = 0.0; self._lat_prog = 0.0; self._yaw_prog = 0.0
        # DR 等级系统：从 0 级开始，按已展示的运动能力逐级打开。
        self._dr_level = int(getattr(self.cfg, "dr_start_level", 0))   # 按门控从 0 升到 3。
        self._dr_gate_count = 0     # 连续满足 DR 升级门槛的日志间隔数。
        _push_s = getattr(self.cfg, f"dr_push_interval_s_{self._dr_level}", 0.0)
        _sdt = self.cfg.dt * self.cfg.decimation
        self._push_steps = max(1, int(_push_s / _sdt)) if _push_s > 0 else int(1e9)
        # IMU 偏置：每个 episode 固定的 gyro+gravity 偏移，模拟传感器标定误差。
        self._imu_bias = torch.zeros((self.num_envs, 6), device=self.device)
        self._dr_warned = set()     # PhysX DR API 不可用时的一次性告警集合。
        self._default_masses = None # mass DR 的默认质量缓存，用于先复位再加扰动。
        self._dr_mat_level = -1        # 已写入摩擦/CoM 的最高 DR 等级。
        self._default_coms = None      # CoM DR 的默认质心缓存，用于先复位再加偏移。
        self._dbg = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._last_gait_match = 0.0
        # 统一训练阶段 phi：协调速度、步态质量、地形和 DR。
        # fresh 默认从 0 开始；从已有行走检查点 resume 时可用 TAILI_INIT_PHASE 跳到已训练阶段。
        # clearance_gate 仍从 0 开始，避免重约束在 resume 时突然生效。
        self._phase = int(os.environ.get("TAILI_INIT_PHASE", getattr(self.cfg, "init_phase", 0)))
        self._phase_count = 0          # 连续满足当前阶段推进门槛的日志间隔数。
        self._penalty_gate = 1.0 if self._phase >= 1 else 0.0   # bootstrap 之后质量惩罚已完全打开。
        self._budget_ratio_ema = 0.0   # 常规惩罚绝对值 / 正向任务奖励。
        self._clearance_gate = 0.0     # 地形 clearance 渐入门控。
        self._terrain_start_phase = int(getattr(self.cfg, "terrain_start_phase", 5))
        self._dr_start_phase = int(getattr(self.cfg, "dr_start_phase", self._terrain_start_phase))
        configured_max_phase = getattr(self.cfg, "max_training_phase", None)
        if configured_max_phase is None:
            try:
                configured_max_phase = max(int(k) for k in getattr(self.cfg, "training_phase_commands", {}).keys())
            except Exception:
                configured_max_phase = 3
        env_max_phase = os.environ.get("TAILI_MAX_PHASE")
        if env_max_phase not in (None, ""):
            configured_max_phase = int(env_max_phase)
        self._max_training_phase = int(configured_max_phase)
        if self._phase > 0:
            print(f"[ENV] init_phase={self._phase} (resume in trained regime; penalty_gate={self._penalty_gate})", flush=True)
        self._advance_ok = True        # 回退保护：能力退化时暂停地形/速度推进。
        self._slip_ema = 0.0           # 足端滑移运行均值，用于清步态阶段门控。
        # 接触缓存：在 _get_observations 中计算，在 _get_rewards 中复用。
        self._in_contact = torch.zeros((self.num_envs, 4), device=self.device)
        # 对称增强：镜像 PPO batch，约束左右对称步态。
        if getattr(self.cfg, "sym_augment", False):
            from . import symmetry
            symmetry.set_mirrors(self._build_hscan_mirror_perm(), self.device)
            symmetry.patch_memory_sample_all()

    def _build_hscan_mirror_perm(self):
        """For each height-scan ray (x,y), index of the ray at (x,-y). Validated involution else identity."""
        P = int(self.cfg.n_height_scan)
        try:
            rs = self._height_scanner.ray_starts
            rs = rs.reshape(-1, rs.shape[-1])[:P].detach().cpu().numpy()   # (P,3) local ray grid
            xy = rs[:, :2]
            perm = np.zeros(P, dtype=np.int64)
            for i in range(P):
                d = (xy[:, 0] - xy[i, 0]) ** 2 + (xy[:, 1] + xy[i, 1]) ** 2
                perm[i] = int(np.argmin(d))
            if np.array_equal(perm[perm], np.arange(P)) and len(set(perm.tolist())) == P:
                return perm
            print("[SYM] height_scan mirror not a clean involution; using identity", flush=True)
        except Exception as e:
            print(f"[SYM] height_scan ray layout unavailable ({e}); using identity", flush=True)
        return np.arange(P, dtype=np.int64)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        # 高度扫描器始终在场景中：用于 terrain_ctx、critic 特权观测和 clearance 奖励。
        # actor 策略不直接读取它，BlindGaussianPolicy 会切掉特权部分。
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        # 接触传感器，用于足端滞空时间和接触奖励。
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # 地形导入器，启用生成器和课程。
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _command_transition_steps(self) -> int:
        step_dt = float(self.cfg.dt) * float(self.cfg.decimation)
        base_period = float(getattr(self.cfg, "gait_period", 0.55))
        cycles = float(getattr(self.cfg, "cmd_transition_cycles", 0.75))
        min_s = float(getattr(self.cfg, "cmd_transition_min_s", 0.25))
        max_s = float(getattr(self.cfg, "cmd_transition_max_s", 0.80))
        duration = min(max(base_period * cycles, min_s), max_s)
        return max(1, int(round(duration / max(step_dt, 1e-6))))

    def _begin_command_transition(self, env_ids, snap: bool = False):
        if len(env_ids) == 0:
            return
        target = self._cmd_target[env_ids]
        yaw = _yaw_from_quat_w(self.robot.data.root_quat_w[env_ids])
        self._cmd_heading_ref[env_ids] = yaw
        self._heading_error[env_ids] = 0.0
        if snap or not bool(getattr(self.cfg, "cmd_transition_enable", True)):
            self.commands[env_ids] = target
            self._cmd_motion_target[env_ids] = target
            self._cmd_transition_start[env_ids] = target
            self._cmd_transition_goal[env_ids] = target
            self._cmd_transition_timer[env_ids] = 0
            self._cmd_transition_total[env_ids] = 1
            self._cmd_transition_via_zero[env_ids] = False
            self._cmd_transition_zero_frac[env_ids] = 0.5
            self._cmd_transition_strength[env_ids] = 0.0
            return

        current = self.commands[env_ids]
        v_thr = float(getattr(self.cfg, "cmd_transition_sign_flip_v", 0.08))
        w_thr = float(getattr(self.cfg, "cmd_transition_sign_flip_w", 0.10))
        cur_lin = torch.linalg.norm(current[:, :2], dim=-1)
        tgt_lin = torch.linalg.norm(target[:, :2], dim=-1)
        cur_move = (cur_lin > v_thr) | (current[:, 2].abs() > w_thr)
        tgt_move = (tgt_lin > v_thr) | (target[:, 2].abs() > w_thr)
        sign_flip = (
            (current[:, 0] * target[:, 0] < -(v_thr * v_thr))
            | (current[:, 1] * target[:, 1] < -(v_thr * v_thr))
            | (current[:, 2] * target[:, 2] < -(w_thr * w_thr))
        )
        current_axis = torch.stack(
            (current[:, 0].abs() / max(v_thr, 1e-6),
             current[:, 1].abs() / max(v_thr, 1e-6),
             current[:, 2].abs() / max(w_thr, 1e-6)),
            dim=-1,
        )
        target_axis = torch.stack(
            (target[:, 0].abs() / max(v_thr, 1e-6),
             target[:, 1].abs() / max(v_thr, 1e-6),
             target[:, 2].abs() / max(w_thr, 1e-6)),
            dim=-1,
        )
        current_family = current_axis.argmax(dim=-1)
        target_family = target_axis.argmax(dim=-1)
        family_change = cur_move & tgt_move & (current_family != target_family)
        through_stop = cur_move & (~tgt_move)
        actual_lin = torch.linalg.norm(self.robot.data.root_lin_vel_b[env_ids, :2], dim=-1)
        actual_yaw = self.robot.data.root_ang_vel_b[env_ids, 2].abs()
        contact_min = float(getattr(self.cfg, "cmd_transition_contact_feet", 3.0))
        contact_count = self._in_contact[env_ids].sum(dim=1) if hasattr(self, "_in_contact") else torch.full_like(cur_lin, 4.0)
        low_v = float(getattr(self.cfg, "cmd_transition_low_speed_v", 0.12))
        low_w = float(getattr(self.cfg, "cmd_transition_low_speed_w", 0.16))
        low_speed_ready = (
            (cur_lin <= low_v)
            & (actual_lin <= low_v)
            & (current[:, 2].abs() <= low_w)
            & (actual_yaw <= low_w)
            & (contact_count >= contact_min)
        )
        abrupt = (sign_flip | family_change | through_stop) & ~low_speed_ready
        direct = ~abrupt
        if bool(direct.any()):
            ids = env_ids[direct]
            self._cmd_motion_target[ids] = self._cmd_target[ids]
            self._cmd_transition_start[ids] = self.commands[ids]
            self._cmd_transition_goal[ids] = self._cmd_target[ids]
            self._cmd_transition_timer[ids] = 0
            self._cmd_transition_total[ids] = 1
            self._cmd_transition_via_zero[ids] = False
            self._cmd_transition_zero_frac[ids] = 0.5
            self._cmd_transition_strength[ids] = 0.0
        if bool(abrupt.any()):
            ids = env_ids[abrupt]
            idx = abrupt
            max_v = max(
                float(getattr(self.cfg, "cmd_fwd_max", 1.0)),
                float(getattr(self.cfg, "cmd_back_max", 0.8)),
                float(getattr(self.cfg, "cmd_lat_max", 0.5)),
                low_v,
            )
            max_w = max(float(getattr(self.cfg, "cmd_yaw_max", 1.0)), low_w)
            delta_lin = torch.linalg.norm((target - current)[:, :2], dim=-1) / max(max_v, 1e-6)
            delta_w = (target[:, 2] - current[:, 2]).abs() / max(max_w, 1e-6)
            delta_norm = torch.maximum(delta_lin, delta_w)
            speed_norm = torch.maximum(
                torch.maximum(cur_lin / max(low_v, 1e-6), actual_lin / max(low_v, 1e-6)),
                torch.maximum(current[:, 2].abs() / max(low_w, 1e-6), actual_yaw / max(low_w, 1e-6)),
            )
            airborne = (contact_count < contact_min).float()
            severity = torch.clamp(0.35 * delta_norm + 0.35 * speed_norm + 0.20 * airborne + 0.10 * sign_flip.float(), 0.0, 1.0)
            step_dt = max(float(self.cfg.dt) * float(self.cfg.decimation), 1e-6)
            fast_s = float(getattr(self.cfg, "cmd_transition_fast_s", 0.16))
            fast_steps = max(1, int(round(fast_s / step_dt)))
            base_steps = self._command_transition_steps()
            max_steps = max(base_steps, int(round(float(getattr(self.cfg, "cmd_transition_max_s", 0.80)) / step_dt)))
            steps = torch.round(
                float(fast_steps) + (float(base_steps) - float(fast_steps)) * severity[idx]
            ).long().clamp(min=fast_steps, max=max_steps)
            z_min = float(getattr(self.cfg, "cmd_transition_zero_frac_min", 0.48))
            z_max = float(getattr(self.cfg, "cmd_transition_zero_frac_max", 0.70))
            stop_z = float(getattr(self.cfg, "cmd_transition_stop_zero_frac", 0.80))
            zero_frac_all = torch.clamp(z_min + (z_max - z_min) * severity, 0.05, 0.95)
            zero_frac_all = torch.where(tgt_move, zero_frac_all, torch.full_like(zero_frac_all, stop_z))
            self._cmd_transition_start[ids] = self.commands[ids]
            self._cmd_transition_goal[ids] = self._cmd_target[ids]
            self._cmd_transition_timer[ids] = steps
            self._cmd_transition_total[ids] = steps
            self._cmd_transition_via_zero[ids] = True
            self._cmd_transition_zero_frac[ids] = zero_frac_all[idx]
            self._cmd_transition_strength[ids] = severity[idx]

    def _update_command_transition(self):
        active = self._cmd_transition_timer > 0
        if bool(active.any()):
            total = self._cmd_transition_total[active].float().clamp(min=1.0)
            timer = self._cmd_transition_timer[active].float()
            p = ((total - timer + 1.0) / total).clamp(0.0, 1.0)
            s = p * p * p * (10.0 + p * (-15.0 + 6.0 * p))
            start = self._cmd_transition_start[active]
            goal = self._cmd_transition_goal[active]
            shaped = start + (goal - start) * s[:, None]
            via_zero = self._cmd_transition_via_zero[active]
            if bool(via_zero.any()):
                s_v = s[via_zero]
                start_v = start[via_zero]
                goal_v = goal[via_zero]
                zfrac = self._cmd_transition_zero_frac[active][via_zero].clamp(0.05, 0.95)
                first = s_v < zfrac
                shaped_v = torch.empty_like(start_v)
                s_first = (s_v / zfrac).clamp(0.0, 1.0)
                s_second = ((s_v - zfrac) / (1.0 - zfrac).clamp(min=1e-6)).clamp(0.0, 1.0)
                s_first = s_first * s_first * s_first * (10.0 + s_first * (-15.0 + 6.0 * s_first))
                s_second = s_second * s_second * s_second * (10.0 + s_second * (-15.0 + 6.0 * s_second))
                shaped_v[first] = start_v[first] * (1.0 - s_first[first, None])
                shaped_v[~first] = goal_v[~first] * s_second[~first, None]
                shaped[via_zero] = shaped_v
            self._cmd_motion_target[active] = shaped
            self._cmd_transition_timer[active] -= 1
            done = active & (self._cmd_transition_timer <= 0)
            if bool(done.any()):
                self._cmd_motion_target[done] = self._cmd_transition_goal[done]
                self._cmd_transition_strength[done] = 0.0

    def _update_heading_error(self):
        yaw = _yaw_from_quat_w(self.robot.data.root_quat_w)
        yaw_active = (self.commands[:, 2].abs() > 0.05) | (self._cmd_target[:, 2].abs() > 0.05)
        self._cmd_heading_ref = torch.where(yaw_active, yaw, self._cmd_heading_ref)
        self._heading_error = _wrap_pi(yaw - self._cmd_heading_ref)

    def _resample_commands(self, env_ids, snap=False):
        # 单轴命令采样：按任务比例写入原始目标命令，self.commands 通过命令缓冲低通靠近它。
        # reset 时 snap=True，直接从新命令开始，避免从旧命令 ramp 过来。
        n = len(env_ids); dev = self.device
        if n == 0:
            return
        phase = int(getattr(self, "_phase", getattr(self.cfg, "init_phase", 0)))
        spec = phase_command_spec(self.cfg, phase) if phase_command_spec is not None else {}
        mode = str(spec.get("command_mode") or getattr(self.cfg, "training_command_mode", "normal") or "normal")
        self._last_command_mode = mode
        self._last_command_spec = spec
        if mode in {"fixed_forward", "forward_range", "stand_only", "single_axis", "mixed"}:
            self._cmd_target[env_ids] = 0.0

            def _range_with_ceiling(name: str, fallback, ceiling_attr: str, max_attr: str):
                lo, hi = _spec_range(spec, name, fallback)
                hi = min(hi, float(getattr(self, ceiling_attr, hi)), float(getattr(self.cfg, max_attr, hi)))
                return lo, max(lo, hi)

            f_lo, f_hi = _range_with_ceiling("fwd_range", self.cfg.cmd_fwd_range, "_vel_max_fwd", "cmd_fwd_max")
            b_lo, b_hi = _range_with_ceiling("back_range", self.cfg.cmd_back_range, "_vel_max_back", "cmd_back_max")
            l_lo, l_hi = _range_with_ceiling("lat_range", self.cfg.cmd_lat_range, "_vel_max_lat", "cmd_lat_max")
            y_lo, y_hi = _range_with_ceiling("yaw_range", self.cfg.cmd_yaw_range, "_vel_max_yaw", "cmd_yaw_max")

            terr_frac = None
            if self.cfg.vel_terrain_decouple and self.cfg.terrain.terrain_type == "generator":
                tmax = max(1.0, self.cfg.terrain.terrain_generator.num_rows - 1)
                terr_frac = (self._terrain.terrain_levels[env_ids].float() / tmax).clamp(0.0, 1.0)

            def _sample_range(lo: float, hi):
                if torch.is_tensor(hi):
                    return torch.rand(n, device=dev) * (hi - float(lo)).clamp(min=0.0) + float(lo)
                return _sample_uniform(n, lo, hi, dev)

            f_hi_eff = f_hi
            b_hi_eff = b_hi
            if terr_frac is not None:
                f_hi_eff = f_lo + (f_hi - f_lo) * (1.0 - terr_frac)
                b_hi_eff = b_lo + (b_hi - b_lo) * (1.0 - terr_frac)

            target = torch.zeros((n, 3), device=dev)
            if mode == "fixed_forward":
                target[:, 0] = _spec_float(spec, "fixed_vx", float(getattr(self.cfg, "training_fixed_vx", 0.5)))
            elif mode == "forward_range":
                lo_v, hi_v = _spec_range(spec, "forward_range", getattr(self.cfg, "training_forward_range", (0.3, 0.7)))
                target[:, 0] = _sample_uniform(n, lo_v, hi_v, dev)
            elif mode == "stand_only":
                target.zero_()
            elif mode == "mixed":
                stand_prob = _spec_float(spec, "stand_prob", float(getattr(self.cfg, "stand_prob", 0.0)))
                near_zero_prob = _spec_float(spec, "near_zero_prob", 0.0)
                axis_prob = _spec_float(spec, "mixed_axis_prob", 1.0)
                active = torch.rand(n, device=dev) >= stand_prob
                near_zero = active & (torch.rand(n, device=dev) < near_zero_prob)
                moving = active & ~near_zero
                p_fwd = max(0.0, _spec_float(spec, "prob_fwd", 0.5))
                p_back = max(0.0, _spec_float(spec, "prob_back", 0.5))
                x_is_fwd = torch.rand(n, device=dev) < (p_fwd / max(1e-9, p_fwd + p_back))
                x_on = torch.rand(n, device=dev) < _spec_float(spec, "x_axis_prob", 1.0)
                lat_on = torch.rand(n, device=dev) < axis_prob
                yaw_on = torch.rand(n, device=dev) < axis_prob
                yaw_on = yaw_on | (~x_on & ~lat_on)
                lat_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                yaw_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                target[:, 0] = torch.where(x_is_fwd, _sample_range(f_lo, f_hi_eff), -_sample_range(b_lo, b_hi_eff))
                target[:, 0] = torch.where(x_on, target[:, 0], torch.zeros(n, device=dev))
                target[:, 1] = torch.where(lat_on, lat_sign * _sample_uniform(n, l_lo, l_hi, dev), torch.zeros(n, device=dev))
                target[:, 2] = torch.where(yaw_on, yaw_sign * _sample_uniform(n, y_lo, y_hi, dev), torch.zeros(n, device=dev))
                target = torch.where(moving[:, None], target, torch.zeros_like(target))
                nz_scale = _spec_float(spec, "near_zero_scale", 0.05)
                near_noise = (torch.rand((n, 3), device=dev) * 2.0 - 1.0) * nz_scale
                target = torch.where(near_zero[:, None], near_noise, target)
            else:
                stand_prob = _spec_float(spec, "stand_prob", float(getattr(self.cfg, "stand_prob", 0.0)))
                active = torch.rand(n, device=dev) >= stand_prob
                weights = torch.tensor([
                    max(0.0, _spec_float(spec, "prob_fwd", float(getattr(self.cfg, "cmd_prob_fwd", 0.25)))),
                    max(0.0, _spec_float(spec, "prob_back", float(getattr(self.cfg, "cmd_prob_back", 0.25)))),
                    max(0.0, _spec_float(spec, "prob_lat", float(getattr(self.cfg, "cmd_prob_lat", 0.25)))),
                    max(0.0, _spec_float(spec, "prob_yaw", float(getattr(self.cfg, "cmd_prob_yaw", 0.25)))),
                ], device=dev)
                if float(weights.sum()) <= 1e-9:
                    weights = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
                u = torch.rand(n, device=dev) * weights.sum()
                c0 = weights[0]
                c1 = c0 + weights[1]
                c2 = c1 + weights[2]
                fwd = active & (u < c0)
                back = active & (u >= c0) & (u < c1)
                lat = active & (u >= c1) & (u < c2)
                yaw = active & (u >= c2)
                lat_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                yaw_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                target[:, 0] = torch.where(fwd, _sample_range(f_lo, f_hi_eff),
                                            torch.where(back, -_sample_range(b_lo, b_hi_eff), torch.zeros(n, device=dev)))
                target[:, 1] = torch.where(lat, lat_sign * _sample_uniform(n, l_lo, l_hi, dev), torch.zeros(n, device=dev))
                target[:, 2] = torch.where(yaw, yaw_sign * _sample_uniform(n, y_lo, y_hi, dev), torch.zeros(n, device=dev))
            self._cmd_target[env_ids] = target
            self._begin_command_transition(env_ids, snap=snap)
            return
        self._cmd_target[env_ids] = 0.0
        active = torch.rand(n, device=dev) >= self.cfg.stand_prob
        u = torch.rand(n, device=dev)
        p_f = self.cfg.cmd_prob_fwd
        p_b = p_f + self.cfg.cmd_prob_back
        p_l = p_b + self.cfg.cmd_prob_lat
        fwd = active & (u < p_f)
        back = active & (u >= p_f) & (u < p_b)
        lat = active & (u >= p_b) & (u < p_l)
        yaw = active & (u >= p_l)

        def uniform(lo, hi):
            return torch.empty(n, device=dev).uniform_(lo, hi)

        # 分方向课程速度上限：每个方向独立调整。
        # 前进：地形解耦，在高难地形上降低速度上限，避免难度叠加。
        fwd_ceil = self._vel_max_fwd
        _terr_frac = None
        if self.cfg.vel_terrain_decouple and self.cfg.terrain.terrain_type == "generator":
            tmax = max(1.0, self.cfg.terrain.terrain_generator.num_rows - 1)
            _terr_frac = (self._terrain.terrain_levels[env_ids].float() / tmax).clamp(0.0, 1.0)
            fwd_ceil = self.cfg.cmd_fwd_range[0] + (self._vel_max_fwd - self.cfg.cmd_fwd_range[0]) * (1.0 - _terr_frac)
        fwd_cmd = torch.rand(n, device=dev) * (fwd_ceil - self.cfg.cmd_fwd_range[0]) + self.cfg.cmd_fwd_range[0]
        # 后退：与前进一样地形解耦，避免下台阶后退时速度过高导致力矩饱和。
        back_hi  = max(self.cfg.cmd_back_range[0], min(self._vel_max_back, self.cfg.cmd_back_max))
        if _terr_frac is not None:
            back_hi = self.cfg.cmd_back_range[0] + (back_hi - self.cfg.cmd_back_range[0]) * (1.0 - _terr_frac)
        back_cmd = torch.rand(n, device=dev) * (back_hi - self.cfg.cmd_back_range[0]) + self.cfg.cmd_back_range[0]
        self._cmd_target[env_ids, 0] = torch.where(
            fwd, fwd_cmd,
            torch.where(back, -back_cmd, torch.zeros(n, device=dev)))
        # 横移速度课程上限。
        lat_hi  = max(self.cfg.cmd_lat_range[0], min(self._vel_max_lat, self.cfg.cmd_lat_max))
        lat_cmd = torch.rand(n, device=dev) * (lat_hi - self.cfg.cmd_lat_range[0]) + self.cfg.cmd_lat_range[0]
        self._cmd_target[env_ids, 1] = torch.where(
            lat, lat_cmd * torch.where(torch.rand(n, device=dev) < 0.5, 1.0, -1.0), torch.zeros(n, device=dev))
        # yaw 速度课程上限。
        yaw_hi  = max(self.cfg.cmd_yaw_range[0], min(self._vel_max_yaw, self.cfg.cmd_yaw_max))
        yaw_cmd = torch.rand(n, device=dev) * (yaw_hi - self.cfg.cmd_yaw_range[0]) + self.cfg.cmd_yaw_range[0]
        self._cmd_target[env_ids, 2] = torch.where(
            yaw, yaw_cmd * torch.where(torch.rand(n, device=dev) < 0.5, 1.0, -1.0), torch.zeros(n, device=dev))
        self._begin_command_transition(env_ids, snap=snap)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        # 命令缓冲：self.commands 平滑靠近 _cmd_target，而不是瞬间跳变。
        # 这会让停步、反向和转向有自然过渡；稳态仍等于目标命令，不改变速度跟踪语义。
        # 这是命令接口的一部分，训练和部署都会执行。
        target_changed = (self._cmd_target - self._cmd_transition_goal).abs().amax(dim=1) > 1e-6
        if bool(target_changed.any()):
            self._begin_command_transition(target_changed.nonzero(as_tuple=False).flatten(), snap=False)
        self._update_command_transition()
        self.commands += (1.0 - self.cfg.cmd_smooth_alpha) * (self._cmd_motion_target - self.commands)
        self._update_heading_error()
        # 每个 env step 推进一次步态时钟；零命令时冻结，使策略自然站立。
        # 周期随命令速度缩短，命令越快步频越高。
        step_dt = self.cfg.dt * self.cfg.decimation
        spd = torch.norm(self.commands[:, :2], dim=1)
        moving = (spd > 0.1) | (self.commands[:, 2].abs() > 0.05)   # 与奖励侧 yaw_cmd_gate 对齐。
        # yaw 感知 cadence：观测中的 gait clock、滞空奖励和 AMP 参考都使用同一速度语义。
        # 转向时把 0.15*|wz| 加入 period_speed，使高 yaw 命令获得更快步频。
        period_speed = spd + 0.15 * self.commands[:, 2].abs()
        period = torch.clamp(self.cfg.gait_period - self.cfg.gait_period_slope * period_speed,
                             min=self.cfg.gait_period_min, max=self.cfg.gait_period)
        self._gait_phase = (self._gait_phase + (step_dt / period) * moving.float()) % 1.0
        # 域随机化：周期性给 base 随机推力速度，训练扰动恢复能力。
        if self.cfg.dr_enable:
            due = (self.episode_length_buf % self._push_steps == 0) & (self.episode_length_buf > 0)
            if bool(due.any()):
                ids = due.nonzero(as_tuple=False).flatten()
                push_vel = float(getattr(self.cfg, f"dr_push_vel_{self._dr_level}", 0.0))
                if push_vel <= 0.0:
                    return
                vel = torch.cat([self.robot.data.root_lin_vel_w[ids],
                                 self.robot.data.root_ang_vel_w[ids]], dim=-1).clone()  # (M, 6)，世界系线速度+角速度。
                vel[:, 0:2] += (torch.rand((len(ids), 2), device=self.device) * 2 - 1) * push_vel   # 线速度推扰。
                # 同时加入 yaw 角速度扰动，使策略学习旋转扰动恢复。
                vel[:, 5] += (torch.rand(len(ids), device=self.device) * 2 - 1) * push_vel * self.cfg.dr_push_ang_scale
                self.robot.write_root_com_velocity_to_sim(vel, ids)

    def _leg_phases(self):
        # 每条腿的 [0,1) 相位：全局相位 + 对角小跑偏置。
        return (self._gait_phase[:, None] + self._trot_offsets[None, :]) % 1.0

    def _ensure_gate_mask(self):
        """构建课程门控 env mask。

        hard 子地形会长期停在低等级；如果直接纳入阶段/速度/DR 门控，会拖慢已经稳定的 env。
        terrain_types 对每个 env 固定，因此该 mask 只需构建一次。
        """
        if getattr(self, "_gate_mask", None) is not None:
            return
        if not (hasattr(self, "_terrain") and hasattr(self._terrain, "terrain_types")):
            self._gate_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._discrete_terrain_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._flat_terrain_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._real_terrain_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            return
        sg = self.cfg.terrain.terrain_generator
        props = np.array([s.proportion for s in sg.sub_terrains.values()], dtype=np.float64)
        props = props / props.sum(); cum = np.cumsum(props); ncol = sg.num_cols
        ct = np.array([int(np.min(np.where(c / ncol + 1e-3 < cum)[0])) for c in range(ncol)])
        self._col_type = torch.as_tensor(ct, device=self.device)
        self._type_names = list(sg.sub_terrains.keys())
        et = self._col_type[self._terrain.terrain_types]
        hard = torch.zeros_like(et, dtype=torch.bool)
        for nm in ("slope_inv", "boxes", "stairs_up"):   # stairs_up 是坑内上楼梯，按 hard 类型处理。
            if nm in self._type_names:
                hard |= (et == self._type_names.index(nm))
        self._gate_mask = ~hard
        # 离散障碍 mask：楼梯/boxes 可能需要比粗糙度门控更高的抬脚目标。
        self._discrete_terrain_mask = torch.zeros_like(et, dtype=torch.bool)
        for nm in ("stairs", "boxes", "stairs_up"):
            if nm in self._type_names:
                self._discrete_terrain_mask |= (et == self._type_names.index(nm))
        self._flat_terrain_mask = torch.zeros_like(et, dtype=torch.bool)
        if "flat" in self._type_names:
            self._flat_terrain_mask = et == self._type_names.index("flat")
        self._real_terrain_mask = ~self._flat_terrain_mask
        print(f"[GATE] curriculum gates exclude hard terrain types {[n for n in ('slope_inv','boxes','stairs_up') if n in self._type_names]}"
              f" -> {int(self._gate_mask.sum())}/{self.num_envs} envs gate the curriculum"
              f" | real_nonflat={int(self._real_terrain_mask.sum())}/{self.num_envs}"
              f" | discrete(stairs/boxes/stairs_up)={int(self._discrete_terrain_mask.sum())}/{self.num_envs}", flush=True)

    def _terrain_level_stats(self) -> dict[str, float | int]:
        stats: dict[str, float | int] = {
            "terrain_mean": 0.0,
            "terrain_max": 0,
            "terrain_flat_mean": 0.0,
            "terrain_flat_max": 0,
            "terrain_real_mean": 0.0,
            "terrain_real_max": 0,
            "terrain_discrete_mean": 0.0,
            "terrain_discrete_max": 0,
            "terrain_real_frac": 0.0,
            "terrain_discrete_frac": 0.0,
        }
        levels = getattr(getattr(self, "_terrain", None), "terrain_levels", None)
        if levels is None:
            return stats
        lv = levels.float()
        stats["terrain_mean"] = float(lv.mean())
        stats["terrain_max"] = int(lv.max().item())
        self._ensure_gate_mask()

        def _fill(prefix: str, mask: torch.Tensor):
            if bool(mask.any()):
                vals = lv[mask]
                stats[f"{prefix}_mean"] = float(vals.mean())
                stats[f"{prefix}_max"] = int(vals.max().item())
                stats[f"{prefix}_frac"] = float(mask.float().mean())

        _fill("terrain_flat", getattr(self, "_flat_terrain_mask", torch.zeros_like(lv, dtype=torch.bool)))
        _fill("terrain_real", getattr(self, "_real_terrain_mask", torch.ones_like(lv, dtype=torch.bool)))
        _fill("terrain_discrete", getattr(self, "_discrete_terrain_mask", torch.zeros_like(lv, dtype=torch.bool)))
        return stats

    def _apply_action(self):
        # 控制延迟：执行上一控制步动作，模拟约 20ms 计算和通信延迟。
        self.robot.set_joint_position_target(self.action_offset + self.action_scale * self._delayed_action)
        self._delayed_action = self.actions.clone()

    def _feet_rel_base(self, foot_pos_w, base_pos_w, base_quat_w, n_foot):
        rel = foot_pos_w - base_pos_w.unsqueeze(1)                       # (M, n_foot, 3)
        q = base_quat_w.unsqueeze(1).expand(-1, n_foot, -1).reshape(-1, 4)
        return quat_apply_inverse(q, rel.reshape(-1, 3)).reshape(-1, n_foot * 3)

    def _compute_terrain_ctx(self):
        if self.cfg.terrain_ctx_dim == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)
        # terrain_ctx_dim == 3：[前后坡度, 横向坡度, 粗糙度]。
        # 坡度在 heading 坐标系下计算；扫描器只跟 yaw 对齐，因此只用 yaw 变换扫描点。
        # 粗糙度是扫描网格上相对 base 地面高度的标准差。
        hits = self._height_scanner.data.ray_hits_w                       # (N, P, 3)
        base = self.robot.data.root_pos_w                                 # (N, 3)
        q = self.robot.data.root_quat_w                                   # (N, 4) wxyz
        w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))   # (N,)
        c, s = torch.cos(-yaw)[:, None], torch.sin(-yaw)[:, None]
        dx = hits[..., 0] - base[:, 0:1]                                  # (N, P)
        dy = hits[..., 1] - base[:, 1:2]
        fx = c * dx - s * dy                                              # heading-frame fore/lat positions
        fy = s * dx + c * dy
        dz = torch.nan_to_num(hits[..., 2] - base[:, 2:3], nan=0.0, posinf=0.0, neginf=0.0)
        fore = (fx * dz).sum(-1) / (fx * fx).sum(-1).clamp(min=1e-6)
        lat  = (fy * dz).sum(-1) / (fy * fy).sum(-1).clamp(min=1e-6)
        # 粗糙度：base 相对地形高度标准差，对应 terrain_ctx 第三维。
        rough = dz.std(dim=-1).clamp(0.0, 0.3)
        ctx = torch.stack([fore.clamp(-0.6, 0.6), lat.clamp(-0.6, 0.6), rough], dim=-1)
        return torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_amp_obs(self):
        jp = self.robot.data.joint_pos
        jv = self.robot.data.joint_vel
        bh = self.robot.data.root_pos_w[:, 2:3] - self._terrain.env_origins[:, 2:3]
        tn = quaternion_to_tangent_and_normal(self.robot.data.root_quat_w)
        rel_b = self._feet_rel_base(self.robot.data.body_pos_w[:, self.foot_indexes],
                                    self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.n_feet)
        self._terrain_ctx = self._compute_terrain_ctx()
        # AMP 只表达风格：纯运动学状态 + terrain_ctx。
        # 速度和命令不进入判别器；速度/命令跟踪由双侧 tracking 奖励负责。
        return torch.cat([jp, jv, bh, tn, rel_b, self._terrain_ctx], dim=-1)

    def _get_observations(self) -> dict:
        if not getattr(self, "use_external_commands", False):
            due = (self.episode_length_buf % self._cmd_resample_steps == 0)
            if due.any():
                self._resample_commands(due.nonzero(as_tuple=False).flatten())
        lp = self._leg_phases()                                       # (N, 4) per-leg phase
        gait_obs = torch.cat([torch.sin(2 * math.pi * lp), torch.cos(2 * math.pi * lp)], dim=-1)  # (N, 8)
        N, dev = self.num_envs, self.device
        # 观测噪声：缩放前加入，使量级接近真实传感器规格。
        jpos_n  = self.robot.data.joint_pos  + torch.randn(N, 12, device=dev) * self.cfg.obs_noise_jpos
        jvel_n  = self.robot.data.joint_vel  + torch.randn(N, 12, device=dev) * self.cfg.obs_noise_jvel
        angv_n  = self.robot.data.root_ang_vel_b  + torch.randn(N, 3, device=dev) * self.cfg.obs_noise_angvel
        grav_n  = self.robot.data.projected_gravity_b + torch.randn(N, 3, device=dev) * self.cfg.obs_noise_gravity
        # IMU 偏置：每个 episode 固定偏移，低 DR 等级下为 0。
        angv_n  = angv_n + self._imu_bias[:, 0:3]
        grav_n  = grav_n + self._imu_bias[:, 3:6]
        # 本体历史 FIFO：当前帧入队，最旧帧出队。
        # 单帧 42 维：jpos-default(12)+jvel(12)+angvel(3)+gravity(3)+lastact(12)。
        prop_now = torch.cat([
            jpos_n - self.action_offset,   # 12
            jvel_n * 0.05,                 # 12
            angv_n * 0.25,                 # 3
            grav_n,                        # 3
            self.last_actions,             # 12
        ], dim=-1)                         # (N, 42)
        self._obs_history = torch.cat([prop_now.unsqueeze(1), self._obs_history[:, :-1]], dim=1)  # FIFO
        # 足端接触缓存，供 _get_rewards 复用，避免重复查询。
        forces_now = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, :].norm(dim=-1)
        self._in_contact = (forces_now > 1.0).float()   # (N, 4)

        # 盲策略观测 473 维：actor 实际使用的信息，只包含电机、IMU、命令和本体历史。
        #   ang_vel(3)+gravity(3)+cmd(3)+jpos(12)+jvel(12)+lastact(12)+gait(8)+history(42×H)
        blind_parts = [
            angv_n * 0.25,
            grav_n,
            self.commands,
            jpos_n - self.action_offset,
            jvel_n * 0.05,
            self.last_actions,
            gait_obs,
            self._obs_history.reshape(N, -1),   # H×42 = 420 dims
        ]
        blind_obs = torch.cat(blind_parts, dim=-1)   # (N, 473)

        # 特权补充 197 维：只供 critic/AMP 使用，actor 不直接读取。
        #   lin_vel(3) + height_scan(187) + terrain_ctx(3) + foot_contact(4)
        hits = self._height_scanner.data.ray_hits_w                          # (N, P, 3)
        hscan = (self.robot.data.root_pos_w[:, 2:3] - hits[:, :, 2]
                 - self.cfg.stand_height).clamp(-1.0, 1.0)
        hscan = torch.nan_to_num(hscan, nan=0.0, posinf=0.0, neginf=0.0)    # (N, 187)
        self._terrain_ctx = self._compute_terrain_ctx()
        priv_obs = torch.cat([
            blind_obs,
            self.robot.data.root_lin_vel_b * 2.0,   # 3
            hscan,                                    # 187
            self._terrain_ctx,                        # 3
            self._in_contact,                         # 4
        ], dim=-1)                                    # (N, 670)

        # 完整 observation 为 670 维；BlindGaussianPolicy 内部切片 [:, :473] 给 actor。
        obs = priv_obs
        # NaN/Inf 保护：物理爆炸可能产生非有限观测，先记录再清理，避免写入网络和 rollout。
        # 对应 env 会在 _get_dones 的非有限检查中 reset。
        finite = torch.isfinite(obs).all(dim=1)
        if not bool(finite.all()):
            bad = (~finite).nonzero(as_tuple=False).flatten()
            print(f"[NANGUARD] step={self._log_step} non-finite policy-obs in {int(bad.numel())} env(s): "
                  f"{bad[:8].tolist()} -> sanitized", flush=True)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_actions = self.actions.clone()
        amp = torch.nan_to_num(self._compute_amp_obs(), nan=0.0, posinf=0.0, neginf=0.0)
        if getattr(self, "_amp_frame_stride", 1) > 1:
            self._push_strided_amp(amp)
        else:
            for i in reversed(range(self.cfg.num_amp_observations - 1)):
                self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
            self.amp_observation_buffer[:, 0] = amp
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}
        self._log_step += 1
        if self._log_step % self.cfg.log_every == 0 and not getattr(self, "use_external_commands", False):
            self._log_training_diag()
        return {"policy": obs}

    def _log_training_diag(self):
        # 周期性 stdout 训练指标：速度跟踪、平均速度、方向覆盖和直立情况。
        with torch.no_grad():
            vb = self.robot.data.root_lin_vel_b
            wb = self.robot.data.root_ang_vel_b
            cmd = self.commands
            lin_err = torch.linalg.norm(cmd[:, :2] - vb[:, :2], dim=1).mean().item()
            ang_err = (cmd[:, 2] - wb[:, 2]).abs().mean().item()
            speed = torch.linalg.norm(vb[:, :2], dim=1).mean().item()
            cmd_mag = torch.linalg.norm(cmd[:, :2], dim=1).mean().item()
            self._ensure_gate_mask()
            upright = (self.robot.data.projected_gravity_b[self._gate_mask, 2] < -0.7).float().mean().item()
            moving = (cmd_mag > 0.05)
            # 命令样本中各方向的粗略占比。
            fwd = (cmd[:, 0] > 0.2).float().mean().item()
            bwd = (cmd[:, 0] < -0.2).float().mean().item()
            lat = (cmd[:, 1].abs() > 0.15).float().mean().item()
            yaw = (cmd[:, 2].abs() > 0.2).float().mean().item()
        # 平面/诊断 TerrainImporter 可能没有课程等级；这里做保护，避免 env.step 崩溃。
        terrain_stats = self._terrain_level_stats()
        tlvl = float(terrain_stats["terrain_mean"])
        real_tlvl = float(terrain_stats["terrain_real_mean"])
        disc_tlvl = float(terrain_stats["terrain_discrete_mean"])
        tctx = self._terrain_ctx.abs().mean(0) if self.cfg.terrain_ctx_dim > 0 else None
        with torch.no_grad():
            # last_air_time 是每只脚已完成摆动相的滞空时长，单位与 air_time_target 一致。
            # current_air_time 是瞬时值，不能用于阶段门控。
            air = float(self._contact_sensor.data.last_air_time[:, self._feet_contact_ids].mean())
            act_std = float(self.actions.std()) if hasattr(self, "actions") else 0.0
            act_mag = float(self.actions.abs().mean()) if hasattr(self, "actions") else 0.0
        lp, ap, rl, ra, rair, gm = self._dbg

        self._ensure_gate_mask()
        # 跌倒 = 低高度且大倾斜。只按绝对 z 会把下坡/下台阶后的直立机器人误判为跌倒。
        # 因此 fall_rate 同时检查低高度和非直立。
        _below = self.robot.data.root_pos_w[self._gate_mask, 2] < self.cfg.termination_height
        _tilted = self.robot.data.projected_gravity_b[self._gate_mask, 2] >= -0.7   # 非直立。
        fall_rate = float((_below & _tilted).float().mean())
        progress_by_dir = {
            "fwd": self._fwd_prog,
            "back": self._back_prog,
            "lat": self._lat_prog,
            "yaw": self._yaw_prog,
        }
        min_prog, active_dirs = active_direction_progress(progress_by_dir, self.cfg, self._phase)
        self._slip_ema = 0.7 * self._slip_ema + 0.3 * getattr(self, "_slip_now", 0.0)
        diag_contact = float(getattr(self, "_diag_contact", 0.0))
        duty_balance = float(getattr(self, "_duty_balance", 0.0))
        C = self.cfg
        # 模仿精度 style_err：阶段门控使用的相对参考关节误差。
        # 该值由 _get_rewards 计算并缓存，这里只读取，避免重复 IK。
        self._style_err = getattr(self, "_style_err", 1.0)

        # 统一训练阶段：0 平地全方向，1 平地 mixed，2 地形/DR，3 扩展部署包络。
        # penalty ramp 从训练开始由预算控制。预算过高时 gate 降低，避免质量惩罚压过任务奖励。
        ramp_step = 1.0 / C.penalty_ramp_intervals
        budget_ratio = float(getattr(self, "_budget_ratio_ema", 0.0))
        budget_max = float(getattr(C, "penalty_budget_ratio_max", 0.8))
        if budget_ratio > budget_max:
            self._penalty_gate = max(0.0, self._penalty_gate - ramp_step)
        else:
            self._penalty_gate = min(1.0, self._penalty_gate + ramp_step)
        # 地形阶段的 clearance 渐入：让已有行走策略逐步提高抬脚要求，避免拖脚卡住地形课程。
        terrain_phase = self._phase >= getattr(self, "_terrain_start_phase", 5)
        if terrain_phase:
            self._clearance_gate = min(1.0, self._clearance_gate + 1.0 / C.clearance_ramp_intervals)
        terrain_slip_thr = float(getattr(C, "terrain_gate_slip_high", 0.22))
        terrain_health_ok = getattr(self, "_slip_high_fraction", 0.0) <= terrain_slip_thr
        self._terrain_health_ok = terrain_health_ok
        # 回退保护：能力退化时暂停地形/速度推进。
        # 地形推进保留低持续滑移约束，但不把平地对角/duty 模板强加到高楼梯或 boxes。
        self._advance_ok = not (fall_rate > C.regress_fall or min_prog < C.regress_prog) and (
            not terrain_phase or terrain_health_ok
        )
        # 阶段推进门控：需要连续 phase_intervals 个日志间隔满足。
        # 阈值按阶段查找，早期使用可达质量条，后期再收紧。
        def _phase_thr(name: str, default: float) -> float:
            for p in range(int(self._phase), -1, -1):
                v = getattr(C, f"phase_gate_{name}_{p}", None)
                if v is not None:
                    return float(v)
            return float(default)

        prog_thr = _phase_thr("prog", 0.70)
        slip_thr = _phase_thr("slip", 0.20)
        diag_thr = _phase_thr("diag", 0.80)
        duty_thr = _phase_thr("duty", 0.80)
        air_thr = _phase_thr("air", 0.0)
        # 质量门控从 phase 0 就生效；_0 阈值是早期可达的平地条。
        flat_quality_ok = (
            self._slip_ema <= slip_thr and diag_contact >= diag_thr and duty_balance >= duty_thr and air >= air_thr
        )
        terrain_quality_ok = self._slip_ema <= max(slip_thr, terrain_slip_thr)
        quality_ok = terrain_quality_ok if terrain_phase else flat_quality_ok
        phase_spec = phase_command_spec(C, self._phase) if phase_command_spec is not None else {}
        command_mode = str(phase_spec.get("command_mode", ""))
        is_mixed_phase = command_mode == "mixed"
        terrain_gate_level = real_tlvl if float(terrain_stats.get("terrain_real_frac", 0.0)) > 0.0 else tlvl
        discrete_gate_thr = float(getattr(C, "phase_gate_discrete_terrain_2", 0.0))
        discrete_gate_ok = discrete_gate_thr <= 0.0 or disc_tlvl >= discrete_gate_thr
        if terrain_phase:
            gate = (
                self._penalty_gate >= 1.0
                and min_prog >= prog_thr
                and quality_ok
                and (
                    not is_mixed_phase
                    or (
                        terrain_gate_level >= C.phase_gate_terrain_2
                        and discrete_gate_ok
                        and fall_rate < C.phase_gate_fall_2
                    )
                )
            )
        else:
            # 平地阶段：要求全方向跟踪、步态质量和 penalty ramp 完整。
            gate = self._penalty_gate >= 1.0 and min_prog >= prog_thr and quality_ok
        # 死锁保护：某一阶段超过 phase_max_steps 仍无法自然推进时，强制进入下一阶段并打印告警。
        # 计时从 penalty_gate 达到 1.0 后开始，避免把正常 ramp 过程误判为死锁。
        _step_now = int(getattr(self, "_log_step", 0))
        if self._penalty_gate >= 1.0 and getattr(self, "_phase_penalty_full_step", None) is None:
            self._phase_penalty_full_step = _step_now
        _clock_start = getattr(self, "_phase_penalty_full_step", None)
        _phase_max_steps = int(getattr(C, "phase_max_steps", 25000))
        _timed_out = (
            os.environ.get("TAILI_NO_PHASE_TIMEOUT") != "1"
            and _clock_start is not None
            and (_step_now - int(_clock_start)) >= _phase_max_steps
        )
        if self._phase < getattr(self, "_max_training_phase", 3):
            self._phase_count = self._phase_count + 1 if gate else max(0, self._phase_count - 1)
            if self._phase_count >= C.phase_intervals or _timed_out:
                _forced = _timed_out and self._phase_count < C.phase_intervals
                self._phase += 1; self._phase_count = 0
                self._phase_penalty_full_step = None   # 新阶段重新计时。
                _tag = " FORCED(deadlock-safeguard: min_prog<prog_thr for phase_max_steps post-ramp)" if _forced else ""
                print(f"[PHASE] -> {self._phase} at step {self._log_step}{_tag} "
                      f"(prog={min_prog:.2f} gait={gm:.2f} slip={self._slip_ema:.2f} terrain={tlvl:.2f})", flush=True)
        # 对称增强属于细化项：phase 0 关闭，phase 1 起打开。
        if getattr(self.cfg, "sym_augment", False):
            from . import symmetry
            symmetry.set_active(self._phase >= 1)

        # 速度课程从 phase 0 开始；命令上限仍受当前阶段范围限制。
        # 只有当前包络跟踪成功时才提高分方向速度上限。
        step = C.vel_cur_step
        if self._vel_curriculum_enable and self._advance_ok:
            if self._fwd_prog  >= C.vel_cur_up:  self._vel_max_fwd  = min(self._vel_max_fwd  + step, C.cmd_fwd_max)
            elif self._fwd_prog <= C.vel_cur_down: self._vel_max_fwd = max(self._vel_max_fwd - step, self._vel_floor_fwd)
            if self._back_prog >= C.vel_cur_up:  self._vel_max_back = min(self._vel_max_back + step, C.cmd_back_max)
            elif self._back_prog <= C.vel_cur_down: self._vel_max_back = max(self._vel_max_back - step, self._vel_floor_back)
            if self._lat_prog  >= C.vel_cur_up:  self._vel_max_lat  = min(self._vel_max_lat  + step, C.cmd_lat_max)
            elif self._lat_prog <= C.vel_cur_down: self._vel_max_lat = max(self._vel_max_lat - step, self._vel_floor_lat)
            # yaw 使用专用增长阈值，避免共享 vel_cur_up 对 yaw 过高导致只能缩小不能扩大。
            _yaw_up = float(getattr(C, "yaw_vel_cur_up", 0.40))
            if self._yaw_prog  >= _yaw_up:  self._vel_max_yaw  = min(self._vel_max_yaw  + step, C.cmd_yaw_max)
            elif self._yaw_prog <= C.vel_cur_down: self._vel_max_yaw = max(self._vel_max_yaw - step, self._vel_floor_yaw)

        # DR 课程：与地形课程并行，但不被地形等级硬绑定。
        # DR 按进展、直立和跌倒率自门控；地形和 DR 各自推进，互不作为唯一前置条件。
        # dr_unlock_terrain 可推迟 DR，让策略先获得一定地形立足能力。
        if (
            self._phase >= getattr(self, "_dr_start_phase", self._terrain_start_phase)
            and self._dr_level < 3
            and terrain_gate_level >= getattr(C, "dr_unlock_terrain", 0.0)
        ):
            gate_prog = [C.dr_gate_progress, C.dr_gate_progress_l2, C.dr_gate_progress_l3][self._dr_level]
            if min_prog >= gate_prog and upright >= 0.97 and fall_rate < 0.05:
                self._dr_gate_count += 1
                if self._dr_gate_count >= C.dr_gate_intervals:
                    self._dr_level += 1; self._dr_gate_count = 0
                    push_s = getattr(C, f"dr_push_interval_s_{self._dr_level}", 0.0)
                    self._push_steps = max(1, int(push_s / (C.dt * C.decimation))) if push_s > 0 else int(1e9)
                    print(f"[DR] Level up -> {self._dr_level} at step {self._log_step}", flush=True)
            else:
                self._dr_gate_count = max(0, self._dr_gate_count - 1)

        tctx_str = ""
        if tctx is not None and len(tctx) >= 3:
            tctx_str = f" terrain_ctx_slope={float(tctx[0]):.2f}/{float(tctx[1]):.2f} rough={float(tctx[2]):.3f}"
        # 按子地形拆分课程等级：平均地形等级会掩盖具体卡在哪类地形。
        # 这里按列映射回子地形，仅用于只读诊断。
        terr_type_str = ""
        if hasattr(self, "_terrain") and hasattr(self._terrain, "terrain_types"):
            if not hasattr(self, "_col_type"):
                sg = self.cfg.terrain.terrain_generator
                props = np.array([s.proportion for s in sg.sub_terrains.values()], dtype=np.float64)
                props = props / props.sum(); cum = np.cumsum(props); ncol = sg.num_cols
                ct = np.array([int(np.min(np.where(c / ncol + 1e-3 < cum)[0])) for c in range(ncol)])
                self._col_type = torch.as_tensor(ct, device=self.device)
                self._type_names = list(sg.sub_terrains.keys())
            env_type = self._col_type[self._terrain.terrain_types]
            lv = self._terrain.terrain_levels.float()
            parts = [f"{nm[:5]}={float(lv[env_type == ti].mean()):.1f}"
                     for ti, nm in enumerate(self._type_names) if bool((env_type == ti).any())]
            terr_type_str = "\n          terr_by_type: " + "  ".join(parts)
        d = getattr(self, "_rew_dbg", {})
        d = {k: d.get(k, 0.0) for k in ("lin", "ang", "gait", "imit", "stand", "height", "slip", "clear",
                                        "hip", "offax", "over", "under", "wrong", "climb", "terr_up", "terr_down",
                                        "terr_support", "terr_quality", "terr_collapse", "land", "torq", "arate", "vz", "wxy")}
        print(f"[ENV s={self._log_step}] PHASE={self._phase} pen_gate={self._penalty_gate:.2f} "
              f"budget={float(getattr(self, '_budget_ratio_ema', 0.0)):.2f} "
              f"clr_gate={self._clearance_gate:.2f} "
              f"adv_ok={int(self._advance_ok)} slip_ema={self._slip_ema:.2f} | speed={speed:.2f} cmd={cmd_mag:.2f} "
              f"fwd_ceil={self._vel_max_fwd:.2f} back_ceil={self._vel_max_back:.2f} "
              f"lat_ceil={self._vel_max_lat:.2f} yaw_ceil={self._vel_max_yaw:.2f} DR_level={self._dr_level}\n"
              f"          prog active={','.join(active_dirs)} "
              f"fwd/back/lat/yaw={self._fwd_prog:.2f}/{self._back_prog:.2f}"
              f"/{self._lat_prog:.2f}/{self._yaw_prog:.2f} "
              f"gait_match={gm:.2f} diag={diag_contact:.2f} duty_bal={duty_balance:.2f} "
              f"style_err={self._style_err:.3f} upright={upright:.2f} air={air:.2f} "
              f"terrain={tlvl:.2f} real={real_tlvl:.2f} discrete={disc_tlvl:.2f}{tctx_str}{terr_type_str}\n"
              f"          rew[task]: lin={d['lin']:+.2f} ang={d['ang']:+.2f} gait={d['gait']:+.2f} imit={d['imit']:+.2f} "
              f"stand={d['stand']:+.2f} height={d['height']:+.2f} slip={d['slip']:+.2f} clear={d['clear']:+.2f} "
              f"hip={d['hip']:+.2f} offax={d['offax']:+.2f} over={d['over']:+.2f} under={d['under']:+.2f} wrong={d['wrong']:+.2f} "
              f"land={d['land']:+.2f} vz={d['vz']:+.2f} "
              f"wxy={d['wxy']:+.2f} torq={d['torq']:+.3f} arate={d['arate']:+.2f}  (+AMP style by skrl)\n"
              f"          rew[terrain]: climb={d['climb']:+.2f} up={d['terr_up']:+.2f} down={d['terr_down']:+.2f} "
              f"support={d['terr_support']:+.2f} quality={d['terr_quality']:+.2f} collapse={d['terr_collapse']:+.2f}\n"
              f"          GATE phi{self._phase}: min_prog={min_prog:.2f}(need>={prog_thr:.2f}) "
              f"terrain_gate={terrain_gate_level:.2f}/{C.phase_gate_terrain_2:.2f} "
              f"disc_gate={disc_tlvl:.2f}/{discrete_gate_thr:.2f} "
              f"quality={int(quality_ok)}(slip<={slip_thr:.2f},diag>={diag_thr:.2f},duty>={duty_thr:.2f},air>={air_thr:.2f}) "
              f"pen_gate={self._penalty_gate:.2f}/1.0 count={self._phase_count}/{C.phase_intervals} | "
              f"act_std={act_std:.3f} cmd_frac={fwd:.2f}/{bwd:.2f}/{lat:.2f}/{yaw:.2f}", flush=True)

    def _get_rewards(self) -> torch.Tensor:
        # 双侧速度跟踪：速度由这里统一负责，欠速和超速都会降低奖励。
        # cmd=0 时奖励零残差，形成干净停步；AMP/参考只判断风格，不负责速度语义。
        vel_lin = self.robot.data.root_lin_vel_b[:, :2]
        wz = self.robot.data.root_ang_vel_b[:, 2]
        cmd_lin = self.commands[:, :2]
        cmd_ang = self.commands[:, 2]
        cmd_lin_m2 = torch.sum(cmd_lin * cmd_lin, dim=1)            # ||cmd_lin||^2，供下方门控使用。
        cmd_ang_2 = cmd_ang * cmd_ang
        thr = 0.0025                                               # (|cmd|>0.05)^2，区分运动/站立。
        mv_l = cmd_lin_m2 > thr
        mv_a = cmd_ang_2 > thr
        # 截断方向进展：bootstrap 时提供“朝命令方向移动”的强梯度。
        # 站着不动在运动命令下没有进展奖励；超速由 r_overshoot 单独刹车。
        # lin_prog / ang_prog 同时用于阶段门控和速度课程。
        lin_prog = torch.clamp(torch.sum(vel_lin * cmd_lin, dim=1) / cmd_lin_m2.clamp(min=thr), 0.0, 1.0)
        ang_prog = torch.clamp(wz * cmd_ang / cmd_ang_2.clamp(min=thr), 0.0, 1.0)
        lin_stand = torch.exp(-torch.sum(vel_lin * vel_lin, dim=1) / self.cfg.stand_sigma)
        ang_stand = torch.exp(-wz * wz / self.cfg.stand_sigma)
        lin_stand_w = torch.where(mv_a & ~mv_l, 0.25, 1.0)   # 纯 yaw 时不对非命令线速度轴给满额站立奖励。
        ang_stand_w = torch.where(mv_l & ~mv_a, 0.25, 1.0)
        r_lin = self.cfg.rew_track_lin * torch.where(mv_l, lin_prog, lin_stand * lin_stand_w)
        r_ang = self.cfg.rew_track_ang * torch.where(mv_a, ang_prog, ang_stand * ang_stand_w)
        r_alive = self.cfg.rew_alive * (~self.reset_terminated).float()
        r_arate = self.cfg.rew_action_rate * torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        r_jacc = self.cfg.rew_joint_acc * torch.sum(torch.square(self.robot.data.joint_acc), dim=1)
        # 电机保护：只在力矩接近关节 effort_limit 的压力区惩罚。
        # 正常步态力矩不受影响，接近饱和时二次增长。
        try:
            tau = self.robot.data.applied_torque                                      # (N, 12)
            eff_lim = self.robot.actuators["legs"].effort_limit                        # (N,12)，每关节 Nm。
            tau_over = torch.clamp(tau.abs() - self.cfg.torque_limit_frac * eff_lim, min=0.0)
            r_torque = self.cfg.rew_torque * torch.sum(tau_over * tau_over, dim=1)
        except Exception as _e:                                                       # 不允许该分支中断奖励计算。
            r_torque = torch.zeros(self.num_envs, device=self.device)
            if "torque" not in self._dr_warned:
                print(f"[REW] torque penalty unavailable ({type(_e).__name__}); skipping", flush=True)
                self._dr_warned.add("torque")
        r_vz = self.cfg.rew_lin_vel_z * torch.square(self.robot.data.root_lin_vel_b[:, 2])
        r_wxy = self.cfg.rew_ang_vel_xy * torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
        # 足端滞空时间：在落脚时结算，目标是接近 air_time_target。
        # 只有机器人沿命令方向产生有效位移或正确转向时才给滞空奖励，避免原地乱抬腿刷分。
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_ids]
        air_time = self._contact_sensor.data.current_air_time[:, self._feet_contact_ids]
        cmd_lin = self.commands[:, :2]
        cmd_lin_mag = torch.norm(cmd_lin, dim=1)
        vb2 = self.robot.data.root_lin_vel_b[:, :2]
        vel_along = (vb2 * cmd_lin).sum(-1) / cmd_lin_mag.clamp(min=0.1)         # 沿命令方向速度。
        yaw_match = self.commands[:, 2] * self.robot.data.root_ang_vel_b[:, 2]   # 大于 0 表示转向方向正确。
        effective = ((vel_along > 0.1) | (yaw_match > 0.05)).float()
        air_credit = torch.clamp(air_time - self.cfg.air_time_min, min=0.0,
                                 max=self.cfg.air_time_target - self.cfg.air_time_min)
        r_air = self.cfg.rew_feet_air_time * torch.sum(air_credit * first_contact.float(), dim=1) * effective
        mv_any = mv_l | mv_a

        # 步态相位接触奖励：对角小跑时钟给出每条腿应处于支撑还是摆动。
        # 实际接触越匹配奖励越高；站立命令下不强加节律。
        in_contact = self._in_contact                                                    # 由 _get_observations 缓存。
        leg_phase = self._leg_phases()
        desired_stance = (leg_phase < self.cfg.gait_duty).float()                       # 1 表示应处于支撑相。
        gait_match = (desired_stance * in_contact + (1.0 - desired_stance) * (1.0 - in_contact)).mean(dim=1)
        # 原始 gait_match 不加 dead-zone，使 bootstrap 时也有接触节律梯度。
        # 粗糙地形上降低固定节律权重，允许策略打破平地小跑节律去跨越地形。
        if self.cfg.terrain_ctx_dim >= 3:
            rough_gate = torch.clamp(self._terrain_ctx[:, 2] / self.cfg.gait_rough_scale, 0.0, 1.0)
        else:
            rough_gate = torch.zeros(self.num_envs, device=self.device)
        r_gait = self.cfg.rew_gait_phase * gait_match * mv_any.float() * (1.0 - rough_gate)

        # 轨迹级 imitation：按当前命令和 gait phase 直接跟踪参考关节。
        # AMP 只是分布先验，imitation 用于把实际步态拉向参考轨迹；粗糙地形上同样放松。
        if self.cfg.rew_imitate != 0.0:
            spd_im = torch.norm(self.commands[:, :2], dim=1)
            T_im = torch.clamp(self.cfg.gait_period - self.cfg.gait_period_slope * spd_im,
                               min=self.cfg.gait_period_min, max=self.cfg.gait_period)
            rough_im = self._terrain_ctx[:, 2] if self.cfg.terrain_ctx_dim >= 3 else None
            jp_ref = flat_reference(self.commands, self._gait_phase * T_im, gait_period=self.cfg.gait_period,
                                    gait_period_slope=self.cfg.gait_period_slope, gait_period_min=self.cfg.gait_period_min,
                                    clearance_base=self.cfg.base_clearance, roughness=rough_im,
                                    clearance_rough_gain=self.cfg.ref_clearance_rough_gain, stance_dx=self.cfg.stance_dx,
                                    jp_only=True, iters=12)            # 快路径：只生成关节位置，12 次 IK。
            jp_ref = jp_ref[:, self.motion_dof_indexes]
            jdiff = self.robot.data.joint_pos - jp_ref                                    # (N,12)
            # turn/strafe 门控：转向和横移更依赖 imitation 锚定足端位置；
            # yaw 按足端半径近似缩放到线速度量级。
            _turn_mag = self.commands[:, 1].abs() + 0.30 * self.commands[:, 2].abs()
            _fwd_mag = self.commands[:, 0].abs()
            turn_frac = _turn_mag / (_turn_mag + _fwd_mag + 1e-3)
            # 前进 imitation floor：纯前进样本也给少量轨迹锚点，用于约束落脚、滑移和对称。
            # 参考轨迹随速度缩放，只约束步态形状，不替代速度跟踪。
            _imit_fwd_floor = float(getattr(self.cfg, "imitate_fwd_floor", 0.0))
            if _imit_fwd_floor > 0.0:
                turn_frac = torch.clamp(turn_frac, min=_imit_fwd_floor)
            r_imitate = self.cfg.rew_imitate * turn_frac \
                        * torch.exp(-jdiff.pow(2).sum(dim=1) / self.cfg.imitate_sigma) \
                        * mv_any.float() * (1.0 - rough_gate)
            self._style_err = float(jdiff.abs().mean(dim=1)[mv_any].mean()) if bool(mv_any.any()) \
                              else getattr(self, "_style_err", 1.0)
        else:
            r_imitate = torch.zeros(self.num_envs, device=self.device)

        # 摆动相拖脚惩罚：摆动窗口内仍承重表示拖脚/滑行。
        # 该项默认关闭；非零时只在运动命令和粗糙地形上生效，避免平地过度抬脚。
        r_swing_drag = (self.cfg.rew_swing_drag * ((1.0 - desired_stance) * in_contact).mean(dim=1)
                        * mv_any.float() * rough_gate)

        # 硬步态约束：惩罚任意腿偏离命令接触时序。
        # 默认关闭；粗糙地形上放松，让策略可以打破平地节律跨越地形。
        off_sched = (1.0 - desired_stance) * in_contact + desired_stance * (1.0 - in_contact)   # (N,4)
        _ge_relax = (1.0 - rough_gate) if (self.cfg.terrain_ctx_dim >= 3) else 1.0
        r_gait_enforce = self.cfg.rew_gait_enforce * off_sched.mean(dim=1) * mv_any.float() * _ge_relax

        # 非命令轴惩罚：惩罚与命令方向垂直的线速度。
        cmd_lin_norm = cmd_lin_mag.clamp(min=1e-4)
        cmd_hat = cmd_lin / cmd_lin_norm[:, None]
        v_along_s = (vel_lin * cmd_hat).sum(-1)                                          # signed speed along cmd
        v_perp = vel_lin - v_along_s[:, None] * cmd_hat
        r_offaxis = self.cfg.rew_offaxis_vel * torch.sum(v_perp * v_perp, dim=1) * mv_l.float()

        # 超速惩罚：超过命令速度的部分直接给刹车梯度。
        lin_over = torch.clamp(v_along_s - cmd_lin_mag, min=0.0)
        ang_over = torch.clamp(wz * torch.sign(cmd_ang) - cmd_ang.abs(), min=0.0)
        r_overshoot = self.cfg.rew_overshoot * (lin_over * lin_over * mv_l.float()
                                                + ang_over * ang_over * mv_a.float())
        # 带容差的欠速惩罚：只惩罚低于验收带的速度，不对所有低于命令的样本施压。
        # 这样后退过慢时有梯度，但跟踪已合格时不会持续推高速度。
        lin_tol = torch.maximum(
            torch.full_like(cmd_lin_mag, self.cfg.speed_tol_abs),
            self.cfg.speed_tol_rel * cmd_lin_mag,
        )
        ang_tol = torch.full_like(cmd_ang, self.cfg.track_sigma_ang)
        lin_under = torch.clamp((cmd_lin_mag - lin_tol) - v_along_s, min=0.0)
        ang_under = torch.clamp((cmd_ang.abs() - ang_tol) - wz * torch.sign(cmd_ang), min=0.0)
        r_underspeed = self.cfg.rew_underspeed * (lin_under * lin_under * mv_l.float()
                                                  + ang_under * ang_under * mv_a.float())
        back_cmd = (
            (self.commands[:, 0] < -0.05)
            & (self.commands[:, 1].abs() < 0.05)
            & (self.commands[:, 2].abs() < 0.05)
        )
        # 后退专用辅助 shaping：只给负 vx 命令额外梯度，避免通用欠速项伤害前进能力。
        back_speed_deficit = torch.clamp((self.commands[:, 0].abs() - lin_tol) - (-vel_lin[:, 0]), min=0.0)
        back_wrong = torch.clamp(vel_lin[:, 0], min=0.0)
        r_backward_aux = (
            self.cfg.rew_backward_underspeed * back_speed_deficit * back_speed_deficit
            + self.cfg.rew_backward_wrong_dir * back_wrong * back_wrong
        ) * back_cmd.float()
        lat_cmd = (
            (self.commands[:, 1].abs() > 0.05)
            & (self.commands[:, 0].abs() < 0.05)
            & (self.commands[:, 2].abs() < 0.05)
        )
        lat_signed_speed = vel_lin[:, 1] * torch.sign(self.commands[:, 1])
        lat_speed_deficit = torch.clamp((self.commands[:, 1].abs() - lin_tol) - lat_signed_speed, min=0.0)
        r_lateral_aux = self.cfg.rew_lateral_underspeed * lat_speed_deficit * lat_speed_deficit * lat_cmd.float()
        # 反向惩罚：对与命令方向相反的速度直接加成本，减少站住或反向移动的捷径。
        lin_wrong = torch.clamp(-v_along_s, min=0.0)
        ang_wrong = torch.clamp(-(wz * torch.sign(cmd_ang)), min=0.0)
        r_wrong_dir = self.cfg.rew_wrong_dir * (lin_wrong * lin_wrong * mv_l.float()
                                                + ang_wrong * ang_wrong * mv_a.float())

        # 站立姿态奖励：零命令下保持直立、名义高度、默认关节和四足接触。
        # 机身高度提前计算，供 r_height 和 r_stand 共用。
        hgt = self.robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]

        # 常开 base height：运动时 r_stand 关闭，因此需要独立高度项防止后腿塌低。
        r_height = self.cfg.rew_base_height * torch.clamp(self.cfg.stand_height - hgt, min=0.0)

        # 站立项由直立、默认关节、高度和四足接触四部分组成。
        standing = (~mv_l) & (~mv_a)
        up = torch.clamp(-self.robot.data.projected_gravity_b[:, 2], 0.0, 1.0)                  # 1 表示直立。
        # jdev 使用较宽 Gaussian，保证趴下时仍有回到默认姿态的梯度。
        jdev = torch.exp(-torch.sum((self.robot.data.joint_pos - self.action_offset) ** 2, dim=1) / 4.0)
        # hdev 使用线性项，保证从低高度到站立高度始终有梯度。
        hdev = torch.clamp(hgt / self.cfg.stand_height, 0.0, 1.0)
        feet_down = in_contact.mean(dim=1)                                                      # frac of 4 feet planted
        r_stand = self.cfg.rew_stand_pose * ((up + jdev + hdev + feet_down) / 4.0) * standing.float()
        # 站立静止：零命令时惩罚关节速度，使机器人平滑回到立正而不是碎步抖动。
        r_stand_still = self.cfg.rew_stand_still * torch.sum(self.robot.data.joint_vel ** 2, dim=1) * standing.float()

        # 髋关节中立：非横移命令下压制髋关节外翻/内扣；横移时关闭该约束。
        lat_active = torch.clamp(self.commands[:, 1].abs() / self.cfg.hip_neutral_lat_scale, 0.0, 1.0)
        hip_dev = torch.sum(self.robot.data.joint_pos[:, self._hip_idx] ** 2, dim=1)
        r_hip = self.cfg.rew_hip_neutral * hip_dev * (1.0 - lat_active)

        # 步态质量奖励：滑移、摆动方向、落脚减速和 clearance。
        foot_vel_w3 = self.robot.data.body_lin_vel_w[:, self.foot_indexes, :]  # (N,4,3)
        q = self.robot.data.root_quat_w[:, None, :].expand(-1, self.n_feet, -1).reshape(-1, 4)
        foot_vel_b = quat_apply_inverse(q, foot_vel_w3.reshape(-1, 3)).reshape(-1, self.n_feet, 3)
        foot_vel_b_xy = foot_vel_b[:, :, :2]                                  # (N,4,2), base frame

        # 落脚释放窗口：clearance 和 swing-direction 只在抬腿/最高点阶段强约束，
        # 落脚后段逐步释放，让脚能单调下降并干净触地。
        s_swing = ((leg_phase - self.cfg.gait_duty) / (1.0 - self.cfg.gait_duty)).clamp(0.0, 1.0)  # (N,4)
        lift_w = torch.clamp((0.65 - s_swing) / 0.20, 0.0, 1.0)            # 1 in lift/apex, 0 by s_swing=0.65
        land_w = torch.clamp((s_swing - 0.70) / 0.30, 0.0, 1.0)           # 0 until touchdown phase, 1 at contact

        # 1. 支撑滑移：惩罚接触脚的水平滑动。
        slip_speed = foot_vel_b_xy.norm(dim=-1)                            # (N,4), m/s
        r_stance_slip = self.cfg.rew_stance_slip * (in_contact * slip_speed).sum(dim=1) * mv_any.float()
        # 平均支撑滑移暴露给阶段门控。
        slip_now_env = (in_contact * slip_speed).sum(dim=1) / in_contact.sum(dim=1).clamp(min=1.0)
        self._slip_now = float(slip_now_env.mean())

        # 2. 摆动方向：摆动脚应沿命令方向移动；纯 yaw 使用绕 base 旋转产生的切向方向。
        foot_rel_b = self._feet_rel_base(self.robot.data.body_pos_w[:, self.foot_indexes],
                                         self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.n_feet)
        foot_rel_b = foot_rel_b.reshape(-1, self.n_feet, 3)
        cmd_hat_2d = cmd_lin / cmd_lin_mag[:, None].clamp(min=1e-4)      # (N,2) unit command dir
        yaw_vec = torch.stack([-cmd_ang[:, None] * foot_rel_b[:, :, 1],
                               cmd_ang[:, None] * foot_rel_b[:, :, 0]], dim=-1)
        yaw_hat = yaw_vec / torch.norm(yaw_vec, dim=-1, keepdim=True).clamp(min=1e-4)
        desired_hat = torch.where(mv_l[:, None, None], cmd_hat_2d[:, None, :], yaw_hat)
        swing_active = (mv_l | mv_a).float()[:, None]
        swing_mask = (1.0 - in_contact) * swing_active                    # (N,4)
        vel_along_cmd = (foot_vel_b_xy * desired_hat).sum(dim=-1)         # (N,4)
        # 相位锚定的落脚释放窗口：bootstrap 阶段保留完整摆动窗口，
        # 之后通过 penalty_gate 渐入到 lift/apex-only，避免早期探索被饿死。
        lift_gate = (1.0 - self._penalty_gate) + self._penalty_gate * lift_w
        r_swing_dir = self.cfg.rew_swing_dir * (vel_along_cmd * swing_mask * lift_gate).mean(dim=1)

        # 2b. 落脚减速：落脚阶段惩罚足端水平速度，使足端在触地前刹住。
        r_land_decel = self.cfg.rew_land_decel * (slip_speed * swing_mask * land_w).mean(dim=1)

        # 3. clearance：摆动脚应相对脚下/附近地形抬起，而不只是相对机身抬起。
        hits = self._height_scanner.data.ray_hits_w                       # (N,P,3), yaw-aligned grid around base
        finite_hits = torch.isfinite(hits[:, :, 2])
        hits_z_valid = torch.where(finite_hits, hits[:, :, 2], torch.zeros_like(hits[:, :, 2]))
        foot_pos = self.robot.data.body_pos_w[:, self.foot_indexes, :]     # (N,4,3)
        dxy = foot_pos[:, :, None, :2] - hits[:, None, :, :2]              # (N,4,P,2)
        nearest = torch.argmin(torch.sum(dxy * dxy, dim=-1), dim=-1)       # (N,4)
        ground_z = torch.gather(hits[:, :, 2], 1, nearest)                 # (N,4)
        foot_clearance = torch.clamp(foot_pos[:, :, 2] - ground_z, min=0.0)
        contact_w = in_contact.clamp(0.0, 1.0)
        contact_sum = contact_w.sum(dim=1)
        support_z_contact = (ground_z * contact_w).sum(dim=1) / contact_sum.clamp(min=1.0)
        support_z_feet = ground_z.median(dim=1).values
        support_z = torch.where(contact_sum > 0.5, support_z_contact, support_z_feet)
        relxy_ahead = hits[:, :, :2] - self.robot.data.root_pos_w[:, None, :2]
        cmdw_ahead = _command_xy_world_from_root_yaw(self.robot.data.root_quat_w, self.commands[:, :2])
        cdir_ahead = cmdw_ahead / cmdw_ahead.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ahead_s = (relxy_ahead * cdir_ahead[:, None, :]).sum(-1)
        ahead_mask = ahead_s > 0.15
        has_ahead = ahead_mask.any(dim=1)
        hz_hi = torch.nan_to_num(hits[:, :, 2], nan=-1e3, posinf=-1e3, neginf=-1e3)
        hz_lo = torch.nan_to_num(hits[:, :, 2], nan=1e3, posinf=1e3, neginf=1e3)
        ahead_max = torch.where(ahead_mask, hz_hi, torch.full_like(hz_hi, -1e3)).max(dim=1).values
        ahead_min = torch.where(ahead_mask, hz_lo, torch.full_like(hz_lo, 1e3)).min(dim=1).values
        ahead_count = ahead_mask.float().sum(dim=1)
        ahead_s_min = torch.where(ahead_mask, ahead_s, torch.full_like(ahead_s, 1e3)).min(dim=1).values
        ahead_s_max = torch.where(ahead_mask, ahead_s, torch.full_like(ahead_s, -1e3)).max(dim=1).values
        rel_x = relxy_ahead[:, :, 0]
        rel_y = relxy_ahead[:, :, 1]
        terrain_scan_probe = {
            "finite_frac": finite_hits.float().mean(),
            "hits_z_min": hits_z_valid.min(),
            "hits_z_max": hits_z_valid.max(),
            "hits_z_span_mean": (hits_z_valid.max(dim=1).values - hits_z_valid.min(dim=1).values).mean(),
            "rel_x_min": rel_x.min(),
            "rel_x_max": rel_x.max(),
            "rel_y_min": rel_y.min(),
            "rel_y_max": rel_y.max(),
            "cmd_world_x_mean": cdir_ahead[:, 0].mean(),
            "cmd_world_y_mean": cdir_ahead[:, 1].mean(),
            "ahead_count_mean": ahead_count.mean(),
            "ahead_count_max": ahead_count.max(),
            "ahead_has_frac": has_ahead.float().mean(),
            "ahead_s_min": torch.where(has_ahead, ahead_s_min, torch.zeros_like(ahead_s_min)).mean(),
            "ahead_s_max": torch.where(has_ahead, ahead_s_max, torch.zeros_like(ahead_s_max)).mean(),
            "support_z_mean": support_z.mean(),
            "support_z_min": support_z.min(),
            "support_z_max": support_z.max(),
            "ahead_max_z_mean": torch.where(has_ahead, ahead_max, torch.zeros_like(ahead_max)).mean(),
            "ahead_min_z_mean": torch.where(has_ahead, ahead_min, torch.zeros_like(ahead_min)).mean(),
            "raw_rise_mean": torch.where(has_ahead, ahead_max - support_z, torch.zeros_like(support_z)).mean(),
            "raw_rise_max": torch.where(has_ahead, ahead_max - support_z, torch.zeros_like(support_z)).max(),
            "raw_drop_mean": torch.where(has_ahead, support_z - ahead_min, torch.zeros_like(support_z)).mean(),
            "raw_drop_max": torch.where(has_ahead, support_z - ahead_min, torch.zeros_like(support_z)).max(),
        }
        terrain_rise_ahead = torch.where(has_ahead, torch.clamp(ahead_max - support_z, min=0.0), torch.zeros_like(support_z))
        terrain_drop_ahead = torch.where(has_ahead, torch.clamp(support_z - ahead_min, min=0.0), torch.zeros_like(support_z))
        # 粗糙度门控 clearance：只在真实粗糙/离散地形上加重抬脚，平地保持低而高效。
        # cg = 阶段渐入门控 x 每个 env 的粗糙度因子。
        rough = self._terrain_ctx[:, 2]                                                  # (N,) terrain height std (m)
        rough_factor = torch.clamp((rough - self.cfg.clr_rough_flat) / self.cfg.clr_rough_span, 0.0, 1.0)  # (N,)
        # 离散障碍：楼梯/boxes 可强制 rough_factor=1，避免低粗糙度读数导致抬脚目标不足。
        if getattr(self.cfg, "discrete_clearance", False):
            self._ensure_gate_mask()
            rough_factor = torch.where(self._discrete_terrain_mask, torch.ones_like(rough_factor), rough_factor)
        cg = self._clearance_gate * rough_factor                                         # (N,) 0 on flat, ->1 on rough
        # 权重：平地轻约束，粗糙地形逐步加重。
        eff_clearance_w = self.cfg.rew_clearance + (self.cfg.rew_clearance_heavy - self.cfg.rew_clearance) * cg  # (N,)
        # 目标：基础抬脚高度加粗糙地形 bonus。
        clearance_target = self.cfg.base_clearance + self.cfg.clr_rough_bonus_max * cg[:, None]  # (N,1)
        clr_deficit = torch.clamp(clearance_target - foot_clearance, min=0.0)
        # lift_gate：早期完整摆动窗口，后期渐入 lift/apex-only，释放自然落脚下降段。
        r_clearance = eff_clearance_w * (swing_mask * lift_gate * clr_deficit).mean(dim=1)

        # 分方向速度课程进展：排除 hard 类型 env，避免它们拖低阶段/速度/DR 门控。
        self._ensure_gate_mask()
        gm_e = self._gate_mask
        fwd_mask  = mv_l & (self.commands[:, 0] > 0.1) & gm_e
        back_mask = mv_l & (self.commands[:, 0] < -0.1) & gm_e
        lat_mask  = mv_l & (self.commands[:, 1].abs() > 0.1) & gm_e
        yaw_mask  = mv_a & gm_e
        if bool(fwd_mask.any()):  self._fwd_prog  = float(lin_prog[fwd_mask].mean())
        if bool(back_mask.any()): self._back_prog = float(lin_prog[back_mask].mean())
        if bool(lat_mask.any()):  self._lat_prog  = float(lin_prog[lat_mask].mean())
        if bool(yaw_mask.any()):  self._yaw_prog  = float(ang_prog[yaw_mask].mean())

        # 最小地形通过驱动：tracking 和 clearance 不会直接奖励向上越过台阶。
        # 该项保持窄范围，避免平地步态被拉成无意义跳跃。
        r_climb = torch.zeros(self.num_envs, device=self.device)
        r_terrain_up = torch.zeros(self.num_envs, device=self.device)
        r_terrain_down = torch.zeros(self.num_envs, device=self.device)
        terrain_probe = {
            "rise_ahead_mean": terrain_rise_ahead.mean(),
            "rise_ahead_max": terrain_rise_ahead.max(),
            "drop_ahead_mean": terrain_drop_ahead.mean(),
            "drop_ahead_max": terrain_drop_ahead.max(),
            "disc_frac": torch.zeros((), device=self.device),
            "moving_lin_frac": torch.zeros((), device=self.device),
            "upright_frac": torch.zeros((), device=self.device),
            "slip_weight_mean": torch.zeros((), device=self.device),
            "common_mean": torch.zeros((), device=self.device),
            "common_max": torch.zeros((), device=self.device),
            "up_active_mean": torch.zeros((), device=self.device),
            "up_active_max": torch.zeros((), device=self.device),
            "down_active_mean": torch.zeros((), device=self.device),
            "down_active_max": torch.zeros((), device=self.device),
            "up_core_mean": torch.zeros((), device=self.device),
            "down_core_mean": torch.zeros((), device=self.device),
        }
        terrain_probe.update(terrain_scan_probe)
        w_climb = float(getattr(self.cfg, "rew_climb", 0.0))
        w_terrain_up = float(getattr(self.cfg, "rew_terrain_up", 0.0))
        w_terrain_down = float(getattr(self.cfg, "rew_terrain_down", 0.0))
        if (w_climb > 0.0 or w_terrain_up > 0.0 or w_terrain_down > 0.0) and self._phase >= getattr(self, "_terrain_start_phase", 5):
            self._ensure_gate_mask()
            disc = getattr(self, "_discrete_terrain_mask", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
            moving_lin_cmd = cmd_lin_mag > 0.10
            upright = (-self.robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0) > 0.85
            # 软滑移权重：目标以下保持满额，超过后逐步衰减，避免硬截断丢梯度。
            slip_target = float(getattr(self.cfg, "climb_slip_gate", 0.30))
            slip_span = max(float(getattr(self.cfg, "climb_slip_soft_span", 0.18)), 1e-6)
            slip_weight = torch.clamp(1.0 - torch.clamp(slip_now_env - slip_target, min=0.0) / slip_span, 0.0, 1.0)
            common = disc.float() * moving_lin_cmd.float() * upright.float() * slip_weight
            eps = float(getattr(self.cfg, "terrain_transition_eps", 0.05))
            span = max(float(getattr(self.cfg, "terrain_transition_span", 0.08)), 1e-6)
            up_active = torch.clamp((terrain_rise_ahead - eps) / span, 0.0, 1.0)
            down_active = torch.clamp((terrain_drop_ahead - eps) / span, 0.0, 1.0)
            up_cap = max(float(getattr(self.cfg, "terrain_up_vz_cap", getattr(self.cfg, "climb_vz_cap", 0.6))), 1e-6)
            down_cap = max(float(getattr(self.cfg, "terrain_down_vz_cap", 0.5)), 1e-6)
            up_v = self.robot.data.root_lin_vel_w[:, 2].clamp(min=0.0, max=up_cap)
            down_v = (-self.robot.data.root_lin_vel_w[:, 2]).clamp(min=0.0, max=down_cap)
            down_target = max(float(getattr(self.cfg, "terrain_down_vz_target", 0.18)), 1e-6)
            down_control = torch.clamp(1.0 - torch.clamp(down_v - down_target, min=0.0) / down_target, 0.0, 1.0)
            up_core = 0.55 * lin_prog + 0.45 * (up_v / up_cap)
            down_core = lin_prog * (0.50 + 0.50 * down_control)
            r_terrain_up = w_terrain_up * up_active * common * up_core
            r_terrain_down = w_terrain_down * down_active * common * down_core
            r_climb = w_climb * up_active * common * up_core
            terrain_probe.update({
                "disc_frac": disc.float().mean(),
                "moving_lin_frac": moving_lin_cmd.float().mean(),
                "upright_frac": upright.float().mean(),
                "slip_weight_mean": slip_weight.mean(),
                "common_mean": common.mean(),
                "common_max": common.max(),
                "up_active_mean": up_active.mean(),
                "up_active_max": up_active.max(),
                "down_active_mean": down_active.mean(),
                "down_active_max": down_active.max(),
                "up_core_mean": up_core.mean(),
                "down_core_mean": down_core.mean(),
            })

        self._dbg = (
            float(lin_prog[mv_l].mean()) if bool(mv_l.any()) else 0.0,
            float(ang_prog[mv_a].mean()) if bool(mv_a.any()) else 0.0,
            float(r_lin.mean()), float(r_ang.mean()), float(r_air.mean()),
            float(gait_match[mv_any].mean()) if bool(mv_any.any()) else 0.0)
        self._last_gait_match = self._dbg[5]
        # [ENV] 日志中的奖励分解均值，用于判断主导项以及任务项是否与 AMP 风格冲突。
        pg = self._penalty_gate
        self._rew_dbg = {
            "lin": float(r_lin.mean()), "ang": float(r_ang.mean()), "gait": float(r_gait.mean()),
            "imit": float(r_imitate.mean()),
            "stand": float(r_stand.mean()), "height": float(r_height.mean()), "slip": float(r_stance_slip.mean()),
            "clear": float(r_clearance.mean()), "hip": float(r_hip.mean()), "offax": float(r_offaxis.mean()),
            "over": float(r_overshoot.mean()), "under": float(r_underspeed.mean()),
            "back_aux": float(r_backward_aux.mean()), "lat_aux": float(r_lateral_aux.mean()), "wrong": float(r_wrong_dir.mean()),
            "climb": float(r_climb.mean()), "terr_up": float(r_terrain_up.mean()), "terr_down": float(r_terrain_down.mean()),
            "terr_probe_rise": float(terrain_probe["rise_ahead_mean"]),
            "terr_probe_up_active": float(terrain_probe["up_active_mean"]),
            "terr_probe_common": float(terrain_probe["common_mean"]),
            "land": float((pg * r_land_decel).mean()), "torq": float(r_torque.mean()),
            "arate": float(r_arate.mean()), "vz": float(r_vz.mean()), "wxy": float(r_wxy.mean()),
            "stand_still": float(r_stand_still.mean()),   # 常开站立静止项，记录未额外 gate 的值。
        }
        # 基础惩罚从 step 0 常开：offaxis、overshoot、hip、slip 和轻量 clearance 是 bootstrap 梯度场。
        # 它们负责压制站立碎步、反向漂移、髋外翻和低抬脚；重 clearance 才是后期渐入项。
        base_penalties = (
            r_offaxis + r_overshoot + r_underspeed + r_backward_aux + r_lateral_aux + r_wrong_dir
            + r_hip + r_stance_slip + r_clearance + r_swing_drag
        )
        # 后期细化项通过 penalty_gate 渐入：land_decel 和 gait_enforce。
        # stand_still 是核心停步技能，只在零命令下生效，因此保持常开。
        refinement_penalties = r_land_decel + r_gait_enforce
        # torque 电机保护常开；正常步态下近似为 0，只惩罚接近饱和压力区。
        reward = (r_lin + r_ang + r_alive + r_arate + r_jacc + r_vz + r_wxy + r_air
                  + r_gait + r_imitate + r_height + r_stand + r_stand_still + r_swing_dir + r_torque
                  + r_climb + r_terrain_up + r_terrain_down
                  + base_penalties
                  + self._penalty_gate * refinement_penalties)
        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)   # 防止 NaN 奖励污染 PPO。

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            # 跌倒高度使用相对局部地面的 base 高度，而不是世界 z。
            # 这样楼梯坑、下坡等低世界坐标地形不会被误判为每帧跌倒。
            if hasattr(self, "_terrain_height_under_base"):
                base_h_local = self.robot.data.root_pos_w[:, 2] - self._terrain_height_under_base()
            else:
                base_h_local = self.robot.data.root_pos_w[:, 2]
            died = base_h_local < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        # 非有限状态保护：物理爆炸会产生 NaN/Inf，普通高度检查捕捉不到。
        # 这里直接终止该 env，避免污染 AMP/PPO 更新。
        nonfinite = ~(torch.isfinite(self.robot.data.root_pos_w).all(dim=1)
                      & torch.isfinite(self.robot.data.root_lin_vel_b).all(dim=1)
                      & torch.isfinite(self.robot.data.root_ang_vel_b).all(dim=1))
        died = died | nonfinite
        return died, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        # 地形课程：reset 重新定位前，按 episode 内前进距离和稳定性决定升/降级。
        if self.cfg.terrain.terrain_type == "generator":
            root_xy = self.robot.data.root_pos_w[env_ids, :2]
            root_z = self.robot.data.root_pos_w[env_ids, 2]
            dist = torch.norm(root_xy - self._terrain.env_origins[env_ids, :2], dim=1)
            cmd_xy = self._episode_start_cmd_xy[env_ids]
            cmd_mag = torch.norm(cmd_xy, dim=1)
            cmd_dir = cmd_xy / cmd_mag.unsqueeze(1).clamp(min=1e-6)
            episode_delta_xy = root_xy - self._episode_start_xy[env_ids]
            forward_dist = torch.sum(episode_delta_xy * cmd_dir, dim=1)
            height_delta = root_z - self._episode_start_root_z[env_ids]
            valid_episode = self.episode_length_buf[env_ids] > 0
            # 地形只在解锁阶段且未触发回退保护时推进；bootstrap 和平地清步态阶段保持低难度。
            terrain_curriculum_active = self._phase >= getattr(self, "_terrain_start_phase", 5)
            terrain_unlocked = terrain_curriculum_active and self._advance_ok
            if hasattr(self, "reset_terminated"):
                terminal_now = self.reset_terminated[env_ids].bool()
            else:
                terminal_now = torch.zeros_like(valid_episode, dtype=torch.bool)
            try:
                base_h_local = root_z - self._terrain_height_under_base()[env_ids]
            except Exception:
                base_h_local = root_z - self._terrain.env_origins[env_ids, 2]
            upright_score = (-self.robot.data.projected_gravity_b[env_ids, 2]).clamp(-1.0, 1.0)
            if hasattr(self, "_in_contact"):
                contact_count = self._in_contact[env_ids].sum(dim=1)
            else:
                contact_count = torch.full_like(root_z, 4.0)
            body_wxy = torch.linalg.norm(self.robot.data.root_ang_vel_b[env_ids, :2], dim=1)
            v_along = torch.sum(self.robot.data.root_lin_vel_b[env_ids, :2] * cmd_dir, dim=1)
            self._ensure_gate_mask()
            eligible_mask = torch.ones_like(valid_episode, dtype=torch.bool)
            if bool(getattr(self.cfg, "terrain_curriculum_ignore_flat", True)):
                eligible_mask = getattr(
                    self,
                    "_real_terrain_mask",
                    torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
                )[env_ids]
            if not hasattr(self, "_terrain_level_peak") or self._terrain_level_peak.shape != self._terrain.terrain_levels.shape:
                self._terrain_level_peak = self._terrain.terrain_levels.clone()
            if not hasattr(self, "_terrain_move_down_streak") or self._terrain_move_down_streak.shape != self._terrain.terrain_levels.shape:
                self._terrain_move_down_streak = torch.zeros_like(self._terrain.terrain_levels)
            if terrain_unlocked:
                peak_next = torch.maximum(self._terrain_level_peak[env_ids], self._terrain.terrain_levels[env_ids])
                self._terrain_level_peak[env_ids] = torch.where(
                    eligible_mask,
                    peak_next,
                    self._terrain_level_peak[env_ids],
                )
            moves = compute_terrain_curriculum_moves(
                dist=dist,
                cmd_mag=cmd_mag,
                forward_dist=forward_dist,
                height_delta=height_delta,
                valid_episode=valid_episode,
                terrain_curriculum_active=bool(terrain_curriculum_active),
                terrain_unlocked=bool(terrain_unlocked),
                terminal_now=terminal_now,
                base_h_local=base_h_local,
                upright_score=upright_score,
                contact_count=contact_count,
                body_wxy=body_wxy,
                v_along=v_along,
                max_episode_length_s=float(self.max_episode_length_s),
                terrain_move_up_dist=float(self.cfg.terrain_move_up_dist),
                height_gain=float(getattr(self.cfg, "terrain_curriculum_height_gain", 0.08)),
                height_loss=float(getattr(self.cfg, "terrain_curriculum_height_loss", 0.08)),
                forward_min=float(getattr(self.cfg, "terrain_curriculum_forward_min", 0.25)),
                stable_h=float(getattr(self.cfg, "terrain_curriculum_stable_h", 0.42)),
                stable_upright=float(getattr(self.cfg, "terrain_curriculum_stable_upright", 0.85)),
                stable_contact_min=float(getattr(self.cfg, "terrain_curriculum_stable_contact_min", 2.0)),
                stable_wxy_max=float(getattr(self.cfg, "terrain_curriculum_stable_wxy_max", 1.50)),
                speed_cap_ratio=float(getattr(self.cfg, "terrain_curriculum_speed_cap_ratio", 1.60)),
                speed_cap_min=float(getattr(self.cfg, "terrain_curriculum_speed_cap_min", 0.75)),
                failure_h=float(getattr(self.cfg, "terrain_curriculum_failure_h", 0.42)),
                failure_upright=float(getattr(self.cfg, "terrain_curriculum_failure_upright", 0.75)),
                failure_wxy=float(getattr(self.cfg, "terrain_curriculum_failure_wxy", 2.00)),
                eligible_mask=eligible_mask,
            )
            controlled_up = moves["controlled_up"]
            controlled_down = moves["controlled_down"]
            move_up = moves["move_up"]
            move_down_low = moves["low_progress_down"]
            failure_down = moves["failure_down"]
            patience = max(1, int(getattr(self.cfg, "terrain_curriculum_move_down_patience", 1)))
            if patience > 1:
                streak = self._terrain_move_down_streak[env_ids]
                streak = torch.where(move_down_low, streak + 1, torch.zeros_like(streak))
                move_down_low = move_down_low & (streak >= patience)
                streak = torch.where(move_up, torch.zeros_like(streak), streak)
                streak = torch.where(failure_down, torch.zeros_like(streak), streak)
                self._terrain_move_down_streak[env_ids] = streak
            floor_after_peak = int(getattr(self.cfg, "terrain_curriculum_floor_after_peak", 0))
            if floor_after_peak > 0 and terrain_curriculum_active:
                level_now = self._terrain.terrain_levels[env_ids]
                peak_drop = max(0, int(getattr(self.cfg, "terrain_curriculum_peak_drop", 0)))
                peak_floor = torch.clamp(self._terrain_level_peak[env_ids] - peak_drop, min=floor_after_peak)
                peaked = self._terrain_level_peak[env_ids] >= floor_after_peak
                at_floor = level_now <= peak_floor
                move_down_low = move_down_low & ~(peaked & at_floor)
            move_down = move_down_low | failure_down
            with torch.no_grad():
                eligible_valid = valid_episode.bool() & eligible_mask.bool()
                n = max(int(eligible_valid.float().sum().item()), 1)
                self._terrain_curriculum_eligible_frac = float(eligible_mask.float().mean())
                self._terrain_curriculum_move_up_rate = float(move_up.float().sum() / n)
                self._terrain_curriculum_move_down_rate = float(move_down.float().sum() / n)
                self._terrain_curriculum_failure_down_rate = float(failure_down.float().sum() / n)
                self._terrain_curriculum_stable_end_rate = float(moves["stable_end"].float().sum() / n)
                self._terrain_curriculum_speed_ok_rate = float(moves["speed_controlled"].float().sum() / n)
                discrete_mask = getattr(
                    self,
                    "_discrete_terrain_mask",
                    torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                )[env_ids] & eligible_valid
                dn = max(int(discrete_mask.float().sum().item()), 1)
                self._terrain_curriculum_discrete_move_up_rate = float((move_up & discrete_mask).float().sum() / dn)
                self._terrain_curriculum_discrete_move_down_rate = float((move_down & discrete_mask).float().sum() / dn)
                self._terrain_curriculum_discrete_failure_down_rate = float((failure_down & discrete_mask).float().sum() / dn)
                self._terrain_curriculum_discrete_stable_end_rate = float((moves["stable_end"] & discrete_mask).float().sum() / dn)
            self._terrain.update_env_origins(env_ids, move_up, move_down)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        # reset 后清空历史，避免 episode 间残留。
        self._obs_history[env_ids] = 0.0
        self._delayed_action[env_ids] = 0.0
        self._in_contact[env_ids] = 0.0
        self._cmd_transition_zero_frac[env_ids] = 0.5
        self._cmd_transition_strength[env_ids] = 0.0
        # 域随机化：按等级选择扰动参数。高等级额外扰动摩擦、base CoM 和 IMU 偏置。
        n = len(env_ids)
        lvl = self._dr_level
        self._imu_bias[env_ids] = 0.0     # 先清零；需要时在下方按等级重采样。
        if self.cfg.dr_enable and lvl >= 1:
            mass_range = getattr(self.cfg, f"dr_mass_range_{lvl}")
            k_range    = getattr(self.cfg, f"dr_stiffness_scale_{lvl}")
            d_range    = getattr(self.cfg, f"dr_damping_scale_{lvl}")
            try:
                # PhysX 质量属性在 CPU 管线；这里全程 CPU，并先恢复默认质量再加扰动，避免跨 reset 漂移。
                masses = self.robot.root_physx_view.get_masses()                 # (N, n_bodies)，CPU tensor。
                if self._default_masses is None:
                    self._default_masses = masses[:, 0].clone()                  # 默认 root 质量，只缓存一次。
                eids = env_ids.detach().cpu()
                delta = torch.empty(n).uniform_(*mass_range)                     # CPU，与质量 tensor 同设备。
                masses[eids, 0] = self._default_masses[eids] + delta             # 恢复默认值后再加扰动。
                self.robot.root_physx_view.set_masses(masses, eids)
            except Exception as e:
                if "mass" not in self._dr_warned:
                    print(f"[DR] mass DR FAILED: {type(e).__name__}: {e}", flush=True); self._dr_warned.add("mass")
            try:
                k_scale = torch.zeros(n, 1, device=self.device).uniform_(*k_range)
                d_scale = torch.zeros(n, 1, device=self.device).uniform_(*d_range)
                base_k = self._actuator_stiffness_base.view(1, -1)
                base_d = self._actuator_damping_base.view(1, -1)
                self.robot.actuators["legs"].stiffness[env_ids] = base_k * k_scale
                self.robot.actuators["legs"].damping[env_ids] = base_d * d_scale
            except Exception:
                if "gain" not in self._dr_warned:
                    print("[DR] gain API unavailable; skipping stiffness/damping DR", flush=True); self._dr_warned.add("gain")
            # 摩擦、CoM 和 IMU 是部署关键 DR 通道；由 dr_full_start_level 控制何时打开，
            # 并随等级逐步扩大扰动范围。
            _dr_full_lvl = int(getattr(self.cfg, "dr_full_start_level", 1))
            if lvl >= _dr_full_lvl:
                # 摩擦和 CoM 写入较贵，因此每个等级只在升级时批量写一次。
                if lvl > self._dr_mat_level:
                    Nenv = self.num_envs
                    fr_range = getattr(self.cfg, f"dr_friction_range_{lvl}", self.cfg.dr_friction_range_3)
                    com_off  = float(getattr(self.cfg, f"dr_com_offset_{lvl}", self.cfg.dr_com_offset_3))
                    try:
                        mats = self.robot.root_physx_view.get_material_properties().clone()   # (N, n_shapes, 3)
                        fr = torch.zeros(Nenv, 1, 1, device=mats.device).uniform_(*fr_range)  # 绝对写入，避免漂移。
                        mats[:, :, 0:2] = fr.expand(-1, mats.shape[1], 2)
                        self.robot.root_physx_view.set_material_properties(
                            mats, torch.arange(Nenv, device=mats.device))
                    except Exception:
                        if "fric" not in self._dr_warned:
                            print("[DR] friction API unavailable; skipping friction DR", flush=True); self._dr_warned.add("fric")
                    try:
                        coms = self.robot.root_physx_view.get_coms().clone()                 # (N, n_bodies, ...)
                        if self._default_coms is None:
                            self._default_coms = coms.clone()                                # 默认 CoM 只缓存一次。
                        coms = self._default_coms.clone()                                    # 每次写入前恢复默认，避免漂移。
                        off = torch.zeros(Nenv, 2, device=coms.device).uniform_(-com_off, com_off)
                        coms[:, 0, 0:2] += off
                        self.robot.root_physx_view.set_coms(coms, torch.arange(Nenv, device=coms.device))
                    except Exception:
                        if "com" not in self._dr_warned:
                            print("[DR] CoM API unavailable; skipping CoM DR", flush=True); self._dr_warned.add("com")
                    self._dr_mat_level = lvl
                    print(f"[DR] level-{lvl} friction {list(fr_range)} + CoM ±{com_off} randomized (all envs)", flush=True)
                # IMU 偏置按 episode 重采样，属于观测层扰动。
                gb = float(getattr(self.cfg, f"dr_imu_gyro_bias_{lvl}", self.cfg.dr_imu_gyro_bias_3))
                vb = float(getattr(self.cfg, f"dr_imu_grav_bias_{lvl}", self.cfg.dr_imu_grav_bias_3))
                self._imu_bias[env_ids, 0:3] = torch.zeros(n, 3, device=self.device).uniform_(-gb, gb)
                self._imu_bias[env_ids, 3:6] = torch.zeros(n, 3, device=self.device).uniform_(-vb, vb)
        num = len(env_ids)
        # reset 顺序：先采样命令，再按命令和地形选择 clip，最后从该 clip 采样初始姿态。
        self._resample_commands(env_ids, snap=True)
        tctx_arg = self._terrain_ctx[env_ids] if self.cfg.terrain_ctx_dim > 0 else None
        clip_ids = self._motion_loader.pick_clips(self.commands[env_ids], tctx_arg)
        start = "start" in self.cfg.reset_strategy
        times = np.zeros(num) if start else self._motion_loader.sample_times(num)
        dp, dv, bp, br, blv, bav = self._motion_loader.sample_frames(clip_ids, times)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, 0:3] = bp[:, self.motion_ref_body_index] + self._terrain.env_origins[env_ids]
        root[:, 2] += 0.03
        root[:, 3:7] = br[:, self.motion_ref_body_index]
        root[:, 7:10] = blv[:, self.motion_ref_body_index]
        root[:, 10:13] = bav[:, self.motion_ref_body_index]
        jp = dp[:, self.motion_dof_indexes]
        jv = dv[:, self.motion_dof_indexes]
        self.robot.write_root_link_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(jp, jv, None, env_ids)
        self.last_actions[env_ids] = 0.0
        self._gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device)   # 打散 env 间步态相位。
        self._episode_start_xy[env_ids] = root[:, 0:2].detach()
        self._episode_start_root_z[env_ids] = root[:, 2].detach()
        self._episode_start_cmd_xy[env_ids] = self.commands[env_ids, :2].detach()
        self._cmd_heading_ref[env_ids] = _yaw_from_quat_w(root[:, 3:7]).detach()
        self._heading_error[env_ids] = 0.0

    def _push_strided_amp(self, amp):
        """策略侧 AMP stride 窗口。

        先把最新原始帧写入深环形缓冲，再按 stride 导出 [t, t-stride, ...]，
        让判别器看到约一个步态周期。仅在 _amp_frame_stride > 1 时调用。
        """
        r = self._amp_raw_ring
        r[:, 1:] = r[:, :-1].clone()      # FIFO 移位；clone 避免原地别名问题。
        r[:, 0] = amp
        stride = self._amp_frame_stride
        for j in range(self.cfg.num_amp_observations):
            self.amp_observation_buffer[:, j] = r[:, j * stride]

    def collect_reference_motions(self, num_samples, current_times=None):
        # 解析参考：根据当前命令连续生成小跑参考，步频和步幅随命令缩放。
        # 参考采样使用实时 rollout 命令和 terrain_ctx，使判别器条件与策略侧一致。
        K = self.cfg.num_amp_observations
        env_sel = torch.randint(0, self.num_envs, (num_samples,), device=self.device)
        cmd_s = self.commands[env_sel]                                             # (num_samples, 3)
        if current_times is None:
            current_times = np.random.uniform(0.0, 2.0, num_samples).astype(np.float32)
        _stride = getattr(self, "_amp_frame_stride", 1)                                       # AMP stride 间隔。
        times = (np.expand_dims(current_times, -1) - (1.0 / 50.0) * _stride * np.arange(K)).flatten()  # 按 stride 取参考帧。
        times_t = torch.as_tensor(times, dtype=torch.float32, device=self.device)
        cmd_rep = cmd_s.repeat_interleave(K, dim=0)                                # (num_samples*K, 3)
        # 地形感知参考：传入粗糙度，使粗糙地形参考轨迹抬脚更高。
        rough_rep = None
        if self.cfg.terrain_ctx_dim >= 3:
            rough_rep = self._terrain_ctx[env_sel, 2].repeat_interleave(K, dim=0)  # (num_samples*K,)
        jp_clip, jv_clip, bh, tn, foot_rel = flat_reference(
            cmd_rep, times_t, gait_period=self.cfg.gait_period,
            gait_period_slope=self.cfg.gait_period_slope, gait_period_min=self.cfg.gait_period_min,
            clearance_base=self.cfg.base_clearance,        # 平地保持较低摆腿，粗糙度项仍会提高地形抬脚。
            roughness=rough_rep, clearance_rough_gain=self.cfg.ref_clearance_rough_gain,
            stance_dx=self.cfg.stance_dx)                  # wider fore-aft stance (blind config 0.05); base 0.0
        jp = jp_clip[:, self.motion_dof_indexes]                                   # clip dof order -> robot order
        jv = jv_clip[:, self.motion_dof_indexes]
        rel_b = foot_rel.reshape(-1, 12)                                           # legs FL,FR,RL,RR x xyz
        # AMP = STYLE ONLY (MUST mirror _compute_amp_obs item-for-item): pure kinematic style + terrain_ctx.
        # Velocity and command are removed — the discriminator no longer judges speed/command (owned by the
        # two-sided tracking reward). terrain_ctx kept so the reference style is terrain-conditioned.
        parts = [jp, jv, bh, tn, rel_b]
        if self.cfg.terrain_ctx_dim > 0:
            tctx_rep = self._terrain_ctx[env_sel].repeat_interleave(K, dim=0)
            parts.append(tctx_rep)
        amp = torch.cat(parts, dim=-1)
        return amp.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_t = torch.zeros_like(q[..., :3])
    ref_n = torch.zeros_like(q[..., :3])
    ref_t[..., 0] = 1
    ref_n[..., -1] = 1
    return torch.cat([quat_apply(q, ref_t), quat_apply(q, ref_n)], dim=-1)
