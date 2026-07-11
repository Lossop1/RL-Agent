"""盲狗 TerrainPerceiver 训练环境。

该类继承 TailiAmpEnv，主要替换观测组装为 Runtime-IO 方案：
tick54 历史 + body53 + critic 特权信息 + 辅助标签。
仿真、传感器、地形、动作延迟、扰动、AMP 缓冲、奖励和 reset 逻辑复用父类。

远端 payload 以 taili_blind_runtime 包形式部署，只需要该包出现在 PYTHONPATH 中。
"""
import math
import os
from dataclasses import asdict, is_dataclass
import types

import numpy as np
import torch

try:                                            # payload 包内导入。
    from .taili_core import (taili_obs, taili_symmetry, taili_amp_reference,
                             taili_reward, taili_terrain_labels)
except ImportError:
    if __package__ == "taili_blind_runtime":
        raise
    try:                                        # 本地源码树导入。
        from autotuner.taili_core import (taili_obs, taili_symmetry, taili_amp_reference,
                                          taili_reward, taili_terrain_labels)
    except ImportError:
        from taili_core import (taili_obs, taili_symmetry, taili_amp_reference,
                                taili_reward, taili_terrain_labels)

try:
    from .telemetry_emit import TrainingTelemetryEmitter
except Exception:  # pragma: no cover - remote deployment may copy files differently
    try:
        from telemetry_emit import TrainingTelemetryEmitter
    except Exception:
        try:
            if __package__ == "taili_blind_runtime":
                raise
            from autotuner.blind_locomotion.telemetry_emit import TrainingTelemetryEmitter
        except Exception:
            TrainingTelemetryEmitter = None

try:
    from .taili_blind_config import active_direction_progress, phase_command_spec
except Exception:  # pragma: no cover - local fallback for unusual import layouts
    try:
        from autotuner.blind_locomotion.taili_blind_config import active_direction_progress, phase_command_spec
    except Exception:
        active_direction_progress = None
        phase_command_spec = None

from .taili_amp_env import TailiAmpEnv, quaternion_to_tangent_and_normal, _command_xy_world_from_root_yaw
from .parametric_ref import flat_reference   # live imitation 使用的解析参考关节轨迹。

HIST_LEN = 25
TICK_DIM = 54


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
        # 感知器历史：最近 25 帧 tick54 的 FIFO。
        self._tick_history = torch.zeros((self.num_envs, HIST_LEN, TICK_DIM), device=self.device)
        self._rcfg = taili_reward.reward_cfg_from_env()          # 奖励配置；可用 TAILI_RW_* 环境变量临时覆盖。
        self._reward_cfg_printed = False
        self._prev_in_contact = None                            # 用于检测摆动到支撑的触地事件。
        self._td_impact = _env_flag("TAILI_TD_IMPACT", bool(getattr(self.cfg, "touchdown_impact_only", False)))
        self._telemetry = TrainingTelemetryEmitter() if TrainingTelemetryEmitter is not None else None
        self._hip_joint_ids, _ = self.robot.find_joints(".*_hip_joint")
        self._stagger_pending = True     # 初始批量 reset 后做一次 episode 相位打散。
        self._cmd_hold = None            # 每个 env 独立的命令保持步数，由 _get_observations 采样。
        self._quality_duty_ema = torch.full((self.num_envs, 4), 0.5, device=self.device)
        self._quality_diag_pair_ema = torch.full((self.num_envs,), 0.5, device=self.device)
        self._quality_slip_speed_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_slip_excess_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_slip_high_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_height_low_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_tilt_high_ema = torch.zeros(self.num_envs, device=self.device)
        self._quality_window_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._terrain_up_latch = torch.zeros(self.num_envs, device=self.device)
        self._terrain_down_latch = torch.zeros(self.num_envs, device=self.device)
        self._terrain_contact_latch = torch.zeros(self.num_envs, device=self.device)

    def _resample_commands(self, env_ids, *args, **kwargs):
        # 课程命令覆盖：用于强制早期学习命令条件，而不是学成与命令无关的爬行/站立。
        # TAILI_CUR_FIXED_VX 会把所有命令固定为 (vx, 0, 0)；未设置时使用正常采样。
        super()._resample_commands(env_ids, *args, **kwargs)
        if len(env_ids) == 0:
            return
        snap = bool(kwargs.get("snap", args[0] if args else False))
        phase = int(getattr(self, "_phase", getattr(self.cfg, "init_phase", 0)))
        spec = phase_command_spec(self.cfg, phase) if phase_command_spec is not None else {}
        mode = str(spec.get("command_mode") or getattr(self.cfg, "training_command_mode", "normal") or "normal")
        self._last_command_mode = mode
        self._last_command_spec = spec
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
                active = torch.rand(n, device=dev) >= stand_prob
                weights = torch.tensor([
                    max(0.0, _spec_float(spec, "prob_fwd", float(getattr(self.cfg, "cmd_prob_fwd", 0.25)))),
                    max(0.0, _spec_float(spec, "prob_back", float(getattr(self.cfg, "cmd_prob_back", 0.25)))),
                    max(0.0, _spec_float(spec, "prob_lat", float(getattr(self.cfg, "cmd_prob_lat", 0.25)))),
                    max(0.0, _spec_float(spec, "prob_yaw", float(getattr(self.cfg, "cmd_prob_yaw", 0.25)))),
                ], device=dev)
                if float(weights.sum()) <= 1e-9:
                    weights[:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
                u = torch.rand(n, device=dev) * weights.sum()
                c0 = weights[0]
                c1 = c0 + weights[1]
                c2 = c1 + weights[2]
                fwd = active & (u < c0)
                back = active & (u >= c0) & (u < c1)
                lat = active & (u >= c1) & (u < c2)
                yaw = active & (u >= c2)
                f_lo, f_hi = _spec_range(spec, "fwd_range", getattr(self.cfg, "cmd_fwd_range", (0.3, 0.7)))
                b_lo, b_hi = _spec_range(spec, "back_range", getattr(self.cfg, "cmd_back_range", (0.2, 0.45)))
                l_lo, l_hi = _spec_range(spec, "lat_range", getattr(self.cfg, "cmd_lat_range", (0.15, 0.35)))
                y_lo, y_hi = _spec_range(spec, "yaw_range", getattr(self.cfg, "cmd_yaw_range", (0.25, 0.8)))
                f_hi = min(f_hi, float(getattr(self, "_vel_max_fwd", f_hi)))
                b_hi = min(b_hi, float(getattr(self, "_vel_max_back", b_hi)))
                l_hi = min(l_hi, float(getattr(self, "_vel_max_lat", l_hi)))
                y_hi = min(y_hi, float(getattr(self, "_vel_max_yaw", y_hi)))
                target[:, 0] = torch.where(fwd, _sample_uniform(n, f_lo, f_hi, dev), target[:, 0])
                target[:, 0] = torch.where(back, -_sample_uniform(n, b_lo, b_hi, dev), target[:, 0])
                lat_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                yaw_sign = torch.where(torch.rand(n, device=dev) < 0.5, torch.ones(n, device=dev), -torch.ones(n, device=dev))
                target[:, 1] = torch.where(lat, lat_sign * _sample_uniform(n, l_lo, l_hi, dev), target[:, 1])
                target[:, 2] = torch.where(yaw, yaw_sign * _sample_uniform(n, y_lo, y_hi, dev), target[:, 2])
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
                target[:, 0] = torch.where(x_is_fwd, _sample_uniform(n, f_lo, f_hi, dev), -_sample_uniform(n, b_lo, b_hi, dev))
                target[:, 0] = torch.where(x_on, target[:, 0], torch.zeros(n, device=dev))
                target[:, 1] = torch.where(lat_on, lat_sign * _sample_uniform(n, l_lo, l_hi, dev), torch.zeros(n, device=dev))
                target[:, 2] = torch.where(yaw_on, yaw_sign * _sample_uniform(n, y_lo, y_hi, dev), torch.zeros(n, device=dev))
                target = torch.where(moving[:, None], target, torch.zeros_like(target))
                nz_scale = _spec_float(spec, "near_zero_scale", 0.05)
                near_noise = (torch.rand((n, 3), device=dev) * 2.0 - 1.0) * nz_scale
                target = torch.where(near_zero[:, None], near_noise, target)
            if target is not None:
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
        self._tick_history = torch.cat([tick.unsqueeze(1), self._tick_history[:, :-1]], dim=1)   # FIFO

        # 足端接触缓存，奖励计算会复用。
        forces_now = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, :].norm(dim=-1)
        contact_threshold = float(getattr(self.cfg, "contact_force_threshold", 10.0))
        self._in_contact = (forces_now > contact_threshold).float()

        # body53 = angv | grav | cmd | jpos_rel | jvel | last_act | gait。
        body = torch.cat([angv_n, grav_n, self.commands, jpos_n - self.action_offset,
                          jvel_n, self.last_actions, gait_obs], dim=-1)                            # (N,53)
        blind_obs = torch.cat([body, self._tick_history.reshape(N, -1)], dim=-1)                   # (N,1403)

        # 特权观测：只供 critic/AMP 使用。
        hits = self._height_scanner.data.ray_hits_w
        hscan = (self.robot.data.root_pos_w[:, 2:3] - hits[:, :, 2] - self.cfg.stand_height).clamp(-1.0, 1.0)
        hscan = torch.nan_to_num(hscan, nan=0.0, posinf=0.0, neginf=0.0)
        self._terrain_ctx = self._compute_terrain_ctx()
        priv = torch.cat([self.robot.data.root_lin_vel_b, hscan, self._terrain_ctx, self._in_contact], dim=-1)
        labels = self._compute_aux_labels()                                                       # (N,22) geom9+mask9+risk2+mask2。
        obs = torch.cat([blind_obs, priv, labels], dim=-1)                                         # (N,1622)

        # AMP 缓冲：当前复用父类 _compute_amp_obs。
        amp = torch.nan_to_num(self._compute_amp_obs(), nan=0.0, posinf=0.0, neginf=0.0)
        if getattr(self, "_amp_frame_stride", 1) > 1:
            self._push_strided_amp(amp)                 # AMP stride 窗口，导出约一个步态周期。
        else:
            for i in reversed(range(self.cfg.num_amp_observations - 1)):
                self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
            self.amp_observation_buffer[:, 0] = amp
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}
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
        """返回 (N,22) = geom_label9 | geom_mask9 | risk_label2 | risk_mask2。

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
        geom_mask = torch.ones(N, 9, device=dev)

        forces = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, :].norm(dim=-1)  # (N,4)
        f_max = forces.max(dim=1).values
        impact_n = taili_terrain_labels.impact_score_norm(f_max, torch.tensor(self.cfg.robot_mass_kg, device=dev)
                                                          if hasattr(self.cfg, "robot_mass_kg") else torch.tensor(39.0, device=dev))
        cc = self._in_contact.sum(dim=1)
        tilt = torch.arccos(torch.clamp(-rd.projected_gravity_b[:, 2], -1.0, 1.0))
        support = torch.clamp((3.0 - cc) / 3.0, 0.0, 1.0)
        support = torch.maximum(support, torch.clamp((tilt - math.radians(10)) / math.radians(20), 0.0, 1.0))
        risk = torch.stack([impact_n, support], dim=-1)                              # (N,2)
        risk_mask = torch.ones(N, 2, device=dev)
        return torch.cat([geom, geom_mask, risk, risk_mask], dim=-1)                 # (N,22)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._tick_history[env_ids] = 0.0
        if hasattr(self, "_prev_support_z") and len(env_ids) > 0:
            try:
                self._prev_support_z[env_ids] = self._terrain_height_under_base()[env_ids].detach()
            except Exception:
                self._prev_support_z[env_ids] = self._terrain.env_origins[env_ids, 2].detach()
        if hasattr(self, "_terrain_up_latch") and len(env_ids) > 0:
            self._terrain_up_latch[env_ids] = 0.0
            self._terrain_down_latch[env_ids] = 0.0
            self._terrain_contact_latch[env_ids] = 0.0
        # reset 后为这些 env 重新采样命令保持时长，维持队列不同步。
        if getattr(self, "_cmd_hold", None) is not None and len(env_ids) > 0:
            self._cmd_hold[env_ids] = self._draw_cmd_hold(len(env_ids))
        # RSI 会把脚放在接触附近；reset 后先标记为已接触，避免第一步误算成新触地。
        if self._prev_in_contact is not None:
            self._prev_in_contact[env_ids] = 1.0
        if hasattr(self, "_quality_window_ready"):
            self._quality_duty_ema[env_ids] = 0.5
            self._quality_diag_pair_ema[env_ids] = 0.5
            self._quality_slip_speed_ema[env_ids] = 0.0
            self._quality_slip_excess_ema[env_ids] = 0.0
            self._quality_slip_high_ema[env_ids] = 0.0
            self._quality_height_low_ema[env_ids] = 0.0
            self._quality_tilt_high_ema[env_ids] = 0.0
            self._quality_window_ready[env_ids] = False

    def _update_quality_windows(self, in_contact, settled_contact, foot_vel_xy, base_h, tilt_rel, moving):
        """更新奖励、阶段门控和遥测共用的步态质量窗口。

        单帧代理过于宽松：四脚爬行可能看起来像对角步态，高 duty 爬行可能看起来平衡，
        单只支撑脚滑移也会被四足均值稀释。EMA 窗口更接近诊断对整段行为的评价。
        """
        f = in_contact.dtype
        if not hasattr(self, "_quality_window_ready"):
            self._quality_duty_ema = torch.full_like(in_contact, 0.5)
            self._quality_diag_pair_ema = torch.full_like(base_h, 0.5)
            self._quality_slip_speed_ema = torch.zeros_like(base_h)
            self._quality_slip_excess_ema = torch.zeros_like(base_h)
            self._quality_slip_high_ema = torch.zeros_like(base_h)
            self._quality_height_low_ema = torch.zeros_like(base_h)
            self._quality_tilt_high_ema = torch.zeros_like(base_h)
            self._quality_window_ready = torch.zeros(base_h.shape[0], dtype=torch.bool, device=base_h.device)

        beta = min(max(float(getattr(self._rcfg, "quality_window_ema_beta", 0.98)), 0.0), 0.999)
        stance = (in_contact > 0.5).to(f)
        settled = (settled_contact > 0.5).to(f)
        moving_mask = moving > 0.5
        moving_col = moving_mask[:, None]

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

        duty_target = min(max(float(getattr(self._rcfg, "duty_target", getattr(self.cfg, "gait_duty", 0.5))), 0.05), 0.95)
        duty_sample = torch.where(moving_col, stance, torch.full_like(stance, duty_target))
        # 对角小跑指标只在适用场景更新：前进、后退或横移命令明显强于 yaw。
        # yaw 主导转向时低 diag_pair 是合理的；不应把它混进直线小跑质量 EMA。
        # 非适用帧保持 EMA 原值，避免对角步态奖励和转向步态互相冲突。
        _cmd = self.commands
        _lin_mag = torch.linalg.norm(_cmd[:, :2], dim=-1)
        _trot_applicable = moving_mask & (_lin_mag > 0.10) & (_lin_mag >= 0.6 * _cmd[:, 2].abs())
        diag_sample = torch.where(_trot_applicable, diag_pair, self._quality_diag_pair_ema)

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
        slip_speed_sample = torch.where(moving_mask, slip_speed_mean, torch.zeros_like(slip_speed_mean))
        slip_excess_sample = torch.where(moving_mask, slip_excess_mean, torch.zeros_like(slip_excess_mean))
        slip_high_sample = torch.where(moving_mask, slip_high_fraction, torch.zeros_like(slip_high_fraction))

        h_ok = float(getattr(self._rcfg, "h_ok", 0.47))
        h_close = float(getattr(self._rcfg, "h_gate_close", 0.42))
        height_low = torch.clamp((h_ok - base_h) / max(h_ok - h_close, 1e-6), 0.0, 1.0)
        tilt_ok = float(getattr(self._rcfg, "tilt_ok_rad", math.radians(15)))
        tilt_close = float(getattr(self._rcfg, "tilt_gate_close_rad", math.radians(40)))
        tilt_high = torch.clamp((tilt_rel - tilt_ok) / max(tilt_close - tilt_ok, 1e-6), 0.0, 1.0)

        cold = ~self._quality_window_ready
        if bool(cold.any()):
            self._quality_duty_ema[cold] = duty_sample[cold]
            self._quality_diag_pair_ema[cold] = diag_sample[cold]
            self._quality_slip_speed_ema[cold] = slip_speed_sample[cold]
            self._quality_slip_excess_ema[cold] = slip_excess_sample[cold]
            self._quality_slip_high_ema[cold] = slip_high_sample[cold]
            self._quality_height_low_ema[cold] = height_low[cold]
            self._quality_tilt_high_ema[cold] = tilt_high[cold]
            self._quality_window_ready[cold] = True

        self._quality_duty_ema.mul_(beta).add_(duty_sample * (1.0 - beta))
        self._quality_diag_pair_ema.mul_(beta).add_(diag_sample * (1.0 - beta))
        self._quality_slip_speed_ema.mul_(beta).add_(slip_speed_sample * (1.0 - beta))
        self._quality_slip_excess_ema.mul_(beta).add_(slip_excess_sample * (1.0 - beta))
        self._quality_slip_high_ema.mul_(beta).add_(slip_high_sample * (1.0 - beta))
        self._quality_height_low_ema.mul_(beta).add_(height_low * (1.0 - beta))
        self._quality_tilt_high_ema.mul_(beta).add_(tilt_high * (1.0 - beta))

        duty_w = self._quality_duty_ema
        front = 0.5 * (duty_w[:, 0] + duty_w[:, 1])
        rear = 0.5 * (duty_w[:, 2] + duty_w[:, 3])
        # 左右对称按前/后腿对分别计算，避免前腿和后腿相反偏置互相抵消。
        # 前后支撑平衡作为独立部分保留。
        lr_pair_skew = (duty_w[:, 0] - duty_w[:, 1]).abs() + (duty_w[:, 2] - duty_w[:, 3]).abs()
        duty_symmetry_score = 1.0 - torch.clamp(lr_pair_skew + (front - rear).abs(), 0.0, 1.0)
        duty_target_error = (duty_w - duty_target).abs().mean(dim=-1)
        duty_target_score = 1.0 - torch.clamp(duty_target_error / max(duty_target, 1.0 - duty_target, 1e-6), 0.0, 1.0)
        duty_quality = torch.clamp(duty_symmetry_score * duty_target_score, 0.0, 1.0)

        return {
            "duty_by_leg": duty_w,
            "duty_mean": duty_w.mean(dim=-1),
            "duty_spread": duty_w.max(dim=-1).values - duty_w.min(dim=-1).values,
            "duty_symmetry_score": duty_symmetry_score,
            "duty_target_score": duty_target_score,
            "duty_target_error": duty_target_error,
            "duty_quality": duty_quality,
            "diag_pair_score": self._quality_diag_pair_ema,
            "slip_speed": self._quality_slip_speed_ema,
            "slip_excess": self._quality_slip_excess_ema,
            "slip_high_fraction": self._quality_slip_high_ema,
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
        mode = taili_amp_reference.mode_onehot(self.commands)
        return torch.cat([jp, jv, bh, tn, rel_b, self.commands, mode], dim=-1)   # 43 + 3 + 5 = 51

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
        rd = self.robot.data
        N, dev = self.num_envs, self.device
        cfg = self._rcfg

        terrain_h = self._terrain_height_under_base()                                 # (N,)
        base_h = rd.root_pos_w[:, 2] - terrain_h                                       # 相对局部地面的 base 高度。
        grav = rd.projected_gravity_b
        tilt_rel = torch.arccos(torch.clamp(-grav[:, 2], -1.0, 1.0))                   # 世界系倾斜，后续可扩展为坡面相对。
        roll = torch.arcsin(torch.clamp(grav[:, 1], -1.0, 1.0))
        pitch = torch.arcsin(torch.clamp(-grav[:, 0], -1.0, 1.0))
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
        if self._td_impact:                                                           # 只在真实触地事件上计算落脚冲击。
            td_mask = ((in_contact > 0.5) & (prev_contact < 0.5)).float()             # 摆动到支撑 = landing。
        # 稳定支撑 = 当前帧接触且上一帧也接触。触地帧由落脚冲击负责，
        # 不计入支撑滑移，避免把同一个事件重复算进 B1 和 B2。
        settled_contact = in_contact * (prev_contact > 0.5).float()
        foot_pos = rd.body_pos_w[:, self.foot_indexes, :]                             # (N,4,3)
        foot_vel = rd.body_lin_vel_w[:, self.foot_indexes, :]                         # (N,4,3)
        # 所有滑移相关项统一使用接触点平面速度。
        # foot link 质心速度包含脚球滚动，不等同于接触点滑移；接触点速度更接近诊断语义。
        # touchdown_vz 仍使用 foot link z 速度，以匹配落脚冲击指标。
        _foot_ang = rd.body_ang_vel_w[:, self.foot_indexes, :]                        # (N,4,3)
        _r_cp = torch.tensor([0.0, 0.0, -0.014], device=dev).view(1, 1, 3)
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
            foot_clearance_terr = torch.clamp(foot_pos[:, :, 2] - _ground_z, min=0.0)  # (N,4) above local ground
            _contact_w = in_contact.clamp(0.0, 1.0)
            _contact_sum = _contact_w.sum(dim=1)
            _support_z_contact = (_ground_z * _contact_w).sum(dim=1) / _contact_sum.clamp(min=1.0)
            _support_z_feet = _ground_z.median(dim=1).values
            _support_z = torch.where(_contact_sum > 0.5, _support_z_contact, _support_z_feet)
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
                # 自适应 clearance：用前方升高量加余量作为目标，上限 0.40m。
                _adaptive = (_rise + 0.06).clamp(0.10, 0.40)                          # 台阶高度 + 6cm 余量。
                local_obstacle_h = torch.where(_ascending, torch.maximum(local_obstacle_h, _adaptive), local_obstacle_h)
        except Exception:
            foot_clearance_terr = torch.clamp(foot_pos[:, :, 2] - terrain_h[:, None], min=0.0)
            local_obstacle_h = torch.zeros(N, device=dev)
            _support_z = terrain_h
            _ground_z = terrain_h[:, None].expand(-1, 4)
        quality = self._update_quality_windows(in_contact, settled_contact, foot_vel_xy, base_h, tilt_rel, moving)
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
        torque_limit = self.robot.actuators["legs"].effort_limit

        inp = types.SimpleNamespace(
            cmd=self.commands,
            base_lin_vel=rd.root_lin_vel_b,
            base_ang_vel=rd.root_ang_vel_b,
            base_h_above_terrain=base_h,
            tilt_rel=tilt_rel,
            stable_motion_gate=gate,
            stand_gate=stand_gate,
            moving_gate=moving,
            terminal_window=terminal_window,
            foot_contact=in_contact,
            stance_slip_contact=settled_contact,
            desired_foot_contact=(self._leg_phases() < self.cfg.gait_duty).float(),
            foot_vel_xy=foot_vel_xy,                                                  # (N,4)
            duty_by_leg_window=quality["duty_by_leg"],
            duty_quality_window=quality["duty_quality"],
            duty_symmetry_window=quality["duty_symmetry_score"],
            duty_target_window=quality["duty_target_score"],
            duty_spread_window=quality["duty_spread"],
            diagonal_pair_window=quality["diag_pair_score"],
            stance_slip_speed_window=quality["slip_speed"],
            stance_slip_high_fraction=quality["slip_high_fraction"],                  # 高滑移脚比例，用于补充滑移尾部风险。
            touchdown_vz=foot_vel[:, :, 2].abs(),                                     # (N,4)，落脚速度代理。
            touchdown_mask=td_mask,                                                   # (N,4)，触地帧为 1；未启用时为 None。
            # 使用 last_air_time 而不是 current_air_time：触地帧 current_air_time 会清零，
            # last_air_time 才保留刚完成摆动相的真实滞空时长。
            foot_air_time=self._contact_sensor.data.last_air_time[:, self._feet_contact_ids],
            # 每个 env 的滞空目标来自 gait clock 的计划摆动时长 period*(1-duty)，
            # 与当前速度自适应周期保持一致，避免反碎步奖励和时钟互相冲突。
            air_time_target=(torch.clamp(
                # yaw 感知 cadence：有效速度包含 yaw 足端杠杆项，
                # 高 yaw 命令会缩短周期，要求更快步频。
                self.cfg.gait_period - self.cfg.gait_period_slope * (
                    torch.norm(self.commands[:, :2], dim=1) + 0.15 * self.commands[:, 2].abs()),
                min=self.cfg.gait_period_min, max=self.cfg.gait_period) * (1.0 - self.cfg.gait_duty)),
            foot_clearance=foot_clearance_terr,                                       # 每只脚相对局部地面的高度。
            local_obstacle_h=local_obstacle_h,                                        # 由粗糙度/离散地形派生；平地约为 0。
            hip_deviation=(rd.joint_pos[:, self._hip_joint_ids]
                           - self.action_offset[:, self._hip_joint_ids]).abs().mean(dim=-1),
            torque=rd.applied_torque,
            torque_limit=torque_limit,
            torque_clamped=(rd.applied_torque.abs() >= torque_limit).float(),
            action=self.actions,
            last_action=self.last_actions,
            default_pose_error=(rd.joint_pos - self.action_offset).abs().mean(dim=1),
            amp_reward=torch.zeros(N, device=dev),                                    # skrl AMP 侧另行加入风格奖励。
            body_collision=torch.zeros(N, device=dev),                               # 后续可接入非足端碰撞采样。
            quality_gate=torch.full((N,), float(getattr(self, "_penalty_gate", 0.0)), device=dev),
            heading_error=getattr(self, "_heading_error", torch.zeros(N, device=dev)),
            terminal_reason=None,                                                     # 每个 env 的 terminal 惩罚在下方处理。
        )
        comp = taili_reward.compute_reward_components(inp, cfg)
        total = comp["total"] - cfg.w_terminal * terminal_window                      # 每个 env 的 terminal 惩罚。
        _w_lateral_foot = float(getattr(self.cfg, "w_lateral_foot_excursion", 0.0))
        if _w_lateral_foot != 0.0:
            from isaaclab.utils.math import quat_apply_inverse
            _rel_foot_w = foot_pos - rd.root_pos_w[:, None, :]
            _qfb = rd.root_quat_w[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
            _foot_b = quat_apply_inverse(_qfb, _rel_foot_w.reshape(-1, 3)).reshape(N, 4, 3)
            _nom_y = 0.2082 * torch.tensor([1.0, -1.0, 1.0, -1.0], device=dev)
            _margin_y = float(getattr(self.cfg, "lateral_foot_margin", 0.035))
            _scale_y = max(float(getattr(self.cfg, "lateral_foot_scale", 0.12)), 1e-6)
            _lat_excess = torch.clamp((_foot_b[:, :, 1] - _nom_y[None, :]).abs() - _margin_y, min=0.0) / _scale_y
            _no_lat_yaw_cmd = ((self.commands[:, 1].abs() <= 0.05) & (self.commands[:, 2].abs() <= 0.05)).float()
            comp["lateral_foot_excursion"] = -_w_lateral_foot * (_lat_excess ** 2).mean(dim=1) * _no_lat_yaw_cmd * gate
        else:
            comp["lateral_foot_excursion"] = torch.zeros(N, device=dev)
        total = total + comp["lateral_foot_excursion"]
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
            comp["lateral_underspeed"] = _lat_aux_w * _lat_deficit * _lat_deficit * _lat_cmd.float()
        else:
            comp["lateral_underspeed"] = torch.zeros(N, device=dev)
        total = total + comp["lateral_underspeed"]
        # 绊脚惩罚：足端水平接触力显著大于竖直力，通常表示脚踹在台阶沿/竖直面上。
        # 只在真实接触且水平力主导时触发，量级为绊脚足占比。
        _ff3 = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, :]
        _stumble = ((_ff3[:, :, :2].norm(dim=-1) > 2.0 * _ff3[:, :, 2].abs().clamp(min=1.0))
                    & (_ff3.norm(dim=-1) > 5.0)).float().mean(dim=1)
        comp["stumble"] = -1.0 * _stumble                                             # 进入 TPPEN 日志。
        total = total + comp["stumble"]
        # 地形事件奖励：只在离散地形、线性移动命令和稳定姿态下启用。
        # 上/下台阶奖励由支撑面高度变化或新触地脚下地面高度变化触发，保持盲狗语义。
        comp["climb"] = torch.zeros(N, device=dev)
        comp["terrain_up"] = torch.zeros(N, device=dev)
        comp["terrain_down"] = torch.zeros(N, device=dev)
        comp["terrain_support_transfer"] = torch.zeros(N, device=dev)
        comp["terrain_contact_quality"] = torch.zeros(N, device=dev)
        comp["terrain_event_collapse"] = torch.zeros(N, device=dev)
        if not hasattr(self, "_prev_support_z") or self._prev_support_z.shape[0] != N:
            self._prev_support_z = _support_z.detach().clone()
        _prev_support = self._prev_support_z.to(device=dev)
        _support_delta = _support_z - _prev_support
        _contact_up_delta = torch.zeros(N, device=dev)
        _contact_down_delta = torch.zeros(N, device=dev)
        try:
            # 接触后的地形事件：新落脚踩到更高/更低地面，是盲狗可通过本体历史推断的信息。
            _land = landing_mask.to(device=dev)
            _contact_up_delta = (torch.clamp(_ground_z - _prev_support[:, None], min=0.0) * _land).max(dim=1).values
            _contact_down_delta = (torch.clamp(_prev_support[:, None] - _ground_z, min=0.0) * _land).max(dim=1).values
        except Exception:
            pass
        _terrain_up_event = torch.maximum(torch.clamp(_support_delta, min=0.0), _contact_up_delta)
        _terrain_down_event = torch.maximum(torch.clamp(-_support_delta, min=0.0), _contact_down_delta)
        terrain_probe = {
            "rise_ahead_mean": terrain_rise_ahead.mean(),
            "rise_ahead_max": terrain_rise_ahead.max(),
            "drop_ahead_mean": terrain_drop_ahead.mean(),
            "drop_ahead_max": terrain_drop_ahead.max(),
            "support_delta_mean": _support_delta.mean(),
            "support_delta_max": _support_delta.max(),
            "support_delta_min": _support_delta.min(),
            "contact_up_delta_mean": _contact_up_delta.mean(),
            "contact_up_delta_max": _contact_up_delta.max(),
            "contact_down_delta_mean": _contact_down_delta.mean(),
            "contact_down_delta_max": _contact_down_delta.max(),
            "disc_frac": torch.zeros((), device=dev),
            "moving_lin_frac": torch.zeros((), device=dev),
            "upright_frac": torch.zeros((), device=dev),
            "slip_weight_mean": torch.zeros((), device=dev),
            "no_stumble_mean": torch.zeros((), device=dev),
            "common_mean": torch.zeros((), device=dev),
            "common_max": torch.zeros((), device=dev),
            "up_active_mean": torch.zeros((), device=dev),
            "up_active_max": torch.zeros((), device=dev),
            "down_active_mean": torch.zeros((), device=dev),
            "down_active_max": torch.zeros((), device=dev),
            "up_latch_mean": torch.zeros((), device=dev),
            "down_latch_mean": torch.zeros((), device=dev),
            "contact_event_mean": torch.zeros((), device=dev),
            "contact_latch_mean": torch.zeros((), device=dev),
            "event_scope_mean": torch.zeros((), device=dev),
            "up_core_mean": torch.zeros((), device=dev),
            "down_core_mean": torch.zeros((), device=dev),
            "event_quality_mean": torch.zeros((), device=dev),
            "support_transfer_mean": torch.zeros((), device=dev),
            "front_duty_mean": torch.zeros((), device=dev),
            "rear_duty_mean": torch.zeros((), device=dev),
            "support_fault_mean": torch.zeros((), device=dev),
            "touchdown_clean_mean": torch.zeros((), device=dev),
            "torque_clean_mean": torch.zeros((), device=dev),
            "event_fault_mean": torch.zeros((), device=dev),
            "event_collapse_fault_mean": torch.zeros((), device=dev),
            "event_collapse_scope_mean": torch.zeros((), device=dev),
        }
        terrain_probe.update(terrain_scan_probe)
        _w_climb = float(getattr(self.cfg, "rew_climb", 0.0))
        _w_up = float(getattr(self.cfg, "rew_terrain_up", 0.0))
        _w_down = float(getattr(self.cfg, "rew_terrain_down", 0.0))
        _w_event_collapse = float(getattr(self.cfg, "rew_terrain_event_collapse", 0.0))
        if (_w_climb > 0.0 or _w_up > 0.0 or _w_down > 0.0 or _w_event_collapse != 0.0) and int(getattr(self, "_phase", getattr(self.cfg, "init_phase", 0))) >= int(getattr(self, "_terrain_start_phase", 5)):
            self._ensure_gate_mask()
            _disc_climb = getattr(self, "_discrete_terrain_mask", None)
            if _disc_climb is not None:
                _slip = quality["slip_speed"].to(device=dev)
                _slip_target = float(getattr(self.cfg, "climb_slip_gate", 0.30))
                _slip_span = max(float(getattr(self.cfg, "climb_slip_soft_span", 0.18)), 1e-6)
                _slip_weight = torch.clamp(1.0 - torch.clamp(_slip - _slip_target, min=0.0) / _slip_span, 0.0, 1.0)
                _cmd_xy = self.commands[:, :2]
                _cmd_xy_norm = _cmd_xy.norm(dim=-1)
                _moving_lin_cmd = _cmd_xy_norm > 0.10
                _v_along = torch.sum(rd.root_lin_vel_b[:, :2] * _cmd_xy, dim=1) / _cmd_xy_norm.clamp(min=1e-6)
                _progress = torch.clamp(_v_along / _cmd_xy_norm.clamp(min=1e-6), 0.0, 1.0)
                _upright = (-rd.projected_gravity_b[:, 2]).clamp(0.0, 1.0) > 0.85
                _no_stumble = torch.clamp(1.0 - _stumble, 0.0, 1.0)
                _common_base = _disc_climb.float() * _moving_lin_cmd.float() * _upright.float() * gate
                _common = _common_base * _slip_weight * _no_stumble
                _eps = float(getattr(self.cfg, "terrain_transition_eps", 0.05))
                _span = max(float(getattr(self.cfg, "terrain_transition_span", 0.08)), 1e-6)
                # 严格盲地形驱动：不按未接触的前方地形直接给奖励。
                # 只有支撑面实际变化后才给台阶事件奖励。
                _up_impulse = torch.clamp((_terrain_up_event - _eps) / _span, 0.0, 1.0)
                _down_impulse = torch.clamp((_terrain_down_event - _eps) / _span, 0.0, 1.0)
                # 真实触地/绊脚事件是盲狗可感知的本体证据；用于约束第一次接触后的稳定性，
                # 但不直接替代上/下台阶的正奖励。
                _contact_impulse = torch.clamp(_td_event.float() + (_stumble > 0.0).float(), 0.0, 1.0)
                _latch_s = float(getattr(self.cfg, "terrain_event_latch_s", 0.0))
                if _latch_s > 0.0:
                    _step_dt = max(float(self.cfg.dt) * float(self.cfg.decimation), 1e-6)
                    _latch_steps = max(1, int(round(_latch_s / _step_dt)))
                    _decay = 1.0 / float(_latch_steps)
                    if not hasattr(self, "_terrain_up_latch") or self._terrain_up_latch.shape[0] != N:
                        self._terrain_up_latch = torch.zeros(N, device=dev)
                        self._terrain_down_latch = torch.zeros(N, device=dev)
                    if not hasattr(self, "_terrain_contact_latch") or self._terrain_contact_latch.shape[0] != N:
                        self._terrain_contact_latch = torch.zeros(N, device=dev)
                    self._terrain_up_latch = torch.maximum(
                        torch.clamp(self._terrain_up_latch.to(device=dev) - _decay, min=0.0),
                        _up_impulse.detach(),
                    )
                    self._terrain_down_latch = torch.maximum(
                        torch.clamp(self._terrain_down_latch.to(device=dev) - _decay, min=0.0),
                        _down_impulse.detach(),
                    )
                    self._terrain_contact_latch = torch.maximum(
                        torch.clamp(self._terrain_contact_latch.to(device=dev) - _decay, min=0.0),
                        _contact_impulse.detach(),
                    )
                    _up_active = torch.maximum(_up_impulse, self._terrain_up_latch)
                    _down_active = torch.maximum(_down_impulse, self._terrain_down_latch)
                    _contact_active = torch.maximum(_contact_impulse, self._terrain_contact_latch)
                else:
                    _up_active = _up_impulse
                    _down_active = _down_impulse
                    _contact_active = _contact_impulse
                _up_cap = max(float(getattr(self.cfg, "terrain_up_vz_cap", getattr(self.cfg, "climb_vz_cap", 0.6))), 1e-6)
                _down_cap = max(float(getattr(self.cfg, "terrain_down_vz_cap", 0.5)), 1e-6)
                _up_v = rd.root_lin_vel_w[:, 2].clamp(min=0.0, max=_up_cap)
                _down_v = (-rd.root_lin_vel_w[:, 2]).clamp(min=0.0, max=_down_cap)
                _down_target = max(float(getattr(self.cfg, "terrain_down_vz_target", 0.18)), 1e-6)
                _down_control = torch.clamp(1.0 - torch.clamp(_down_v - _down_target, min=0.0) / _down_target, 0.0, 1.0)
                _support_up_mag = torch.clamp(_terrain_up_event / _span, 0.0, 1.0)
                _support_down_mag = torch.clamp(_terrain_down_event / _span, 0.0, 1.0)
                # 事件局部质量：地形奖励不仅看高度变化，还耦合接触质量。
                # 使用的信号都来自接触后的本体证据：duty、触地速度、滑移和力矩余量。
                _duty_ev = quality["duty_by_leg"].to(device=dev)
                _front_duty = 0.5 * (_duty_ev[:, 0] + _duty_ev[:, 1])
                _rear_duty = 0.5 * (_duty_ev[:, 2] + _duty_ev[:, 3])
                _front_margin = float(getattr(self.cfg, "terrain_front_duty_margin", 0.12))
                _rear_floor = float(getattr(self.cfg, "terrain_rear_duty_floor", 0.48))
                _support_scale = max(float(getattr(self.cfg, "terrain_support_scale", 0.22)), 1e-6)
                _front_fault = torch.clamp((_front_duty - _rear_duty - _front_margin) / _support_scale, 0.0, 1.0)
                _rear_fault = torch.clamp((_rear_floor - _rear_duty) / _support_scale, 0.0, 1.0)
                _support_fault = torch.clamp(_front_fault + _rear_fault, 0.0, 1.0)
                _support_transfer = 1.0 - _support_fault

                _td_m = landing_mask.to(device=dev)
                _td_event = _td_m.sum(dim=-1) > 0.5
                _impact_scale = max(float(getattr(cfg, "impact_speed_scale", 1.0)), 1e-6)
                _td_over = torch.clamp((foot_vel[:, :, 2].abs() - 0.10) / _impact_scale, 0.0, 1.0)
                _td_bad = (_td_over * _td_m).max(dim=-1).values
                _touchdown_clean = torch.where(_td_event, 1.0 - _td_bad, torch.ones_like(_td_bad))

                _torque_limit_ev = torch.as_tensor(torque_limit, device=dev).float()
                if _torque_limit_ev.ndim == 1:
                    _torque_limit_ev = _torque_limit_ev.unsqueeze(0).expand_as(rd.applied_torque)
                _torque_util_ev = rd.applied_torque.abs() / _torque_limit_ev.clamp_min(1e-6)
                _torque_soft = min(max(float(getattr(self.cfg, "terrain_torque_soft_frac", getattr(self.cfg, "torque_limit_frac", 0.85))), 0.05), 0.99)
                _torque_excess = torch.clamp((_torque_util_ev - _torque_soft) / max(1.0 - _torque_soft, 1e-6), 0.0, 1.0)
                _torque_sat = (_torque_util_ev >= 1.0).float()
                _torque_stress = torch.clamp(0.5 * _torque_excess.mean(dim=-1) + 0.5 * _torque_sat.mean(dim=-1), 0.0, 1.0)
                _torque_clean = 1.0 - _torque_stress

                _event_quality_raw = torch.clamp(
                    0.25 * _touchdown_clean + 0.25 * _torque_clean + 0.25 * _support_transfer + 0.25 * _slip_weight,
                    0.0,
                    1.0,
                )
                _event_quality_floor = min(max(float(getattr(self.cfg, "terrain_event_quality_floor", 0.35)), 0.0), 1.0)
                _event_quality = torch.clamp(
                    _event_quality_floor + (1.0 - _event_quality_floor) * _event_quality_raw,
                    0.0,
                    1.0,
                )
                _event_active = torch.maximum(torch.maximum(_up_active, _down_active), _contact_active)
                _slip_fault = 1.0 - _slip_weight
                _event_fault = torch.clamp(
                    0.30 * _slip_fault + 0.25 * (1.0 - _touchdown_clean) + 0.25 * (1.0 - _torque_clean) + 0.20 * _support_fault,
                    0.0,
                    1.0,
                )
                if _w_event_collapse != 0.0:
                    _collapse_h = float(getattr(self.cfg, "terrain_event_collapse_height", 0.42))
                    _collapse_wxy = max(float(getattr(self.cfg, "terrain_event_collapse_wxy", 1.50)), 1e-6)
                    _collapse_speed_ratio = float(getattr(self.cfg, "terrain_event_collapse_speed_ratio", 1.60))
                    _collapse_speed_min = float(getattr(self.cfg, "terrain_event_collapse_speed_min", 0.75))
                    _collapse_speed_scale = max(float(getattr(self.cfg, "terrain_event_collapse_speed_scale", 0.60)), 1e-6)
                    _speed_limit = torch.maximum(
                        _cmd_xy_norm * _collapse_speed_ratio,
                        torch.full_like(_cmd_xy_norm, _collapse_speed_min),
                    )
                    _height_bad = torch.clamp(
                        (_collapse_h - base_h) / max(_collapse_h * 0.25, 0.08),
                        0.0,
                        1.0,
                    )
                    _wxy_bad = torch.clamp((rd.root_ang_vel_b[:, :2].norm(dim=-1) - _collapse_wxy) / _collapse_wxy, 0.0, 1.0)
                    _speed_bad = torch.clamp((_v_along.abs() - _speed_limit) / _collapse_speed_scale, 0.0, 1.0)
                    _collapse_fault = torch.clamp(
                        torch.maximum(torch.maximum(_height_bad, _wxy_bad), torch.maximum(_speed_bad, terminal_window)),
                        0.0,
                        1.0,
                    )
                    _collapse_scope = _event_active * _disc_climb.float() * _moving_lin_cmd.float()
                    comp["terrain_event_collapse"] = -_w_event_collapse * _collapse_scope * (_collapse_fault ** 2)

                _up_core = (0.45 * _progress + 0.35 * (_up_v / _up_cap) + 0.20 * _support_up_mag) * _event_quality
                _down_core = _progress * (0.50 + 0.50 * _down_control)
                _down_core = (0.80 * _down_core + 0.20 * _support_down_mag) * _event_quality
                comp["terrain_up"] = _w_up * _up_active * _common * _up_core
                comp["terrain_down"] = _w_down * _down_active * _common * _down_core
                comp["climb"] = _w_climb * _up_active * _common * _up_core
                _w_support_transfer = float(getattr(self.cfg, "rew_terrain_support_transfer", 0.0))
                _w_contact_quality = float(getattr(self.cfg, "rew_terrain_contact_quality", 0.0))
                _quality_scope = _event_active * _common_base
                comp["terrain_support_transfer"] = -_w_support_transfer * _quality_scope * (_support_fault ** 2)
                comp["terrain_contact_quality"] = -_w_contact_quality * _quality_scope * (_event_fault ** 2)
                terrain_probe.update({
                    "disc_frac": _disc_climb.float().mean(),
                    "moving_lin_frac": _moving_lin_cmd.float().mean(),
                    "upright_frac": _upright.float().mean(),
                    "slip_weight_mean": _slip_weight.mean(),
                    "no_stumble_mean": _no_stumble.mean(),
                    "common_mean": _common.mean(),
                    "common_max": _common.max(),
                    "up_active_mean": _up_active.mean(),
                    "up_active_max": _up_active.max(),
                    "down_active_mean": _down_active.mean(),
                    "down_active_max": _down_active.max(),
                    "up_latch_mean": getattr(self, "_terrain_up_latch", torch.zeros(N, device=dev)).mean(),
                    "down_latch_mean": getattr(self, "_terrain_down_latch", torch.zeros(N, device=dev)).mean(),
                    "contact_event_mean": _contact_impulse.mean(),
                    "contact_latch_mean": getattr(self, "_terrain_contact_latch", torch.zeros(N, device=dev)).mean(),
                    "event_scope_mean": (_event_active * _disc_climb.float() * _moving_lin_cmd.float()).mean(),
                    "up_core_mean": _up_core.mean(),
                    "down_core_mean": _down_core.mean(),
                    "event_quality_mean": _event_quality.mean(),
                    "support_transfer_mean": _support_transfer.mean(),
                    "front_duty_mean": _front_duty.mean(),
                    "rear_duty_mean": _rear_duty.mean(),
                    "support_fault_mean": _support_fault.mean(),
                    "touchdown_clean_mean": _touchdown_clean.mean(),
                    "torque_clean_mean": _torque_clean.mean(),
                    "event_fault_mean": _event_fault.mean(),
                    "event_collapse_fault_mean": (
                        _collapse_fault.mean() if _w_event_collapse != 0.0 else torch.zeros((), device=dev)
                    ),
                    "event_collapse_scope_mean": (
                        _collapse_scope.mean() if _w_event_collapse != 0.0 else torch.zeros((), device=dev)
                    ),
                })
                total = (
                    total
                    + comp["climb"]
                    + comp["terrain_up"]
                    + comp["terrain_down"]
                    + comp["terrain_support_transfer"]
                    + comp["terrain_contact_quality"]
                    + comp["terrain_event_collapse"]
                )
        # live joint imitation：朝干净解析参考步态的正向吸引项。
        # 它直接约束关节轨迹，使软落脚、足端回收和左右对称进入奖励，而不只依赖 AMP 判别器。
        _w_imitate = float(getattr(self.cfg, "w_imitate_live", 0.0))
        if _w_imitate != 0.0:
            # 相位对齐：flat_reference 内部用平面速度重算周期。
            # 这里传入 self._gait_phase * T_im，使解析参考相位与环境 gait clock 对齐。
            # T_im 必须保持平面速度周期，以匹配 flat_reference 内部定义。
            _spd_im = torch.norm(self.commands[:, :2], dim=1)
            _T_im = torch.clamp(self.cfg.gait_period - self.cfg.gait_period_slope * _spd_im,
                                min=self.cfg.gait_period_min, max=self.cfg.gait_period)
            _rough_im = self._terrain_ctx[:, 2] if self.cfg.terrain_ctx_dim >= 3 else None
            _jp_ref = flat_reference(
                self.commands, self._gait_phase * _T_im,
                gait_period=self.cfg.gait_period, gait_period_slope=self.cfg.gait_period_slope,
                gait_period_min=self.cfg.gait_period_min, clearance_base=self.cfg.base_clearance,
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
            comp["imitate"] = (_w_imitate * _turn_frac_floored
                               * torch.exp(-_jdiff.pow(2).sum(dim=1) / _sigma_im) * moving)   # 只在运动时生效的正向吸引项。
            total = total + comp["imitate"]
            # 阶段门控风格指标：运动 env 中相对参考的平均关节误差。
            _mv = moving > 0.5
            self._style_err = (float(_jdiff.abs().mean(dim=1)[_mv].mean()) if bool(_mv.any())
                               else getattr(self, "_style_err", 1.0))
        # 摆动方向吸引项：约束空中摆动脚的前后速度，避免非前进命令下仍套用前进摆腿。
        # 它只作用于空中脚，且只看机体系前后方向，不压制地形抬脚。
        _w_swing = float(getattr(self.cfg, "w_swing_dir", 0.0))
        if _w_swing != 0.0:
            from isaaclab.utils.math import quat_apply_inverse
            _qsw = rd.root_quat_w[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
            _v_rel_w = foot_vel - rd.root_lin_vel_w[:, None, :]                           # (N,4,3)，足端相对 base 的世界速度。
            _v_fwd_b = quat_apply_inverse(_qsw, _v_rel_w.reshape(-1, 3)).reshape(self.num_envs, 4, 3)[:, :, 0]  # 机体系前后速度。
            _foot_y0 = 0.2082 * torch.tensor([1., -1., 1., -1.], device=self.device)      # FL,FR,RL,RR 的名义足端横向位置。
            _vfx_ref = self.commands[:, 0:1] - self.commands[:, 2:3] * _foot_y0[None, :]   # (N,4)，命令对应的前后摆动速率。
            _margin_sw = float(getattr(self.cfg, "swing_dir_margin", 0.15))
            # 允许参考中段摆动峰值加固定余量，只抓非命令方向的前向漂移。
            _excess_sw = torch.clamp(_v_fwd_b - 2.0 * _vfx_ref.clamp(min=0.0) - _margin_sw, min=0.0)
            _swing_m = (in_contact < 0.5).float()                                         # (N,4)，空中脚。
            _sw_err = (_swing_m * _excess_sw.pow(2)).sum(dim=1) / _swing_m.sum(dim=1).clamp(min=1.0)
            _sig_sw = float(getattr(self.cfg, "swing_dir_sigma", 0.08))
            comp["swing_dir"] = _w_swing * moving * torch.exp(-_sw_err / _sig_sw)          # 正向项，进入 pos_sum。
            total = total + comp["swing_dir"]
            self._swing_excess = float((_swing_m * _excess_sw).sum() / _swing_m.sum().clamp(min=1.0))  # 可用于遥测 EMA。
        # 停步刹车奖励：零命令但仍在运动时，奖励速度下降。
        # stand 项只奖励最终低速，不直接奖励减速过程；该项用于减少长时间滑行停步。
        _w_brake = float(getattr(self.cfg, "w_settle_brake", 0.0))
        if _w_brake != 0.0:
            if not hasattr(self, "_prev_settle_speed"):
                self._prev_settle_speed = torch.zeros(self.num_envs, device=self.device)
            _spd_now = torch.linalg.norm(rd.root_lin_vel_b[:, :2], dim=-1)
            _cmd_still = ((torch.norm(self.commands[:, :2], dim=1) <= 0.1)
                          & (self.commands[:, 2].abs() <= 0.05))
            _decel = torch.clamp(self._prev_settle_speed - _spd_now, min=0.0)          # 大于 0 表示正在减速。
            comp["settle_brake"] = (_w_brake * _decel * _cmd_still.to(rd.joint_pos.dtype)
                                    * (_spd_now > 0.08).to(rd.joint_pos.dtype))
            total = total + comp["settle_brake"]
            self._prev_settle_speed = _spd_now.detach()
        with torch.no_grad():
            cmd_xy = self.commands[:, :2]
            cmd_xy_norm_sq = torch.sum(cmd_xy * cmd_xy, dim=1)
            cmd_wz_sq = self.commands[:, 2] * self.commands[:, 2]
            progress_thr = 0.0025
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
            # 分方向进展使用与父类相同的课程 gate mask，用于阶段推进/回退、DR 升级和速度上限调整。
            # 这样可避免硬地形 env 把平地能力进展统计拖低。
            self._ensure_gate_mask()
            gm_e = self._gate_mask
            fwd_mask = moving_lin & (self.commands[:, 0] > 0.1) & gm_e
            back_mask = moving_lin & (self.commands[:, 0] < -0.1) & gm_e
            lat_mask = moving_lin & (self.commands[:, 1].abs() > 0.1) & gm_e
            yaw_mask = moving_ang & gm_e
            if bool(fwd_mask.any()):
                self._fwd_prog = float(lin_prog[fwd_mask].mean())
            if bool(back_mask.any()):
                self._back_prog = float(lin_prog[back_mask].mean())
            if bool(lat_mask.any()):
                self._lat_prog = float(lin_prog[lat_mask].mean())
            if bool(yaw_mask.any()):
                self._yaw_prog = float(ang_prog[yaw_mask].mean())
            gait_match = torch.zeros(N, device=dev)
            moving_any = moving > 0.5
            try:
                leg_phase = self._leg_phases()
                desired_stance = (leg_phase < self.cfg.gait_duty).float()
                gait_match = (desired_stance * in_contact + (1.0 - desired_stance) * (1.0 - in_contact)).mean(dim=1)
                gait_mean = float(gait_match[moving_any].mean()) if bool(moving_any.any()) else 0.0
                fl, fr, rl, rr = in_contact[:, 0], in_contact[:, 1], in_contact[:, 2], in_contact[:, 3]
                diag = 1.0 - 0.5 * ((fl - rr).abs() + (fr - rl).abs())
                diag_pair_inst_mean = float(quality["diag_pair_inst"][moving_any].mean()) if bool(moving_any.any()) else 0.0
                front = 0.5 * (fl + fr)
                rear = 0.5 * (rl + rr)
                left = 0.5 * (fl + rl)
                right = 0.5 * (fr + rr)
                duty_balance = 1.0 - torch.clamp((rear - front).abs() + (left - right).abs(), 0.0, 1.0)
                diag_mean = float(quality["diag_pair_score"][moving_any].mean()) if bool(moving_any.any()) else 0.0
                duty_balance_mean = float(quality["duty_quality"][moving_any].mean()) if bool(moving_any.any()) else 0.0
                duty_balance_inst_mean = float(duty_balance[moving_any].mean()) if bool(moving_any.any()) else 0.0
            except Exception:
                gait_mean = 0.0
                diag_mean = 0.0
                diag_pair_inst_mean = 0.0
                duty_balance_mean = 0.0
                duty_balance_inst_mean = 0.0
            foot_slip_xy = foot_vel_xy
            self._slip_now = float(quality["slip_speed"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._slip_inst = float(quality["slip_speed_inst"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._slip_high_fraction = float(quality["slip_high_fraction"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._diag_contact = diag_mean
            self._diag_pair_instant = diag_pair_inst_mean
            self._duty_balance = duty_balance_mean
            self._duty_balance_instant = duty_balance_inst_mean
            self._duty_spread = float(quality["duty_spread"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._duty_target_score = float(quality["duty_target_score"][moving_any].mean()) if bool(moving_any.any()) else 0.0
            self._duty_symmetry_score = float(quality["duty_symmetry_score"][moving_any].mean()) if bool(moving_any.any()) else 0.0
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
                float(lin_prog[moving_lin].mean()) if bool(moving_lin.any()) else 0.0,
                float(ang_prog[moving_ang].mean()) if bool(moving_ang.any()) else 0.0,
                float(comp["tracking_lin"].mean()),
                float(comp["tracking_yaw"].mean()),
                0.0,
                gait_mean,
            )
            self._last_gait_match = gait_mean
            # 预算监控：常规惩罚绝对值 / 正向任务奖励。
            # 碰撞、饱和和 terminal 这类事件惩罚不计入常规预算。
            # 超过阈值时父类会降低 penalty ramp，避免质量惩罚让行走本身变得不划算。
            _tier_s = ("torque_saturation", "terminal_penalty", "terrain_event_collapse")
            _skip = ("total", "stable_motion_gate", "stand_gate", "moving_gate", "quality_gate")
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
                "ang": float(comp["tracking_yaw"].mean()),
                "gait": float(comp["gait_anchor"].mean()),
                "diag": float(comp["diagonal_contact"].mean()),
                "duty": float(comp["duty_balance"].mean()),
                "imit": 0.0,
                "stand": float(comp["stand"].mean()),
                "height": 0.0,
                "slip": float(comp["stance_slip"].mean()),
                "clear": float((comp["clearance_under"] + comp["clearance_over"]).mean()),
                "hip": 0.0,
                "offax": float(comp["off_axis"].mean()),
                "purity": float(comp.get("planar_purity", torch.zeros(N, device=dev)).mean()),
                "heading": float(comp.get("heading_hold", torch.zeros(N, device=dev)).mean()),
                "lat_aux": float(comp["lateral_underspeed"].mean()),
                "climb": float(comp["climb"].mean()),
                "terr_up": float(comp["terrain_up"].mean()),
                "terr_down": float(comp["terrain_down"].mean()),
                "terr_support": float(comp["terrain_support_transfer"].mean()),
                "terr_quality": float(comp["terrain_contact_quality"].mean()),
                "terr_collapse": float(comp["terrain_event_collapse"].mean()),
                "terr_probe_rise": float(terrain_probe["rise_ahead_mean"]),
                "terr_probe_up_active": float(terrain_probe["up_active_mean"]),
                "terr_probe_common": float(terrain_probe["common_mean"]),
                "land": float(comp["landing_impact"].mean()),
                "torq": float((comp["torque_margin"] + comp["torque_saturation"]).mean()),
                "arate": float(comp["action_rate"].mean()),
                "vz": 0.0,
                "wxy": 0.0,
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
            with torch.no_grad():
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
                reward_payload = {
                    "total": float(total.mean()),
                    "lin_err": lin_err,
                    "speed": speed,
                    "gait": gait,
                    "base_h": float(base_h.mean()),
                    "upright": float((-rd.projected_gravity_b[:, 2]).clamp(0.0, 1.0).mean()),
                }
                for name, value in comp.items():
                    if name == "total":
                        continue
                    if torch.is_tensor(value) and value.numel() > 0:
                        reward_payload[name] = float(value.mean())
                for name, value in terrain_probe.items():
                    if torch.is_tensor(value) and value.numel() > 0:
                        reward_payload[f"terrain_probe_{name}"] = float(value.mean())
                if not self._reward_cfg_printed:
                    self._reward_cfg_printed = True
                    try:
                        reward_cfg_payload = (
                            asdict(cfg)
                            if is_dataclass(cfg)
                            else {
                                name: float(getattr(cfg, name))
                                for name in dir(cfg)
                                if not name.startswith("_") and isinstance(getattr(cfg, name), (int, float))
                            }
                        )
                    except Exception:
                        reward_cfg_payload = {}
                    for key, value in reward_cfg_payload.items():
                        if isinstance(value, (int, float)):
                            reward_payload[f"cfg_{key}"] = float(value)
                command_payload = {
                    "cmd_vx": float(self.commands[:, 0].mean()),
                    "cmd_vy": float(self.commands[:, 1].mean()),
                    "cmd_wz": float(self.commands[:, 2].mean()),
                    "actual_vx": float(vb[:, 0].mean()),
                    "actual_vy": float(vb[:, 1].mean()),
                    "actual_wz": float(rd.root_ang_vel_b[:, 2].mean()),
                    "v_along": float(v_along.mean()),
                    "speed_xy": speed,
                    "lin_err": lin_err,
                    "yaw_err": yaw_err,
                    "progress_ratio": float(progress_ratio.mean()),
                    "heading_error_abs": float(getattr(self, "_heading_error", torch.zeros(N, device=dev)).abs().mean()),
                    # 分方向进展是阶段门控的实际阻塞项，需要写入 JSONL，而不是只靠人工读 console。
                    "progress_fwd": progress_by_dir["fwd"],
                    "progress_back": progress_by_dir["back"],
                    "progress_lat": progress_by_dir["lat"],
                    "progress_yaw": progress_by_dir["yaw"],
                    "progress_min_active": float(progress_gate),
                    "transition_active_frac": float((getattr(self, "_cmd_transition_timer", torch.zeros(N, device=dev)) > 0).float().mean()),
                    "transition_strength": float(getattr(self, "_cmd_transition_strength", torch.zeros(N, device=dev)).mean()),
                    "transition_zero_frac": float(getattr(self, "_cmd_transition_zero_frac", torch.full((N,), 0.5, device=dev)).mean()),
                    "progress_lagging_dir": min(
                        (name for name in (active_dirs or progress_by_dir.keys())),
                        key=lambda name: progress_by_dir.get(str(name), 0.0),
                    ) if progress_by_dir else "",
                    "gait_match": float(getattr(self, "_last_gait_match", 0.0)),
                    "diagonal_contact": float(getattr(self, "_diag_contact", 0.0)),
                    "duty_balance": float(getattr(self, "_duty_balance", 0.0)),
                    "stance_slip": float(getattr(self, "_slip_now", 0.0)),
                    "diagonal_pair_instant": float(getattr(self, "_diag_pair_instant", 0.0)),
                    "duty_balance_instant": float(getattr(self, "_duty_balance_instant", 0.0)),
                    "duty_spread_window": float(getattr(self, "_duty_spread", 0.0)),
                    "duty_target_score": float(getattr(self, "_duty_target_score", 0.0)),
                    "duty_symmetry_score": float(getattr(self, "_duty_symmetry_score", 0.0)),
                    "stance_slip_instant": float(getattr(self, "_slip_inst", 0.0)),
                    "stance_slip_high_fraction": float(getattr(self, "_slip_high_fraction", 0.0)),
                }
                duty_by_leg = getattr(self, "_duty_by_leg", {})
                if isinstance(duty_by_leg, dict):
                    for leg, value in duty_by_leg.items():
                        command_payload[f"duty_{str(leg).lower()}"] = float(value)
                curriculum_payload = {
                    "phase": f"phi{phase}" if phase is not None else "",
                    "command_mode": getattr(self, "_last_command_mode", ""),
                    "terrain_mean": terrain_mean,
                    "terrain_max": terrain_max,
                    "terrain_flat_mean": float(terrain_stats.get("terrain_flat_mean", 0.0)),
                    "terrain_flat_max": int(terrain_stats.get("terrain_flat_max", 0)),
                    "terrain_real_mean": float(terrain_stats.get("terrain_real_mean", 0.0)),
                    "terrain_real_max": int(terrain_stats.get("terrain_real_max", 0)),
                    "terrain_discrete_mean": float(terrain_stats.get("terrain_discrete_mean", 0.0)),
                    "terrain_discrete_max": int(terrain_stats.get("terrain_discrete_max", 0)),
                    "dr_level": dr_level,
                    "terrain_move_up_rate": float(getattr(self, "_terrain_curriculum_move_up_rate", 0.0)),
                    "terrain_move_down_rate": float(getattr(self, "_terrain_curriculum_move_down_rate", 0.0)),
                    "terrain_failure_down_rate": float(getattr(self, "_terrain_curriculum_failure_down_rate", 0.0)),
                    "terrain_stable_end_rate": float(getattr(self, "_terrain_curriculum_stable_end_rate", 0.0)),
                    "terrain_speed_ok_rate": float(getattr(self, "_terrain_curriculum_speed_ok_rate", 0.0)),
                    "terrain_eligible_frac": float(getattr(self, "_terrain_curriculum_eligible_frac", 0.0)),
                    "terrain_discrete_move_up_rate": float(getattr(self, "_terrain_curriculum_discrete_move_up_rate", 0.0)),
                    "terrain_discrete_move_down_rate": float(getattr(self, "_terrain_curriculum_discrete_move_down_rate", 0.0)),
                    "terrain_discrete_failure_down_rate": float(getattr(self, "_terrain_curriculum_discrete_failure_down_rate", 0.0)),
                    "terrain_discrete_stable_end_rate": float(getattr(self, "_terrain_curriculum_discrete_stable_end_rate", 0.0)),
                    "phase_count": float(getattr(self, "_phase_count", 0)),
                    "penalty_gate": float(getattr(self, "_penalty_gate", 0.0)),
                    "budget_ratio": float(getattr(self, "_budget_ratio_ema", 0.0)),
                    "clearance_gate": float(getattr(self, "_clearance_gate", 0.0)),
                    "terrain_health_ok": 1.0 if getattr(self, "_terrain_health_ok", True) else 0.0,
                }
                curriculum_payload.update(terrain_type_payload)
                if progress_gate is not None:
                    curriculum_payload["progress_gate"] = progress_gate
                    curriculum_payload["progress_fwd"] = progress_by_dir["fwd"]
                    curriculum_payload["progress_back"] = progress_by_dir["back"]
                    curriculum_payload["progress_lat"] = progress_by_dir["lat"]
                    curriculum_payload["progress_yaw"] = progress_by_dir["yaw"]
                    curriculum_payload["yaw_ceil"] = float(getattr(self, "_vel_max_yaw", 0.0))   # yaw 速度上限诊断。
                    curriculum_payload["progress_lagging_dir"] = command_payload["progress_lagging_dir"]
                    if active_dirs:
                        curriculum_payload["active_dirs"] = ",".join(active_dirs)
                    cfg_obj = getattr(self, "cfg", None)
                    progress_threshold = None
                    if cfg_obj is not None and phase is not None:
                        progress_threshold = getattr(cfg_obj, f"phase_gate_prog_{int(phase)}", None)
                    if progress_threshold is not None and progress_gate < float(progress_threshold):
                        curriculum_payload["blocked_by"] = "progress"
                        curriculum_payload["next_gate"] = "phase_progress"
                if gait_gate is not None:
                    curriculum_payload["gait_gate"] = gait_gate
                    curriculum_payload["diagonal_gate"] = float(getattr(self, "_diag_contact", 0.0))
                    curriculum_payload["duty_balance_gate"] = float(getattr(self, "_duty_balance", 0.0))
                    curriculum_payload["slip_gate"] = float(getattr(self, "_slip_now", 0.0))
                    curriculum_payload["duty_spread_gate"] = float(getattr(self, "_duty_spread", 0.0))
                if fall_rate is not None:
                    cfg_obj = getattr(self, "cfg", None)
                    fall_threshold = getattr(cfg_obj, "phase_gate_fall_2", None) if cfg_obj is not None else None
                    curriculum_payload["fall_gate"] = fall_rate
                    if fall_threshold is not None and fall_rate >= float(fall_threshold):
                        curriculum_payload["blocked_by"] = curriculum_payload.get("blocked_by", "fall")
                        curriculum_payload["next_gate"] = curriculum_payload.get("next_gate", "fall_rate")
                health_payload = {
                    "stable_motion_gate": float(gate.mean()),
                    "moving_gate": float(moving.mean()),
                    "stand_gate": float(stand_gate.mean()),
                    "base_h": float(base_h.mean()),
                    "upright": float((-rd.projected_gravity_b[:, 2]).clamp(0.0, 1.0).mean()),
                    "tilt_deg": float(torch.rad2deg(tilt_rel).mean()),
                    "support_instability": float(support_instab.mean()),
                    "height_low_risk_window": float(getattr(self, "_height_low_risk", 0.0)),
                    "tilt_high_risk_window": float(getattr(self, "_tilt_high_risk", 0.0)),
                    "base_h_min": float(base_h.min()),
                    "tilt_deg_max": float(torch.rad2deg(tilt_rel).max()),
                    "contact_count": float(cc.mean()),
                    "contacts_mean": float(in_contact.mean()),
                    "torque_util": float(torque_util),
                    "terminal_rate": float(terminal_window.mean()),
                }
                if fall_rate is not None:
                    health_payload["fall_rate"] = fall_rate
                if self._telemetry is not None:
                    self._telemetry.emit(
                        step=self._rew_log_step,
                        total_steps=total_steps or None,
                        reward=reward_payload,
                        curriculum=curriculum_payload,
                        health=health_payload,
                        command=command_payload,
                    )
                else:
                    print("[TPREW] step %d rew %.3f lin_err %.3f speed %.3f gate %.2f tracking_lin %.3f stand %.3f"
                          % (self._rew_log_step, float(total.mean()), lin_err,
                             speed, gait, float(comp["tracking_lin"].mean()), float(comp["stand"].mean())), flush=True)
        # 多 critic：把同一套奖励分量拆成 [N,K] 目标向量并暂存。
        # 按 K 求和必须等于标量 total；环境侧 terminal 惩罚通过 extra_by_group 放入 stab 组。
        # _get_observations 重置 extras 后会把该向量暴露出去；未启用 TAILI_MULTI_CRITIC 时跳过。
        self._prev_support_z = _support_z.detach().clone()
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
