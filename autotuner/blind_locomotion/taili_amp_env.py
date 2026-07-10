# Taili quadruped AMP locomotion env. Task=velocity tracking; AMP style from trot reference.
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
from .motions import MotionLoader  # noqa: F401 (kept for compat)
from .multi_motion_loader import MultiMotionLoader
from .parametric_ref import flat_reference
try:
    from .taili_blind_config import active_direction_progress, phase_command_spec
except ImportError:  # local source layout: env_edit is nested one level deeper
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
    """Rotate body-frame XY commands into world XY using root yaw only."""
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
        # feet in the CONTACT SENSOR's own body ordering (NOT robot's — they differ; see memory).
        self._feet_contact_ids, _ = self._contact_sensor.find_bodies(self.cfg.foot_body_names)
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_foot_indexes = self._motion_loader.get_body_index(self.cfg.foot_body_names)
        self.n_feet = len(self.foot_indexes)
        # AMP STRIDE WINDOW (TAILI_AMP_STRIDE=1): make the style window span ~one gait period (~400ms)
        # instead of 2 adjacent frames. We raise num_amp_observations (2->6) and subsample the AMP obs
        # ring by amp_frame_stride (~4 steps ~80ms) on BOTH the policy and reference sides, so the exported
        # window is [t, t-stride, t-2*stride, ...]. The discriminator input auto-scales to
        # amp_observation_space * num_amp_observations. FLAG UNSET => stride=1, no cfg/buffer change,
        # byte-identical behavior.
        self._amp_frame_stride = 1
        if os.environ.get("TAILI_AMP_STRIDE") == "1":
            self._amp_frame_stride = int(getattr(self.cfg, "amp_frame_stride", 4))
            self.cfg.num_amp_observations = int(getattr(self.cfg, "amp_stride_frames", 6))
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device)
        # Deep raw ring holding enough ADJACENT frames to build the strided export window (stride>1 only).
        if self._amp_frame_stride > 1:
            self._amp_raw_depth = (self.cfg.num_amp_observations - 1) * self._amp_frame_stride + 1
            self._amp_raw_ring = torch.zeros(
                (self.num_envs, self._amp_raw_depth, self.cfg.amp_observation_space), device=self.device)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)        # SMOOTHED command (what the policy tracks)
        self._cmd_target = torch.zeros((self.num_envs, 3), device=self.device)      # raw target; commands low-passes toward it
        self.last_actions = torch.zeros((self.num_envs, 12), device=self.device)
        self._cmd_resample_steps = max(1, int(self.cfg.cmd_resample_s / (self.cfg.dt * self.cfg.decimation)))
        self._log_step = 0          # for periodic stdout training diagnostics
        self._terrain_ctx = torch.zeros((self.num_envs, self.cfg.terrain_ctx_dim), device=self.device)
        self._episode_start_xy = self.robot.data.root_pos_w[:, :2].detach().clone()
        self._episode_start_root_z = self.robot.data.root_pos_w[:, 2].detach().clone()
        self._episode_start_cmd_xy = torch.zeros((self.num_envs, 2), device=self.device)
        # GAIT-PHASE clock: a per-env phase in [0,1) advancing only when commanded to move; per-leg phase =
        # phase + diagonal trot offsets (FL,FR,RL,RR). Drives the contact-schedule reward + a policy obs.
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)
        self._trot_offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device)   # FL,FR,RL,RR diagonal
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
        # PROPRIOCEPTIVE HISTORY buffer — FIFO ring of the last H proprio frames (all real-robot-measurable).
        # Enables implicit stair sensing + velocity estimation without lin_vel or height scanner.
        H, P = self.cfg.obs_history_len, self.cfg.obs_history_dim
        self._obs_history = torch.zeros((self.num_envs, H, P), device=self.device)
        # CONTROL DELAY buffer — apply action from previous step to simulate ~20ms compute+comm latency.
        self._delayed_action = torch.zeros((self.num_envs, 12), device=self.device)
        # PER-DIRECTION velocity curriculum: each direction has its own speed ceiling + progress tracker.
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
        # YAW starts EASY and ramps up. In-place rotation at the full [0.25, 0.90] range from step 0
        # is far harder than the linear ranges — the progress metric averaged in the 0.90 rad/s
        # commands and pinned yaw_prog ~0.55, which then gated all of phi0. Start the yaw ceiling at
        # an achievable rate (~range_lo + 0.20) so competence builds on easy turns; the velocity
        # curriculum ramps it toward cmd_yaw_max as yaw_prog succeeds. (fwd/back/lat handle their
        # ranges fine, so they still start at the phase high.)
        yaw_lo = _phase_lo("yaw_range", self.cfg.cmd_yaw_range)
        yaw_hi = _phase_hi("yaw_range", self.cfg.cmd_yaw_range)
        # YAW REACHABLE RAMP (0707 v2): the yaw curriculum had two failure modes — the ORIGINAL "build from
        # easy" collapsed the ceiling to 0.30 (yaw_prog plateaus ~0.50 < vel_cur_down 0.55 → shrinks; grow
        # needs yaw_prog>=vel_cur_up 0.85, NEVER reached), so the robot only practiced yaw≈0.3 while A2 tests
        # 0.4/0.8 (out-of-distribution). The v1 fix (TAILI_YAW_FULLRANGE, pin BOTH to yaw_hi=0.9) over-corrected:
        # a from-scratch policy gets ~0 reward on 0.9-yaw, earns NOTHING, and ABANDONS yaw for linear (measured:
        # tracking_yaw peaked ~14k then declined). v2 = a REACHABLE easy→hard ramp: start easy (yaw_lo+0.20),
        # FLOOR at 0.45 (covers A2 yaw04=0.4, can't collapse to the too-easy 0.3), and GROW the ceiling toward
        # yaw_hi on a REACHABLE yaw-specific threshold (yaw_vel_cur_up, default 0.40) so competence at moderate
        # yaw expands the range up to 0.9 — bootstrapping easy yaw wins first, then covering the test range.
        self._vel_max_yaw = float(self.cfg.cmd_yaw_max) if not self._vel_curriculum_enable else min(yaw_hi, yaw_lo + 0.20)
        # Floors = phase range LOW (was = the ceiling, a bug: the curriculum could never ease a
        # struggling direction below its starting ceiling). Now a regressing direction can drop
        # back toward its easy end and rebuild.
        self._vel_floor_fwd = _phase_lo("fwd_range", self.cfg.cmd_fwd_range)
        self._vel_floor_back = _phase_lo("back_range", self.cfg.cmd_back_range)
        self._vel_floor_lat = _phase_lo("lat_range", self.cfg.cmd_lat_range)
        # yaw floor = 0.45 (covers A2 yaw04=0.4; prevents the collapse-to-0.30 that made yaw practice too easy
        # and out-of-distribution), but never above yaw_hi.
        self._vel_floor_yaw = min(yaw_hi, max(yaw_lo, 0.45))
        self._fwd_prog = 0.0; self._back_prog = 0.0; self._lat_prog = 0.0; self._yaw_prog = 0.0
        # DR LEVEL SYSTEM: starts at 0 (no DR), gates on demonstrated locomotion capability.
        self._dr_level = int(getattr(self.cfg, "dr_start_level", 0))   # gates up 0->1->2->3
        self._dr_gate_count = 0     # consecutive log intervals above level gate
        _push_s = getattr(self.cfg, f"dr_push_interval_s_{self._dr_level}", 0.0)
        _sdt = self.cfg.dt * self.cfg.decimation
        self._push_steps = max(1, int(_push_s / _sdt)) if _push_s > 0 else int(1e9)
        # IMU bias (level-3 DR): per-episode constant gyro+gravity offset (sensor miscalibration). [angv(3), grav(3)]
        self._imu_bias = torch.zeros((self.num_envs, 6), device=self.device)
        self._dr_warned = set()     # one-time warnings for unavailable physx DR APIs
        self._default_masses = None # stored once (CPU) for mass DR: reset-to-default + delta (no drift)
        self._dr_mat_level = -1        # highest DR level whose friction/CoM was written (re-fire on level-up; ROOT-4)
        self._default_coms = None      # stored once for CoM DR: reset-to-default + offset (no drift on per-level re-fire)
        self._dbg = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._last_gait_match = 0.0
        # UNIFIED TRAINING PHASE phi (0..3): coordinates all four curricula in strict dependency order
        # 0 bootstrap locomotion -> 1 clean gait -> 2 scale speed+terrain -> 3 harden DR.
        # Quality penalties / velocity ramp / terrain move-up / DR are each gated on phi (see _log_training_diag).
        # init_phase: fresh starts at 0 (full bootstrap). On RESUME of an already-walking checkpoint, set env var
        # TAILI_INIT_PHASE=2 to continue in the walker's TRAINED regime (penalties full, terrain on) instead of
        # dropping it back to phi0 (penalties off) — that phase-reset is a distribution shift AND wastes ~20k steps
        # re-climbing. clearance_gate still starts 0 so the new heavy clearance fades in gently even on resume.
        self._phase = int(os.environ.get("TAILI_INIT_PHASE", getattr(self.cfg, "init_phase", 0)))
        self._phase_count = 0          # consecutive intervals meeting the current phase's advance gate
        self._penalty_gate = 1.0 if self._phase >= 1 else 0.0   # past bootstrap -> quality penalties already full
        self._budget_ratio_ema = 0.0   # |regular penalties| / positive task reward (set by the reward step)
        self._clearance_gate = 0.0     # phi2 terrain clearance ramp: clearance -1.5->-8, base 0.07->0.09 (ramps 0->1 over phase 2)
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
        self._advance_ok = True        # regression guard: pause terrain/speed advance when capability regresses
        self._slip_ema = 0.0           # running mean foot slip speed (for the phi1->phi2 clean-gait gate)
        # Contact cache (computed in _get_observations, used in _get_rewards to avoid double query)
        self._in_contact = torch.zeros((self.num_envs, 4), device=self.device)
        # SYMMETRY augmentation: mirror the PPO batch to force a left-right symmetric gait.
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
        # Height scanner — always in scene: used for terrain_ctx (AMP conditioning), privileged critic
        # obs, and clearance reward. The ACTOR policy never sees it (BlindGaussianPolicy slices it out).
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        # contact sensor (foot air-time for the stepping reward)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # terrain importer with generator + curriculum (anymal_c direct pattern)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _resample_commands(self, env_ids, snap=False):
        # SINGLE-AXIS commands with explicit task-demand proportions. Writes the raw TARGET; self.commands low-passes
        # toward it (command buffer). snap=True (on reset) starts the env AT the command (no ramp from a stale value).
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
            if snap:
                self.commands[env_ids] = self._cmd_target[env_ids]
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

        # Per-direction curriculum ceilings — each direction ramps independently.
        # Forward: terrain-decoupled (scale down ceiling on hard terrain to avoid double difficulty).
        fwd_ceil = self._vel_max_fwd
        _terr_frac = None
        if self.cfg.vel_terrain_decouple and self.cfg.terrain.terrain_type == "generator":
            tmax = max(1.0, self.cfg.terrain.terrain_generator.num_rows - 1)
            _terr_frac = (self._terrain.terrain_levels[env_ids].float() / tmax).clamp(0.0, 1.0)
            fwd_ceil = self.cfg.cmd_fwd_range[0] + (self._vel_max_fwd - self.cfg.cmd_fwd_range[0]) * (1.0 - _terr_frac)
        fwd_cmd = torch.rand(n, device=dev) * (fwd_ceil - self.cfg.cmd_fwd_range[0]) + self.cfg.cmd_fwd_range[0]
        # Back: curriculum ceiling ramps independently — but TERRAIN-decoupled like forward (0706
        # D3p): backward was commanded at its FULL ceiling down stairs, and eccentric braking on the
        # 110 N·m thigh hit the torque clamp (terrain diag: torque_clamp_bout, backward@stairs).
        back_hi  = max(self.cfg.cmd_back_range[0], min(self._vel_max_back, self.cfg.cmd_back_max))
        if _terr_frac is not None:
            back_hi = self.cfg.cmd_back_range[0] + (back_hi - self.cfg.cmd_back_range[0]) * (1.0 - _terr_frac)
        back_cmd = torch.rand(n, device=dev) * (back_hi - self.cfg.cmd_back_range[0]) + self.cfg.cmd_back_range[0]
        self._cmd_target[env_ids, 0] = torch.where(
            fwd, fwd_cmd,
            torch.where(back, -back_cmd, torch.zeros(n, device=dev)))
        # Lateral: curriculum ceiling
        lat_hi  = max(self.cfg.cmd_lat_range[0], min(self._vel_max_lat, self.cfg.cmd_lat_max))
        lat_cmd = torch.rand(n, device=dev) * (lat_hi - self.cfg.cmd_lat_range[0]) + self.cfg.cmd_lat_range[0]
        self._cmd_target[env_ids, 1] = torch.where(
            lat, lat_cmd * torch.where(torch.rand(n, device=dev) < 0.5, 1.0, -1.0), torch.zeros(n, device=dev))
        # Yaw: curriculum ceiling
        yaw_hi  = max(self.cfg.cmd_yaw_range[0], min(self._vel_max_yaw, self.cfg.cmd_yaw_max))
        yaw_cmd = torch.rand(n, device=dev) * (yaw_hi - self.cfg.cmd_yaw_range[0]) + self.cfg.cmd_yaw_range[0]
        self._cmd_target[env_ids, 2] = torch.where(
            yaw, yaw_cmd * torch.where(torch.rand(n, device=dev) < 0.5, 1.0, -1.0), torch.zeros(n, device=dev))
        if snap:                              # reset: snap commands to target (no ramp from a stale prior value)
            self.commands[env_ids] = self._cmd_target[env_ids]

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        # COMMAND BUFFER (low-pass): self.commands smoothly ramps toward the raw target _cmd_target instead of
        # stepping instantly -> NATURAL decel on stop (spec-5 "crisp but natural": no hard brake) + smooth accel/turn.
        # Steady-state commands==target so velocity-tracking accuracy (A1) is unchanged; only transients are smoothed.
        # Part of the command INTERFACE -> runs in both training and deploy (external mode sets _cmd_target).
        self.commands += (1.0 - self.cfg.cmd_smooth_alpha) * (self._cmd_target - self.commands)
        # advance the gait clock ONCE per env step, ONLY when commanded to move (frozen at zero command ->
        # no rhythm -> policy stands still). SPEED-ADAPTIVE: period shortens with the commanded planar speed
        # (step frequency rises with command), so the contact clock matches the high-speed reference clips.
        step_dt = self.cfg.dt * self.cfg.decimation
        spd = torch.norm(self.commands[:, :2], dim=1)
        moving = (spd > 0.1) | (self.commands[:, 2].abs() > 0.05)   # yaw thr 0.1->0.05, match yaw_cmd_gate (0704)
        # YAW-INCLUSIVE OBSERVED CADENCE (0707 arch-fix #1): the actor's OBSERVED sin/cos gait clock was the LAST
        # component still ticking at PLANAR-only cadence, while the air-time reward (blind_tp_env.py:702) and the
        # AMP reference (taili_amp_reference.py period_speed) already use spd + 0.15*|wz|. During turns the clock
        # fed the actor a SLOW phase while it was rewarded + style-matched to a FASTER turn cadence — an
        # observation/reward inconsistency on exactly the failing A2-yaw gate. Match them (0.15 = validated; 0.30
        # broke F2). Keep the `moving` mask on planar spd.
        period_speed = spd + 0.15 * self.commands[:, 2].abs()
        period = torch.clamp(self.cfg.gait_period - self.cfg.gait_period_slope * period_speed,
                             min=self.cfg.gait_period_min, max=self.cfg.gait_period)
        self._gait_phase = (self._gait_phase + (step_dt / period) * moving.float()) % 1.0
        # DOMAIN RANDOMIZATION — random base PUSHES (disturbance-rejection robustness for sim2real). Every
        # push_steps, kick due envs with a random horizontal base velocity.
        if self.cfg.dr_enable:
            due = (self.episode_length_buf % self._push_steps == 0) & (self.episode_length_buf > 0)
            if bool(due.any()):
                ids = due.nonzero(as_tuple=False).flatten()
                push_vel = float(getattr(self.cfg, f"dr_push_vel_{self._dr_level}", 0.0))
                if push_vel <= 0.0:
                    return
                vel = torch.cat([self.robot.data.root_lin_vel_w[ids],
                                 self.robot.data.root_ang_vel_w[ids]], dim=-1).clone()  # (M, 6) world lin+ang
                vel[:, 0:2] += (torch.rand((len(ids), 2), device=self.device) * 2 - 1) * push_vel   # linear shove
                # YAW angular kick (rad/s) too, so the policy also recovers from ROTATIONAL disturbances, not just
                # translational pushes. Scaled by dr_push_ang_scale x push_vel.
                vel[:, 5] += (torch.rand(len(ids), device=self.device) * 2 - 1) * push_vel * self.cfg.dr_push_ang_scale
                self.robot.write_root_com_velocity_to_sim(vel, ids)

    def _leg_phases(self):
        # per-leg phase in [0,1): global phase + diagonal trot offsets -> (N, 4) for FL,FR,RL,RR.
        return (self._gait_phase[:, None] + self._trot_offsets[None, :]) % 1.0

    def _ensure_gate_mask(self):
        """Build (once) the CURRICULUM-GATE env mask = envs NOT on the hard sub-terrain types (slope_inv, boxes).
        Those types stay stuck at low terrain levels (blind discrete-obstacle / pit-spawn limits), so including
        their low prog / higher falls in the metrics that gate DR / velocity / phase / regression DRAGS the whole
        curriculum for the 75% of envs that are fine. Gate on the mastered-terrain envs instead. terrain_types is
        fixed per env, so this mask is constant -> build once."""
        if getattr(self, "_gate_mask", None) is not None:
            return
        if not (hasattr(self, "_terrain") and hasattr(self._terrain, "terrain_types")):
            self._gate_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._discrete_terrain_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return
        sg = self.cfg.terrain.terrain_generator
        props = np.array([s.proportion for s in sg.sub_terrains.values()], dtype=np.float64)
        props = props / props.sum(); cum = np.cumsum(props); ncol = sg.num_cols
        ct = np.array([int(np.min(np.where(c / ncol + 1e-3 < cum)[0])) for c in range(ncol)])
        self._col_type = torch.as_tensor(ct, device=self.device)
        self._type_names = list(sg.sub_terrains.keys())
        et = self._col_type[self._terrain.terrain_types]
        hard = torch.zeros_like(et, dtype=torch.bool)
        for nm in ("slope_inv", "boxes", "stairs_up"):   # stairs_up = pit-spawn ascend, hard like slope_inv
            if nm in self._type_names:
                hard |= (et == self._type_names.index(nm))
        self._gate_mask = ~hard
        # (B) DISCRETE-OBSTACLE mask (stairs/boxes): these need a high foot lift the roughness-gated clearance
        # under-provisions when stuck at low level. Used to FORCE the clearance demand there (break the catch-22).
        self._discrete_terrain_mask = torch.zeros_like(et, dtype=torch.bool)
        for nm in ("stairs", "boxes", "stairs_up"):
            if nm in self._type_names:
                self._discrete_terrain_mask |= (et == self._type_names.index(nm))
        print(f"[GATE] curriculum gates exclude hard terrain types {[n for n in ('slope_inv','boxes','stairs_up') if n in self._type_names]}"
              f" -> {int(self._gate_mask.sum())}/{self.num_envs} envs gate the curriculum"
              f" | discrete(stairs/boxes/stairs_up) types present: {[n for n in ('stairs','boxes','stairs_up') if n in self._type_names]}", flush=True)

    def _apply_action(self):
        # CONTROL DELAY: apply the action from the PREVIOUS step (simulates ~20ms compute+comm latency).
        self.robot.set_joint_position_target(self.action_offset + self.action_scale * self._delayed_action)
        self._delayed_action = self.actions.clone()

    def _feet_rel_base(self, foot_pos_w, base_pos_w, base_quat_w, n_foot):
        rel = foot_pos_w - base_pos_w.unsqueeze(1)                       # (M, n_foot, 3)
        q = base_quat_w.unsqueeze(1).expand(-1, n_foot, -1).reshape(-1, 4)
        return quat_apply_inverse(q, rel.reshape(-1, 3)).reshape(-1, n_foot * 3)

    def _compute_terrain_ctx(self):
        if self.cfg.terrain_ctx_dim == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)
        # terrain_ctx_dim == 3: [fore_slope, lat_slope, roughness]
        # roughness = std of base-relative ground height over the scan grid
        # TRUE terrain slope (fore, lat) in the HEADING frame. The scanner is attach_yaw_only (horizontal,
        # yaw-aligned) so it measures terrain independent of body pitch/roll -> transform hits by YAW ONLY
        # (using full quat would cancel the slope when the body aligns to it). Grid symmetric -> decoupled
        # least-squares slope = Sum(fx*dz)/Sum(fx^2), Sum(fy*dz)/Sum(fy^2).
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
        # Roughness: std of base-relative terrain height (3rd dim of terrain_ctx)
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
        # AMP = STYLE ONLY: pure kinematic style (jp,jv,bh,tn,foot_rel) conditioned ONLY on terrain_ctx, so the
        # discriminator learns "style | terrain" (high lift on rough/stairs, tilted posture on slope). Velocity
        # and command are DELIBERATELY EXCLUDED — speed/command tracking is owned by the two-sided tracking reward,
        # not the discriminator. (The old achieved-velocity channel demanded no-slip command speed → penalized real
        # slip as "off-style" and was too weak to brake overspeed; physeval-confirmed it didn't work.)
        return torch.cat([jp, jv, bh, tn, rel_b, self._terrain_ctx], dim=-1)

    def _get_observations(self) -> dict:
        if not getattr(self, "use_external_commands", False):
            due = (self.episode_length_buf % self._cmd_resample_steps == 0)
            if due.any():
                self._resample_commands(due.nonzero(as_tuple=False).flatten())
        lp = self._leg_phases()                                       # (N, 4) per-leg phase
        gait_obs = torch.cat([torch.sin(2 * math.pi * lp), torch.cos(2 * math.pi * lp)], dim=-1)  # (N, 8)
        N, dev = self.num_envs, self.device
        # OBSERVATION NOISE — added before scaling so magnitudes match real sensor specs.
        jpos_n  = self.robot.data.joint_pos  + torch.randn(N, 12, device=dev) * self.cfg.obs_noise_jpos
        jvel_n  = self.robot.data.joint_vel  + torch.randn(N, 12, device=dev) * self.cfg.obs_noise_jvel
        angv_n  = self.robot.data.root_ang_vel_b  + torch.randn(N, 3, device=dev) * self.cfg.obs_noise_angvel
        grav_n  = self.robot.data.projected_gravity_b + torch.randn(N, 3, device=dev) * self.cfg.obs_noise_gravity
        # IMU BIAS (level-3 DR): add the per-episode constant offset (zero below level 3)
        angv_n  = angv_n + self._imu_bias[:, 0:3]
        grav_n  = grav_n + self._imu_bias[:, 3:6]
        # PROPRIOCEPTIVE HISTORY update (FIFO): push current frame to front, drop oldest.
        # History dim = 42: jpos-default(12)+jvel(12)+angvel(3)+gravity(3)+lastact(12) — all motor+IMU.
        prop_now = torch.cat([
            jpos_n - self.action_offset,   # 12
            jvel_n * 0.05,                 # 12
            angv_n * 0.25,                 # 3
            grav_n,                        # 3
            self.last_actions,             # 12
        ], dim=-1)                         # (N, 42)
        self._obs_history = torch.cat([prop_now.unsqueeze(1), self._obs_history[:, :-1]], dim=1)  # FIFO
        # FOOT CONTACT (cached here for use in _get_rewards without double query)
        forces_now = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, :].norm(dim=-1)
        self._in_contact = (forces_now > 1.0).float()   # (N, 4)

        # BLIND OBS (473 dims) — what the actor policy actually uses (motor+IMU+history only):
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

        # PRIVILEGED EXTRAS (197 dims) — only for critic/AMP; actor NEVER sees these.
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

        # Full observation = privileged (670 dims). BlindGaussianPolicy slices [:, :473] internally.
        obs = priv_obs
        # NaN/Inf DIAGNOSTIC + GUARD: a physics blowup can make obs non-finite; log it once (to find the
        # culprit) then sanitize so it never reaches the network or the rollout buffer (prevents the silent
        # update-time CUDA deadlock). The blown-up env is reset by the non-finite check in _get_dones.
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
        # periodic stdout reference metrics (goes to the train log via tee) so progress is visible:
        # how well velocity is tracked, mean speed reached, per-direction command coverage, uprightness.
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
            # rough per-direction fractions of the commanded population
            fwd = (cmd[:, 0] > 0.2).float().mean().item()
            bwd = (cmd[:, 0] < -0.2).float().mean().item()
            lat = (cmd[:, 1].abs() > 0.15).float().mean().item()
            yaw = (cmd[:, 2].abs() > 0.2).float().mean().item()
        # plane/flat TerrainImporter (diagnostics) has no curriculum levels — guard, don't crash env.step
        _tl = getattr(getattr(self, "_terrain", None), "terrain_levels", None)
        tlvl = float(_tl.float().mean()) if _tl is not None else 0.0
        tctx = self._terrain_ctx.abs().mean(0) if self.cfg.terrain_ctx_dim > 0 else None
        with torch.no_grad():
            # last_air_time = per-foot COMPLETED stride air duration (held between touchdowns), the
            # same unit as air_time_target and phase_gate_air_* (~0.27 s for a clean trot). The
            # instantaneous current_air_time mean reads ~T_air/2 × swing-fraction (~0.07 for a
            # perfect trot) and would make the air gate unreachable.
            air = float(self._contact_sensor.data.last_air_time[:, self._feet_contact_ids].mean())
            act_std = float(self.actions.std()) if hasattr(self, "actions") else 0.0
            act_mag = float(self.actions.abs().mean()) if hasattr(self, "actions") else 0.0
        lp, ap, rl, ra, rair, gm = self._dbg

        self._ensure_gate_mask()
        # FALL = body LOW **and** TILTED. The old `root_pos_w[z] < termination_height` used ABSOLUTE world height,
        # so an UPRIGHT robot standing on low terrain (descended a slope/stair — abs z < 0.35 while perfectly fine)
        # was counted as "fallen". At phi0 that pinned fall_rate at ~0.10 (10% of gate-mask envs upright-but-low)
        # while upright≈0.995 — a contradiction that structurally BLOCKED the phi0→phi1 gate (needs fall_rate<0.05),
        # regardless of policy/imitation. A real fall is low AND not-upright; gate the height check by tilt so
        # upright-on-low-terrain no longer masquerades as a fall (0707).
        _below = self.robot.data.root_pos_w[self._gate_mask, 2] < self.cfg.termination_height
        _tilted = self.robot.data.projected_gravity_b[self._gate_mask, 2] >= -0.7   # NOT upright
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
        # IMITATION FIDELITY (style_err) — the phi1->phi2 gate metric (mean joint error vs reference). Now computed
        # in _get_rewards (the r_imitate reward shares the same flat_reference call) and stored on self._style_err,
        # so it's just read here (no duplicate flat_reference IK).
        self._style_err = getattr(self, "_style_err", 1.0)

        # ── UNIFIED TRAINING PHASE (v3: 0 flat all-dirs -> 1 flat mixed -> 2 +terrain/DR -> 3 envelope) ──
        # Penalty ramp (budget-controlled) runs from STEP 0: there is no unpenalized bootstrap
        # phase anymore — the budget controller IS the adaptive bootstrap (a weak early policy has
        # a high penalty/positive ratio, so the gate self-holds near 0 and rises as the policy can
        # afford quality). Invariant 4a: |regular penalties| <= penalty_budget_ratio_max × positive
        # task reward (tier-S excluded); above it the gate steps DOWN, so the ramp can never make
        # "not walking" the more profitable policy (the v1-v4 stand-still trap).
        ramp_step = 1.0 / C.penalty_ramp_intervals
        budget_ratio = float(getattr(self, "_budget_ratio_ema", 0.0))
        budget_max = float(getattr(C, "penalty_budget_ratio_max", 0.8))
        if budget_ratio > budget_max:
            self._penalty_gate = max(0.0, self._penalty_gate - ramp_step)
        else:
            self._penalty_gate = min(1.0, self._penalty_gate + ramp_step)
        # phi2 terrain clearance ramp: -1.5->-8 / base 0.07->0.09 so a WALKER lifts feet high enough to break the
        # terrain~5 ceiling (light clearance -> drag-shuffle stalls the terrain curriculum).
        terrain_phase = self._phase >= getattr(self, "_terrain_start_phase", 5)
        if terrain_phase:
            self._clearance_gate = min(1.0, self._clearance_gate + 1.0 / C.clearance_ramp_intervals)
        terrain_slip_thr = float(getattr(C, "terrain_gate_slip_high", 0.22))
        terrain_health_ok = getattr(self, "_slip_high_fraction", 0.0) <= terrain_slip_thr
        self._terrain_health_ok = terrain_health_ok
        # Regression guard (phase 2): pause terrain/speed advance on capability regress.
        # Terrain advance keeps low persistent slip as the shared contact constraint, but
        # does not force the flat diagonal/duty template onto high stairs or boxes.
        self._advance_ok = not (fall_rate > C.regress_fall or min_prog < C.regress_prog) and (
            not terrain_phase or terrain_health_ok
        )
        # Phase advance gate (sustained over phase_intervals). Thresholds are per-phase with
        # walk-down (phase_gate_<name>_<p> for the highest p <= current phase): early flat phases
        # can use reachable quality bars while later phases re-tighten toward the spec bars.
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
        # v3: quality gates apply from phase 0 (the _0 keys are the reachable flat bars) —
        # there is no quality-exempt bootstrap phase anymore.
        flat_quality_ok = (
            self._slip_ema <= slip_thr and diag_contact >= diag_thr and duty_balance >= duty_thr and air >= air_thr
        )
        terrain_quality_ok = self._slip_ema <= max(slip_thr, terrain_slip_thr)
        quality_ok = terrain_quality_ok if terrain_phase else flat_quality_ok
        phase_spec = phase_command_spec(C, self._phase) if phase_command_spec is not None else {}
        command_mode = str(phase_spec.get("command_mode", ""))
        is_mixed_phase = command_mode == "mixed"
        if terrain_phase:
            gate = (
                self._penalty_gate >= 1.0
                and min_prog >= prog_thr
                and quality_ok
                and (not is_mixed_phase or (tlvl >= C.phase_gate_terrain_2 and fall_rate < C.phase_gate_fall_2))
            )
        else:
            # Flat phases (phi0 command-conditioning + phi1 mixed): tracking in all directions +
            # gait quality bars + penalty ramp full. Quality gates apply from phi0 (the _0 keys are
            # the reachable flat bars). The phi0 blocker was never quality — it was YAW progress
            # stuck in a reward dead zone (fixed via sigma_yaw / yaw_far), not the quality bars.
            gate = self._penalty_gate >= 1.0 and min_prog >= prog_thr and quality_ok
        # DEADLOCK SAFEGUARD (0707): yaw progress is the historical phi0 blocker (anti-symmetric, hard to
        # explore — it plateaued ~0.50 vs prog_thr 0.65). Since progress_gate = MIN over active dirs, a yaw that
        # can't clear prog_thr makes `gate` permanently False → the phase NEVER advances → a from-scratch run
        # burns GPU forever. Cap time-in-phase: after phase_max_steps in one phase without advancing, FORCE the
        # advance with a loud warning. Safe because yaw is active in ALL phases, so forcing phi0->phi1 keeps
        # training yaw (+ adds mixed commands) rather than abandoning it — it only stops the hardest single
        # direction from hard-blocking the whole curriculum. Set TAILI_NO_PHASE_TIMEOUT=1 to disable.
        # Timeout CLOCK starts only AFTER the penalty ramp completes (penalty_gate first hits 1.0 in this
        # phase). The `gate` requires penalty_gate>=1.0, which on a from-scratch phi0 can't happen until the
        # ramp finishes (~step 12500). A raw step-in-phase budget of 12000 therefore fired BEFORE the earliest
        # possible natural advance and force-marched every phi0. Measuring from ramp-completion makes the
        # safeguard fire on a genuine POST-ramp stall, not the ramp itself. Clock resets on phase advance.
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
                self._phase_penalty_full_step = None   # restart the clock for the new phase
                _tag = " FORCED(deadlock-safeguard: min_prog<prog_thr for phase_max_steps post-ramp)" if _forced else ""
                print(f"[PHASE] -> {self._phase} at step {self._log_step}{_tag} "
                      f"(prog={min_prog:.2f} gait={gm:.2f} slip={self._slip_ema:.2f} terrain={tlvl:.2f})", flush=True)
        # SYMMETRY is a refinement (like the penalties): off during phase-0 bootstrap, on from phase 1.
        if getattr(self.cfg, "sym_augment", False):
            from . import symmetry
            symmetry.set_active(self._phase >= 1)

        # VELOCITY curriculum runs from phase 0. Command maxima are still bounded by the
        # active phase ranges, so this only raises the per-direction ceiling when the
        # policy already tracks the current envelope.
        step = C.vel_cur_step
        if self._vel_curriculum_enable and self._advance_ok:
            if self._fwd_prog  >= C.vel_cur_up:  self._vel_max_fwd  = min(self._vel_max_fwd  + step, C.cmd_fwd_max)
            elif self._fwd_prog <= C.vel_cur_down: self._vel_max_fwd = max(self._vel_max_fwd - step, self._vel_floor_fwd)
            if self._back_prog >= C.vel_cur_up:  self._vel_max_back = min(self._vel_max_back + step, C.cmd_back_max)
            elif self._back_prog <= C.vel_cur_down: self._vel_max_back = max(self._vel_max_back - step, self._vel_floor_back)
            if self._lat_prog  >= C.vel_cur_up:  self._vel_max_lat  = min(self._vel_max_lat  + step, C.cmd_lat_max)
            elif self._lat_prog <= C.vel_cur_down: self._vel_max_lat = max(self._vel_max_lat - step, self._vel_floor_lat)
            # YAW uses a REACHABLE grow threshold (yaw_vel_cur_up, default 0.40) — the shared vel_cur_up=0.85
            # is never reached by yaw (plateaus ~0.50), so the ceiling could only shrink, never grow. A 0.40
            # bar lets moderate yaw competence EXPAND the practiced range toward cmd_yaw_max (0.9→1.5).
            _yaw_up = float(getattr(C, "yaw_vel_cur_up", 0.40))
            if self._yaw_prog  >= _yaw_up:  self._vel_max_yaw  = min(self._vel_max_yaw  + step, C.cmd_yaw_max)
            elif self._yaw_prog <= C.vel_cur_down: self._vel_max_yaw = max(self._vel_max_yaw - step, self._vel_floor_yaw)

        # DR curriculum: runs in phase >= 2, IN PARALLEL with the terrain curriculum (DECOUPLED from terrain level).
        # Rationale: DR (sim2real dynamics robustness: mass->10kg, friction, CoM, IMU bias) and terrain (external
        # difficulty) are INDEPENDENT axes — a policy can be robust to mass/friction on flat ground. The old design
        # gated DR behind phase 3 (= terrain>=6), making DR a HOSTAGE of the terrain milestone: if terrain plateaus
        # below 6 (observed: stuck ~4.8), DR would NEVER activate — yet for a BLIND real 39kg robot, DR matters MORE
        # than terrain level 6. DR still self-gates on TERRAIN-INDEPENDENT capability (prog/upright/fall, NOT
        # gait_match which is suppressed on rough terrain), so it only steps up when the policy is stable; terrain
        # and DR each advance on their own merits, neither blocking the other.
        # DR-DEFER: don't START stacking DR until the policy has some terrain footing (tlvl >= dr_unlock_terrain).
        # Reduces the phi2 load (terrain + DR + clearance at once); DR (sim2real robustness) is the LAST axis to add.
        if (
            self._phase >= getattr(self, "_dr_start_phase", self._terrain_start_phase)
            and self._dr_level < 3
            and tlvl >= getattr(C, "dr_unlock_terrain", 0.0)
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
        # PER-SUB-TERRAIN-TYPE level breakdown: the MEAN terrain level is capped ~num_rows/2 by the curriculum's
        # bounce-to-random (maxed envs -> random level), so it HIDES which sub-terrains the policy actually masters
        # (~rows/2 = mastered+cycling) vs caps below (genuinely stuck). cols map to sub-terrains by cumulative
        # proportion (same logic as IsaacLab _generate_curriculum_terrains). Read-only diagnostic.
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
                                        "hip", "offax", "over", "under", "wrong", "land", "torq", "arate", "vz", "wxy")}
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
              f"terrain={tlvl:.2f}{tctx_str}{terr_type_str}\n"
              f"          rew[task]: lin={d['lin']:+.2f} ang={d['ang']:+.2f} gait={d['gait']:+.2f} imit={d['imit']:+.2f} "
              f"stand={d['stand']:+.2f} height={d['height']:+.2f} slip={d['slip']:+.2f} clear={d['clear']:+.2f} "
              f"hip={d['hip']:+.2f} offax={d['offax']:+.2f} over={d['over']:+.2f} under={d['under']:+.2f} wrong={d['wrong']:+.2f} "
              f"land={d['land']:+.2f} vz={d['vz']:+.2f} "
              f"wxy={d['wxy']:+.2f} torq={d['torq']:+.3f} arate={d['arate']:+.2f}  (+AMP style by skrl)\n"
              f"          GATE phi{self._phase}: min_prog={min_prog:.2f}(need>={prog_thr:.2f}) "
              f"quality={int(quality_ok)}(slip<={slip_thr:.2f},diag>={diag_thr:.2f},duty>={duty_thr:.2f},air>={air_thr:.2f}) "
              f"pen_gate={self._penalty_gate:.2f}/1.0 count={self._phase_count}/{C.phase_intervals} | "
              f"act_std={act_std:.3f} cmd_frac={fwd:.2f}/{bwd:.2f}/{lat:.2f}/{yaw:.2f}", flush=True)

    def _get_rewards(self) -> torch.Tensor:
        # TWO-SIDED velocity tracking (A1/A2/A3 — speed's SOLE owner). exp(-||achieved-command||^2/sigma):
        # peaks at exact match, falls off for BOTH under- AND over-speed. The old capped-progress left overshoot
        # UNPENALIZED -> physeval measured +30~67% overspeed and "can't go slow". Unified for moving AND standing:
        # cmd=0 -> rewards zero residual = clean stop (A3). The 2D linear error also pins lateral drift (vy->0 when
        # commanded fwd). Speed is now owned HERE only; AMP/reference judge style, not speed.
        vel_lin = self.robot.data.root_lin_vel_b[:, :2]
        wz = self.robot.data.root_ang_vel_b[:, 2]
        cmd_lin = self.commands[:, :2]
        cmd_ang = self.commands[:, 2]
        cmd_lin_m2 = torch.sum(cmd_lin * cmd_lin, dim=1)            # ||cmd_lin||^2 (used by gates below)
        cmd_ang_2 = cmd_ang * cmd_ang
        thr = 0.0025                                               # (|cmd|>0.05)^2 — moving vs standing split
        mv_l = cmd_lin_m2 > thr
        mv_a = cmd_ang_2 > thr
        # CAPPED directional progress (0..1) — the PROVEN bootstrap reward: a strong "move that way from any speed"
        # gradient, and ZERO reward for standing when commanded to move (forces locomotion). The exp two-sided form
        # (P4) OVER-rewarded standing (cmd0.5,v0 -> 0.37*w) -> phi0 bootstrap STALLED at min_prog~0.18. Overspeed is
        # braked SEPARATELY by r_overshoot below -> the two-sided effect WITHOUT the bootstrap-killing stand reward.
        # lin_prog/ang_prog also feed the phi gate (min_prog) + velocity curriculum.
        lin_prog = torch.clamp(torch.sum(vel_lin * cmd_lin, dim=1) / cmd_lin_m2.clamp(min=thr), 0.0, 1.0)
        ang_prog = torch.clamp(wz * cmd_ang / cmd_ang_2.clamp(min=thr), 0.0, 1.0)
        lin_stand = torch.exp(-torch.sum(vel_lin * vel_lin, dim=1) / self.cfg.stand_sigma)
        ang_stand = torch.exp(-wz * wz / self.cfg.stand_sigma)
        lin_stand_w = torch.where(mv_a & ~mv_l, 0.25, 1.0)   # don't pay full standing on a non-commanded axis
        ang_stand_w = torch.where(mv_l & ~mv_a, 0.25, 1.0)
        r_lin = self.cfg.rew_track_lin * torch.where(mv_l, lin_prog, lin_stand * lin_stand_w)
        r_ang = self.cfg.rew_track_ang * torch.where(mv_a, ang_prog, ang_stand * ang_stand_w)
        r_alive = self.cfg.rew_alive * (~self.reset_terminated).float()
        r_arate = self.cfg.rew_action_rate * torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        r_jacc = self.cfg.rew_joint_acc * torch.sum(torch.square(self.robot.data.joint_acc), dim=1)
        # MOTOR PROTECTION: penalize torque ONLY in the stress zone (|tau| above torque_limit_frac of the per-joint
        # effort limit). Normal-gait torque -> 0 penalty (gait untouched); near-saturation/violent torque -> grows
        # quadratically -> discourages motor-damaging commands. effort_limit is per-joint in the correct DOF order.
        try:
            tau = self.robot.data.applied_torque                                      # (N, 12)
            eff_lim = self.robot.actuators["legs"].effort_limit                        # (N, 12) per-joint Nm
            tau_over = torch.clamp(tau.abs() - self.cfg.torque_limit_frac * eff_lim, min=0.0)
            r_torque = self.cfg.rew_torque * torch.sum(tau_over * tau_over, dim=1)
        except Exception as _e:                                                       # never let it kill the reward
            r_torque = torch.zeros(self.num_envs, device=self.device)
            if "torque" not in self._dr_warned:
                print(f"[REW] torque penalty unavailable ({type(_e).__name__}); skipping", flush=True)
                self._dr_warned.add("torque")
        r_vz = self.cfg.rew_lin_vel_z * torch.square(self.robot.data.root_lin_vel_b[:, 2])
        r_wxy = self.cfg.rew_ang_vel_xy * torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
        # feet_air_time: reward each foot for being airborne ~air_time_target, credited on landing.
        # BOUND TO EFFECTIVE DISPLACEMENT (principle 4): only credited when the robot is actually making
        # progress on its command (linear velocity ALONG the commanded direction, OR turning the right
        # way). Marching/flailing in place (high air_time, zero displacement) earns NOTHING -> the gait
        # reward cannot be gamed without real locomotion.
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_ids]
        air_time = self._contact_sensor.data.current_air_time[:, self._feet_contact_ids]
        cmd_lin = self.commands[:, :2]
        cmd_lin_mag = torch.norm(cmd_lin, dim=1)
        vb2 = self.robot.data.root_lin_vel_b[:, :2]
        vel_along = (vb2 * cmd_lin).sum(-1) / cmd_lin_mag.clamp(min=0.1)         # speed along commanded dir
        yaw_match = self.commands[:, 2] * self.robot.data.root_ang_vel_b[:, 2]   # >0 if turning the right way
        effective = ((vel_along > 0.1) | (yaw_match > 0.05)).float()
        air_credit = torch.clamp(air_time - self.cfg.air_time_min, min=0.0,
                                 max=self.cfg.air_time_target - self.cfg.air_time_min)
        r_air = self.cfg.rew_feet_air_time * torch.sum(air_credit * first_contact.float(), dim=1) * effective
        mv_any = mv_l | mv_a

        # GAIT-PHASE TROT CONTACT reward (HARD contact-timing imitation; AMP is blind to contact -> couldn't
        # do this). The diagonal trot clock says which legs should be in stance (phase<duty) vs swing; reward
        # the fraction of legs whose ACTUAL contact matches. A dragging leg (loaded during its swing phase)
        # mismatches -> penalized. Gated to moving commands (no rhythm imposed while standing).
        in_contact = self._in_contact                                                    # cached from _get_observations
        leg_phase = self._leg_phases()
        desired_stance = (leg_phase < self.cfg.gait_duty).float()                       # 1 = should be planted
        gait_match = (desired_stance * in_contact + (1.0 - desired_stance) * (1.0 - in_contact)).mean(dim=1)
        # RAW gait_match (NO dead-zone): this is V1's proven formula (rew*gait_match), which reached gait_match
        # 0.93. The dead-zone clamp((gait_match-0.5)/0.5) gave ZERO gradient at the bootstrap point (gait_match
        # ~0.5) -> the trot rhythm never established. Raw gait_match gives a strong gradient (rew) everywhere.
        # TERRAIN RELAXATION kept: scale DOWN on rough terrain so the policy can break the clock to climb
        # (gait_match dropping on hard terrain is CORRECT; flat keeps the full reward).
        if self.cfg.terrain_ctx_dim >= 3:
            rough_gate = torch.clamp(self._terrain_ctx[:, 2] / self.cfg.gait_rough_scale, 0.0, 1.0)
        else:
            rough_gate = torch.zeros(self.num_envs, device=self.device)
        r_gait = self.cfg.rew_gait_phase * gait_match * mv_any.float() * (1.0 - rough_gate)

        # (A) IMITATION reward (trajectory-level): directly track the reference JOINTS at each env's current command
        # + gait phase. AMP is a weak 2-frame distributional prior; this forces the ACTUAL gait toward the reference
        # (gait_match only checks contact timing). exp(-||Δjoint||²/sigma)*weight, moving-only, rough-gated like
        # r_gait so the policy can still deviate to climb. Also sets self._style_err (the phi1->phi2 gate metric).
        if self.cfg.rew_imitate != 0.0:
            spd_im = torch.norm(self.commands[:, :2], dim=1)
            T_im = torch.clamp(self.cfg.gait_period - self.cfg.gait_period_slope * spd_im,
                               min=self.cfg.gait_period_min, max=self.cfg.gait_period)
            rough_im = self._terrain_ctx[:, 2] if self.cfg.terrain_ctx_dim >= 3 else None
            jp_ref = flat_reference(self.commands, self._gait_phase * T_im, gait_period=self.cfg.gait_period,
                                    gait_period_slope=self.cfg.gait_period_slope, gait_period_min=self.cfg.gait_period_min,
                                    clearance_base=self.cfg.base_clearance, roughness=rough_im,
                                    clearance_rough_gain=self.cfg.ref_clearance_rough_gain, stance_dx=self.cfg.stance_dx,
                                    jp_only=True, iters=12)            # FAST PATH: jp only, 12 IK iters (~6x cheaper)
            jp_ref = jp_ref[:, self.motion_dof_indexes]
            jdiff = self.robot.data.joint_pos - jp_ref                                    # (N,12)
            # TURN-GATE (0706 D3s): weight imitation by how much the command is a turn/strafe, so it anchors
            # foot placement exactly where AMP's forward-dense style field is blind, and stays ~0 for pure
            # forward (where AMP already covers it and the old global tracker over-constrained). yaw scaled by
            # ~foot radius (0.30) to compare with linear m/s. Pure fwd→0, pure yaw/lat→1.
            _turn_mag = self.commands[:, 1].abs() + 0.30 * self.commands[:, 2].abs()
            _fwd_mag = self.commands[:, 0].abs()
            turn_frac = _turn_mag / (_turn_mag + _fwd_mag + 1e-3)
            # RANK-2A (0707): a MODEST forward imitation FLOOR so pure-forward buckets get a trajectory anchor
            # for gait SHAPE (B1 touchdown vz / B2 slip / B4 duty symmetry) — the exact forward gait-quality
            # gates that fail when turn_frac→0 leaves forward unanchored (the mechanism behind the yaw-lesson:
            # the turn-gated anchor fixed TURNING placement but left FORWARD's foot fine-dynamics ungoverned).
            # The reference is speed-COVARIANT (period/stride scale with |cmd|), so it constrains gait shape,
            # NOT net speed → A1-forward (passing) is protected. Config-driven, default 0.0 = off (backward
            # compatible); set imitate_fwd_floor≈0.3 to enable. Kept modest to bound the A4 tradeoff risk.
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

        # SWING-DRAG penalty (framework hardening for heavier robots): a foot loaded (in contact) during its
        # commanded SWING phase = dragging/skating instead of stepping. physdiag swing_z: 39 kg Taili lifts
        # 9-19 cm, 75 kg B2 slides with feet on the ground (swing_z≈0). Dimensionless per-leg fraction [0,1];
        # default weight 0.0 → exact no-op for robots that already lift. A nonzero (negative) weight forces
        # every leg to actually leave the ground during its swing window. Gated to moving commands.
        # TERRAIN-GATED: swing_drag (collision-triggered lift) is OFF on flat (rough_gate=0) — else it penalizes
        # flat-gait timing imperfections and pushes a wasteful OVER-LIFT (physeval: 7cm vs the 4cm reference). It
        # engages only on rough/stairs (rough_gate->1), where hitting a riser SHOULD trigger a higher, adaptive lift.
        r_swing_drag = (self.cfg.rew_swing_drag * ((1.0 - desired_stance) * in_contact).mean(dim=1)
                        * mv_any.float() * rough_gate)

        # GAIT ENFORCEMENT (hard, both directions): penalize a foot OFF its commanded contact schedule —
        # loaded during swing (drag) OR airborne during stance (premature lift). r_gait reward is SOFT
        # (rew*gait_match) and asymptotes ~0.89-0.93; this hard penalty pushes gait_match toward 0.95 by
        # punishing every off-clock leg. Dimensionless per-leg [0,1]; default 0 = no-op. Relaxed on rough
        # terrain (like gait_match) so the policy can break the clock to climb.
        off_sched = (1.0 - desired_stance) * in_contact + desired_stance * (1.0 - in_contact)   # (N,4)
        _ge_relax = (1.0 - rough_gate) if (self.cfg.terrain_ctx_dim >= 3) else 1.0
        r_gait_enforce = self.cfg.rew_gait_enforce * off_sched.mean(dim=1) * mv_any.float() * _ge_relax

        # OFF-AXIS penalty: velocity PERPENDICULAR to the commanded direction (physdiag: fwd cmd -> vy 0.53).
        cmd_lin_norm = cmd_lin_mag.clamp(min=1e-4)
        cmd_hat = cmd_lin / cmd_lin_norm[:, None]
        v_along_s = (vel_lin * cmd_hat).sum(-1)                                          # signed speed along cmd
        v_perp = vel_lin - v_along_s[:, None] * cmd_hat
        r_offaxis = self.cfg.rew_offaxis_vel * torch.sum(v_perp * v_perp, dim=1) * mv_l.float()

        # OVERSHOOT penalty: speed BEYOND the commanded magnitude (physdiag: back cmd 0.4 -> vx 1.10). The
        # capped-progress reward gives no bonus past cmd but also no penalty; this is the missing brake.
        lin_over = torch.clamp(v_along_s - cmd_lin_mag, min=0.0)
        ang_over = torch.clamp(wz * torch.sign(cmd_ang) - cmd_ang.abs(), min=0.0)
        r_overshoot = self.cfg.rew_overshoot * (lin_over * lin_over * mv_l.float()
                                                + ang_over * ang_over * mv_a.float())
        # BANDED UNDERSPEED penalty: only punish falling below the A1 tolerance band, not every
        # sample below the command. The previous raw underspeed term pushed forward commands past
        # the target while trying to fix backward. Keeping the no-penalty band aligned with the
        # acceptance definition gives backward a gradient when it is truly too slow, while avoiding
        # an always-on "go faster" bias once tracking is already acceptable.
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
        # BACKWARD-ONLY auxiliary shaping: diagnostics showed generic underspeed can collapse forward
        # locomotion into a cautious standing solution. Backward is the weak axis; apply extra gradient
        # only to negative-vx commands and leave forward retention owned by capped progress + overshoot.
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
        # WRONG-DIRECTION penalty: a backward command solved by standing still or moving forward keeps
        # capped progress near zero, but previously had little explicit cost. Penalize signed velocity
        # opposite the requested axis so backward/yaw commands get a usable gradient away from shortcuts.
        lin_wrong = torch.clamp(-v_along_s, min=0.0)
        ang_wrong = torch.clamp(-(wz * torch.sign(cmd_ang)), min=0.0)
        r_wrong_dir = self.cfg.rew_wrong_dir * (lin_wrong * lin_wrong * mv_l.float()
                                                + ang_wrong * ang_wrong * mv_a.float())

        # STAND-POSTURE reward (only when commanded ~0): hold an upright, nominal-height, default-joint,
        # ALL-4-FEET-DOWN stance so it ACTIVELY returns to standing (physdiag: was balancing on 3 legs, FR up,
        # and recovering slowly). Four terms in [0,1]: uprightness, joints-near-default, height, feet-down.
        # body height — hoisted here so r_height (always-on) and r_stand both use it
        hgt = self.robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]

        # ALWAYS-ON BASE HEIGHT: during movement r_stand is fully gated off, so the policy has no incentive to
        # keep the body up -> rear legs collapse ("后腿卧前腿直立"). Linear penalty fires any time body drops
        # below stand_height regardless of command, preventing the degenerate rear-leg-lying strategy.
        r_height = self.cfg.rew_base_height * torch.clamp(self.cfg.stand_height - hgt, min=0.0)

        # STAND-POSTURE reward (only when commanded ~0): hold an upright, nominal-height, default-joint,
        # ALL-4-FEET-DOWN stance so it ACTIVELY returns to standing (physdiag: was balancing on 3 legs, FR up,
        # and recovering slowly). Four terms in [0,1]: uprightness, joints-near-default, height, feet-down.
        standing = (~mv_l) & (~mv_a)
        up = torch.clamp(-self.robot.data.projected_gravity_b[:, 2], 0.0, 1.0)                  # 1 = upright
        # jdev: WIDE Gaussian (sigma=4.0 instead of 0.5) so even a prone robot (large joint deviation) gets
        # nonzero gradient toward the default pose; exp(-sum/0.5) -> ~0 in prone = no recovery gradient.
        jdev = torch.exp(-torch.sum((self.robot.data.joint_pos - self.action_offset) ** 2, dim=1) / 4.0)
        # hdev: LINEAR (not sharp Gaussian) so there is always a gradient to rise from prone height to stand_height.
        hdev = torch.clamp(hgt / self.cfg.stand_height, 0.0, 1.0)
        feet_down = in_contact.mean(dim=1)                                                      # frac of 4 feet planted
        r_stand = self.cfg.rew_stand_pose * ((up + jdev + hdev + feet_down) / 4.0) * standing.float()
        # STAND STILLNESS: when commanded to stand, penalize joint velocity so the robot SETTLES smoothly into the
        # nominal "立正" stance (esp. on a sudden move->stop) instead of fidgeting. Refinement -> gated to phi>=1.
        r_stand_still = self.cfg.rew_stand_still * torch.sum(self.robot.data.joint_vel ** 2, dim=1) * standing.float()

        # ANTI-PIGEON-TOE: hold hips NEUTRAL (wide nominal stance) except when laterally commanded (lateral
        # legitimately needs abduction). Penalize sum(hip_joint^2), gated off as |vy| rises.
        lat_active = torch.clamp(self.commands[:, 1].abs() / self.cfg.hip_neutral_lat_scale, 0.0, 1.0)
        hip_dev = torch.sum(self.robot.data.joint_pos[:, self._hip_idx] ** 2, dim=1)
        r_hip = self.cfg.rew_hip_neutral * hip_dev * (1.0 - lat_active)

        # GAIT QUALITY REWARDS (directly fix physdiag slip=54-85% issue)
        foot_vel_w3 = self.robot.data.body_lin_vel_w[:, self.foot_indexes, :]  # (N,4,3)
        q = self.robot.data.root_quat_w[:, None, :].expand(-1, self.n_feet, -1).reshape(-1, 4)
        foot_vel_b = quat_apply_inverse(q, foot_vel_w3.reshape(-1, 3)).reshape(-1, self.n_feet, 3)
        foot_vel_b_xy = foot_vel_b[:, :, :2]                                  # (N,4,2), base frame

        # DESCENT-RELEASE window: clearance + swing-direction shaping apply only during the LIFT/APEX part of
        # swing, then taper to 0 over the final ~35% (the touchdown descent). Without this, clearance keeps
        # penalizing the foot for being below target as it descends to land (-> the policy lifts it back up
        # mid-descent = "略微上升") and swing_dir keeps pushing it sideways (-> "平移") — both stop the foot
        # from dropping straight to a clean plant. Releasing them lets the foot descend monotonically.
        s_swing = ((leg_phase - self.cfg.gait_duty) / (1.0 - self.cfg.gait_duty)).clamp(0.0, 1.0)  # (N,4)
        lift_w = torch.clamp((0.65 - s_swing) / 0.20, 0.0, 1.0)            # 1 in lift/apex, 0 by s_swing=0.65
        land_w = torch.clamp((s_swing - 0.70) / 0.30, 0.0, 1.0)           # 0 until touchdown phase, 1 at contact

        # 1. STANCE SLIP: penalize feet sliding horizontally during contact
        slip_speed = foot_vel_b_xy.norm(dim=-1)                            # (N,4), m/s
        r_stance_slip = self.cfg.rew_stance_slip * (in_contact * slip_speed).sum(dim=1) * mv_any.float()
        # mean stance-slip (m/s) exposed for the phi1->phi2 clean-gait gate
        slip_now_env = (in_contact * slip_speed).sum(dim=1) / in_contact.sum(dim=1).clamp(min=1.0)
        self._slip_now = float(slip_now_env.mean())

        # 2. SWING DIRECTION: swing foot should move along the commanded body-frame direction. For yaw-only
        # commands, use the tangential direction induced by rotating around the base.
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
        # PHASE-ANCHORED descent-release window. lift_w (descent-release) is a REFINEMENT (fixes the touchdown
        # "顿挫") that did NOT exist in the fresh-proven 245k. lift_gate fades it in via penalty_gate: in phi0 it is
        # 1.0 (PLAIN full-swing window == 245k), over phi1 it tapers to lift_w (descent-release active). Applied to
        # BOTH swing_dir and clearance (the two shaping terms 245k ran un-gated). A fresh policy barely lifts its
        # feet, so gating these to lift/apex-only (lift_w~0) at bootstrap starves them -> never explores stepping.
        lift_gate = (1.0 - self._penalty_gate) + self._penalty_gate * lift_w
        r_swing_dir = self.cfg.rew_swing_dir * (vel_along_cmd * swing_mask * lift_gate).mean(dim=1)

        # 2b. LANDING DECELERATION: in the touchdown phase (land_w), penalize the foot's HORIZONTAL speed so it
        # arrests its sideways/forward motion BEFORE planting -> clean plant, no drift ("平移") / landing slip.
        # Complements lift_w (lift+steer early; brake horizontal late).
        r_land_decel = self.cfg.rew_land_decel * (slip_speed * swing_mask * land_w).mean(dim=1)

        # 3. CLEARANCE: swing foot should clear the terrain under/near the foot, not just move relative to body.
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
        # ROUGHNESS-GATED phi2 clearance (per-env): engage heavy lift ONLY where the terrain is actually rough
        # (boxes/stairs/rough tiles); flat stays the efficient 245k base. gate cg = clr_gate(phi2 time fade) x
        # rough_factor(per-env terrain roughness). In phi0/phi1 clr_gate=0 -> cg=0 -> EXACTLY 245k (-1.5, 7cm).
        rough = self._terrain_ctx[:, 2]                                                  # (N,) terrain height std (m)
        rough_factor = torch.clamp((rough - self.cfg.clr_rough_flat) / self.cfg.clr_rough_span, 0.0, 1.0)  # (N,)
        # (B) DISCRETE OBSTACLES (stairs/boxes): FORCE rough_factor=1 regardless of measured roughness, so the
        # clearance target/weight ramps to full (target -> base+clr_rough_bonus_max ≈ 0.30m, weight -> -8). This
        # breaks the catch-22: a stuck-low stair tile reads low roughness -> low lift demand -> can't clear the
        # step -> never climbs. Demanding ~30cm lift lets the foot clear up to a 25cm step.
        if getattr(self.cfg, "discrete_clearance", False):
            self._ensure_gate_mask()
            rough_factor = torch.where(self._discrete_terrain_mask, torch.ones_like(rough_factor), rough_factor)
        cg = self._clearance_gate * rough_factor                                         # (N,) 0 on flat, ->1 on rough
        # weight: flat -1.5 (efficient) -> -8 on full-rough (forces foot high enough to clear obstacles)
        eff_clearance_w = self.cfg.rew_clearance + (self.cfg.rew_clearance_heavy - self.cfg.rew_clearance) * cg  # (N,)
        # target: base 0.07 + up to +6cm on full-rough (so it AIMS over the obstacle: 13cm clears L5 boxes/stairs)
        clearance_target = self.cfg.base_clearance + self.cfg.clr_rough_bonus_max * cg[:, None]  # (N,1)
        clr_deficit = torch.clamp(clearance_target - foot_clearance, min=0.0)
        # lift_gate: phi0 = full swing window (== 245k, plain); phi1+ tapers to lift/apex only (descent-release,
        # stops penalizing the natural touchdown descent = the "略微上升" hitch).
        r_clearance = eff_clearance_w * (swing_mask * lift_gate * clr_deficit).mean(dim=1)

        # PER-DIRECTION velocity curriculum progress — TERRAIN-GATED: exclude hard-type envs (slope_inv, boxes)
        # so they don't drag the prog that gates DR / velocity / phase (those types are a separate, capped axis).
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

        # Minimal terrain-throughput drive: tracking and clearance alone do not pay for committing upward over
        # stairs/boxes. Keep this narrow so flat gait is not pulled into wasteful hopping.
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
            # Soft slip weight: a hard cutoff at the target made climb reward disappear exactly when the
            # policy still needed gradient. Keep full credit below the target and taper to zero over ~0.18 m/s.
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
        # FULL reward-component breakdown (means) for the [ENV] log — so it's clear WHICH term dominates and
        # whether any task term is fighting the AMP style. _penalty_gate-scaled refinements shown post-gate.
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
            "stand_still": float(r_stand_still.mean()),   # P5: now ALWAYS-ON (no longer pg-gated) -> log un-gated
        }
        # 245k-PROVEN BASE penalties (ALWAYS ON from step 0): offaxis/overshoot/hip/slip/clearance(@-1.5) were ALL
        # ON in the fresh-bootstrapped 245k run -> they ARE the bootstrap gradient field (punish standing-shuffle,
        # wrong-direction drift, pigeon-toe, foot under-lift), NOT refinements. The phase system WRONGLY gated them
        # to 0 in phi0 (penalty_gate=0), deleting 245k's proven stack -> the policy got paid to stand perfectly
        # still (gait_match 0.5, upright 1.0, prog~0.1 for 8k+ steps). The leg-holding degeneration that motivated
        # the gate came from clearance -8 (HEAVY), not from penalties existing; clearance is back at 245k's -1.5.
        base_penalties = (
            r_offaxis + r_overshoot + r_underspeed + r_backward_aux + r_lateral_aux + r_wrong_dir
            + r_hip + r_stance_slip + r_clearance + r_swing_drag
        )
        # TRUE post-245k REFINEMENTS (fade in phi1+ on a walking policy): land_decel + gait_enforce.
        # r_gait_enforce is a REFINEMENT (push gait_match past 0.93 AFTER the gait forms) — pen-gated so it does
        # NOT fight the phi0 bootstrap (un-gated -2.0 punished the messy early gait → gait_match dropped to 0.61).
        # r_stand_still MOVED to ALWAYS-ON (P5): clean-stop/立正 is a CORE skill (physeval: 0.19 m/s creep, feet not
        # planted) → must train from phi0, not be a late refinement. Fires only when commanded ~0, so always-on is safe.
        refinement_penalties = r_land_decel + r_gait_enforce
        # r_torque (motor protection) is ALWAYS ON: it is ~0 in normal gait (only the near-saturation stress zone
        # is penalized), so it is bootstrap-safe and protects the (sim) motors from the first step.
        reward = (r_lin + r_ang + r_alive + r_arate + r_jacc + r_vz + r_wxy + r_air
                  + r_gait + r_imitate + r_height + r_stand + r_stand_still + r_swing_dir + r_torque
                  + r_climb + r_terrain_up + r_terrain_down
                  + base_penalties
                  + self._penalty_gate * refinement_penalties)
        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)   # never let a NaN reward poison PPO

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            # FALL = base too low ABOVE LOCAL TERRAIN, not absolute world z (0706). The old
            # `root_pos_w[:,2] < termination_height` used raw world height, so on any terrain whose ground
            # sits below the world origin the robot "died" every step even while perfectly upright: the
            # inverted-pyramid up-stairs pit (spawn z≈-1.55) terminated+reset on EVERY frame → a frozen
            # diagnostic + pose_stable=0; descending stairs/slopes hit the same false-fall. Use height above
            # the ground directly under the base (the canonical reward-side terrain height) so 0.35 means
            # "0.35 m of clearance", terrain-agnostic.
            if hasattr(self, "_terrain_height_under_base"):
                base_h_local = self.robot.data.root_pos_w[:, 2] - self._terrain_height_under_base()
            else:
                base_h_local = self.robot.data.root_pos_w[:, 2]
            died = base_h_local < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        # CRITICAL: terminate any env whose root state went NON-FINITE (physics blowup). The height check
        # above MISSES these: `NaN < termination_height` evaluates to False, so a blown-up env would NOT be
        # reset, its NaN state poisons the AMP/PPO update -> silent CUDA deadlock at a learning boundary.
        nonfinite = ~(torch.isfinite(self.robot.data.root_pos_w).all(dim=1)
                      & torch.isfinite(self.robot.data.root_lin_vel_b).all(dim=1)
                      & torch.isfinite(self.robot.data.root_ang_vel_b).all(dim=1))
        died = died | nonfinite
        return died, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        # Terrain curriculum (terrain_levels_vel) — grade by distance walked BEFORE repositioning:
        # walked > half a terrain tile -> harder; < half the commanded distance -> easier.
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
            # TERRAIN only advances in phase >= 2 AND while not regressed (the regression guard). Frozen at the
            # easiest level during bootstrap (phi 0) + gait-cleaning (phi 1) so those happen on near-flat ground.
            terrain_unlocked = (self._phase >= getattr(self, "_terrain_start_phase", 5)) and self._advance_ok
            height_gain = float(getattr(self.cfg, "terrain_curriculum_height_gain", 0.08))
            height_loss = float(getattr(self.cfg, "terrain_curriculum_height_loss", 0.08))
            controlled_up = (height_delta > height_gain) & (forward_dist > 0.25) & valid_episode
            controlled_down = (height_delta < -height_loss) & (forward_dist > 0.25) & valid_episode
            move_up = ((dist > self.cfg.terrain_move_up_dist) | controlled_up | controlled_down) & valid_episode & terrain_unlocked
            move_down = (
                (dist < cmd_mag * self.max_episode_length_s * 0.5)
                & ~controlled_up
                & ~controlled_down
                & ~move_up
                & valid_episode
            )
            self._terrain.update_env_origins(env_ids, move_up, move_down)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        # HISTORY: zero out for reset envs (fresh episode, no carryover)
        self._obs_history[env_ids] = 0.0
        self._delayed_action[env_ids] = 0.0
        self._in_contact[env_ids] = 0.0
        # DOMAIN RANDOMIZATION — level-gated. Params selected dynamically by level (supports 1/2/3).
        # Level 3 additionally randomizes friction, base CoM, and IMU bias (the comprehensive sim2real menu).
        n = len(env_ids)
        lvl = self._dr_level
        self._imu_bias[env_ids] = 0.0     # cleared here; re-drawn below only at level >= 3
        if self.cfg.dr_enable and lvl >= 1:
            mass_range = getattr(self.cfg, f"dr_mass_range_{lvl}")
            k_range    = getattr(self.cfg, f"dr_stiffness_scale_{lvl}")
            d_range    = getattr(self.cfg, f"dr_damping_scale_{lvl}")
            try:
                # PhysX mass is a CPU-pipeline property -> get_masses() returns a CPU tensor. The old code used a
                # GPU delta + GPU index on it -> device mismatch -> caught -> silently skipped (mass DR was FAKE).
                # Also it did "+= delta" on the CURRENT mass -> drift across resets. Fix: all-CPU, reset-to-default.
                masses = self.robot.root_physx_view.get_masses()                 # (N, n_bodies) CPU
                if self._default_masses is None:
                    self._default_masses = masses[:, 0].clone()                  # default root mass (CPU), once
                eids = env_ids.detach().cpu()
                delta = torch.empty(n).uniform_(*mass_range)                     # CPU, matches the CPU mass tensor
                masses[eids, 0] = self._default_masses[eids] + delta             # reset-to-default + delta (no drift)
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
            # ROOT-4 FIX (terrain-dr workflow 0707): friction/CoM/IMU are the deploy-critical channels for E3(μ)/E5.
            # They USED to be gated to lvl>=3 (a rung fresh/stall-restarted runs never reach -> E3/E5 got ZERO gradient,
            # violating spec §3.5 "DR always-on"). Now gated to a config knob dr_full_start_level (default 1) so they
            # train from the first DR level, RAMPING the envelope per level (dr_*_range_{lvl}) so a full μ=0.4 is not
            # dumped on a weak gait at level-1 entry.
            _dr_full_lvl = int(getattr(self.cfg, "dr_full_start_level", 1))
            if lvl >= _dr_full_lvl:
                # FRICTION + CoM: physx-view writes are EXPENSIVE (per-reset was 3.7x slower). Do them ONCE per LEVEL
                # -- re-fire on level-up (lvl > _dr_mat_level) to WIDEN the envelope, so still only ~3 writes per run.
                if lvl > self._dr_mat_level:
                    Nenv = self.num_envs
                    fr_range = getattr(self.cfg, f"dr_friction_range_{lvl}", self.cfg.dr_friction_range_3)
                    com_off  = float(getattr(self.cfg, f"dr_com_offset_{lvl}", self.cfg.dr_com_offset_3))
                    try:
                        mats = self.robot.root_physx_view.get_material_properties().clone()   # (N, n_shapes, 3)
                        fr = torch.zeros(Nenv, 1, 1, device=mats.device).uniform_(*fr_range)  # ABSOLUTE set -> no drift
                        mats[:, :, 0:2] = fr.expand(-1, mats.shape[1], 2)
                        self.robot.root_physx_view.set_material_properties(
                            mats, torch.arange(Nenv, device=mats.device))
                    except Exception:
                        if "fric" not in self._dr_warned:
                            print("[DR] friction API unavailable; skipping friction DR", flush=True); self._dr_warned.add("fric")
                    try:
                        coms = self.robot.root_physx_view.get_coms().clone()                 # (N, n_bodies, ...)
                        if self._default_coms is None:
                            self._default_coms = coms.clone()                                # capture default ONCE
                        coms = self._default_coms.clone()                                    # reset-to-default -> no drift on re-fire
                        off = torch.zeros(Nenv, 2, device=coms.device).uniform_(-com_off, com_off)
                        coms[:, 0, 0:2] += off
                        self.robot.root_physx_view.set_coms(coms, torch.arange(Nenv, device=coms.device))
                    except Exception:
                        if "com" not in self._dr_warned:
                            print("[DR] CoM API unavailable; skipping CoM DR", flush=True); self._dr_warned.add("com")
                    self._dr_mat_level = lvl
                    print(f"[DR] level-{lvl} friction {list(fr_range)} + CoM ±{com_off} randomized (all envs)", flush=True)
                # IMU BIAS stays per-episode (cheap, obs-level): constant gyro + gravity offset, per-level range
                gb = float(getattr(self.cfg, f"dr_imu_gyro_bias_{lvl}", self.cfg.dr_imu_gyro_bias_3))
                vb = float(getattr(self.cfg, f"dr_imu_grav_bias_{lvl}", self.cfg.dr_imu_grav_bias_3))
                self._imu_bias[env_ids, 0:3] = torch.zeros(n, 3, device=self.device).uniform_(-gb, gb)
                self._imu_bias[env_ids, 3:6] = torch.zeros(n, 3, device=self.device).uniform_(-vb, vb)
        num = len(env_ids)
        # reorder (user-confirmed): 1) sample command, 2) pick the matching clip (command + terrain),
        # 3) sample the initial pose FROM that clip -> reset pose is coherent with command + terrain.
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
        self._gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device)   # phase diversity across envs
        self._episode_start_xy[env_ids] = root[:, 0:2].detach()
        self._episode_start_root_z[env_ids] = root[:, 2].detach()
        self._episode_start_cmd_xy[env_ids] = self.commands[env_ids, :2].detach()

    def _push_strided_amp(self, amp):
        """AMP STRIDE WINDOW policy side: push the newest raw frame into the deep adjacent-frame ring, then
        export a stride-subsampled window [t, t-stride, t-2*stride, ...] into amp_observation_buffer so the
        discriminator sees ~one gait period. Called only when _amp_frame_stride > 1."""
        r = self._amp_raw_ring
        r[:, 1:] = r[:, :-1].clone()      # FIFO shift oldest-out (clone avoids in-place aliasing)
        r[:, 0] = amp
        stride = self._amp_frame_stride
        for j in range(self.cfg.num_amp_observations):
            self.amp_observation_buffer[:, j] = r[:, j * stride]

    def collect_reference_motions(self, num_samples, current_times=None):
        # PARAMETRIC reference: analytic flat trot computed CONTINUOUSLY from each sample's command (step
        # frequency + stride scale with the command, same speed-adaptive period as the env gait clock) — no
        # discrete clips, no coverage gaps. Validated to reproduce the clip kinematics to <3e-4 (parametric_ref).
        # Sampled from the live rollout commands + their terrain_ctx, so the conditional discriminator's
        # context channels (terrain_ctx, command) match the policy side by construction.
        K = self.cfg.num_amp_observations
        env_sel = torch.randint(0, self.num_envs, (num_samples,), device=self.device)
        cmd_s = self.commands[env_sel]                                             # (num_samples, 3)
        if current_times is None:
            current_times = np.random.uniform(0.0, 2.0, num_samples).astype(np.float32)
        _stride = getattr(self, "_amp_frame_stride", 1)                                       # AMP STRIDE WINDOW: space
        times = (np.expand_dims(current_times, -1) - (1.0 / 50.0) * _stride * np.arange(K)).flatten()  # ref frames by stride
        times_t = torch.as_tensor(times, dtype=torch.float32, device=self.device)
        cmd_rep = cmd_s.repeat_interleave(K, dim=0)                                # (num_samples*K, 3)
        # TERRAIN-AWARE reference: pass each sampled env's roughness so the reference lifts higher on rough
        # terrain -> the conditional discriminator rewards a high-lift climbing style (not flat shuffle).
        rough_rep = None
        if self.cfg.terrain_ctx_dim >= 3:
            rough_rep = self._terrain_ctx[env_sel, 2].repeat_interleave(K, dim=0)  # (num_samples*K,)
        jp_clip, jv_clip, bh, tn, foot_rel = flat_reference(
            cmd_rep, times_t, gait_period=self.cfg.gait_period,
            gait_period_slope=self.cfg.gait_period_slope, gait_period_min=self.cfg.gait_period_min,
            clearance_base=self.cfg.base_clearance,        # 0.07 (was default 0.09): lower FLAT foot lift -> the
                                                           # flat gait is less "high"/wasteful (user feedback); the
                                                           # roughness term still raises the lift on rough terrain.
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
