"""Multi-clip reference loader with condition-matched soft sampling.

Wraps N MotionLoaders sharing the same dof/body layout. Each clip carries a condition:
  - cond_command: the commanded velocity (vx, vy, wz)
  - cond_slope:   the clip's terrain context. Old clips may store only
                  (slope_x, slope_y); those are padded with roughness=0.
When the env collects reference motions for a batch of rollout conditions (command + terrain_ctx),
each sample picks a clip by
    w_i = softmax( -||cmd - clip_cmd_i||^2/sigma  - ||tctx - clip_slope_i||^2/slope_sigma )
so the reference-pool condition distribution tracks the policy rollout condition distribution
(skrl AMP trains the discriminator on a random sample of this pool -> distribution-level match).
"""
from __future__ import annotations

import numpy as np
import torch

from .motions import MotionLoader


class MultiMotionLoader:
    def __init__(self, motion_files, device, sigma: float = 0.3, slope_sigma: float = 0.1,
                 cmd_scale=(1.0, 0.6, 1.0)):
        self.device = device
        self.loaders = [MotionLoader(motion_file=f, device=device) for f in motion_files]
        self.num_clips = len(self.loaders)
        ref = self.loaders[0]
        for L in self.loaders[1:]:
            assert L.dof_names == ref.dof_names, "clip dof layout mismatch"
            assert L.body_names == ref.body_names, "clip body layout mismatch"
        self.dt = ref.dt
        # per-clip conditions (npz cond_command / cond_slope); fallback zeros
        cmds, slopes = [], []
        for f in motion_files:
            d = np.load(f)
            cmds.append(np.asarray(d["cond_command"], np.float32) if "cond_command" in d
                        else np.zeros(3, np.float32))
            slope = np.asarray(d["cond_slope"], np.float32) if "cond_slope" in d else np.zeros(2, np.float32)
            slope = np.ravel(slope).astype(np.float32)
            if slope.shape[0] < 3:
                slope = np.pad(slope, (0, 3 - slope.shape[0]), constant_values=0.0)
            elif slope.shape[0] > 3:
                slope = slope[:3]
            slopes.append(slope)
        self.clip_cmd = torch.tensor(np.stack(cmds), dtype=torch.float32, device=device)    # (C, 3)
        self.clip_slope = torch.tensor(np.stack(slopes), dtype=torch.float32, device=device)  # (C, 3)
        self.sigma = float(sigma)
        self.slope_sigma = float(slope_sigma)
        self.cmd_scale = torch.tensor(cmd_scale, dtype=torch.float32, device=device)         # (3,)
        self.durations = np.asarray([float(loader.duration) for loader in self.loaders], dtype=np.float32)
        self._gait_phase_maps = None

    # ---- delegation (all clips share layout / duration) ----
    def get_dof_index(self, names):
        return self.loaders[0].get_dof_index(names)

    def get_body_index(self, names):
        return self.loaders[0].get_body_index(names)

    def sample_times(self, n, clip_ids=None):
        """按被选 clip 的真实时长采样，避免短 clip 被夹在最后一帧。"""
        if clip_ids is None:
            return self.loaders[0].sample_times(n)
        cid = clip_ids.detach().cpu().numpy() if torch.is_tensor(clip_ids) else np.asarray(clip_ids)
        if len(cid) != int(n):
            raise ValueError(f"clip_ids 长度 {len(cid)} 与采样数 {n} 不一致")
        return self.durations[cid] * np.random.uniform(low=0.0, high=1.0, size=int(n))

    def configure_gait_phase_maps(
        self,
        reference_fn,
        *,
        gait_period: float,
        gait_period_slope: float,
        gait_period_min: float,
        yaw_speed_equiv: float,
        clearance_base: float,
        clearance_rough_gain: float,
        stance_dx: float,
        grid_size: int = 128,
    ):
        """预计算每个 clip 帧在生产解析参考中的最相近 gait phase。

        reset 仍然随机采样 clip 时间，因此保留相位多样性；区别只是 gait clock
        不再与刚写入仿真的关节姿态相互矛盾。映射只在环境初始化时计算一次。
        """
        grid_size = max(int(grid_size), 16)
        phases = torch.arange(grid_size, dtype=torch.float32, device=self.device) / float(grid_size)
        maps = []
        with torch.no_grad():
            for clip_index, loader in enumerate(self.loaders):
                command = self.clip_cmd[clip_index].unsqueeze(0).expand(grid_size, -1)
                speed = torch.linalg.norm(command[0, :2]) + float(yaw_speed_equiv) * command[0, 2].abs()
                period = torch.clamp(
                    torch.as_tensor(float(gait_period), dtype=torch.float32, device=self.device)
                    - float(gait_period_slope) * speed,
                    min=float(gait_period_min),
                    max=float(gait_period),
                )
                roughness = self.clip_slope[clip_index, 2].expand(grid_size)
                reference = reference_fn(
                    command,
                    phases * period,
                    gait_period=float(gait_period),
                    gait_period_slope=float(gait_period_slope),
                    gait_period_min=float(gait_period_min),
                    yaw_speed_equiv=float(yaw_speed_equiv),
                    clearance_base=float(clearance_base),
                    roughness=roughness,
                    clearance_rough_gain=float(clearance_rough_gain),
                    stance_dx=float(stance_dx),
                    jp_only=True,
                    iters=12,
                )
                mse = (loader.dof_positions[:, None, :] - reference[None, :, :]).square().mean(dim=-1)
                maps.append(phases[mse.argmin(dim=1)].detach())
        self._gait_phase_maps = maps

    def sample_gait_phases(self, clip_ids, times):
        """按 clip 时间插值预计算相位，使用圆周插值正确处理 1 -> 0 回绕。"""
        if self._gait_phase_maps is None:
            raise RuntimeError("gait phase maps 尚未配置")
        cid = clip_ids.detach().cpu().numpy() if torch.is_tensor(clip_ids) else np.asarray(clip_ids)
        tm = times.detach().cpu().numpy() if torch.is_tensor(times) else np.asarray(times)
        result = torch.zeros(len(cid), dtype=torch.float32, device=self.device)
        for clip_index, loader in enumerate(self.loaders):
            selected = np.where(cid == clip_index)[0]
            if len(selected) == 0:
                continue
            index_0, index_1, blend = loader._compute_frame_blend(tm[selected])
            index_0_t = torch.as_tensor(index_0, dtype=torch.long, device=self.device)
            index_1_t = torch.as_tensor(index_1, dtype=torch.long, device=self.device)
            blend_t = torch.as_tensor(blend, dtype=torch.float32, device=self.device)
            phase_map = self._gait_phase_maps[clip_index]
            angle_0 = 2.0 * torch.pi * phase_map[index_0_t]
            angle_1 = 2.0 * torch.pi * phase_map[index_1_t]
            x = (1.0 - blend_t) * torch.cos(angle_0) + blend_t * torch.cos(angle_1)
            y = (1.0 - blend_t) * torch.sin(angle_0) + blend_t * torch.sin(angle_1)
            phase = torch.remainder(torch.atan2(y, x) / (2.0 * torch.pi), 1.0)
            result[torch.as_tensor(selected, dtype=torch.long, device=self.device)] = phase
        return result

    def clip_terrain_ctx(self, clip_ids):
        """per-sample terrain context for the given clip ids -> (M, 3)."""
        return self.clip_slope[clip_ids]

    def clip_command_ctx(self, clip_ids):
        """per-sample command context for the given clip ids -> (M, 3)."""
        return self.clip_cmd[clip_ids]

    # ---- condition-matched clip selection ----
    def pick_clips(self, commands: torch.Tensor, terrain_ctx: torch.Tensor | None = None) -> torch.Tensor:
        """(commands (n,3), terrain_ctx (n,2/3)) -> clip_ids (n,) by softmax over command + terrain distance."""
        # GUARD: sanitize inputs — a non-finite command/terrain_ctx (from a blown-up env upstream) would make
        # d2 -> softmax -> NaN weights, and torch.multinomial asserts ("probability tensor contains nan") and
        # crashes the whole training with a device-side assert. nan_to_num here is the last line of defense.
        commands = torch.nan_to_num(commands, nan=0.0, posinf=0.0, neginf=0.0)
        c = (commands / self.cmd_scale)[:, None, :]              # (n,1,3)
        cc = (self.clip_cmd / self.cmd_scale)[None, :, :]        # (1,C,3)
        d2 = ((c - cc) ** 2).sum(-1) / self.sigma                # (n, C)
        if terrain_ctx is not None:
            terrain_ctx = torch.nan_to_num(terrain_ctx, nan=0.0, posinf=0.0, neginf=0.0)
            k = min(terrain_ctx.shape[-1], self.clip_slope.shape[-1])
            ts = terrain_ctx[:, None, :k]                         # (n,1,k)
            cs = self.clip_slope[None, :, :k]                     # (1,C,k)
            d2 = d2 + ((ts - cs) ** 2).sum(-1) / self.slope_sigma
        w = torch.softmax(-d2, dim=-1)                          # (n, C)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        w = w + 1e-8                                            # ensure no all-zero row -> multinomial valid
        return torch.multinomial(w, 1).squeeze(-1)              # (n,)

    def sample_frames(self, clip_ids, times):
        """clip_ids (M,), times (M,) local times -> 6 tensors (M, ...). Frame i from clip clip_ids[i]."""
        cid = clip_ids.detach().cpu().numpy() if torch.is_tensor(clip_ids) else np.asarray(clip_ids)
        tm = times.detach().cpu().numpy() if torch.is_tensor(times) else np.asarray(times)
        M = len(cid)
        proto = self.loaders[0].sample(num_samples=1, times=np.array([0.0]))
        bufs = [torch.zeros((M,) + tuple(p.shape[1:]), dtype=p.dtype, device=self.device) for p in proto]
        for c in range(self.num_clips):
            idx = np.where(cid == c)[0]
            if len(idx) == 0:
                continue
            res = self.loaders[c].sample(num_samples=len(idx), times=tm[idx])
            for b, r in zip(bufs, res):
                b[idx] = r
        return tuple(bufs)
