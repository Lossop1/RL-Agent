"""Taili 盲行走环境配置。

环境保留统一的步态、净空、课程和域随机化契约；部署 actor 只接收本体可测信息。
特权地形信息只用于训练感知器监督，不能进入 actor 输入。
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

from isaaclab.utils import configclass
from .blind_tp_env_cfg import TailiBlindTPEnvCfg
from .taili_blind_config import apply_env_config_to_cfg

BLIND_MOTIONS_DIR = os.environ.get("TAILI_MOTIONS_DIR", str(Path(__file__).resolve().parent / "motions"))
_GAITS = ["fwd_030", "fwd_060", "fwd_090", "fwd_120", "fwd_160", "fwd_200",
          "back_030", "back_060", "left_025", "left_045", "right_025", "right_045",
          "yawl_040", "yawl_080", "yawr_040", "yawr_080",
          "uphill_fwd", "downhill_fwd", "uphill_yawl", "uphill_yawr", "cross_fwd", "cross_back"]


@configclass
class TailiBlindEnvCfg(TailiBlindTPEnvCfg):
    # 对称惩罚接触时序偏差：摆动期落地或支撑期离地都会扣分。
    # 惩罚从 phi1 以后渐入，避免阻塞初始探索；楼梯上按 rough gate 放宽，给地形响应留出空间。
    rew_gait_enforce = -2.0
    # 平地参考净空约 8cm；粗糙和楼梯再根据障碍高度增加要求。
    base_clearance = 0.08
    # 前后站距同时写入参考片段和运行时裁剪，保证判别器目标与训练片段一致。
    stance_dx = 0.05
    # HARD (structural) symmetry lives in the actor: TerrainPerceiverPolicy is constructed equivariant
    # (mean = 1/2[net(x)+M_act net(M_in x)], z=[u(h),u(M_hist h)]) -> pi(Mx)=M pi(x) BY CONSTRUCTION, walk AND
    # stand (decision D; unit-tested 0 error). The soft mirror-AUGMENTATION is therefore redundant -> OFF (it only
    # doubled the PPO batch). physeval had shown soft aug fixed walking but NOT the asymmetric stand; structural
    # equivariance fixes both.
    sym_augment = False
    # COLLISION-TRIGGERED LIFT (stairs, spec 4 + 3): penalize a foot in contact during its SWING phase = it hit a
    # riser/obstacle. With 3-5cm nominal lift, this + the terrain latent push an ADAPTIVE higher lift only where
    # the foot actually hits something -> climb without a fixed high lift everywhere. ~0 on flat (feet lift clean).
    rew_swing_drag = -0.5
    # #5 clearance scales with ACTUAL roughness (was: flat 0.008/span 0.012 → saturates at 2 cm height-std →
    #    demands 28 cm on any stair). Widen so small bumps ask small lift, tall obstacles ask tall lift.
    clr_rough_flat = 0.01
    clr_rough_span = 0.12
    clr_rough_bonus_max = 0.26          # full-rough target = 0.04 + 0.26 = 0.30 m → clears 25 cm + margin
    # speed ceiling 1.5
    cmd_fwd_max = 1.5
    # #4 DR-defer + stronger DR (mass -5..+20)
    dr_unlock_terrain = 2.0
    dr_mass_range_1 = (-2.0, 4.0)
    dr_mass_range_2 = (-4.0, 12.0)
    dr_mass_range_3 = (-5.0, 20.0)
    dr_push_vel_3 = 1.0
    dr_friction_range_3 = (0.4, 1.4)
    dr_com_offset_3 = 0.05
    dr_stiffness_scale_3 = (0.6, 1.4)
    dr_damping_scale_3 = (0.5, 1.5)

    # references: 4 cm lift + wider stance (STANCE_DX 0.05)
    motion_files = [os.path.join(BLIND_MOTIONS_DIR, "clips", f"taili_{g}.npz") for g in _GAITS]

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Isolate terrain from the shared Taili object before applying edits from the single config.
        self.terrain = copy.deepcopy(self.terrain)
        apply_env_config_to_cfg(self)
