"""Taili blind locomotion environment configuration.

Subclasses TailiAmpEnvCfg with the coordinated redesign (replaces the over-coupled hand-tuned web):
  #2 HARD gait enforcement   rew_gait_enforce -2.0  → push gait_match past ~0.93 soft ceiling toward 0.95
  #3 wider fore-aft stance    refs regenerated with STANCE_DX 0.05 (wheelbase 0.61→0.71 m, pitch-stable)
  #4 DR-defer                 dr_unlock_terrain 2.0  → don't stack DR until terrain footing (reduce phi2 load)
  #5 clearance SCALES w/ roughness  clr_rough_flat/span widened so an 8 cm stair asks ~10 cm lift (not 28 cm),
                                    a 25 cm obstacle asks ~30 cm — fixes "stair=0" (was: any rough → demand 28 cm)
  #7 stairs moderate          stairs proportion 25%→12% (practiced, not dominating), step to 0.28 (25 cm+)
  + flat 8 cm lift, speed 1.5, DR mass -5..+20.

The deployed actor is blind: privileged terrain data is used only as a training target for the terrain
perceiver and is never part of the actor input.
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
    # #2 RESTORED -2.0 (AMP-CORE audit): r_gait_enforce is the SYMMETRIC contact-mismatch penalty (penalize a foot
    # planted during its swing phase OR airborne during stance). It is the RIGHT lever to push gait_match past the
    # raw-r_gait ceiling WITHOUT the standing-floor trap (raw r_gait pays a planted policy 0.5*weight; gait_enforce
    # PENALIZES that same standing). And it is COHERENT with AMP, not against it: the gait clock is verified
    # identical to the AMP reference (trot offsets/duty/period), so enforcing contact==clock == enforcing the
    # reference's stance/swing pattern. pen-gated (phi1+ fade-in) so it never punishes the phi0 bootstrap; rough_gate
    # relaxes it on stairs (AMP↔terrain decoupling valve). (My earlier 0.0 was a wrong call — the clock IS the ref.)
    rew_gait_enforce = -2.0
    # flat lift ~8 cm (user 2026-06-26). Sets BOTH the reference clearance_base (AMP shows an 8cm-lift style) and
    # the flat rew_clearance target -> the foot aims for ~8cm above ground on flat (rough/stairs add bonus on top).
    base_clearance = 0.08
    # #3 wider fore-aft stance now lives IN the reference (flat_reference stance_dx) so discriminator target ==
    # the RSI clips (both widened). Was baked into regenerated clips only; now consistent on both sides.
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
