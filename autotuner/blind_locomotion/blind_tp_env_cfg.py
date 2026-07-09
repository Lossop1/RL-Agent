"""Blind-dog (TerrainPerceiver) env cfg — strict per taili_strategy_decisions.md.

Subclasses the existing AMP env cfg (reference) and only changes the OBSERVATION layout:
  blind actor obs = body53 + history[25,54]=1350  -> 1403
  + privileged(197 = lin_vel3 + height_scan187 + terrain_ctx3 + foot_contact4)
  + aux labels(22 = geom9 + geom_mask9 + risk2 + risk_mask2) -> total 1622
Everything else (sim, sensors, terrain, DR, AMP machinery) is inherited unchanged. Deployed as
part of the self-contained taili_blind_runtime package.
"""
from isaaclab.utils import configclass

from .taili_amp_env_cfg import TailiAmpEnvCfg


@configclass
class TailiBlindTPEnvCfg(TailiAmpEnvCfg):
    # Keep parent bookkeeping buffers aligned with the real blind runtime layout.
    obs_history_len = 25
    obs_history_dim = 54
    # observation_space = body53 + history(25*54=1350) + privileged(197) + aux labels(22)
    #   aux labels = geom9 + geom_mask9 + risk2 + risk_mask2 (training-only, for the perceiver aux hook)
    observation_space = 53 + 25 * 54 + (3 + 17 * 11 + 3 + 4) + 22   # = 1622
    # AMP discriminator: frame51 = motion43 + command3 + mode_onehot5 (D); two-frame -> 102.
    # num_amp_observations is inherited from TailiAmpEnvCfg and may be overridden by env.amp.frames.
    amp_observation_space = 43 + 3 + 5                          # = 51 (terrain_ctx dropped)
