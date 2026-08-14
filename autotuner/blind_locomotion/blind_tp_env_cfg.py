"""Blind-dog (TerrainPerceiver) env cfg — strict per taili_strategy_decisions.md.

Subclasses the existing AMP env cfg (reference) and only changes the OBSERVATION layout:
  blind actor obs = body57 + history[25,54]=1350  -> 1407
  + privileged(197 = lin_vel3 + height_scan187 + terrain_ctx3 + foot_contact4)
  + aux labels(34 = geom9 + geom_mask9 + risk8 + risk_mask8) -> total 1638
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
    # Fresh策略按因果顺序读取约1秒本体历史；旧检查点不得加载此观测契约。
    obs_history_stride = 2
    obs_history_order = "oldest_first"
    # observation_space = body57 + history(25*54=1350) + privileged(197) + aux labels(34)
    #   aux labels = geom9 + geom_mask9 + risk8 + risk_mask8，仅供训练期感知器辅助损失。
    observation_space = 57 + 25 * 54 + (3 + 17 * 11 + 3 + 4) + 34   # = 1638
    # AMP discriminator: frame51 = motion43 + command3 + mode_onehot5；帧数和步长由 YAML 管理。
    # num_amp_observations is inherited from TailiAmpEnvCfg and may be overridden by env.amp.frames.
    amp_observation_space = 43 + 3 + 5                          # = 51 (terrain_ctx dropped)
