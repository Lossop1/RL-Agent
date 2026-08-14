"""Taili 机器人资产的几何常量。

这些数值描述 URDF 碰撞几何和默认关节姿态，不是可调奖励参数。训练、诊断、
导出和物理评估必须共同使用本文件，避免把足端 link 原点误当成足底接触点。
"""
from __future__ import annotations


# URDF 中四个 foot link 的碰撞球半径。
FOOT_RADIUS = 0.042

# q_default=(hip=0, thigh=0.7, calf=-1.4) 时，base 到足端球心的竖直距离。
DEFAULT_FOOT_CENTER_HEIGHT = 0.5051961541175842

# 默认姿态下足底恰好接触平地时的 base 高度。
NOMINAL_BASE_HEIGHT = DEFAULT_FOOT_CENTER_HEIGHT + FOOT_RADIUS


def sole_clearance(foot_center_height, terrain_height, foot_radius: float = FOOT_RADIUS):
    """返回足底相对局部地面的净空，适用于标量、NumPy 和 Torch 张量。"""
    return foot_center_height - terrain_height - float(foot_radius)
