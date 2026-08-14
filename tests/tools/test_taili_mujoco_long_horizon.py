"""MuJoCo 部署观测契约测试。"""

import numpy as np

from tools.taili_mujoco_long_horizon import _mask_standing_gait_clock


def test_standing_gait_clock_mask_preserves_motion_inputs() -> None:
    commands = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.09, 0.0, 0.04],
            [0.1, 0.0, 0.05],
            [0.1001, 0.0, 0.0],
            [-0.1001, 0.0, 0.0],
            [0.0, 0.1001, 0.0],
            [0.0, 0.0, -0.0501],
        ],
        dtype=np.float32,
    )
    clocks = np.arange(56, dtype=np.float32).reshape(7, 8) / 57.0

    masked = np.stack([
        _mask_standing_gait_clock(command, clock)
        for command, clock in zip(commands, clocks, strict=True)
    ])

    np.testing.assert_array_equal(masked[:3], np.zeros((3, 8), dtype=np.float32))
    np.testing.assert_array_equal(masked[3:], clocks[3:])
