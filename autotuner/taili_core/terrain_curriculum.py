"""Pure terrain-curriculum decisions for Taili training.

The simulator owns terrain generation and reset, but the success/failure math
is kept here so it can be tested without IsaacLab.
"""
from __future__ import annotations

from typing import Any

import torch


def compute_terrain_curriculum_moves(
    *,
    dist: torch.Tensor,
    cmd_mag: torch.Tensor,
    forward_dist: torch.Tensor,
    height_delta: torch.Tensor,
    valid_episode: torch.Tensor,
    terrain_curriculum_active: bool,
    terrain_unlocked: bool,
    terminal_now: torch.Tensor,
    base_h_local: torch.Tensor,
    upright_score: torch.Tensor,
    contact_count: torch.Tensor,
    body_wxy: torch.Tensor,
    v_along: torch.Tensor,
    max_episode_length_s: float,
    terrain_move_up_dist: float,
    height_gain: float = 0.08,
    height_loss: float = 0.08,
    forward_min: float = 0.25,
    stable_h: float = 0.42,
    stable_upright: float = 0.85,
    stable_contact_min: float = 2.0,
    stable_wxy_max: float = 1.50,
    speed_cap_ratio: float = 1.60,
    speed_cap_min: float = 0.75,
    failure_h: float = 0.42,
    failure_upright: float = 0.75,
    failure_wxy: float = 2.00,
    eligible_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return terrain curriculum move decisions.

    `move_up` means a stable traversal happened. A height change alone is not
    success: it must be paired with a non-terminal, upright, supported and
    speed-controlled final state. Catastrophic contact/fall becomes an immediate
    `failure_down`, so rushing down stairs cannot masquerade as terrain skill.
    """
    if not terrain_curriculum_active:
        zeros = torch.zeros_like(valid_episode, dtype=torch.bool)
        return {
            "move_up": zeros,
            "low_progress_down": zeros,
            "failure_down": zeros,
            "move_down": zeros,
            "controlled_up": zeros,
            "controlled_down": zeros,
            "distance_success": zeros,
            "stable_end": zeros,
            "speed_controlled": zeros,
            "catastrophic": zeros,
        }

    valid = valid_episode.bool()
    if eligible_mask is not None:
        valid = valid & eligible_mask.bool()
    terminal = terminal_now.bool()
    stable_end = (
        valid
        & ~terminal
        & (base_h_local >= float(stable_h))
        & (upright_score >= float(stable_upright))
        & (contact_count >= float(stable_contact_min))
        & (body_wxy <= float(stable_wxy_max))
    )

    speed_limit = torch.maximum(
        cmd_mag * float(speed_cap_ratio),
        torch.full_like(cmd_mag, float(speed_cap_min)),
    )
    speed_controlled = v_along.abs() <= speed_limit
    forward_ok = forward_dist > float(forward_min)

    controlled_up = (
        (height_delta > float(height_gain))
        & forward_ok
        & stable_end
        & speed_controlled
    )
    controlled_down = (
        (height_delta < -float(height_loss))
        & forward_ok
        & stable_end
        & speed_controlled
    )
    distance_success = (
        (dist > float(terrain_move_up_dist))
        & stable_end
        & speed_controlled
    )
    unlocked = bool(terrain_unlocked)
    move_up = (distance_success | controlled_up | controlled_down) & valid & unlocked

    catastrophic = (
        terminal
        | (base_h_local < float(failure_h))
        | (upright_score < float(failure_upright))
        | (body_wxy > float(failure_wxy))
    ) & valid
    failure_down = catastrophic

    expected_dist = cmd_mag * float(max_episode_length_s) * 0.5
    low_progress_down = (
        (dist < expected_dist)
        & ~controlled_up
        & ~controlled_down
        & ~move_up
        & valid
    )
    move_down = (low_progress_down | failure_down) & ~move_up

    return {
        "move_up": move_up,
        "low_progress_down": low_progress_down,
        "failure_down": failure_down,
        "move_down": move_down,
        "controlled_up": controlled_up,
        "controlled_down": controlled_down,
        "distance_success": distance_success,
        "stable_end": stable_end,
        "speed_controlled": speed_controlled,
        "catastrophic": catastrophic,
    }
