"""Early-stop rules — kill a trial before it burns its budget on a hopeless config.

Saves ~30% wall time on rung-0 by aborting NaN/stuck/short-episode trials early.

Rules (evaluated against the rolling sequence of (iter, snapshot) tuples):
  R1 — NaN in Mean reward → abort immediately, status=early_killed, log_score=0
  R2 — terrain_levels == 0 for ≥50 iter after iter 80 → abort (curriculum stuck)
  R3 — mean_episode_length < 100 at iter ≥ 150 → abort (locomotion not formed)

Each rule is pure-function: input list of (iter, snapshot_dict) → bool, reason.
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple


class EarlyStopReason:
    NAN_REWARD = "nan_reward"
    TERRAIN_STUCK = "terrain_stuck"
    LOCOMOTION_NOT_FORMED = "locomotion_not_formed"


_FIELD_REWARD = "Mean reward"
_FIELD_TERRAIN = "Curriculum/terrain_levels"
_FIELD_LENGTH = "Mean episode length"


def _isnan(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float):
        return math.isnan(v)
    if isinstance(v, str) and v.lower() in ("nan", "inf", "-inf"):
        return True
    return False


def check_nan_reward(snapshots: List[Tuple[int, dict]]) -> Optional[str]:
    """R1 — return reason if any snapshot's mean_reward is NaN. """
    for it, snap in snapshots:
        v = snap.get(_FIELD_REWARD)
        if _isnan(v):
            return f"{EarlyStopReason.NAN_REWARD}: NaN at iter {it}"
    return None


def check_terrain_stuck(snapshots: List[Tuple[int, dict]],
                         min_iter_threshold: int = 80,
                         stuck_window: int = 50) -> Optional[str]:
    """R2 — terrain_levels stuck at 0 for too long.

    Triggers if:
      - latest iter >= min_iter_threshold + stuck_window
      - all snapshots with iter >= min_iter_threshold have terrain_levels == 0
    """
    if not snapshots:
        return None
    latest_iter = snapshots[-1][0]
    if latest_iter < min_iter_threshold + stuck_window:
        return None
    relevant = [
        (it, snap) for it, snap in snapshots
        if it >= min_iter_threshold and snap.get(_FIELD_TERRAIN) is not None
    ]
    if len(relevant) < 3:
        return None
    if all(float(snap.get(_FIELD_TERRAIN, 0) or 0) <= 0.01 for _, snap in relevant):
        return (
            f"{EarlyStopReason.TERRAIN_STUCK}: terrain_levels<=0.01 "
            f"for {len(relevant)} snaps after iter {min_iter_threshold}"
        )
    return None


def check_short_episode(snapshots: List[Tuple[int, dict]],
                         min_iter_threshold: int = 150,
                         length_floor: int = 100) -> Optional[str]:
    """R3 — episode length too short (locomotion not formed).

    Triggers if latest iter >= min_iter_threshold and mean_episode_length < length_floor.
    """
    if not snapshots:
        return None
    latest_iter, latest = snapshots[-1]
    if latest_iter < min_iter_threshold:
        return None
    length = latest.get(_FIELD_LENGTH)
    if length is None:
        return None
    try:
        length = float(length)
    except (TypeError, ValueError):
        return None
    if length < length_floor:
        return (
            f"{EarlyStopReason.LOCOMOTION_NOT_FORMED}: "
            f"mean_episode_length={length:.1f} < {length_floor} at iter {latest_iter}"
        )
    return None


class EarlyStopChecker:
    """Aggregates all rules. Stateless: pass full snapshot history each check."""

    def __init__(self,
                 nan_check: bool = True,
                 terrain_check: bool = True,
                 short_episode_check: bool = True):
        self.checks = []
        if nan_check:
            self.checks.append(check_nan_reward)
        if terrain_check:
            self.checks.append(check_terrain_stuck)
        if short_episode_check:
            self.checks.append(check_short_episode)

    def should_stop(self, snapshots: List[Tuple[int, dict]]) -> Optional[str]:
        """Return the first triggered reason, or None."""
        for rule in self.checks:
            reason = rule(snapshots)
            if reason:
                return reason
        return None
