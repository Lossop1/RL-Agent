from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .schema import DiagnosticThresholds, LEGS
from .events import (
    add_derived_event_columns,
    command_settled_mask,
    pose_stable_mask,
    before_first_major_event_mask,
    after_first_major_event_mask,
)
from .util import bool_series, group_key_columns, numeric_col


@dataclass
class SliceBundle:
    masks: dict[str, pd.Series]
    description: dict[str, str]


def build_slices(df: pd.DataFrame, thresholds: DiagnosticThresholds | None = None) -> SliceBundle:
    thresholds = thresholds or DiagnosticThresholds()
    d = add_derived_event_columns(df, thresholds)
    all_frames = pd.Series(True, index=d.index)
    command_settled = command_settled_mask(d, thresholds)
    pose_stable = pose_stable_mask(d, thresholds)
    before_major = before_first_major_event_mask(d, thresholds)
    after_major = after_first_major_event_mask(d, thresholds)

    masks: dict[str, pd.Series] = {
        "all_frames": all_frames,
        "command_settled": command_settled,
        "pose_stable": pose_stable,
        "pose_stable_command_settled": pose_stable & command_settled,
        "before_first_major_event": before_major,
        "pose_stable_before_first_major_event": pose_stable & before_major,
        "after_first_major_event": after_major,
    }

    # Event-window slices around any major event.
    event_window = thresholds.event_window_s
    if "time" in d.columns:
        t = numeric_col(d, "time", 0.0)
        pre = pd.Series(False, index=d.index)
        post = pd.Series(False, index=d.index)
        group_cols = group_key_columns(d)
        groups = d.groupby(group_cols, dropna=False, sort=False) if group_cols else [(None, d)]
        for _, g in groups:
            major = bool_series(g, "major_event_observed")
            if not major.any():
                continue
            event_times = numeric_col(g[major], "time", 0.0).to_numpy()
            for et in event_times:
                tg = numeric_col(g, "time", 0.0)
                pre.loc[g.index] |= ((tg >= et - event_window) & (tg < et)).fillna(False)
                post.loc[g.index] |= ((tg > et) & (tg <= et + event_window)).fillna(False)
        masks["pre_major_event_0p5s"] = pre
        masks["post_major_event_0p5s"] = post

    # Contact slices.
    stance_any = pd.Series(False, index=d.index)
    swing_any = pd.Series(False, index=d.index)
    for leg in LEGS:
        c = bool_series(d, f"foot_{leg}_contact")
        if f"foot_{leg}_contact" in d.columns:
            stance_any |= c
            swing_any |= ~c
            masks[f"stance_{leg}"] = c
            masks[f"swing_{leg}"] = ~c
    masks["stance_any"] = stance_any
    masks["swing_any"] = swing_any

    description = {
        "all_frames": "all recorded frames",
        "command_settled": f"frames at least {thresholds.command_settle_s}s after command segment start",
        "pose_stable": "frames without large roll/pitch, height drop, reset/done observation",
        "pose_stable_command_settled": "intersection of command_settled and pose_stable",
        "before_first_major_event": "frames before first major event in each segment",
        "pose_stable_before_first_major_event": "pose-stable frames before first major event",
        "after_first_major_event": "frames after first major event in each segment",
        "pre_major_event_0p5s": "0.5s before any major event",
        "post_major_event_0p5s": "0.5s after any major event",
        "stance_any": "frames where at least one foot is in contact",
        "swing_any": "frames where at least one foot is not in contact",
    }
    return SliceBundle(masks=masks, description=description)
