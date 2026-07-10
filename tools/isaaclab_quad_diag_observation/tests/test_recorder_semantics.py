from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from isaaclab_quad_diag.events import extract_event_timeline, segment_event_summary
from isaaclab_quad_diag.record import (
    apply_terrain_case_best_effort,
    collect_terminal_rows,
    contact_transition_flags,
    install_terminal_capture_hook,
    load_evaluation_checkpoint,
    normalize_pushes,
    restore_terminal_capture_hook,
)
from isaaclab_quad_diag.slices import build_slices


def test_disabled_push_mapping_is_normalized_to_empty_list():
    assert normalize_pushes({"enabled": False, "note": "disabled suite"}) == []


def test_empty_time_since_command_switch_is_derived():
    df = pd.DataFrame({
        "case_id": [0, 0],
        "env_id": [0, 0],
        "episode_id": [0, 0],
        "cmd_segment_id": [0, 0],
        "time": [0.0, 1.0],
        "time_since_command_switch": [float("nan"), float("nan")],
    })
    settled = build_slices(df).masks["command_settled"]
    assert settled.tolist() == [False, True]


def test_case_id_keeps_independent_terrain_rollouts_separate():
    df = pd.DataFrame({
        "case_id": [0, 0, 1, 1],
        "env_id": [0, 0, 0, 0],
        "episode_id": [0, 0, 0, 0],
        "cmd_segment_id": [0, 0, 0, 0],
        "time": [0.0, 1.0, 0.0, 1.0],
        "terrain_type": ["flat", "flat", "stairs", "stairs"],
        "cmd_target_mode": ["forward"] * 4,
        "cmd_target_vx": [0.5] * 4,
        "cmd_target_vy": [0.0] * 4,
        "cmd_target_wz": [0.0] * 4,
        "base_lin_vel_b_x": [0.1] * 4,
        "base_lin_vel_b_y": [0.0] * 4,
    })
    timeline = extract_event_timeline(df)
    segments = segment_event_summary(df, timeline)
    assert len(segments) == 2
    assert {(s["case_id"], s["terrain_type"]) for s in segments} == {(0, "flat"), (1, "stairs")}


def test_terrain_case_is_applied_by_real_generator_proportion():
    sub = {
        "flat": SimpleNamespace(proportion=0.5),
        "stairs": SimpleNamespace(proportion=0.5),
    }
    cfg = SimpleNamespace(
        terrain=SimpleNamespace(
            terrain_type="generator",
            terrain_generator=SimpleNamespace(sub_terrains=sub, curriculum=True),
        )
    )
    result = apply_terrain_case_best_effort(cfg, {"type": "stairs_up", "level": 2}, [])
    assert result["applied"] is True
    assert sub["flat"].proportion == 0.0
    assert sub["stairs"].proportion == 1.0


def test_contact_transitions_require_a_real_previous_sample():
    assert contact_transition_flags(False, True, False) == (0, 0)
    assert contact_transition_flags(False, True, True) == (1, 0)
    assert contact_transition_flags(True, False, True) == (0, 1)


def test_terminal_hook_captures_before_reset_mutates_state():
    class Dummy:
        value = 7

        def _reset_idx(self, env_ids):
            self.value = -1

    base = Dummy()
    state = install_terminal_capture_hook(base)
    state["callback"] = lambda env_ids: [{"value": base.value, "env_id": int(env_ids[0])}]
    base._reset_idx([3])
    rows = collect_terminal_rows(state)
    restore_terminal_capture_hook(base, state)

    assert rows == [{"value": 7, "env_id": 3}]
    assert base.value == -1


def test_evaluation_checkpoint_loads_only_policy_and_state_preprocessor(tmp_path):
    import torch

    source_policy = torch.nn.Linear(2, 1)
    source_preprocessor = torch.nn.Linear(1, 1)
    path = tmp_path / "agent.pt"
    torch.save({
        "policy": source_policy.state_dict(),
        "state_preprocessor": source_preprocessor.state_dict(),
        "optimizer": {"not": "needed for inference"},
    }, path)

    target_policy = torch.nn.Linear(2, 1)
    target_preprocessor = torch.nn.Linear(1, 1)
    agent = SimpleNamespace(
        checkpoint_modules={
            "policy": target_policy,
            "state_preprocessor": target_preprocessor,
        }
    )
    load_evaluation_checkpoint(agent, str(path))

    for expected, actual in zip(source_policy.parameters(), target_policy.parameters()):
        assert torch.equal(expected, actual)
    for expected, actual in zip(source_preprocessor.parameters(), target_preprocessor.parameters()):
        assert torch.equal(expected, actual)
