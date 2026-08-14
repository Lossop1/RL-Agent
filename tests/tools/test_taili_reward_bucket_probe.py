from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tools.taili_reward_bucket_probe import (
    _ContinuousQualityCounterfactualAudit,
    _counterfactual_cfg_from_runtime,
    _float_grid,
    _reward_config_overrides,
)


def _snapshot(support: list[float], collision: list[float], credit: list[float]):
    count = len(support)
    zeros = torch.zeros(count, dtype=torch.bool)
    return {
        "components": {
            "terrain_supported_progress_quality": torch.tensor(support),
            "terrain_supported_progress": torch.tensor(credit),
        },
        "terrain_clearance_trace": torch.tensor(collision)[:, None].repeat(1, 4),
        "stairs_down": zeros.clone(),
        "stairs_up": torch.ones(count, dtype=torch.bool),
        "foot_contact": torch.ones(count, 4),
        "base_height": torch.full((count,), 0.50),
        "base_ang_vel": torch.zeros(count, 3),
    }


def _masks(count: int):
    zeros = torch.zeros(count, dtype=torch.bool)
    return {
        "flat_moving": zeros.clone(),
        "flat_stop": zeros.clone(),
        "stairs_down_frontier": zeros.clone(),
        "stairs_down_replay": zeros.clone(),
        "stairs_up_frontier": torch.ones(count, dtype=torch.bool),
        "stairs_up_replay": zeros.clone(),
    }


def test_float_grid_rejects_empty_and_out_of_range_values():
    assert _float_grid("0.5, 0.8", name="tau", lower=0.0, upper=1.0) == (0.5, 0.8)
    with pytest.raises(ValueError):
        _float_grid("", name="tau", lower=0.0, upper=1.0)
    with pytest.raises(ValueError):
        _float_grid("1.1", name="tau", lower=0.0, upper=1.0)


def test_counterfactual_config_preserves_runtime_values_and_applies_only_payload_diff():
    @dataclass
    class Config:
        scheduled_weight: float = 1.0
        changed_gain: float = 1.0
        candidate_only: float = 0.0

    overrides = _reward_config_overrides(
        {"scheduled_weight": 1.0, "changed_gain": 1.0},
        {"scheduled_weight": 1.0, "changed_gain": 1.2, "candidate_only": 0.5},
    )
    assert overrides == {"changed_gain": 1.2, "candidate_only": 0.5}

    runtime = SimpleNamespace(scheduled_weight=2.75, changed_gain=1.0)
    candidate, copied, applied = _counterfactual_cfg_from_runtime(
        Config, runtime, overrides
    )
    assert candidate.scheduled_weight == 2.75
    assert candidate.changed_gain == 1.2
    assert candidate.candidate_only == 0.5
    assert set(copied) == {"scheduled_weight", "changed_gain"}
    assert set(applied) == {"changed_gain", "candidate_only"}


def test_continuous_quality_keeps_recent_support_and_collision_risk():
    base = SimpleNamespace(device=torch.device("cpu"), num_envs=1, step_dt=0.1)
    audit = _ContinuousQualityCounterfactualAudit(
        base, taus_s=(0.5,), floors=(0.3,)
    )
    done = torch.tensor([False])
    audit.update(
        _snapshot([0.2], [0.0], [0.2]),
        _masks(1),
        done,
        done,
        None,
        record_frame=True,
        record_episode=False,
    )
    key = "tau_0.50_floor_0.30"
    assert audit.risk[key].item() == pytest.approx(0.8)

    audit.update(
        _snapshot([1.0], [0.0], [1.0]),
        _masks(1),
        done,
        done,
        None,
        record_frame=True,
        record_episode=False,
    )
    expected_risk = 0.8 * torch.exp(torch.tensor(-0.1 / 0.5)).item()
    assert audit.risk[key].item() == pytest.approx(expected_risk)
    expected_quality = 0.3 + 0.7 * (1.0 - expected_risk)
    report = audit.report()["frames"]["stairs_up_frontier"]["metrics"]
    assert report[f"{key}/history_quality"]["min"] == pytest.approx(0.44)
    assert report[f"{key}/history_quality"]["max"] == pytest.approx(expected_quality)


def test_continuous_quality_reports_success_failure_credit_separation():
    base = SimpleNamespace(device=torch.device("cpu"), num_envs=3, step_dt=0.1)
    audit = _ContinuousQualityCounterfactualAudit(
        base, taus_s=(0.5,), floors=(0.3,)
    )
    masks = _masks(3)
    no_done = torch.tensor([False, False, False])
    audit.update(
        _snapshot([1.0, 1.0, 0.2], [0.0, 0.0, 1.0], [1.0, 1.0, 0.2]),
        masks,
        no_done,
        no_done,
        None,
        record_frame=True,
        record_episode=True,
    )
    done = torch.tensor([True, False, True])
    outcome = {
        "env_ids": torch.tensor([0, 2]),
        "valid": torch.tensor([True, True]),
        "direction": torch.tensor([1.0, 1.0]),
        "move_up": torch.tensor([True, False]),
        "failure_down": torch.tensor([False, True]),
        "stable_end": torch.tensor([True, False]),
        "controlled_height": torch.tensor([True, False]),
    }
    audit.update(
        _snapshot([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        masks,
        done,
        torch.tensor([False, False, False]),
        outcome,
        record_frame=True,
        record_episode=True,
    )
    episodes = audit.report()["episodes"]["stairs_up"]
    key = "tau_0.50_floor_0.30/counterfactual_credit_mean"
    assert episodes["success"]["samples"] == 1
    assert episodes["failure"]["samples"] == 1
    assert episodes["success"]["metrics"][key]["mean"] > episodes["failure"]["metrics"][key]["mean"]
    assert audit.risk["tau_0.50_floor_0.30"].sum().item() == 0.0
