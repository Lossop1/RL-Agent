import math
from types import SimpleNamespace

import torch

from autotuner.blind_locomotion.diagnose_taili import (
    RECORD_SCHEMA_VERSION,
    _build_columns,
    _configure_terrain,
    _enable_external_commands,
    _extend_diagnostic_episode_horizon,
    _force_env_reset,
    _set_external_command,
    _sync_external_command_observation,
)
from autotuner.blind_locomotion.diagnose_taili_cases import RECORD_SCHEMA_VERSION as MERGED_SCHEMA_VERSION


def test_diagnostic_records_center_and_sole_clearance_separately():
    columns = _build_columns()
    assert "foot_radius" in columns
    assert "foot_FL_center_clearance_local" in columns
    assert "foot_FL_clearance_local" in columns


def test_single_and_merged_diagnostics_share_the_record_schema():
    assert RECORD_SCHEMA_VERSION == "ilqd_observation_record_v0.6.0"
    assert MERGED_SCHEMA_VERSION == RECORD_SCHEMA_VERSION


def test_external_command_mode_is_created_even_when_the_env_does_not_predeclare_it():
    base = SimpleNamespace()

    _enable_external_commands(base)

    assert base.use_external_commands is True


def test_external_command_updates_target_runtime_command_and_current_actor_observation():
    base = SimpleNamespace(
        num_envs=2,
        _cmd_target=torch.full((2, 3), -1.0),
        commands=torch.full((2, 3), -2.0),
    )
    target = torch.tensor([0.3, -0.2, 0.4])
    observation = torch.zeros(2, 20)

    _set_external_command(base, target)
    returned = _sync_external_command_observation(observation, base)

    expected = target.unsqueeze(0).expand(2, -1)
    torch.testing.assert_close(base._cmd_target, expected)
    torch.testing.assert_close(base.commands, expected)
    torch.testing.assert_close(observation[:, 6:9], expected)
    assert returned is observation


def test_external_command_sync_supports_policy_observation_dict():
    base = SimpleNamespace(num_envs=1, commands=torch.tensor([[0.0, 0.0, -0.5]]))
    observation = {"policy": torch.ones(1, 12), "critic": torch.ones(1, 5)}

    _sync_external_command_observation(observation, base)

    torch.testing.assert_close(observation["policy"][:, 6:9], base.commands)
    torch.testing.assert_close(observation["critic"], torch.ones(1, 5))


class _ScalarBuffer:
    def __init__(self, value: int):
        self.value = value

    def max(self):
        return self

    def item(self):
        return self.value


class _Base:
    def __init__(self, max_episode_steps: int, current_step: int, step_dt: float = 0.02):
        self.step_dt = step_dt
        self.cfg = SimpleNamespace(episode_length_s=max_episode_steps * step_dt)
        self.episode_length_buf = _ScalarBuffer(current_step)

    @property
    def max_episode_length(self):
        return math.ceil(self.cfg.episode_length_s / self.step_dt)


class _CachedResetWrapper:
    def __init__(self):
        self._reset_once = False
        self.underlying_reset_calls = 0

    def reset(self):
        if self._reset_once:
            self.underlying_reset_calls += 1
            self._reset_once = False
        return "observation", {}


def test_diagnostic_episode_horizon_covers_the_whole_plan_without_timeout_reset():
    base = _Base(max_episode_steps=360, current_step=0)

    horizon = _extend_diagnostic_episode_horizon(base, planned_steps=810, margin_steps=100)

    assert horizon["previous_max_episode_steps"] == 360
    assert horizon["effective_max_episode_steps"] == 911
    assert base.max_episode_length == 911


def test_diagnostic_episode_horizon_accounts_for_a_nonzero_starting_step():
    base = _Base(max_episode_steps=1000, current_step=638)

    horizon = _extend_diagnostic_episode_horizon(base, planned_steps=810, margin_steps=100)

    assert horizon["starting_episode_step"] == 638
    assert base.max_episode_length == 1549


def test_force_env_reset_bypasses_skrl_reset_cache_every_time():
    env = _CachedResetWrapper()

    assert _force_env_reset(env) == ("observation", {})
    assert _force_env_reset(env) == ("observation", {})
    assert env.underlying_reset_calls == 2


def test_stairs_up_diagnostic_keeps_the_training_terrain_type():
    stairs_down = SimpleNamespace(
        proportion=0.5,
        step_height_range=(0.04, 0.30),
    )
    stairs_up = SimpleNamespace(
        proportion=0.5,
        step_height_range=(0.04, 0.30),
    )
    generator = SimpleNamespace(
        curriculum=False,
        sub_terrains={"stairs": stairs_down, "stairs_up": stairs_up},
    )
    env_cfg = SimpleNamespace(
        terrain=SimpleNamespace(
            terrain_type="generator",
            terrain_generator=generator,
            max_init_terrain_level=0,
        )
    )

    effective = _configure_terrain(
        env_cfg,
        {
            "type": "stairs_up",
            "level": 5,
            "params": {"direction": "up", "step_height": 0.18},
        },
    )

    selected = env_cfg.terrain.terrain_generator.sub_terrains
    assert effective == "stairs_up"
    assert selected["stairs_up"].proportion == 1.0
    assert selected["stairs"].proportion == 0.0
    assert selected["stairs_up"].step_height_range == (0.18, 0.18)
    assert env_cfg.terrain.max_init_terrain_level == 5
