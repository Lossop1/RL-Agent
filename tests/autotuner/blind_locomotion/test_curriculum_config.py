"""Internal-consistency tests for the curriculum/config loader (pure Python, no sim).

These lock the behavior of the YAML->runtime mapping that every strategy edit flows through:
phase resolution, active-direction gating (so curriculum gates never take the min over
directions the current recipe does not train), and reward-config passthrough.
"""
from types import SimpleNamespace

from autotuner.blind_locomotion import taili_blind_config as C


def _cfg_phase_curriculum():
    return SimpleNamespace(
        training_command_mode="phase_curriculum",
        training_phase_commands={
            0: {"command_mode": "single_axis", "active_dirs": ["fwd", "back", "lat", "yaw"], "prob_yaw": 0.40},
            1: {"command_mode": "mixed"},
            2: {"command_mode": "mixed"},
            3: {"command_mode": "mixed"},
        },
        init_phase=0,
    )


def test_phase_spec_selects_highest_phase_at_or_below_current():
    cfg = _cfg_phase_curriculum()
    assert C.phase_command_spec(cfg, 0)["command_mode"] == "single_axis"
    assert C.phase_command_spec(cfg, 2)["command_mode"] == "mixed"
    # phase beyond the last defined clamps to the last defined spec
    assert C.phase_command_spec(cfg, 9)["command_mode"] == "mixed"


def test_non_phase_mode_passthrough():
    cfg = SimpleNamespace(training_command_mode="normal")
    assert C.phase_command_spec(cfg)["command_mode"] == "normal"


def test_active_directions_from_explicit_active_dirs():
    cfg = _cfg_phase_curriculum()
    assert C.active_command_directions(cfg, 0) == ("fwd", "back", "lat", "yaw")


def test_active_directions_mixed_is_all_four():
    cfg = _cfg_phase_curriculum()
    assert set(C.active_command_directions(cfg, 1)) == {"fwd", "back", "lat", "yaw"}


def test_active_directions_ignores_untrained_dirs_via_probs():
    # A single-axis phase that only samples fwd/yaw must not gate on back/lat progress.
    cfg = SimpleNamespace(
        training_command_mode="phase_curriculum",
        training_phase_commands={0: {"command_mode": "single_axis",
                                     "prob_fwd": 0.5, "prob_yaw": 0.5,
                                     "prob_back": 0.0, "prob_lat": 0.0}},
        init_phase=0,
    )
    active = C.active_command_directions(cfg, 0)
    assert set(active) == {"fwd", "yaw"}
    assert "back" not in active and "lat" not in active


def test_active_direction_progress_takes_min_over_active_only():
    cfg = _cfg_phase_curriculum()
    progress = {"fwd": 0.9, "back": 0.8, "lat": 0.7, "yaw": 0.5}
    val, active = C.active_direction_progress(progress, cfg, 0)
    assert val == 0.5  # yaw is the min and yaw is active
    # If yaw were not trained, its low progress must not drag the gate down.
    cfg_noyaw = SimpleNamespace(
        training_command_mode="phase_curriculum",
        training_phase_commands={0: {"command_mode": "single_axis", "prob_fwd": 1.0,
                                     "prob_back": 0.0, "prob_lat": 0.0, "prob_yaw": 0.0}},
        init_phase=0,
    )
    val2, active2 = C.active_direction_progress(progress, cfg_noyaw, 0)
    assert active2 == ("fwd",) and val2 == 0.9


def test_reward_config_mapping_derives_height_gates():
    data = {"reward": {"nominal_base_h": 0.52, "w_tracking_lin": 2.5}}
    out = C.reward_config_mapping(data)
    assert out["w_tracking_lin"] == 2.5
    # single-source height: h_ok = nominal-0.05, h_gate_close = nominal-0.10
    assert abs(out["h_ok"] - 0.47) < 1e-9
    assert abs(out["h_gate_close"] - 0.42) < 1e-9


def test_real_yaml_loads_and_reward_maps():
    # The shipped strategy contract must load and expose a well-formed reward block.
    data = C.load_taili_blind_config()
    rc = C.reward_config_mapping(data)
    assert rc["sigma_lin_abs"] == 0.10   # spec A1 band (forward tracking passes, so left as-is)
    assert rc["sigma_yaw"] == 0.08       # 0705 structural fix (D3m): a FRACTION of the 0.15 spec band,
    #                                      not equal to it — sigma==band was gradient-dead for the yaw tail.
    assert rc["nominal_base_h"] == 0.52
    assert abs(rc["h_ok"] - 0.47) < 1e-9 and abs(rc["h_gate_close"] - 0.42) < 1e-9
