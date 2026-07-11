"""Regression: a single malformed telemetry line must be DROPPED, not raised.

A raised parse error propagates up and is misclassified as a remote-SSH failure, putting a healthy
box into a self-inflicted cooldown outage (the bad line stays in `tail -n 240` and re-fails every
poll, escalating the cooldown). parse_telemetry_jsonl must be resilient to bad numeric fields.
"""
from autotuner.locomotion_console.telemetry import (
    _actual_phase_gate_from_config,
    parse_telemetry_jsonl,
)


def test_bad_numeric_field_is_dropped_not_raised():
    text = "\n".join([
        '{"step": 1500, "reward": {"total": 1.1}}',        # good
        '{"step": "N/A", "total_steps": "N/A", "fps": "N/A"}',  # bad numerics — must not raise
        '{"step": 1600, "reward": {"total": 1.2}}',        # good
        'not json at all',                                  # bad json — must not raise
        '{"step": 1700, "fps": "oops", "reward": {"total": 1.3}}',  # bad fps — must not raise
    ])
    points = parse_telemetry_jsonl(text)          # must not raise
    steps = [p.step for p in points]
    assert 1500 in steps and 1600 in steps        # good lines survive
    assert all(isinstance(s, int) for s in steps)


def test_empty_and_blank_lines_ignored():
    assert parse_telemetry_jsonl("") == []
    assert parse_telemetry_jsonl("\n\n  \n") == []


def test_playback_keeps_one_real_env_across_cases():
    """Multi-terrain suites must play EVERY case — the old global-max kept a single case's
    trajectory, so a flat/slope/rough/stairs suite silently showed only one terrain."""
    from autotuner.locomotion_console.diagnostics import _best_continuous_rows

    def row(case, env, t):
        return {"case_id": str(case), "env_id": str(env), "episode_id": "1",
                "time": str(t), "step": str(int(t * 50)),
                "post_step_state_may_be_after_reset": "False"}

    rows = ([row(0, 0, t / 10) for t in range(10)]        # case0 env0: 10 rows
            + [row(1, 0, t / 10) for t in range(8)]        # case1 env0: 8 rows
            + [row(1, 1, t / 10) for t in range(12)])      # case1 env1: 12 rows (case1's best)
    out = _best_continuous_rows(rows)
    cases = [int(r["case_id"]) for r in out]
    assert set(cases) == {0, 1}                            # both cases present
    assert cases == sorted(cases)                          # case order preserved
    assert len([r for r in out if r["case_id"] == "0"]) == 10
    assert len([r for r in out if r["case_id"] == "1"]) == 8
    assert {int(r["env_id"]) for r in out} == {0}


def test_gate_thresholds_carry_config_vs_default_provenance():
    """Each gate threshold must be labelled 'config' (this run's real strategy value)
    or 'default' (UI fallback), so the console never passes a default off as fact."""
    text = "\n".join([
        "env:",
        "  curriculum:",
        "    phase_gate_prog_1: 0.72",   # present -> config
        "    phase_gate_diag_1: 0.83",   # present -> config
        "    phase_max_steps: 30000",    # present -> config
        # phase_gate_slip_* / duty_* / phase_intervals absent -> default
    ])
    gate = _actual_phase_gate_from_config(text, phase=1)
    assert gate["available"] is True
    cond = gate["conditions"]
    src = gate["conditions_source"]

    assert cond["progress_min"] == 0.72 and src["progress_min"] == "config"
    assert cond["diagonal_min"] == 0.83 and src["diagonal_min"] == "config"
    assert src["phase_max_steps"] == "config"
    assert src["slip_max"] == "default"          # no phase_gate_slip_* in config
    assert src["duty_min"] == "default"
    assert src["phase_intervals"] == "default"
    assert src["penalty_gate_min"] == "constant"  # definitional, not a per-strategy knob
    # phase 1 < terrain_start (5): terrain-only conditions must be absent, not defaulted
    assert "fall_max" not in cond


def test_gate_thresholds_include_terrain_conditions_in_terrain_phase():
    text = "\n".join([
        "env:",
        "  curriculum:",
        "    terrain_start_phase: 5",
        "    phase_gate_fall_2: 0.04",   # present -> config
        # phase_gate_terrain_2 absent -> default 0.0
    ])
    gate = _actual_phase_gate_from_config(text, phase=5)
    cond = gate["conditions"]
    src = gate["conditions_source"]
    assert cond["fall_max"] == 0.04 and src["fall_max"] == "config"
    assert cond["terrain_min"] == 0.0 and src["terrain_min"] == "default"


def test_gate_unavailable_when_config_missing():
    gate = _actual_phase_gate_from_config("", phase=1)
    assert gate["available"] is False
    assert gate["conditions"] == {}


def test_playback_keeps_same_env_after_reset():
    from autotuner.locomotion_console.diagnostics import _best_continuous_rows

    def row(t, episode):
        return {
            "case_id": "0",
            "env_id": "0",
            "episode_id": str(episode),
            "time": str(t),
            "step": str(int(t * 50)),
            "post_step_state_may_be_after_reset": "False",
        }

    rows = [row(t / 50, 1) for t in range(226)] + [row(t / 50, 2) for t in range(226, 540)]
    out = _best_continuous_rows(rows)
    assert len(out) == 540
    assert {r["episode_id"] for r in out} == {"1", "2"}
