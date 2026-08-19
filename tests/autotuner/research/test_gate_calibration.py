"""Pure tests for effective gate extraction and trace qualification."""
from __future__ import annotations

from autotuner.research.gate_calibration import (
    extract_gate_definitions,
    gate_trace_rows,
)


def test_legacy_dr_quality_gates_keep_runtime_comparison_semantics():
    config = {
        "env": {
            "curriculum": {"terrain_start_phase": 2, "dr_start_phase": 2},
            "domain_randomization": {
                "gate_gait_min": 0.78,
                "gate_duty_min": 0.45,
                "gate_slip_max": 0.08,
                "gate_tilt_deg_max": 10.0,
            },
        }
    }
    definitions = {item.gate_id: item for item in extract_gate_definitions(config)}

    expected = {
        "env.domain_randomization.gate_gait_min": ("dr_gate_gait_value", "gte", 0.78),
        "env.domain_randomization.gate_duty_min": ("dr_gate_duty_value", "gte", 0.45),
        "env.domain_randomization.gate_slip_max": ("dr_gate_slip_value", "lte", 0.08),
        "env.domain_randomization.gate_tilt_deg_max": ("dr_gate_tilt_deg_value", "lte", 10.0),
    }
    assert set(definitions) == set(expected)
    for gate_id, (field, comparison, threshold) in expected.items():
        item = definitions[gate_id]
        assert (item.metric_field, item.comparison, item.configured_threshold) == (
            field,
            comparison,
            threshold,
        )
        assert item.phase_min == 2
        assert item.purpose == "dr_progress"

    rows = gate_trace_rows(
        [
            {
                "step": 500,
                "curriculum": {
                    "phase": "phi1",
                    "dr_level": 0,
                    "phase_gate_eval_step": 500,
                    "dr_gate_gait_value": 0.9,
                },
            },
            {
                "step": 1000,
                "curriculum": {
                    "phase": "phi2",
                    "dr_level": 0,
                    "phase_gate_eval_step": 1000,
                    "dr_gate_gait_value": 0.80,
                    "dr_gate_duty_value": 0.50,
                    "dr_gate_slip_value": 0.07,
                    "dr_gate_tilt_deg_value": 9.0,
                },
            },
        ],
        definitions.values(),
    )
    assert len(rows) == 4
    assert {item["gate_eval_step"] for item in rows} == {1000}
    assert all(item["passed"] for item in rows)
