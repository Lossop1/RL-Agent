from __future__ import annotations

import csv

from autotuner.blind_locomotion.stair_validation import compute_stair_validation


def _write_case(path, *, rear_complete: bool) -> None:
    fields = [
        "case_id", "time", "cmd_target_mode", "terrain_type_requested",
        "base_pos_w_z", "base_height_local", "would_terminate",
    ]
    for leg in ("FL", "FR", "RL", "RR"):
        fields += [
            f"foot_{leg}_terrain_height", f"foot_{leg}_contact",
            f"foot_{leg}_stance_slip_xy",
        ]
    for index in range(12):
        fields.append(f"torque_utilization_{index}")
    rows = []
    for index in range(100):
        late = index >= 50
        row = {
            "case_id": 0,
            "time": index * 0.02,
            "cmd_target_mode": "forward",
            "terrain_type_requested": "stairs_up",
            "base_pos_w_z": 0.55 + (0.18 if late else 0.0),
            "base_height_local": 0.55,
            "would_terminate": 0,
        }
        for leg in ("FL", "FR", "RL", "RR"):
            complete = late and (rear_complete or leg in {"FL", "FR"})
            row[f"foot_{leg}_terrain_height"] = 0.18 if complete else 0.0
            row[f"foot_{leg}_contact"] = 1
            row[f"foot_{leg}_stance_slip_xy"] = 0.05
        for joint in range(12):
            row[f"torque_utilization_{joint}"] = 0.5
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _suite():
    return {
        "terrains": [{"type": "stairs_up", "level": 5, "params": {"step_height": 0.18}}],
        "dr_cases": [{"level": 0}],
        "criteria": {"min_height_m": 0.40},
    }


def test_complete_support_transfer_passes(tmp_path):
    record = tmp_path / "record.csv"
    _write_case(record, rear_complete=True)
    result = compute_stair_validation(record, _suite())
    assert result["summary"] == {"cases": 1, "passed": 1, "pass_rate": 1.0}
    assert result["cases"][0]["all_feet_completed_layer_change"] is True


def test_front_only_step_fails(tmp_path):
    record = tmp_path / "record.csv"
    _write_case(record, rear_complete=False)
    result = compute_stair_validation(record, _suite())
    case = result["cases"][0]
    assert case["success"] is False
    assert case["legs"]["FL"]["completed_layer_change"] is True
    assert case["legs"]["RL"]["completed_layer_change"] is False
    assert "feet_did_not_change_layer:RL,RR" in case["failure_reasons"]
