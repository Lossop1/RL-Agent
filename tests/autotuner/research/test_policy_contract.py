"""Parity tests for the shared policy/deployment contract."""
from __future__ import annotations

from autotuner.research.policy_contract import (
    compare_policy_contracts,
    contract_from_export_metadata,
)


def _metadata():
    return {
        "input_shape": [1, 1403],
        "output_shape": [1, 12],
        "observation": {
            "body_dim": 53,
            "history_len": 25,
            "history_tick_dim": 54,
            "history_order": "newest_first",
            "history_stride": 1,
            "actor_inputs": ["body", "history", "command"],
            "critic_inputs": ["privileged"],
        },
        "timing": {"physics_dt": 0.0025, "policy_dt": 0.02, "decimation": 8},
        "control": {
            "joint_order": ["j0", "j1"],
            "action_scale": 0.25,
            "q_default": [0.0, 0.1],
            "stiffness": [100.0, 100.0],
            "damping": [10.0, 10.0],
            "effort_limit": [20.0, 20.0],
        },
    }


def test_export_metadata_maps_to_shared_contract():
    contract = contract_from_export_metadata(_metadata(), contract_id="isaac@1", version="1")
    assert contract.observation["dimension"] == 1403
    assert contract.action["dimension"] == 12
    assert contract.controller["physics_hz"] == 400.0
    assert contract.privileged_inputs["actor"] == ["body", "history", "command"]


def test_equal_contracts_are_proven():
    left = contract_from_export_metadata(_metadata(), contract_id="isaac@1")
    right = contract_from_export_metadata(_metadata(), contract_id="mujoco@1")
    report = compare_policy_contracts(left, right)
    assert report["status"] == "proven"
    assert report["mismatches"] == []


def test_control_or_history_mismatch_blocks_parity():
    left = contract_from_export_metadata(_metadata(), contract_id="isaac@1")
    changed = _metadata()
    changed["observation"]["history_order"] = "oldest_first"
    changed["control"]["damping"] = [8.0, 10.0]
    right = contract_from_export_metadata(changed, contract_id="mujoco@1")
    report = compare_policy_contracts(left, right)
    assert report["status"] == "blocked"
    paths = {item["path"] for item in report["mismatches"]}
    assert "observation.history_order" in paths
    assert "controller.kd[0]" in paths
