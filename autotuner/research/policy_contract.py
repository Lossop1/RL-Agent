"""Policy/deployment contract construction and parity checks.

The exporter and each deployment runtime may describe the same policy in
different files.  This module compares the resolved contracts rather than
assuming that matching checkpoint names imply matching behavior.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .research_ledger import PolicyDeploymentContract, content_hash


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing required field {key}")
    return mapping[key]


def contract_from_export_metadata(metadata: dict[str, Any], *, contract_id: str, version: str = "") -> PolicyDeploymentContract:
    """Convert ``export_taili_deployment.py`` metadata into the shared contract."""
    observation = dict(_required(metadata, "observation", "deployment metadata"))
    control = dict(_required(metadata, "control", "deployment metadata"))
    timing = dict(_required(metadata, "timing", "deployment metadata"))
    output_shape = metadata.get("output_shape")
    input_shape = metadata.get("input_shape")
    if not isinstance(input_shape, list) or len(input_shape) != 2:
        raise ValueError("deployment metadata input_shape must be [batch, dimension]")
    if not isinstance(output_shape, list) or len(output_shape) != 2:
        raise ValueError("deployment metadata output_shape must be [batch, dimension]")
    return PolicyDeploymentContract(
        id=contract_id,
        version=version,
        observation={
            **observation,
            "dimension": int(input_shape[1]),
            "history_length": observation.get("history_len"),
            "history_stride": observation.get("history_stride", observation.get("history_update_interval_policy_steps")),
        },
        action={
            "dimension": int(output_shape[1]),
            "joint_order": control.get("joint_order", []),
            "scale": control.get("action_scale"),
            "offset": control.get("q_default", []),
        },
        controller={
            **timing,
            "policy_hz": 1.0 / float(timing["policy_dt"]),
            "physics_hz": 1.0 / float(timing["physics_dt"]),
            "kp": control.get("stiffness", []),
            "kd": control.get("damping", []),
            "torque_limits": control.get("effort_limit", []),
        },
        privileged_inputs={
            "actor": list(observation.get("actor_inputs", [])),
            "critic": list(observation.get("critic_inputs", [])),
            "training_supervision": list(observation.get("training_supervision", [])),
        },
        isaaclab_ref=str(metadata.get("isaaclab_ref") or metadata.get("source_checkpoint") or ""),
        mujoco_ref=str(metadata.get("mujoco_ref") or ""),
        real_adapter_ref=str(metadata.get("real_adapter_ref") or ""),
        source="deployment_export_metadata",
    )


def _diff(expected: Any, actual: Any, path: str, out: list[dict[str, Any]], tolerance: float) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                out.append({"path": f"{path}.{key}", "expected": "<missing>", "actual": actual[key]})
            elif key not in actual:
                out.append({"path": f"{path}.{key}", "expected": expected[key], "actual": "<missing>"})
            else:
                _diff(expected[key], actual[key], f"{path}.{key}" if path else key, out, tolerance)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append({"path": path, "expected": {"length": len(expected)}, "actual": {"length": len(actual)}})
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _diff(left, right, f"{path}[{index}]", out, tolerance)
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(expected, bool) and not isinstance(actual, bool):
        if not (math.isfinite(float(expected)) and math.isfinite(float(actual))) or abs(float(expected) - float(actual)) > tolerance:
            out.append({"path": path, "expected": expected, "actual": actual})
        return
    if expected != actual:
        out.append({"path": path, "expected": expected, "actual": actual})


def compare_policy_contracts(expected: PolicyDeploymentContract, actual: PolicyDeploymentContract, *, tolerance: float = 1.0e-9) -> dict[str, Any]:
    """Return a conservative parity report; any mismatch blocks deployment claims."""
    left = expected.model_dump(mode="json", exclude={"id", "version", "source"})
    right = actual.model_dump(mode="json", exclude={"id", "version", "source"})
    mismatches: list[dict[str, Any]] = []
    _diff(left, right, "", mismatches, tolerance)
    return {
        "schema_version": "rl-agent.policy-parity/v1",
        "status": "proven" if not mismatches else "blocked",
        "expected_ref": expected.id,
        "actual_ref": actual.id,
        "expected_hash": content_hash(left),
        "actual_hash": content_hash(right),
        "checked_fields": sorted(left),
        "mismatches": mismatches,
        "tolerance": tolerance,
    }


def load_contract(path: str | Path) -> PolicyDeploymentContract:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict) and "contract" in value:
        value = value["contract"]
    return PolicyDeploymentContract.model_validate(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two resolved RL deployment contracts.")
    parser.add_argument("--expected", required=True)
    parser.add_argument("--actual", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = compare_policy_contracts(load_contract(args.expected), load_contract(args.actual))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "proven" else 2


if __name__ == "__main__":
    raise SystemExit(main())
