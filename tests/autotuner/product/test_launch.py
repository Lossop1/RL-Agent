from __future__ import annotations

import pytest

from autotuner.product import (
    ProductAdapterError,
    TrainingLaunchError,
    TrainingLaunchRequest,
    build_training_kill_command,
    build_training_launch_plan,
    resolve_product_adapter,
    resolve_product_contract,
)


def test_product_adapter_projects_runtime_layout_and_process_identity() -> None:
    contract = resolve_product_contract("taili")
    adapter = resolve_product_adapter(contract)

    assert adapter.runtime_package == "taili_blind_runtime"
    assert adapter.runs_root == "/root/gpufree-data/taili_runs"
    assert adapter.payload_file("launcher") == "launch_taili_train.py"
    assert adapter.process_pattern() == (
        r"[t]aili_blind_runtime\.launch_taili_train|"
        r"[t]aili_blind_runtime\.train_taili"
    )


def test_product_adapter_rejects_unsafe_remote_layout() -> None:
    contract = resolve_product_contract("taili").to_dict()
    contract["deployment"]["remote_root"] = "/root/data/../escape"

    with pytest.raises(ProductAdapterError, match="safe absolute remote path"):
        resolve_product_adapter(contract)


def test_launch_plan_is_structured_and_product_driven() -> None:
    contract = resolve_product_contract("taili")
    plan = build_training_launch_plan(
        contract,
        TrainingLaunchRequest(
            payload_root="/root/gpufree-data/rl-agent/payloads/" + "a" * 64,
            run_id="taili_train_20260819_resume",
            checkpoint="/root/gpufree-data/taili_runs/source/checkpoints/agent_20000.pt",
            source_run="/root/gpufree-data/taili_runs/source",
            remote_boot_id="boot-id",
            resume=True,
            num_envs=512,
            resume_state={"phase": 2},
        ),
    )

    assert plan.run_dir == "/root/gpufree-data/taili_runs/taili_train_20260819_resume"
    assert plan.argv[:3] == (
        "/opt/conda/envs/isaaclab/bin/python",
        "-m",
        "taili_blind_runtime.launch_taili_train",
    )
    assert plan.argv[-3:] == ("--", "--num_envs", "512")
    assert plan.environment["TAILI_INIT_PHASE"] == "2"
    assert "cd /root/gpufree-data/rl-agent/payloads/" in plan.terminal_command()
    assert plan.start_record()["checkpoint"].endswith("agent_20000.pt")
    assert "console_start.json" in plan.marker_command()


def test_launch_plan_rejects_ambiguous_resume_and_control_characters() -> None:
    contract = resolve_product_contract("taili")
    with pytest.raises(TrainingLaunchError, match="requires a checkpoint"):
        build_training_launch_plan(
            contract,
            TrainingLaunchRequest(payload_root="/data/payload", run_id="run-1", resume=True),
        )
    with pytest.raises(TrainingLaunchError, match="safe id"):
        build_training_launch_plan(
            contract,
            TrainingLaunchRequest(payload_root="/data/payload", run_id="run; shutdown"),
        )


def test_kill_command_uses_only_declared_product_processes() -> None:
    product_id, session, command = build_training_kill_command(resolve_product_contract("taili"))

    assert product_id == "taili"
    assert session == "rl_train"
    assert "[t]aili_blind_runtime" in command
    assert "diagnose_taili" not in command
