from __future__ import annotations

import json
from pathlib import Path
import shutil

from fastapi.testclient import TestClient

from autotuner.llm_gateway.client import LLMResponse
from autotuner.llm_gateway import task_intake
from autotuner.locomotion_console.task_intake_service import TaskIntakeError, TaskIntakeService


REPO_ROOT = Path(__file__).resolve().parents[3]


def _response(*, confidence: float = 0.94) -> LLMResponse:
    return LLMResponse(
        parsed={
            "instance_id": "api-task",
            "objective": "prepare a deployable locomotion task",
            "goals": [{"id": "quality", "objective": "preserve the protected baseline"}],
            "constraints": {"deployment_requires_privileged_terrain": False},
            "protected_capabilities": ["flat"],
            "training": {"experiment": {"mode": "evidence_driven"}},
            "telemetry": {"cadence_s": 10},
            "diagnostics": {"presets": ["directions"]},
            "simulation": {"sim2sim": {"required": True}},
            "deployment": {},
            "clarifications": [],
            "confidence": confidence,
        },
        raw_text="{}",
        model="test-model",
        elapsed_s=0.01,
        attempt=0,
    )


def _patch_service(monkeypatch, root: Path):
    from autotuner.locomotion_console import app as app_module

    monkeypatch.setattr(
        app_module,
        "_task_intake_service",
        lambda: TaskIntakeService(root),
    )
    return app_module


def test_task_intake_api_materializes_one_traceable_handoff(monkeypatch) -> None:
    root = REPO_ROOT / ".pytest-tmp" / "api-intake-draft"
    shutil.rmtree(root, ignore_errors=True)
    monkeypatch.setattr(task_intake, "call_llm_with_schema", lambda **_: _response())
    app_module = _patch_service(monkeypatch, root)

    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/tasks/intake",
                json={"user_text": "prepare a stable locomotion task", "product_id": "taili", "run_id": "api-draft"},
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ok"] is True
        assert body["contract_status"] == "draft"
        assert body["launch_plan"] is None
        assert {item["kind"] for item in body["artifacts"]} == {
            "training", "telemetry", "diagnostics", "simulation", "deployment",
        }
        assert body["bundle_digest"]
        assert body["payload"]["payload_digest"]
        assert body["run_manifest"]
        assert body["ledger_event_ids"]

        manifest = json.loads((REPO_ROOT / body["run_manifest"]).read_text(encoding="utf-8"))
        assert manifest["execution"]["task_contract_ref"] == body["contract_ref"]
        assert manifest["execution"]["task_bundle_digest"] == body["bundle_digest"]
        assert manifest["execution"]["payload_digest"] == body["payload"]["payload_digest"]
        assert all((REPO_ROOT / item["path"]).is_file() for item in body["artifacts"])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_task_intake_api_requires_explicit_approval_before_launch(monkeypatch) -> None:
    root = REPO_ROOT / ".pytest-tmp" / "api-intake-launch-guard"
    shutil.rmtree(root, ignore_errors=True)
    app_module = _patch_service(monkeypatch, root)

    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/tasks/intake",
                json={
                    "user_text": "start training now",
                    "product_id": "taili",
                    "run_id": "api-draft-launch",
                    "launch": {"payload_root": "/remote/payload"},
                },
            )
        assert response.status_code == 400
        assert "approved" in response.json()["detail"]
        assert not root.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_approved_task_can_prepare_launch_plan_without_starting_process(monkeypatch) -> None:
    root = REPO_ROOT / ".pytest-tmp" / "api-intake-approved"
    shutil.rmtree(root, ignore_errors=True)
    monkeypatch.setattr(task_intake, "call_llm_with_schema", lambda **_: _response())
    app_module = _patch_service(monkeypatch, root)

    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/tasks/intake",
                json={
                    "user_text": "prepare an approved training run",
                    "product_id": "taili",
                    "run_id": "api-approved",
                    "approved": True,
                    "approved_by": "operator",
                    "launch": {"payload_root": "/remote/payload"},
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ok"] is True
        assert body["contract_status"] == "approved"
        assert body["launch_plan"]["run_id"] == "api-approved"
        assert body["launch_plan"]["argv"]
        assert "started" not in body
        manifest = json.loads((REPO_ROOT / body["run_manifest"]).read_text(encoding="utf-8"))
        assert manifest["launch_plan"]["run_id"] == "api-approved"
        assert manifest["execution"]["task_contract_digest"] == body["contract_digest"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_task_intake_output_root_is_workspace_bound() -> None:
    outside = REPO_ROOT.parent / "task-intake-outside"
    try:
        TaskIntakeService(outside)
    except TaskIntakeError as exc:
        assert "inside the project workspace" in str(exc)
    else:
        raise AssertionError("external task intake root was accepted")
