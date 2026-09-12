"""Controlled entry point from user task text to a prepared task handoff.

This module owns the console boundary only. Product contracts, artifact generation,
payload building, and launch-plan validation remain in their existing product-neutral
modules. The service never starts a remote process.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
from pathlib import Path
from typing import Any, Mapping
import uuid

from autotuner.llm_gateway import task_intake
from autotuner.llm_gateway.task_intake import DynamicTaskIntakeResult

from .config import PROJECT_ROOT


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class TaskIntakeError(ValueError):
    """The console request violates the preparation boundary."""


def _workspace_path(value: str | Path, name: str) -> Path:
    workspace = PROJECT_ROOT.resolve()
    target = Path(value).expanduser().resolve()
    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise TaskIntakeError(f"{name} must stay inside the project workspace") from exc
    return target


def _workspace_ref(value: str | Path, name: str) -> str:
    target = _workspace_path(value, name)
    return target.relative_to(PROJECT_ROOT.resolve()).as_posix()


def _safe_run_id(value: str | None) -> str:
    result = str(value or "").strip() or f"intake-{uuid.uuid4().hex}"
    if _RUN_ID.fullmatch(result) is None:
        raise TaskIntakeError("run_id contains unsafe characters or is too long")
    return result


@dataclass(frozen=True)
class PreparedTaskIntake:
    """A serializable preparation result plus the controlled local run directory."""

    result: DynamicTaskIntakeResult
    run_id: str
    output_root: Path

    def response_dict(self) -> dict[str, Any]:
        result = self.result
        bundle = result.bundle
        pipeline = result.pipeline
        contract = bundle.contract if bundle is not None else None
        payload: dict[str, Any] = {}
        artifacts: list[dict[str, Any]] = []
        run_manifest = ""
        ledger_event_ids: list[str] = []
        launch_plan: dict[str, Any] | None = None
        if pipeline is not None:
            payload = {
                "archive": _workspace_ref(pipeline.payload.archive, "payload archive"),
                "manifest": _workspace_ref(pipeline.payload.manifest, "payload manifest"),
                "payload_digest": pipeline.payload.payload_digest,
                "file_count": pipeline.payload.file_count,
                "task_contract_ref": pipeline.payload.task_contract_ref,
                "task_bundle_digest": pipeline.payload.task_bundle_digest,
                "task_artifact_manifest_ref": pipeline.payload.task_artifact_manifest_ref,
            }
            artifacts = [
                {
                    "kind": item.kind,
                    "path": _workspace_ref(item.path, f"{item.kind} artifact"),
                    "ref": pipeline.materialized.artifact_ref(item.kind),
                    "content_digest": item.content_digest,
                    "file_digest": item.file_digest,
                }
                for item in pipeline.materialized.artifacts
            ]
            run_manifest = _workspace_ref(pipeline.run_manifest, "run manifest")
            ledger_event_ids = [event.event_id for event in pipeline.ledger_events]
            if pipeline.launch_plan is not None:
                launch_plan = pipeline.launch_plan.to_dict()

        return {
            "ok": bool(result.ready and pipeline is not None),
            "ready": result.ready,
            "product_id": result.product_id,
            "run_id": self.run_id,
            "confidence": result.confidence,
            "model": result.model,
            "clarifications": list(result.clarifications),
            "error": result.error,
            "contract_ref": result.contract_ref,
            "contract_status": contract.status if contract is not None else "",
            "contract_digest": contract.contract_digest if contract is not None else "",
            "bundle_digest": bundle.bundle_digest if bundle is not None else "",
            "output_root": _workspace_ref(self.output_root, "task intake output"),
            "materialization_manifest": (
                _workspace_ref(pipeline.materialized.manifest, "materialization manifest")
                if pipeline is not None
                else ""
            ),
            "artifacts": artifacts,
            "payload": payload,
            "run_manifest": run_manifest,
            "launch_plan": launch_plan,
            "ledger_event_ids": ledger_event_ids,
        }


class TaskIntakeService:
    """Prepare a user task using a workspace-owned output root; never execute it."""

    def __init__(self, output_root: str | Path | None = None) -> None:
        self.output_root = _workspace_path(
            output_root or PROJECT_ROOT / "output" / "task_intake",
            "task intake output root",
        )

    def prepare(
        self,
        user_text: str,
        *,
        product_id: str = "",
        run_id: str | None = None,
        approved: bool = False,
        approved_by: str = "",
        launch: Mapping[str, Any] | None = None,
    ) -> PreparedTaskIntake:
        text = str(user_text or "").strip()
        if not text or len(text) > 12000:
            raise TaskIntakeError("user_text must contain 1 to 12000 characters")
        approver = str(approved_by or "").strip()
        if approved and not approver:
            raise TaskIntakeError("approved preparation requires approved_by")
        if launch is not None and not approved:
            raise TaskIntakeError("a launch plan requires an approved task")
        if launch is not None and not isinstance(launch, Mapping):
            raise TaskIntakeError("launch must be an object")

        selected_run_id = _safe_run_id(run_id)
        run_root = _workspace_path(self.output_root / selected_run_id, "task intake run output")
        if run_root.exists():
            raise TaskIntakeError(f"task intake run already exists: {selected_run_id}")

        launch_request: Mapping[str, Any] | None = None
        if launch is not None:
            launch_data = dict(launch)
            declared_run_id = str(launch_data.get("run_id") or "").strip()
            if declared_run_id and declared_run_id != selected_run_id:
                raise TaskIntakeError("launch.run_id must match run_id")
            launch_data["run_id"] = selected_run_id
            launch_request = launch_data

        try:
            result = task_intake.translate_dynamic(
                text,
                product_id=product_id or None,
                approved=approved,
                approved_by=approver,
                output_root=run_root,
                run_id=selected_run_id,
                launch_request=launch_request,
            )
            prepared = PreparedTaskIntake(result, selected_run_id, run_root)
            if not result.ready or result.pipeline is None:
                shutil.rmtree(run_root, ignore_errors=True)
            return prepared
        except Exception:
            shutil.rmtree(run_root, ignore_errors=True)
            raise


__all__ = ["PreparedTaskIntake", "TaskIntakeError", "TaskIntakeService"]
