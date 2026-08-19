"""任务合同到可执行交接的产品无关编排层。

本模块只协调已经声明的产品适配器，不实现训练、奖励、仿真或诊断逻辑。
它把合同、五类 artifact、payload、运行 manifest 和研究 ledger 绑定为同一
条可追溯谱系；远程执行仍由 execution 层负责，LLM 不能绕过批准直接执行。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping

from autotuner.execution import DeploymentSpec
from autotuner.research.research_ledger import (
    ArtifactRef,
    ContractBundle,
    LedgerEvent,
    ResearchLedgerStore,
)

from .launch import (
    TrainingLaunchPlan,
    TrainingLaunchRequest,
    build_training_launch_plan,
)
from .payload import (
    ProductPayload,
    build_product_payload,
    make_deployment_spec,
)
from .task_contract import (
    ResolvedTaskContract,
    TaskContractBundle,
    TaskContractError,
    revise_task_bundle,
)
from .task_materializer import MaterializedTaskBundle, TaskBundleMaterializer
from .task_store import StoredTaskContract, TaskContractStore


TASK_PIPELINE_SCHEMA = "rl-agent.task-pipeline/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]*$")


class TaskPipelineError(ValueError):
    """任务编排输入、谱系或产物不一致时抛出。"""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_id(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not _SAFE_ID.fullmatch(result):
        raise TaskPipelineError(f"{name} is unsafe or empty: {result!r}")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """原子写入运行 manifest，避免中断时留下半份谱系记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{path.stem}.tmp")
    temporary.write_text(_canonical(value) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _product_data(product: Any) -> Mapping[str, Any]:
    value = product.to_dict() if hasattr(product, "to_dict") else product
    if not isinstance(value, Mapping):
        raise TaskPipelineError("product contract must be a mapping")
    return value


def _task_ref(bundle: TaskContractBundle) -> str:
    contract = bundle.contract
    return f"{contract.contract_id}@{contract.contract_version}"


def _materialization_root(root: Path, bundle: TaskContractBundle) -> Path:
    """在独立 artifact 仓库中定位版本，避免合同目录与临时路径过深。"""
    contract_id = re.sub(r"[^A-Za-z0-9._-]+", "_", bundle.contract.contract_id)
    return root / "task_artifacts" / contract_id / f"v{bundle.contract.contract_version:06d}"


def _deployment_dict(spec: DeploymentSpec) -> dict[str, Any]:
    data = asdict(spec)
    for key in ("payload_archive", "payload_manifest", "run_manifest"):
        if isinstance(data.get(key), Path):
            data[key] = str(data[key])
    return data


@dataclass(frozen=True)
class TaskPipelineResult:
    """一次准备阶段的完整、不可变产物索引。"""

    stored_contract: StoredTaskContract
    materialized: MaterializedTaskBundle
    payload: ProductPayload
    deployment_spec: DeploymentSpec
    run_manifest: Path
    launch_plan: TrainingLaunchPlan | None = None
    ledger_events: tuple[LedgerEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TASK_PIPELINE_SCHEMA,
            "contract": self.stored_contract.to_dict(),
            "materialization": self.materialized.to_dict(),
            "payload": self.payload.to_dict(),
            "deployment": _deployment_dict(self.deployment_spec),
            "run_manifest": str(self.run_manifest),
            "launch_plan": self.launch_plan.to_dict() if self.launch_plan else None,
            "ledger_event_ids": [event.event_id for event in self.ledger_events],
        }


class TaskExecutionPipeline:
    """把合同准备成一次可审计的产品交接，不负责实际远程执行。"""

    def __init__(
        self,
        output_root: str | Path,
        *,
        contract_store: TaskContractStore | None = None,
        ledger: ResearchLedgerStore | None = None,
        materializer: TaskBundleMaterializer | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.contract_store = contract_store or TaskContractStore(self.output_root / "task_contracts")
        self.ledger = ledger or ResearchLedgerStore(self.output_root / "research" / "ledger")
        self.materializer = materializer or TaskBundleMaterializer()

    @staticmethod
    def _validate_bundle_product(bundle: TaskContractBundle, product: Any) -> None:
        product_data = _product_data(product)
        contract = bundle.contract
        if contract.product_id != str(product_data.get("product_id") or ""):
            raise TaskPipelineError("task bundle and product contract identify different products")
        if contract.product_contract_digest != str(product_data.get("contract_digest") or ""):
            raise TaskPipelineError("task bundle was compiled from a different product contract")

    def _register_contract(
        self,
        stored: StoredTaskContract,
        bundle: TaskContractBundle,
        materialized: MaterializedTaskBundle,
        *,
        actor: str,
    ) -> list[LedgerEvent]:
        contract = bundle.contract
        ledger_id = f"task:{stored.ref}"
        goals = [
            str(goal.get("objective") or goal.get("description") or "")
            for goal in contract.goals
            if isinstance(goal, Mapping)
        ]
        artifacts = [
            ArtifactRef(
                id=f"artifact:{materialized.bundle_ref}:manifest",
                kind="task_artifact_manifest",
                path=materialized.manifest_ref,
                sha256=materialized.manifest_digest,
                role="task_artifacts",
                exists=True,
            )
        ]
        artifacts.extend(
            ArtifactRef(
                id=f"artifact:{materialized.bundle_ref}:{item.kind}",
                kind=item.kind,
                path=materialized.artifact_ref(item.kind),
                sha256=item.file_digest,
                role="task_artifact",
                exists=True,
            )
            for item in materialized.artifacts
        )
        revision = contract.provenance.get("revision", {})
        revision = revision if isinstance(revision, Mapping) else {}
        approval_source = str(
            revision.get("approved_by")
            or contract.provenance.get("approval_source")
            or ""
        )
        record = ContractBundle(
            id=ledger_id,
            version=str(contract.contract_version),
            status="approved" if contract.status == "approved" else "draft",
            approved_by=approval_source,
            goals=goals,
            protected_capabilities=list(contract.protected_capabilities),
            generated_artifacts=artifacts,
            supersedes=[
                value
                for value in (stored.parent_ref, stored.supersedes_ref)
                if value
            ],
            source="task_execution_pipeline",
        )
        current = self.ledger.latest("contract_bundle", ledger_id)
        if current == record.model_dump(mode="json"):
            return []
        return [self.ledger.append("contract_bundle", record, actor=actor)]

    def prepare(
        self,
        product: Any,
        bundle: TaskContractBundle,
        *,
        run_id: str,
        parent_ref: str = "",
        supersedes_ref: str = "",
        actor: str = "task-pipeline",
        launch_request: TrainingLaunchRequest | Mapping[str, Any] | None = None,
    ) -> TaskPipelineResult:
        """生成合同、artifact、payload 和运行交接；不启动远程进程。"""
        self._validate_bundle_product(bundle, product)
        run_id = _safe_id(run_id, "run_id")
        stored = self.contract_store.save(
            bundle,
            parent_ref=parent_ref,
            supersedes_ref=supersedes_ref,
            actor=actor,
        )
        materialized = self.materializer.materialize(
            bundle,
            _materialization_root(self.output_root, bundle),
            product=product,
            bundle_ref=stored.ref,
        )
        payload = build_product_payload(
            contract=product,
            output_dir=self.output_root / "payloads" / run_id,
            task_bundle=bundle,
            task_artifacts=materialized,
        )

        launch_plan: TrainingLaunchPlan | None = None
        if launch_request is not None:
            # 生成远程启动计划会改变外部状态，draft 合同只能物料化，不能进入启动路径。
            bundle.contract.require_approved()
            request = (
                launch_request
                if isinstance(launch_request, TrainingLaunchRequest)
                else TrainingLaunchRequest.from_mapping(launch_request)
            )
            if request.run_id != run_id:
                raise TaskPipelineError("launch_request.run_id must match run_id")
            launch_plan = build_training_launch_plan(product, request)
        run_manifest = self.output_root / "runs" / run_id / "runtime_manifest.json"
        lineage: dict[str, Any] = {}
        if launch_plan is not None:
            lineage = {
                "resume": launch_plan.resume,
                "source_run": launch_plan.source_run,
                "checkpoint": launch_plan.checkpoint,
                "remote_boot_id": launch_plan.remote_boot_id,
                "resume_state": dict(
                    (launch_request.resume_state if isinstance(launch_request, TrainingLaunchRequest) else {})
                ),
            }
            resume_state = dict(lineage["resume_state"])
            lineage["source_task_contract_ref"] = str(
                resume_state.get("source_task_contract_ref") or ""
            )
            lineage["source_task_bundle_digest"] = str(
                resume_state.get("source_task_bundle_digest") or ""
            )
        manifest: dict[str, Any] = {
            "schema_version": TASK_PIPELINE_SCHEMA,
            "generated_at": _utc_now(),
            "product": {
                "id": str(_product_data(product).get("product_id") or ""),
                "version": str(_product_data(product).get("product_version") or ""),
                "contract_digest": str(_product_data(product).get("contract_digest") or ""),
            },
            "resolved_contract": {
                "product_id": str(_product_data(product).get("product_id") or ""),
                "product_version": str(_product_data(product).get("product_version") or ""),
                "config_digest": str(_product_data(product).get("config_digest") or ""),
                "source_digest": str(_product_data(product).get("source_digest") or ""),
                "runtime": dict(_product_data(product).get("runtime") or {}),
                "compatibility": dict(_product_data(product).get("compatibility") or {}),
                "training": dict(_product_data(product).get("training") or {}),
            },
            "run": {
                "run_id": run_id,
                "task": bundle.contract.contract_id,
                "task_contract_ref": stored.ref,
                "resume_checkpoint": lineage.get("checkpoint", ""),
            },
            "contract_bundle_ref": stored.ref,
            "task_contract_ref": stored.ref,
            "task_bundle_digest": bundle.bundle_digest,
            "task_artifact_manifest_ref": materialized.manifest_ref,
            "task_artifacts": materialized.to_dict(),
            "payload": payload.to_dict(),
            "execution": {
                "task_contract_ref": stored.ref,
                "task_contract_digest": bundle.contract.contract_digest,
                "task_bundle_digest": bundle.bundle_digest,
                "task_artifact_manifest_ref": materialized.manifest_ref,
                "payload_digest": payload.payload_digest,
                "runtime_digest": str(_product_data(product).get("runtime", {}).get("digest", "")),
            },
            "lineage": lineage,
            "configuration": {
                "training_artifact_ref": materialized.artifact_ref("training"),
                "telemetry_artifact_ref": materialized.artifact_ref("telemetry"),
                "diagnostics_artifact_ref": materialized.artifact_ref("diagnostics"),
                "simulation_artifact_ref": materialized.artifact_ref("simulation"),
                "deployment_artifact_ref": materialized.artifact_ref("deployment"),
            },
        }
        if launch_plan is not None:
            manifest["launch_plan"] = launch_plan.to_dict()
            if launch_plan.resume:
                restored = {
                    str(key): bool(value)
                    for key, value in lineage.get("resume_state", {}).items()
                    if isinstance(value, bool)
                }
                manifest["resume_edge"] = {
                    "parent_checkpoint_ref": launch_plan.checkpoint,
                    "parent_task_contract_ref": str(
                        lineage.get("source_task_contract_ref") or ""
                    ),
                    "parent_task_contract_digest": str(
                        resume_state.get("source_task_contract_digest") or ""
                    ),
                    "child_task_contract_ref": stored.ref,
                    "child_task_contract_digest": bundle.contract.contract_digest,
                    "child_bundle_digest": bundle.bundle_digest,
                    "child_payload_digest": payload.payload_digest,
                    "restored": restored,
                    "reset_reason": {},
                    # 这里只能证明“已声明”，运行时 preflight 成功后再升级为 proven。
                    "status": "declared",
                }

        # 先写入路径，再构造 spec，保证执行层拿到的 manifest 已经存在。
        _atomic_json(run_manifest, manifest)
        deployment_spec = make_deployment_spec(
            product,
            payload,
            run_id=run_id,
            run_manifest=run_manifest,
            task_bundle=bundle,
            task_artifacts=materialized,
        )
        deployment_spec.validate()
        manifest["deployment"] = _deployment_dict(deployment_spec)
        _atomic_json(run_manifest, manifest)

        ledger_events = self._register_contract(stored, bundle, materialized, actor=actor)
        ledger_events.extend(self.ledger.import_runtime_manifest(run_manifest, actor=actor))
        return TaskPipelineResult(
            stored_contract=stored,
            materialized=materialized,
            payload=payload,
            deployment_spec=deployment_spec,
            run_manifest=run_manifest,
            launch_plan=launch_plan,
            ledger_events=tuple(ledger_events),
        )

    def revise(
        self,
        product: Any,
        base: TaskContractBundle,
        approved_change: Mapping[str, Any],
        *,
        evidence_refs: tuple[str, ...] | list[str],
        approved_by: str,
        actor: str = "task-pipeline",
    ) -> tuple[TaskContractBundle, StoredTaskContract]:
        """保存一条证据驱动的合同修订谱系，旧版本保持不可变。"""
        revised = revise_task_bundle(
            product,
            base,
            approved_change,
            evidence_refs=evidence_refs,
            approved_by=approved_by,
        )
        base_ref = _task_ref(base)
        stored = self.contract_store.save(
            revised,
            parent_ref=base_ref,
            supersedes_ref=base_ref,
            actor=actor,
        )
        return revised, stored


__all__ = [
    "TASK_PIPELINE_SCHEMA",
    "TaskExecutionPipeline",
    "TaskPipelineError",
    "TaskPipelineResult",
]
