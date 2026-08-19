"""任务合同编译器。

本模块是产品输入与系统各子系统之间的确定性投影层。LLM 或人工可以
提供结构化的任务意图，但不能通过本模块直接执行代码、修改奖励实现或
改变验收规则。编译结果会固定训练、遥测、诊断、仿真和部署规格，并用
内容摘要绑定它们，避免不同子系统各自猜测同一任务的含义。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .contracts import ResolvedProductContract


TASK_CONTRACT_SCHEMA = "rl-agent.task-contract/v1"
BUNDLE_SCHEMA = "rl-agent.task-bundle/v1"
COMPILER_VERSION = "task-contract-compiler/1"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")

# 这些字段描述产品实现和远程运行布局，任务意图只能消费，不能改写。
_PRODUCT_OWNED_FIELDS: Mapping[str, frozenset[str]] = {
    "training": frozenset(
        {
            "task_id",
            "runtime_task_ids",
            "task_family",
            "framework_id",
            "train_entrypoint",
            "config_path",
            "source_roots",
            "sources",
            "requirements",
            "knowledge",
        }
    ),
    "telemetry": frozenset({"source", "contract"}),
    "diagnostics": frozenset({"entrypoint", "spec", "spec_asset", "suite_root"}),
    "simulation": frozenset({"worlds"}),
    "deployment": frozenset(
        {
            "payload_builder",
            "payload_package",
            "runtime_package",
            "remote_root",
            "legacy_payload_root",
            "runs_root",
            "data_root",
            "runtime_python",
            "launcher_protocol",
            "payload_entrypoints",
            "privileged_information_required",
        }
    ),
}


class TaskContractError(ValueError):
    """任务意图或编译结果不满足合同边界时抛出。"""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TaskContractError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _id(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not _ID_RE.fullmatch(result):
        raise TaskContractError(f"{name} has unsafe or empty id: {result!r}")
    return result


def _unique_strings(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TaskContractError(f"{name} must be a list")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    if len(result) != len(set(result)):
        raise TaskContractError(f"{name} contains duplicates")
    return result


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并配置；标量和列表由任务意图整体替换，规则保持可预测。"""
    result = {str(key): value for key, value in base.items()}
    for key, value in overlay.items():
        key = str(key)
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _reject_product_owned_overrides(section: str, overlay: Mapping[str, Any]) -> None:
    protected = _PRODUCT_OWNED_FIELDS.get(section, frozenset())
    conflicts = sorted(str(key) for key in overlay if str(key) in protected)
    if conflicts:
        paths = ", ".join(f"{section}.{key}" for key in conflicts)
        raise TaskContractError(f"task request cannot override product-owned fields: {paths}")


def _merge_hard_constraints(
    required: Mapping[str, Any],
    requested: Mapping[str, Any],
    *,
    path: str = "constraints",
) -> dict[str, Any]:
    """合并硬约束；任务可以补充约束，但不能削弱产品已声明的要求。"""
    result = {str(key): value for key, value in required.items()}
    for raw_key, value in requested.items():
        key = str(raw_key)
        child_path = f"{path}.{key}"
        if key not in result:
            result[key] = value
            continue
        current = result[key]
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _merge_hard_constraints(current, value, path=child_path)
        elif current != value:
            raise TaskContractError(
                f"task constraint conflicts with product requirement at {child_path}: "
                f"required={current!r}, requested={value!r}"
            )
    return result


def _product_data(product: ResolvedProductContract | Mapping[str, Any]) -> dict[str, Any]:
    if hasattr(product, "to_dict") and callable(product.to_dict):
        product = product.to_dict()
    if not isinstance(product, Mapping):
        raise TaskContractError("product contract must be a mapping")
    return dict(product)


@dataclass(frozen=True)
class TaskRequest:
    """LLM/人工提交的结构化任务意图，不是可直接执行的配置。"""

    instance_id: str
    objective: str
    goals: tuple[Mapping[str, Any], ...] = ()
    constraints: Mapping[str, Any] = field(default_factory=dict)
    protected_capabilities: tuple[str, ...] = ()
    baseline: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] = field(default_factory=dict)
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    simulation: Mapping[str, Any] = field(default_factory=dict)
    deployment: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    approved: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskRequest":
        data = _mapping(value, "task_request")
        task = _mapping(data.get("task"), "task_request.task") if "task" in data else data
        raw_goals = task.get("goals", ())
        if isinstance(raw_goals, str):
            raw_goals = (raw_goals,)
        if not isinstance(raw_goals, (list, tuple)):
            raise TaskContractError("task_request.goals must be a list")
        goals: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_goals):
            if isinstance(raw, str):
                goal = {"id": f"goal-{index + 1}", "objective": raw.strip()}
            else:
                goal = _mapping(raw, f"task_request.goals[{index}]")
            goal_id = _id(goal.get("id", f"goal-{index + 1}"), f"task_request.goals[{index}].id")
            text = str(goal.get("objective") or goal.get("description") or "").strip()
            if not text:
                raise TaskContractError(f"task_request.goals[{index}] needs objective")
            if goal_id in seen:
                raise TaskContractError(f"duplicate goal id: {goal_id}")
            seen.add(goal_id)
            goals.append({**goal, "id": goal_id, "objective": text})
        objective = str(task.get("objective") or "").strip()
        if not objective:
            raise TaskContractError("task_request.objective is required")
        return cls(
            instance_id=_id(task.get("instance_id"), "task_request.instance_id"),
            objective=objective,
            goals=tuple(goals),
            constraints=_mapping(task.get("constraints"), "task_request.constraints"),
            protected_capabilities=_unique_strings(
                task.get("protected_capabilities"), "task_request.protected_capabilities"
            ),
            baseline=_mapping(task.get("baseline"), "task_request.baseline"),
            training=_mapping(task.get("training"), "task_request.training"),
            telemetry=_mapping(task.get("telemetry"), "task_request.telemetry"),
            diagnostics=_mapping(task.get("diagnostics"), "task_request.diagnostics"),
            simulation=_mapping(task.get("simulation"), "task_request.simulation"),
            deployment=_mapping(task.get("deployment"), "task_request.deployment"),
            metadata=_mapping(task.get("metadata"), "task_request.metadata"),
            approved=bool(task.get("approved", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class ResolvedTaskContract:
    """由产品合同和任务意图共同确定的不可变任务合同。"""

    schema_version: str
    compiler_version: str
    contract_id: str
    contract_version: int
    status: str
    product_id: str
    product_contract_digest: str
    request_digest: str
    objective: str
    goals: tuple[Mapping[str, Any], ...]
    constraints: Mapping[str, Any]
    protected_capabilities: tuple[str, ...]
    baseline: Mapping[str, Any]
    training: Mapping[str, Any]
    telemetry: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    simulation: Mapping[str, Any]
    deployment: Mapping[str, Any]
    runtime: Mapping[str, Any]
    compatibility: Mapping[str, Any]
    provenance: Mapping[str, Any]
    contract_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["goals"] = [dict(goal) for goal in self.goals]
        return data

    def _unsigned(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("contract_digest", None)
        return data

    def digest(self) -> str:
        return _digest(self._unsigned())

    def with_digest(self) -> "ResolvedTaskContract":
        return ResolvedTaskContract(**{**self.to_dict(), "goals": self.goals, "contract_digest": self.digest()})

    def require_approved(self) -> None:
        """所有会改变远程状态的执行交接都必须经过显式批准。"""
        if self.status != "approved":
            raise TaskContractError(
                f"task contract {self.contract_id!r} is {self.status!r}; approved contract required for execution"
            )

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_canonical_json(self.to_dict()) + "\n", encoding="utf-8")
        return target


@dataclass(frozen=True)
class TaskContractBundle:
    """合同及其派生规格的可复现产物集合。"""

    contract: ResolvedTaskContract
    specs: Mapping[str, Mapping[str, Any]]
    bundle_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BUNDLE_SCHEMA,
            "bundle_digest": self.bundle_digest,
            "contract_digest": self.contract.contract_digest,
            "contract_id": self.contract.contract_id,
            "specs": {key: dict(value) for key, value in self.specs.items()},
        }

    def write(self, root: str | Path) -> Path:
        """写入生成目录；只写确定性文件，不覆盖产品源文件。"""
        target = Path(root)
        target.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}
        contract_path = self.contract.write(target / "task_contract.json")
        written[contract_path.name] = _digest(json.loads(contract_path.read_text(encoding="utf-8")))
        for name, spec in sorted(self.specs.items()):
            path = target / f"{name}_spec.json"
            path.write_text(_canonical_json(spec) + "\n", encoding="utf-8")
            written[path.name] = _digest(spec)
        manifest = {
            "schema_version": BUNDLE_SCHEMA,
            "bundle_digest": self.bundle_digest,
            "contract_digest": self.contract.contract_digest,
            "contract_id": self.contract.contract_id,
            "files": written,
        }
        (target / "bundle_manifest.json").write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        return target

    def deployment_spec(
        self,
        *,
        payload_archive: str | Path,
        payload_manifest: str | Path,
        run_id: str,
        run_manifest: Mapping[str, Any] | str | Path,
        payload_digest: str = "",
    ) -> Any:
        """把已构建 payload 绑定到通用执行层，不让执行层反向依赖产品。"""
        from autotuner.execution import DeploymentSpec

        self.contract.require_approved()
        return DeploymentSpec(
            runtime=self.contract.runtime,
            runtime_digest=str(self.contract.runtime.get("digest") or ""),
            payload_archive=payload_archive,
            payload_manifest=payload_manifest,
            run_id=run_id,
            run_manifest=run_manifest,
            payload_digest=payload_digest,
        )


class TaskContractCompiler:
    """将任务意图编译为合同；没有副作用，也不执行 LLM 输出。"""

    def __init__(self, *, compiler_version: str = COMPILER_VERSION):
        self.compiler_version = compiler_version

    def compile(
        self,
        product: ResolvedProductContract | Mapping[str, Any],
        request: TaskRequest | Mapping[str, Any],
        *,
        contract_version: int = 1,
        contract_id: str | None = None,
    ) -> ResolvedTaskContract:
        product_data = _product_data(product)
        product_id = _id(product_data.get("product_id"), "product_id")
        product_digest = str(product_data.get("contract_digest") or "").strip()
        if not product_digest:
            raise TaskContractError("product contract_digest is required")
        task = request if isinstance(request, TaskRequest) else TaskRequest.from_mapping(request)
        if contract_version <= 0:
            raise TaskContractError("contract_version must be positive")
        resolved_id = _id(contract_id or f"{task.instance_id}.{product_id}", "contract_id")
        for section, overlay in (
            ("training", task.training),
            ("telemetry", task.telemetry),
            ("diagnostics", task.diagnostics),
            ("simulation", task.simulation),
            ("deployment", task.deployment),
        ):
            _reject_product_owned_overrides(section, overlay)

        product_training = _mapping(product_data.get("training"), "product.training")
        training = _deep_merge(product_training, task.training)
        telemetry = _deep_merge(_mapping(product_data.get("telemetry"), "product.telemetry"), task.telemetry)
        diagnostics = _deep_merge(_mapping(product_data.get("diagnostics"), "product.diagnostics"), task.diagnostics)
        simulation = _deep_merge(_mapping(product_data.get("simulation"), "product.simulation"), task.simulation)
        deployment = _deep_merge(_mapping(product_data.get("deployment"), "product.deployment"), task.deployment)

        product_constraints = _merge_hard_constraints(
            _mapping(product_training.get("requirements"), "product.training.requirements"),
            _mapping(product_training.get("constraints"), "product.training.constraints"),
        )
        requested_constraints = _merge_hard_constraints(
            _mapping(task.training.get("constraints"), "task_request.training.constraints"),
            task.constraints,
        )
        constraints = _merge_hard_constraints(product_constraints, requested_constraints)
        training["constraints"] = constraints
        protected = tuple(dict.fromkeys(task.protected_capabilities))
        status = "approved" if task.approved else "draft"
        provenance = {
            "compiler": self.compiler_version,
            "product_contract_digest": product_digest,
            "request_digest": task.digest(),
            "approval_required": not task.approved,
            "source": "structured_task_request",
        }
        contract = ResolvedTaskContract(
            schema_version=TASK_CONTRACT_SCHEMA,
            compiler_version=self.compiler_version,
            contract_id=resolved_id,
            contract_version=int(contract_version),
            status=status,
            product_id=product_id,
            product_contract_digest=product_digest,
            request_digest=task.digest(),
            objective=task.objective,
            goals=task.goals,
            constraints=constraints,
            protected_capabilities=protected,
            baseline=dict(task.baseline),
            training=training,
            telemetry=telemetry,
            diagnostics=diagnostics,
            simulation=simulation,
            deployment=deployment,
            runtime=_mapping(product_data.get("runtime"), "product.runtime"),
            compatibility=_mapping(product_data.get("compatibility"), "product.compatibility"),
            provenance=provenance,
        )
        return contract.with_digest()

    def compile_bundle(
        self,
        product: ResolvedProductContract | Mapping[str, Any],
        request: TaskRequest | Mapping[str, Any],
        *,
        contract_version: int = 1,
        contract_id: str | None = None,
    ) -> TaskContractBundle:
        contract = self.compile(
            product,
            request,
            contract_version=contract_version,
            contract_id=contract_id,
        )
        shared = {
            "schema_version": BUNDLE_SCHEMA,
            "contract_id": contract.contract_id,
            "contract_digest": contract.contract_digest,
        }
        specs = {
            name: {**shared, "kind": name, "spec": dict(value)}
            for name, value in (
                ("training", contract.training),
                ("telemetry", contract.telemetry),
                ("diagnostics", contract.diagnostics),
                ("simulation", contract.simulation),
                ("deployment", contract.deployment),
            )
        }
        bundle_digest = _digest(
            {
                "compiler": self.compiler_version,
                "contract_digest": contract.contract_digest,
                "specs": specs,
            }
        )
        return TaskContractBundle(contract=contract, specs=specs, bundle_digest=bundle_digest)


def compile_task_bundle(
    product: ResolvedProductContract | Mapping[str, Any],
    request: TaskRequest | Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    contract_version: int = 1,
    contract_id: str | None = None,
) -> TaskContractBundle:
    """公共入口：编译合同，必要时写入生成产物目录。"""
    bundle = TaskContractCompiler().compile_bundle(
        product,
        request,
        contract_version=contract_version,
        contract_id=contract_id,
    )
    if output_root is not None:
        bundle.write(output_root)
    return bundle


__all__ = [
    "BUNDLE_SCHEMA",
    "COMPILER_VERSION",
    "ResolvedTaskContract",
    "TASK_CONTRACT_SCHEMA",
    "TaskContractBundle",
    "TaskContractCompiler",
    "TaskContractError",
    "TaskRequest",
    "compile_task_bundle",
]
