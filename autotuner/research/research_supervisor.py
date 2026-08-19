"""Research lifecycle guards and the bounded experiment supervisor.

The pure policy functions remain usable by the API without side effects.  The
execution classes below add an explicit, leased boundary for experiments:
candidate files are copied into an isolated directory, the backend is the only
process launcher, and promotion changes a managed pointer atomically.  No
checkpoint or payload is deleted as part of rollback.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from .research_ledger import (
    BaselineSet,
    ContractBundle,
    EvidenceRecord,
    ExperimentPlan,
)


_EXPERIMENT_ENV_EXACT = frozenset({"CUDA_VISIBLE_DEVICES"})
_EXPERIMENT_ENV_PREFIXES = ("RL_",)


def _now_epoch() -> float:
    return time.time()


def _safe_name(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:100] or "experiment"


@dataclass(frozen=True)
class ResourceLease:
    lease_id: str
    resources: tuple[str, ...]
    owner: str
    expires_at: float


class ResourceLeaseStore:
    """Small filesystem lease preventing concurrent GPU/Isaac experiments."""

    def __init__(self, root: str | os.PathLike[str] = "output/research_leases") -> None:
        self.root = Path(root)
        self.path = self.root / "active.json"

    @contextmanager
    def _locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".lock"
        deadline = _now_epoch() + 5.0
        fd: int | None = None
        while fd is None:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as exc:
                try:
                    stale = _now_epoch() - lock.stat().st_mtime > 120.0
                except FileNotFoundError:
                    continue
                if stale:
                    lock.unlink(missing_ok=True)
                    continue
                if _now_epoch() >= deadline:
                    raise RuntimeError("resource lease store is busy") from exc
                time.sleep(0.05)
        try:
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
            yield
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)

    def _read_leases(self) -> list[ResourceLease]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("resource lease state is corrupt; manual review required") from exc
        raw = payload.get("leases", []) if isinstance(payload, dict) and "leases" in payload else [payload]
        leases: list[ResourceLease] = []
        for item in raw:
            if not isinstance(item, dict):
                raise RuntimeError("resource lease state is corrupt; manual review required")
            try:
                leases.append(ResourceLease(
                    lease_id=str(item["lease_id"]),
                    resources=tuple(str(value) for value in item["resources"]),
                    owner=str(item["owner"]),
                    expires_at=float(item["expires_at"]),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("resource lease state is corrupt; manual review required") from exc
        return leases

    def _write_leases(self, leases: Sequence[ResourceLease]) -> None:
        if not leases:
            self.path.unlink(missing_ok=True)
            return
        payload = {
            "version": 1,
            "leases": [
                {
                    "lease_id": item.lease_id,
                    "resources": list(item.resources),
                    "owner": item.owner,
                    "expires_at": item.expires_at,
                }
                for item in leases
            ],
        }
        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, self.path)

    def acquire(self, resources: Sequence[str], *, owner: str, ttl_s: float = 3600.0) -> ResourceLease:
        requested = tuple(sorted(set(str(item) for item in resources if str(item).strip())))
        if not requested:
            raise ValueError("at least one resource is required")
        if not owner.strip():
            raise ValueError("lease owner is required")
        with self._locked():
            active = [item for item in self._read_leases() if item.expires_at > _now_epoch()]
            conflicts = sorted({
                resource
                for item in active
                for resource in set(requested) & set(item.resources)
            })
            if conflicts:
                raise RuntimeError(f"resources already leased: {', '.join(conflicts)}")
            lease = ResourceLease(
                lease_id=f"lease:{uuid.uuid4().hex}", resources=requested,
                owner=owner, expires_at=_now_epoch() + max(float(ttl_s), 1.0),
            )
            self._write_leases([*active, lease])
            return lease

    def release(self, lease: ResourceLease) -> None:
        with self._locked():
            active = [
                item for item in self._read_leases()
                if item.lease_id != lease.lease_id and item.expires_at > _now_epoch()
            ]
            self._write_leases(active)


@dataclass(frozen=True)
class BackendHandle:
    id: str
    started_at: float


@dataclass(frozen=True)
class BackendStatus:
    state: Literal["running", "succeeded", "failed", "stopped"]
    step: int = 0
    message: str = ""
    metrics: Mapping[str, Any] | None = None


class ExperimentBackend(Protocol):
    def start(self, plan: ExperimentPlan, workspace: Path, environment: Mapping[str, str]) -> BackendHandle: ...
    def poll(self, handle: BackendHandle) -> BackendStatus: ...
    def stop(self, handle: BackendHandle, reason: str) -> None: ...
    def evaluate(self, plan: ExperimentPlan, workspace: Path) -> Mapping[str, Any]: ...


class CommandExperimentBackend:
    """Reference backend for bounded local/remote command wrappers.

    Commands are argument arrays from an approved plan; shell interpretation is
    intentionally disabled.  Real SSH/GPU adapters can implement the same
    protocol without changing the supervisor.
    """

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._streams: dict[str, Any] = {}
        self._log_paths: dict[str, Path] = {}
        self._environments: dict[str, dict[str, str]] = {}
        self._workspace_handles: dict[str, str] = {}

    @staticmethod
    def _command(plan: ExperimentPlan, key: str) -> list[str]:
        value = plan.training_window.get(key, [])
        if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
            raise ValueError(f"approved experiment requires training_window.{key} as a non-empty argv list")
        return value

    def start(self, plan: ExperimentPlan, workspace: Path, environment: Mapping[str, str]) -> BackendHandle:
        command = self._command(plan, "command")
        handle = BackendHandle(id=f"process:{uuid.uuid4().hex}", started_at=_now_epoch())
        log_path = workspace / "training.log"
        stream = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            command, cwd=workspace, env=dict(environment), stdout=stream,
            stderr=subprocess.STDOUT, text=True,
        )
        self._processes[handle.id] = proc
        self._streams[handle.id] = stream
        self._log_paths[handle.id] = log_path
        self._environments[handle.id] = dict(environment)
        self._workspace_handles[str(workspace.resolve())] = handle.id
        return handle

    def _close_stream(self, handle_id: str) -> None:
        stream = self._streams.pop(handle_id, None)
        if stream is not None:
            stream.flush()
            stream.close()

    def _log_tail(self, handle_id: str) -> str:
        path = self._log_paths.get(handle_id)
        if path is None or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")[-2000:]

    def poll(self, handle: BackendHandle) -> BackendStatus:
        proc = self._processes[handle.id]
        code = proc.poll()
        if code is None:
            return BackendStatus("running")
        self._close_stream(handle.id)
        return BackendStatus("succeeded" if code == 0 else "failed", message=self._log_tail(handle.id))

    def stop(self, handle: BackendHandle, reason: str) -> None:
        proc = self._processes.get(handle.id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._close_stream(handle.id)

    def evaluate(self, plan: ExperimentPlan, workspace: Path) -> Mapping[str, Any]:
        command = plan.evaluation_plan.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError("independent evaluator command is missing")
        handle_id = self._workspace_handles.get(str(workspace.resolve()))
        environment = self._environments.get(handle_id or "", os.environ.copy())
        proc = subprocess.run(
            command,
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=plan.resource_budget.get("evaluation_timeout_s", 900),
        )
        if proc.returncode != 0:
            return {"success": False, "reason": "evaluator_failed", "stderr": proc.stderr[-2000:]}
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            return {"success": False, "reason": "evaluator_output_not_json", "stdout": proc.stdout[-2000:], "error": str(exc)}


@dataclass(frozen=True)
class ExperimentExecution:
    experiment_ref: str
    disposition: Literal["promote", "rollback", "inconclusive"]
    workspace: str
    lease_id: str = ""
    backend_state: str = ""
    stop_reason: str = ""
    evaluation: Mapping[str, Any] | None = None
    protected_results: Mapping[str, Any] | None = None


class ResearchSupervisor:
    """Execute one approved experiment with explicit resource and artifact boundaries."""

    def __init__(
        self,
        root: str | os.PathLike[str] = "output/research_experiments",
        *,
        lease_store: ResourceLeaseStore | None = None,
        backend: ExperimentBackend | None = None,
    ) -> None:
        self.root = Path(root)
        self.lease_store = lease_store or ResourceLeaseStore(self.root / "leases")
        self.backend = backend or CommandExperimentBackend()

    def _workspace(self, plan: ExperimentPlan) -> Path:
        return self.root / "runs" / f"{_safe_name(plan.id)}-{uuid.uuid4().hex[:10]}"

    @staticmethod
    def _copy_candidate(candidate_artifact: str | os.PathLike[str], workspace: Path) -> Path:
        source = Path(candidate_artifact).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        destination = workspace / "candidate"
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.mkdir(parents=True)
            shutil.copy2(source, destination / source.name)
        return destination

    @staticmethod
    def _protected_results(evaluation: Mapping[str, Any], plan: ExperimentPlan) -> dict[str, Any]:
        declared = evaluation.get("protected", {})
        results: dict[str, Any] = {}
        for capability in plan.protected_capabilities:
            item = declared.get(capability) if isinstance(declared, Mapping) else None
            results[capability] = item if isinstance(item, Mapping) else {"passed": False, "reason": "missing evaluator result"}
        return results

    @staticmethod
    def _environment(
        overrides: Mapping[str, str] | None,
        plan: ExperimentPlan,
    ) -> dict[str, str]:
        declaration = plan.training_window.get("environment_policy", {})
        declaration = declaration if isinstance(declaration, Mapping) else {}
        raw_prefixes = declaration.get("prefixes", ("RL_",))
        raw_exact = declaration.get("exact", ("CUDA_VISIBLE_DEVICES",))
        prefixes = tuple(str(item).strip() for item in raw_prefixes) if isinstance(raw_prefixes, (list, tuple)) else ()
        exact = tuple(str(item).strip() for item in raw_exact) if isinstance(raw_exact, (list, tuple)) else ()
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*_", item) for item in prefixes):
            raise ValueError("training_window.environment_policy.prefixes must be safe names ending in '_'")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item) for item in exact):
            raise ValueError("training_window.environment_policy.exact must contain safe names")
        supplied = {str(key): str(value) for key, value in (overrides or {}).items()}
        invalid = sorted(
            key for key in supplied
            if key not in exact
            and not key.startswith(prefixes)
        )
        if invalid:
            raise ValueError(f"experiment environment keys are not allowlisted: {', '.join(invalid)}")
        result = os.environ.copy()
        result.update(supplied)
        return result

    @staticmethod
    def _decide(evaluation: Mapping[str, Any], protected: Mapping[str, Any]) -> Literal["promote", "rollback", "inconclusive"]:
        if not evaluation:
            return "inconclusive"
        if any(not bool(item.get("passed")) for item in protected.values()):
            return "rollback"
        if evaluation.get("success") is True and evaluation.get("evidence_complete", True) is True:
            return "promote"
        if evaluation.get("success") is False:
            return "rollback"
        return "inconclusive"

    def _set_active_pointer(self, candidate: Path, *, experiment_ref: str) -> None:
        pointer = self.root / "ACTIVE_MECHANISM.json"
        temp = pointer.with_name(f".{pointer.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps({"experiment_ref": experiment_ref, "candidate": str(candidate), "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, pointer)

    def execute(
        self,
        plan: ExperimentPlan,
        candidate_artifact: str | os.PathLike[str],
        *,
        actor: str,
        environment: Mapping[str, str] | None = None,
        poll_interval_s: float = 1.0,
    ) -> ExperimentExecution:
        if plan.status not in {"approved", "executing"}:
            raise RuntimeError(f"experiment is not approved: {plan.status}")
        resources = plan.resource_budget.get("resources", ["gpu:0"])
        if not isinstance(resources, list):
            raise ValueError("resource_budget.resources must be a list")
        ttl = float(plan.resource_budget.get("lease_ttl_s", 3600.0))
        lease = self.lease_store.acquire(resources, owner=actor, ttl_s=ttl)
        try:
            workspace = self._workspace(plan)
            workspace.mkdir(parents=True, exist_ok=False)
            candidate = self._copy_candidate(candidate_artifact, workspace)
            (workspace / "execution.json").write_text(json.dumps({
                "experiment_ref": plan.id, "candidate_artifact": str(candidate_artifact),
                "lease_id": lease.lease_id, "actor": actor,
            }, indent=2) + "\n", encoding="utf-8")
            handle: BackendHandle | None = None
            status = BackendStatus("failed", message="not started")
            stop_reason = ""
            env = self._environment(environment, plan)
            env["RL_RESEARCH_EXPERIMENT_REF"] = plan.id
            env["RL_RESEARCH_CANDIDATE_ROOT"] = str(candidate)
            mechanism_bundle = candidate / "mechanisms.json"
            if not mechanism_bundle.is_file():
                raise FileNotFoundError(f"candidate mechanism bundle is missing: {mechanism_bundle}")
            env["RL_MECHANISM_BUNDLE"] = str(mechanism_bundle)
            handle = self.backend.start(plan, workspace, env)
            max_seconds = float(plan.training_window.get("max_seconds", plan.resource_budget.get("max_seconds", 3600.0)))
            started = _now_epoch()
            while True:
                status = self.backend.poll(handle)
                if status.state != "running":
                    break
                if _now_epoch() - started >= max_seconds:
                    stop_reason = "training_window_timeout"
                    self.backend.stop(handle, stop_reason)
                    status = BackendStatus("stopped", status.step, stop_reason, status.metrics)
                    break
                if poll_interval_s > 0:
                    time.sleep(poll_interval_s)
            if status.state in {"failed", "stopped"}:
                return ExperimentExecution(plan.id, "rollback", str(workspace), lease.lease_id, status.state, stop_reason or status.message)
            evaluation = self.backend.evaluate(plan, workspace)
            protected = self._protected_results(evaluation, plan)
            disposition = self._decide(evaluation, protected)
            if disposition == "promote":
                self._set_active_pointer(candidate, experiment_ref=plan.id)
            return ExperimentExecution(plan.id, disposition, str(workspace), lease.lease_id, status.state, evaluation=evaluation, protected_results=protected)
        finally:
            self.lease_store.release(lease)


LIFECYCLE_TRANSITIONS: dict[str, dict[str, set[str]]] = {
    "run": {
        "CREATED": {"VALIDATED", "FAILED"},
        "VALIDATED": {"RUNNING", "FAILED"},
        "RUNNING": {"STOPPING", "FAILED"},
        "STOPPING": {"STOPPED", "FAILED"},
        "STOPPED": set(),
        "FAILED": set(),
    },
    "research_case": {
        "OBSERVED": {"ATTRIBUTING", "EVIDENCE_NEEDED", "OPEN"},
        "ATTRIBUTING": {"EVIDENCE_NEEDED", "PROPOSED", "OPEN"},
        "EVIDENCE_NEEDED": {"PROPOSED", "OPEN"},
        "PROPOSED": {"EXPERIMENTING", "OPEN"},
        "EXPERIMENTING": {"EVALUATING", "OPEN"},
        "EVALUATING": {"RESOLVED", "OPEN"},
        "OPEN": {"ATTRIBUTING", "EVIDENCE_NEEDED", "PROPOSED", "EXPERIMENTING"},
        "RESOLVED": set(),
    },
    "job": {
        "QUEUED": {"LEASED", "CANCELLED"},
        "LEASED": {"RUNNING", "CANCELLED", "FAILED"},
        "RUNNING": {"SUCCEEDED", "FAILED", "CANCELLED"},
        "SUCCEEDED": set(),
        "FAILED": set(),
        "CANCELLED": set(),
    },
}


def lifecycle_transition(kind: str, current: str, target: str) -> dict[str, Any]:
    allowed = LIFECYCLE_TRANSITIONS.get(kind, {}).get(current, set())
    return {
        "kind": kind,
        "current": current,
        "target": target,
        "status": "allowed" if target in allowed else "blocked",
        "allowed_targets": sorted(allowed),
        "reason": "" if target in allowed else f"{kind} cannot transition {current} -> {target}",
    }


class EvidencePolicyDecision:
    def __init__(self, *, required_kinds: list[str], default_action: str, rationale: str):
        self.required_kinds = required_kinds
        self.default_action = default_action
        self.rationale = rationale

    def model_dump(self) -> dict[str, Any]:
        return {
            "required_kinds": self.required_kinds,
            "default_action": self.default_action,
            "rationale": self.rationale,
        }


def evidence_policy(*, log_quality: Literal["bad", "good", "unknown"], behavior_quality: Literal["known", "unknown", "contradictory"], deployment_mismatch: bool = False, high_risk: bool = False) -> EvidencePolicyDecision:
    """Choose the minimum evidence class; never force a full battery by default."""
    if deployment_mismatch:
        return EvidencePolicyDecision(
            required_kinds=["source_audit", "sim2sim"],
            default_action="block_intervention",
            rationale="IsaacLab/MuJoCo/deployment parity must be established before tuning attribution.",
        )
    if log_quality == "bad" and behavior_quality == "known":
        return EvidencePolicyDecision(
            required_kinds=["telemetry", "source_audit"],
            default_action="inspect_mechanism_without_full_diagnostic",
            rationale="A clearly bad log is already sufficient to reject the current progression claim.",
        )
    if behavior_quality in {"unknown", "contradictory"} or high_risk:
        return EvidencePolicyDecision(
            required_kinds=["physical_diag", "video_observation", "telemetry"],
            default_action="block_intervention",
            rationale="Behavior and metric disagreement requires aligned raw traces and observation context.",
        )
    return EvidencePolicyDecision(
        required_kinds=["telemetry"],
        default_action="continue_observation",
        rationale="Use the smallest evidence window that can distinguish trend from noise.",
    )


def experiment_readiness(
    plan: ExperimentPlan,
    contract: ContractBundle,
    baseline: BaselineSet,
    *,
    policy_parity_status: str = "unknown",
    evidence: list[EvidenceRecord] | None = None,
) -> dict[str, Any]:
    """Check protected capabilities and deployment preconditions before execution."""
    blockers: list[str] = []
    if contract.status != "approved":
        blockers.append("contract bundle is not approved")
    if plan.status not in {"approved", "executing"}:
        blockers.append(f"experiment status is {plan.status}, not approved")
    if policy_parity_status != "proven":
        blockers.append("policy/deployment parity is not proven")
    protected = set(baseline.protected_capabilities) | {
        capability for anchor in baseline.anchors for capability in anchor.capabilities
    }
    planned = set(plan.protected_capabilities)
    missing = sorted(protected - planned)
    if missing:
        blockers.append(f"experiment does not declare protected capabilities: {', '.join(missing)}")
    if not plan.rollback_condition:
        blockers.append("rollback condition is missing")
    evidence = evidence or []
    evidence_ids = {item.id for item in evidence}
    missing_refs = sorted(set(plan.hypotheses_addressed) - evidence_ids) if plan.hypotheses_addressed else []
    # Hypothesis ids are allowed to live in the ledger independently; only an
    # explicit evidence ref is required when the plan claims one.
    if plan.evaluation_plan.get("required_evidence_refs"):
        for ref in plan.evaluation_plan["required_evidence_refs"]:
            if ref not in evidence_ids:
                blockers.append(f"required evaluation evidence is missing: {ref}")
    return {
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "protected_capabilities": sorted(protected),
        "missing_protected_declarations": missing,
        "evidence_ids": sorted(evidence_ids),
        "unresolved_hypothesis_refs": missing_refs,
    }
