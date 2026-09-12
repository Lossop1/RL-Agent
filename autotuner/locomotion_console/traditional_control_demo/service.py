"""Local traditional-control demo lifecycle and artifact management."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping
from uuid import uuid4

from ..config import LocomotionConsoleSettings
from ..schemas import (
    DiagnosticPlayback,
    TraditionalControlCatalog,
    TraditionalControlHistory,
    TraditionalControlJobStatus,
    TraditionalControlResult,
    TraditionalControlRunRequest,
)
from .contracts import DemoRunCancelled, TraditionalControlDemoProvider
from .registry import (
    ProviderRegistryError,
    TraditionalControlProviderRegistry,
    discover_traditional_control_providers,
)


_MANIFEST_NAME = "manifest.json"
_PLAYBACK_NAME = "playback.json"
_MANIFEST_SCHEMA = "traditional_control_demo_manifest_v1"


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def _optional_int(value: Any) -> int | None:
    parsed = _optional_float(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


@dataclass
class _DemoJob:
    run_id: str
    request: TraditionalControlRunRequest
    provider_id: str
    output_dir: Path
    state: str = "starting"
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    progress: float = 0.0
    message: str = "Preparing traditional-control demo"
    error: str = ""
    result_path: Path | None = None
    manifest_path: Path | None = None
    playback: DiagnosticPlayback | None = None
    provenance: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)


class TraditionalControlDemoService:
    """Own one local demo job while keeping providers stateless and replaceable."""

    def __init__(
        self,
        settings: LocomotionConsoleSettings,
        *,
        registry: TraditionalControlProviderRegistry | None = None,
        output_root: str | Path | None = None,
    ) -> None:
        configured_root = output_root or getattr(settings, "traditional_control_output_root", "")
        self.output_root = Path(configured_root or Path(settings.local_state_root) / "traditional_control")
        self.output_root = self.output_root.expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.registry = registry or discover_traditional_control_providers()
        self._job: _DemoJob | None = None
        self._lock = threading.RLock()
        self._task: asyncio.Task[None] | None = None
        self._recover_and_restore_latest()

    async def catalog(self) -> TraditionalControlCatalog:
        products_by_id = {}
        sources = []
        errors = list(self.registry.load_errors)
        for item in self.registry.catalogs():
            provider_sources = [
                self._catalog_source(
                    source,
                    product_id=item.product.id,
                    provider_id=source.provider_id or self._provider_id_for_catalog(item),
                )
                for source in item.data_sources
            ]
            catalog_product = item.product.model_copy(deep=True)
            catalog_product.data_sources = provider_sources
            product = products_by_id.get(item.product.id)
            if product is None:
                product = catalog_product
                products_by_id[product.id] = product
            else:
                if product.robot.id != catalog_product.robot.id:
                    errors.append(f"{product.id}: providers declare different robot ids")
                    continue
                errors.extend(self._merge_catalog_items(product.controllers, catalog_product.controllers, "controller"))
                errors.extend(self._merge_catalog_items(product.scenes, catalog_product.scenes, "scene"))
                errors.extend(self._merge_catalog_items(product.data_sources, catalog_product.data_sources, "data source"))
            for source in provider_sources:
                if not any(
                    existing.id == source.id and existing.provider_id == source.provider_id
                    for existing in sources
                ):
                    sources.append(source)
        products = [products_by_id[key] for key in sorted(products_by_id)]
        message = "; ".join(errors)
        return TraditionalControlCatalog(
            available=bool(products),
            message=message,
            products=products,
            data_sources=sources,
            recent_runs=self._history_rows(limit=12),
        )

    async def start(self, request: TraditionalControlRunRequest) -> TraditionalControlJobStatus:
        with self._lock:
            if self._job is not None and self._job.state in {"starting", "running"}:
                raise RuntimeError("A traditional-control demo is already running")
        provider, _provider_catalog, source = self.registry.resolve(
            product_id=request.product_id,
            controller_id=request.controller_id,
            scene_id=request.scene_id,
            data_source_id=request.data_source_id,
        )
        if not source.available:
            raise ProviderRegistryError(f"selected data source is unavailable: {source.id}")
        if source.kind == "replay":
            run_dir = self._resolve_replay_dir(
                request.replay_id or source.replay_id,
                product_id=request.product_id,
                controller_id=request.controller_id,
                scene_id=request.scene_id,
                provider_id=provider.provider_id,
            )
            manifest = self._read_manifest(run_dir)
            stored_request = manifest.get("request") if isinstance(manifest.get("request"), Mapping) else {}
            for key in ("product_id", "controller_id", "scene_id"):
                if str(stored_request.get(key) or "") != str(getattr(request, key)):
                    raise ProviderRegistryError(f"replay does not match selected {key}")
            stored_provider_id = str(manifest.get("provider_id") or "")
            if stored_provider_id != provider.provider_id:
                raise ProviderRegistryError("replay provider does not match the selected data source")
            started_at = self._parse_timestamp(manifest.get("created_at"), fallback=run_dir.stat().st_ctime)
            updated_at = self._parse_timestamp(manifest.get("updated_at"), fallback=started_at)
            job = _DemoJob(
                run_id=run_dir.name,
                request=request,
                provider_id=provider.provider_id,
                output_dir=run_dir,
                state="complete",
                started_at=started_at,
                updated_at=updated_at,
                finished_at=updated_at,
                progress=1.0,
                message="Loaded existing traditional-control playback",
                manifest_path=run_dir / _MANIFEST_NAME,
                result_path=run_dir / "result.json",
            )
            job.playback = await asyncio.to_thread(provider.load_playback, run_dir)
            with self._lock:
                if self._job is not None and self._job.state in {"starting", "running"}:
                    raise RuntimeError("A traditional-control demo is already running")
                self._job = job
            return self._status(job)

        with self._lock:
            if self._job is not None and self._job.state in {"starting", "running"}:
                raise RuntimeError("A traditional-control demo is already running")
            run_id = self._new_run_id()
            output_dir = self.output_root / run_id
            output_dir.mkdir(parents=True, exist_ok=False)
            job = _DemoJob(
                run_id=run_id,
                request=request,
                provider_id=provider.provider_id,
                output_dir=output_dir,
                manifest_path=output_dir / _MANIFEST_NAME,
            )
            self._job = job
            self._write_manifest(job)
            self._task = asyncio.create_task(self._execute(job, provider))
        return self._status(job)

    async def status(self, run_id: str | None = None) -> TraditionalControlJobStatus:
        job = self._job_for(run_id)
        if job is None:
            return TraditionalControlJobStatus(state="idle", message="No traditional-control demo has been started")
        return self._status(job)

    async def playback(self, max_frames: int = 900, run_id: str | None = None) -> DiagnosticPlayback:
        max_frames = max(1, min(int(max_frames or 900), 4000))
        job = self._job_for(run_id)
        if job is None:
            return DiagnosticPlayback(
                available=False,
                source="local",
                message="No traditional-control demo has been started",
            )
        if job.state not in {"complete", "error"}:
            return DiagnosticPlayback(
                available=False,
                source="local",
                output_dir=str(job.output_dir),
                manifest_path=str(job.manifest_path) if job.manifest_path else None,
                message="Traditional-control playback is available after the run finishes",
            )
        playback_path = job.output_dir / _PLAYBACK_NAME
        if playback_path.is_file():
            raw = json.loads(playback_path.read_text(encoding="utf-8"))
            playback = DiagnosticPlayback.model_validate(raw)
            # Older artifacts predate the command-vector field. Let the
            # provider reconstruct the trace when possible so historical
            # records still show the requested direction in the viewer.
            if not playback.command:
                try:
                    provider = self.registry.provider(job.provider_id)
                    reconstructed = await asyncio.to_thread(
                        provider.load_playback, job.output_dir, max_frames
                    )
                    if reconstructed.available and reconstructed.command:
                        playback = reconstructed
                except Exception:
                    pass
        elif job.playback is not None:
            playback = job.playback
        else:
            provider = self.registry.provider(job.provider_id)
            playback = await asyncio.to_thread(provider.load_playback, job.output_dir, 4000)
        return self._limit_playback(playback, max_frames)

    async def history(self, limit: int = 30) -> TraditionalControlHistory:
        rows = self._history_rows(limit=max(1, min(int(limit or 30), 100)))
        return TraditionalControlHistory(
            items=[TraditionalControlJobStatus.model_validate(row) for row in rows]
        )

    async def result(self, run_id: str | None = None) -> TraditionalControlResult:
        job = self._job_for(run_id)
        if job is None or job.result_path is None:
            raise RuntimeError("traditional-control result is not available")
        path = job.result_path
        if not path.is_file():
            raise RuntimeError("traditional-control result is not available")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return TraditionalControlResult.model_validate(raw)

    async def manifest(self, run_id: str | None = None) -> dict[str, Any]:
        job = self._job_for(run_id)
        if job is None or job.manifest_path is None or not job.manifest_path.is_file():
            raise RuntimeError("traditional-control manifest is not available")
        return self._read_manifest(job.output_dir)

    async def cancel(self) -> TraditionalControlJobStatus:
        job = self._job_for(None)
        if job is None:
            return TraditionalControlJobStatus(state="idle", message="No traditional-control demo has been started")
        if job.state in {"starting", "running"}:
            job.cancel_event.set()
            job.message = "Cancellation requested"
            job.updated_at = time.time()
            self._write_manifest(job)
        return self._status(job)

    async def _execute(self, job: _DemoJob, provider: TraditionalControlDemoProvider) -> None:
        job.state = "running"
        job.message = "Running local headless traditional-control demo"
        job.updated_at = time.time()
        self._write_manifest(job)
        last_progress_value = -1.0
        last_progress_message = ""
        last_progress_write = 0.0

        def progress(value: float, message: str) -> None:
            nonlocal last_progress_value, last_progress_message, last_progress_write
            job.progress = max(0.0, min(1.0, float(value)))
            job.message = str(message)
            job.updated_at = time.time()
            if (
                job.progress - last_progress_value < 0.02
                and job.message == last_progress_message
                and job.updated_at - last_progress_write < 0.25
            ):
                return
            self._write_manifest(job)
            last_progress_value = job.progress
            last_progress_message = job.message
            last_progress_write = job.updated_at

        try:
            result = await asyncio.to_thread(
                provider.run,
                controller_id=job.request.controller_id,
                scene_id=job.request.scene_id,
                data_source_id=job.request.data_source_id,
                duration_s=job.request.duration_s,
                output_dir=job.output_dir,
                progress=progress,
                cancelled=job.cancel_event.is_set,
            )
            result_path = job.output_dir / "result.json"
            result_path.write_text(
                json.dumps(result.result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            job.result_path = result_path
            job.playback = result.playback
            (job.output_dir / _PLAYBACK_NAME).write_text(
                json.dumps(result.playback.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            job.provenance = {str(key): str(value) for key, value in result.provenance.items()}
            for name, path in result.raw_artifacts.items():
                resolved = Path(path).resolve()
                try:
                    resolved.relative_to(job.output_dir.resolve())
                except ValueError as exc:
                    raise RuntimeError(f"provider artifact escapes run directory: {name}") from exc
                job.artifacts[str(name)] = resolved
            job.state = "complete"
            job.progress = 1.0
            job.message = "Traditional-control demo completed"
        except DemoRunCancelled:
            job.state = "cancelled"
            job.message = "Traditional-control demo cancelled"
            job.error = ""
        except Exception as exc:  # noqa: BLE001 - expose provider failure in job manifest/API
            job.state = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "Traditional-control demo failed"
        finally:
            job.updated_at = time.time()
            job.finished_at = job.updated_at
            self._write_manifest(job)

    def _job_for(self, run_id: str | None) -> _DemoJob | None:
        with self._lock:
            if run_id:
                if self._job is not None and self._job.run_id == run_id:
                    return self._job
                run_dir = self._resolve_replay_dir(run_id, missing_ok=True)
                if run_dir is None:
                    return None
                manifest = self._read_manifest(run_dir)
                return self._job_from_manifest(run_dir, manifest)
            return self._job

    def _resolve_replay_dir(
        self,
        replay_id: str | None,
        *,
        product_id: str | None = None,
        controller_id: str | None = None,
        scene_id: str | None = None,
        provider_id: str | None = None,
        missing_ok: bool = False,
    ) -> Path | None:
        value = str(replay_id or "").strip()
        candidate: Path | None = None
        if value:
            raw = Path(value)
            candidate = raw if raw.is_absolute() else self.output_root / raw
            candidate = candidate.resolve()
            try:
                candidate.relative_to(self.output_root)
            except ValueError as exc:
                raise ProviderRegistryError("replay path must remain inside the traditional-control output root") from exc
        else:
            directories = []
            for manifest_path in self.output_root.glob(f"*/{_MANIFEST_NAME}"):
                try:
                    raw = self._read_manifest(manifest_path.parent)
                    request = raw.get("request") if isinstance(raw.get("request"), Mapping) else {}
                    if str(raw.get("state") or "") != "complete":
                        continue
                    if product_id and str(request.get("product_id") or "") != product_id:
                        continue
                    if controller_id and str(request.get("controller_id") or "") != controller_id:
                        continue
                    if scene_id and str(request.get("scene_id") or "") != scene_id:
                        continue
                    if provider_id and str(raw.get("provider_id") or "") != provider_id:
                        continue
                    directories.append(manifest_path.parent)
                except (OSError, ProviderRegistryError, json.JSONDecodeError):
                    continue
            candidate = max(directories, key=lambda item: item.stat().st_mtime, default=None)
        if candidate is None or not (candidate / _MANIFEST_NAME).is_file():
            if missing_ok:
                return None
            raise ProviderRegistryError("no local traditional-control replay was found")
        return candidate

    def _new_run_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{stamp}_{uuid4().hex[:8]}"

    def _read_manifest(self, output_dir: Path) -> dict[str, Any]:
        try:
            raw = json.loads((output_dir / _MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderRegistryError(f"invalid traditional-control manifest: {output_dir}") from exc
        return raw if isinstance(raw, dict) else {}

    def _job_from_manifest(self, output_dir: Path, manifest: Mapping[str, Any]) -> _DemoJob:
        request = TraditionalControlRunRequest.model_validate(manifest.get("request") or {})
        state = str(manifest.get("state") or "complete")
        started_at = self._parse_timestamp(manifest.get("created_at"), fallback=output_dir.stat().st_ctime)
        updated_at = self._parse_timestamp(manifest.get("updated_at"), fallback=started_at)
        result_path = output_dir / "result.json"
        return _DemoJob(
            run_id=output_dir.name,
            request=request,
            provider_id=str(manifest.get("provider_id") or ""),
            output_dir=output_dir,
            state=state,
            started_at=started_at,
            updated_at=updated_at,
            finished_at=updated_at if state in {"complete", "error", "cancelled"} else None,
            progress=1.0 if state == "complete" else float(manifest.get("progress") or 0.0),
            message=str(manifest.get("message") or ""),
            error=str(manifest.get("error") or ""),
            manifest_path=output_dir / _MANIFEST_NAME,
            result_path=result_path if result_path.is_file() else None,
            provenance={
                str(key): str(value)
                for key, value in (manifest.get("provenance") or {}).items()
            } if isinstance(manifest.get("provenance"), Mapping) else {},
        )

    def _recover_and_restore_latest(self) -> None:
        manifests = sorted(
            self.output_root.glob(f"*/{_MANIFEST_NAME}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        now = datetime.now(timezone.utc).isoformat()
        for manifest_path in manifests:
            try:
                raw = self._read_manifest(manifest_path.parent)
                if raw.get("state") in {"starting", "running"}:
                    raw.update(
                        state="error",
                        message="Traditional-control demo was interrupted before completion",
                        error="console process ended while the local provider was active",
                        updated_at=now,
                    )
                    temporary = manifest_path.with_suffix(".json.recover.tmp")
                    temporary.write_text(
                        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    temporary.replace(manifest_path)
                if self._job is None:
                    self._job = self._job_from_manifest(manifest_path.parent, raw)
            except Exception:
                continue

    def _write_manifest(self, job: _DemoJob) -> None:
        with self._lock:
            if job.manifest_path is None:
                job.manifest_path = job.output_dir / _MANIFEST_NAME
            payload = {
                "schema_version": _MANIFEST_SCHEMA,
                "run_id": job.run_id,
                "state": job.state,
                "provider_id": job.provider_id,
                "request": job.request.model_dump(mode="json"),
                "output_dir": str(job.output_dir),
                "progress": job.progress,
                "message": job.message,
                "error": job.error,
                "provenance": dict(job.provenance),
                "created_at": datetime.fromtimestamp(job.started_at, timezone.utc).isoformat(),
                "updated_at": datetime.fromtimestamp(job.updated_at, timezone.utc).isoformat(),
                "artifacts": {
                    "manifest": self._artifact_record(job.manifest_path, include_digest=False),
                    "trace": self._artifact_record(job.artifacts.get("trace", job.output_dir / "control_trace.jsonl")),
                    "result": self._artifact_record(job.output_dir / "result.json"),
                    "playback": self._artifact_record(job.output_dir / _PLAYBACK_NAME),
                    **{
                        name: self._artifact_record(path)
                        for name, path in sorted(job.artifacts.items())
                        if name not in {"trace", "result", "playback", "manifest"}
                    },
                },
            }
            temporary = job.manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(job.manifest_path)

    def _status(self, job: _DemoJob) -> TraditionalControlJobStatus:
        elapsed = max(0.0, (job.finished_at or time.time()) - job.started_at)
        result_summary = self._result_status_fields(job.result_path)
        return TraditionalControlJobStatus(
            state=job.state,  # type: ignore[arg-type]
            run_id=job.run_id,
            product_id=job.request.product_id,
            controller_id=job.request.controller_id,
            scene_id=job.request.scene_id,
            data_source_id=job.request.data_source_id,
            output_dir=str(job.output_dir),
            manifest_path=str(job.manifest_path) if job.manifest_path else None,
            result_path=str(job.result_path) if job.result_path else None,
            playback_available=bool(
                (job.playback and job.playback.available)
                or (job.output_dir / _PLAYBACK_NAME).is_file()
            ),
            progress=job.progress,
            elapsed_s=elapsed,
            message=job.message,
            error=job.error,
            created_at=job.started_at,
            finished_at=job.finished_at,
            requested_duration_s=job.request.duration_s,
            **result_summary,
        )

    @staticmethod
    def _result_status_fields(result_path: Path | None) -> dict[str, Any]:
        if result_path is None or not result_path.is_file():
            return {}
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, Mapping):
            return {}
        raw = payload.get("raw") if isinstance(payload.get("raw"), Mapping) else {}
        failure_reasons = payload.get("failure_reasons")
        if not isinstance(failure_reasons, list):
            failure_reasons = []
        runtime_failure = raw.get("runtime_failure") if isinstance(raw.get("runtime_failure"), Mapping) else {}
        failure_code = str(runtime_failure.get("code") or "")
        failure_message = str(runtime_failure.get("message") or "")
        termination_reason = ": ".join(part for part in (failure_code, failure_message) if part)
        if not termination_reason and failure_reasons:
            termination_reason = ", ".join(str(item) for item in failure_reasons)
        verdict = str(payload.get("verdict") or "unknown")
        if verdict not in {"passed", "failed", "unknown"}:
            verdict = "unknown"
        return {
            "simulated_duration_s": _optional_float(raw.get("elapsed_s")),
            "completed_steps": _optional_int(raw.get("steps")),
            "requested_steps": _optional_int(raw.get("requested_steps")),
            "verdict": verdict,
            "failure_reasons": [str(item) for item in failure_reasons],
            "termination_reason": termination_reason,
        }

    @staticmethod
    def _artifact_record(path: Path, *, include_digest: bool = True) -> dict[str, Any]:
        available = path.is_file()
        record: dict[str, Any] = {
            "path": str(path),
            "available": available,
            "bytes": path.stat().st_size if available else 0,
        }
        if available and include_digest:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            record["sha256"] = digest.hexdigest()
        else:
            record["sha256"] = ""
        return record

    def _history_rows(self, limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for manifest_path in self.output_root.glob(f"*/{_MANIFEST_NAME}"):
            try:
                raw = self._read_manifest(manifest_path.parent)
                request = TraditionalControlRunRequest.model_validate(raw.get("request") or {})
                provider_id = str(raw.get("provider_id") or "")
                state = str(raw.get("state") or "complete")
                started_at = self._parse_timestamp(raw.get("created_at"), fallback=manifest_path.stat().st_ctime)
                updated_at = self._parse_timestamp(raw.get("updated_at"), fallback=started_at)
                rows.append(
                    self._status(
                        _DemoJob(
                            run_id=manifest_path.parent.name,
                            request=request,
                            provider_id=provider_id,
                            output_dir=manifest_path.parent,
                            state=state,
                            progress=float(raw.get("progress") or (1.0 if state == "complete" else 0.0)),
                            message=str(raw.get("message") or ""),
                            error=str(raw.get("error") or ""),
                            started_at=started_at,
                            updated_at=updated_at,
                            finished_at=updated_at if state in {"complete", "error", "cancelled"} else None,
                            manifest_path=manifest_path,
                            result_path=manifest_path.parent / "result.json",
                        )
                    ).model_dump(mode="json")
                )
            except Exception:
                continue
        rows.sort(key=lambda item: str(item.get("run_id") or ""), reverse=True)
        return rows[:limit]

    def _provider_id_for_catalog(self, catalog: Any) -> str:
        """Return the provider id without coupling the service to a product type."""

        controllers = getattr(catalog.product, "controllers", ())
        for controller in controllers:
            provider_ids = getattr(controller, "provider_ids", ())
            if provider_ids:
                return str(provider_ids[0])
            provider_id = str(getattr(controller, "provider_id", "") or "")
            if provider_id:
                return provider_id
        return ""

    def _catalog_source(
        self,
        source: Any,
        *,
        product_id: str,
        provider_id: str,
    ) -> Any:
        """Expose replay availability from the local artifact store.

        Providers declare the source capability; the service adds the current
        local-state fact so the UI does not offer an empty replay action.
        """

        if str(getattr(source, "kind", "")) != "replay":
            return source.model_copy(deep=True)
        available = bool(getattr(source, "available", True)) and self._has_replay(
            product_id=product_id,
            provider_id=provider_id,
            replay_id=getattr(source, "replay_id", None),
        )
        return source.model_copy(update={"available": available}, deep=True)

    def _has_replay(
        self,
        *,
        product_id: str,
        provider_id: str,
        replay_id: str | None,
    ) -> bool:
        if replay_id:
            try:
                return self._resolve_replay_dir(replay_id, missing_ok=True) is not None
            except ProviderRegistryError:
                return False
        for manifest_path in self.output_root.glob(f"*/{_MANIFEST_NAME}"):
            try:
                raw = self._read_manifest(manifest_path.parent)
                request = raw.get("request") if isinstance(raw.get("request"), Mapping) else {}
                if str(raw.get("state") or "") != "complete":
                    continue
                if str(raw.get("provider_id") or "") != provider_id:
                    continue
                if str(request.get("product_id") or "") != product_id:
                    continue
                run_dir = manifest_path.parent
                if (run_dir / _PLAYBACK_NAME).is_file() or (run_dir / "control_trace.jsonl").is_file():
                    return True
            except (OSError, ProviderRegistryError, json.JSONDecodeError):
                continue
        return False

    @staticmethod
    def _parse_timestamp(value: Any, *, fallback: float) -> float:
        try:
            return datetime.fromisoformat(str(value)).timestamp()
        except (TypeError, ValueError):
            return float(fallback)

    @staticmethod
    def _merge_catalog_items(target: list[Any], incoming: list[Any], kind: str) -> list[str]:
        errors: list[str] = []
        for item in incoming:
            existing = next((value for value in target if value.id == item.id), None)
            if existing is None:
                target.append(item.model_copy(deep=True))
                continue
            if kind == "data source":
                errors.append(
                    f"duplicate data source id {item.id!r}; ids must be unique within a product"
                )
                continue
            excluded = {"provider_id", "provider_ids"}
            if kind == "scene":
                excluded.add("controller_ids")
            current_values = existing.model_dump(exclude=excluded)
            incoming_values = item.model_dump(exclude=excluded)
            if current_values != incoming_values:
                errors.append(
                    f"providers declare incompatible {kind} metadata for id {item.id!r}"
                )
                continue
            existing.provider_ids = sorted(set(existing.provider_ids + item.provider_ids))
            if kind == "scene":
                existing.controller_ids = sorted(set(existing.controller_ids + item.controller_ids))
        return errors

    @staticmethod
    def _limit_playback(playback: DiagnosticPlayback, max_frames: int) -> DiagnosticPlayback:
        count = len(playback.frames)
        if count <= max_frames:
            return playback
        if max_frames == 1:
            frames = [playback.frames[0]]
            stride = count
        else:
            indices = [round(index * (count - 1) / (max_frames - 1)) for index in range(max_frames)]
            frames = [playback.frames[index] for index in indices]
            stride = max(1, int(round(count / max_frames)))
        return playback.model_copy(update={
            "frames": frames,
            "stride": max(1, playback.stride * stride),
        })


__all__ = ["TraditionalControlDemoService"]
