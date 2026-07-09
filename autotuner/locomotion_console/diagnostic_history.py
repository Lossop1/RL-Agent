"""Persistent diagnostic job registry.

The registry is intentionally small: one JSON index on the local console machine.
It gives diagnostics a stable management layer without committing the whole console
to SQLite before the job/artifact model settles.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import LocomotionConsoleSettings
from .schemas import ArtifactRefInfo, DiagnosticHistory, DiagnosticHistoryItem, DiagnosticJobStatus


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _checkpoint_name(path: str | None) -> str:
    if not path:
        return ""
    return path.rstrip("/").rsplit("/", 1)[-1]


def _artifacts_for(status: DiagnosticJobStatus, source: str) -> list[ArtifactRefInfo]:
    output_dir = status.output_dir or ""
    report_ready = status.state == "complete"
    has_output = bool(output_dir)
    artifacts = [
        ArtifactRefInfo(
            kind="report",
            label="Diagnostic report",
            uri="/diagnostics/report" if report_ready else "",
            available=report_ready,
            note="Generated from diagnostic metrics." if report_ready else "Available after completion.",
        ),
        ArtifactRefInfo(
            kind="playback",
            label="Rollout playback",
            uri="/diagnostics/playback" if report_ready else "",
            available=report_ready,
            note="Rendered from diagnostic record.csv frames." if report_ready else "Available after record.csv exists.",
        ),
        ArtifactRefInfo(
            kind="output_dir",
            label="Output directory",
            uri=output_dir,
            available=has_output,
            note="Remote path" if source == "real" else "Local/demo path",
        ),
        ArtifactRefInfo(
            kind="job_log",
            label="Job log",
            uri=f"{output_dir}/job.log" if has_output else "",
            available=has_output and source == "real",
            note="Remote tmux job log." if source == "real" else "Demo mode does not write a remote job log.",
        ),
    ]
    return artifacts


class DiagnosticHistoryStore:
    def __init__(self, settings: LocomotionConsoleSettings):
        self.root = Path(settings.local_state_root) / "diagnostics"
        self.index_path = self.root / "index.json"
        self.source = "real" if settings.source == "real" else "fake"

    def _matches_runtime_source(self, item: dict) -> bool:
        source = str(item.get("source") or "fake")
        if source != self.source:
            return False
        checkpoint = str(item.get("checkpoint") or "")
        if self.source == "real" and checkpoint.startswith("fake/"):
            # Guard against stale demo jobs that were restored after switching
            # the backend to real mode and then re-written as source=real.
            return False
        return True

    def list(self, limit: int = 30) -> DiagnosticHistory:
        items = [
            DiagnosticHistoryItem.model_validate(item)
            for item in self._read_items()
            if self._matches_runtime_source(item)
        ]
        items.sort(key=lambda item: item.updated_at or item.created_at, reverse=True)
        return DiagnosticHistory(items=items[: max(1, min(int(limit or 30), 200))])

    def get(self, job_id: str) -> DiagnosticHistoryItem | None:
        for item in self._read_items():
            if item.get("job_id") == job_id and self._matches_runtime_source(item):
                return DiagnosticHistoryItem.model_validate(item)
        return None

    def upsert_status(
        self,
        status: DiagnosticJobStatus,
        *,
        source: str,
        config_set_id: str,
        diagnostic_task: str,
    ) -> DiagnosticHistoryItem | None:
        if not status.job_id:
            return None
        now = _utc_now()
        items = self._read_items()
        existing = next(
            (
                item
                for item in items
                if item.get("job_id") == status.job_id
                and str(item.get("source") or "fake") == source
            ),
            None,
        )
        created_at = str((existing or {}).get("created_at") or now)
        item = DiagnosticHistoryItem(
            job_id=status.job_id,
            state=status.state,
            preset=status.preset or "",
            preset_label=status.preset_label or "",
            checkpoint=status.checkpoint or "",
            checkpoint_name=_checkpoint_name(status.checkpoint),
            framework_id=status.framework_id or "",
            diagnostic_task=diagnostic_task,
            config_set_id=config_set_id,
            source=source,  # type: ignore[arg-type]
            output_dir=status.output_dir or "",
            progress=status.progress,
            message=status.message,
            created_at=created_at,
            updated_at=now,
            elapsed_s=status.elapsed_s,
            artifacts=_artifacts_for(status, source),
        )
        next_items = [
            item_data
            for item_data in items
            if not (
                item_data.get("job_id") == status.job_id
                and str(item_data.get("source") or "fake") == source
            )
        ]
        next_items.append(item.model_dump())
        next_items.sort(key=lambda data: str(data.get("updated_at") or data.get("created_at") or ""), reverse=True)
        self._write_items(next_items[:200])
        return item

    def update(self, job_id: str, *, note: str | None = None, archived: bool | None = None) -> DiagnosticHistoryItem | None:
        items = self._read_items()
        updated = None
        now = _utc_now()
        for item in items:
            if item.get("job_id") != job_id:
                continue
            if note is not None:
                item["note"] = note[:2000]
            if archived is not None:
                item["archived"] = bool(archived)
            item["updated_at"] = now
            updated = item
            break
        if updated is None:
            return None
        self._write_items(items)
        return DiagnosticHistoryItem.model_validate(updated)

    def delete(self, job_id: str) -> bool:
        items = self._read_items()
        next_items = [item for item in items if item.get("job_id") != job_id]
        if len(next_items) == len(items):
            return False
        self._write_items(next_items)
        return True

    def _read_items(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(data, dict):
            raw_items = data.get("items", [])
        else:
            raw_items = data
        return [item for item in raw_items if isinstance(item, dict)]

    def _write_items(self, items: Iterable[dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "items": list(items)}
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.index_path)
