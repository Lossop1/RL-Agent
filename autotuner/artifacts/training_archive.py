"""可查询的训练运行归档。

大文件仍留在原运行目录或归档目录中；SQLite 只保存可复现所需的索引和来源信息。
这样不会把 checkpoint 复制进源码，也不会依赖文件名猜测产品、任务或 resume 关系。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT / "output" / "training_archive"


def _safe_id(value: str, field_name: str) -> str:
    value = str(value or "").strip()
    if not value or not _ID_RE.fullmatch(value):
        raise ValueError(f"{field_name} has unsafe value: {value!r}")
    return value


@dataclass(frozen=True)
class TrainingRunRecord:
    """一个训练运行的不可变身份和可更新状态。"""

    run_id: str
    product_id: str
    product_version: str
    task_id: str
    status: str
    run_dir: str
    source_revision: str = ""
    config_digest: str = ""
    asset_digest: str = ""
    payload_digest: str = ""
    started_at: str = ""
    finished_at: str = ""
    parent_run_id: str = ""
    parent_checkpoint: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("run_id", "product_id", "task_id"):
            _safe_id(getattr(self, name), name)
        if not str(self.product_version).strip():
            raise ValueError("product_version is required")
        if not str(self.status).strip():
            raise ValueError("status is required")
        if self.parent_run_id:
            _safe_id(self.parent_run_id, "parent_run_id")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrainingRunRecord":
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        return cls(
            run_id=str(data.get("run_id", "")),
            product_id=str(data.get("product_id", "")),
            product_version=str(data.get("product_version", "")),
            task_id=str(data.get("task_id", "")),
            status=str(data.get("status", "unknown")),
            run_dir=str(data.get("run_dir", "")),
            source_revision=str(data.get("source_revision", "")),
            config_digest=str(data.get("config_digest", "")),
            asset_digest=str(data.get("asset_digest", "")),
            payload_digest=str(data.get("payload_digest", "")),
            started_at=str(data.get("started_at", "")),
            finished_at=str(data.get("finished_at", "")),
            parent_run_id=str(data.get("parent_run_id", "")),
            parent_checkpoint=str(data.get("parent_checkpoint", "")),
            metadata=dict(metadata),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


class TrainingArchive:
    """以 SQLite 索引训练运行，并保留 resume 父链。"""

    def __init__(self, root: str | Path = DEFAULT_ARCHIVE_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    product_version TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    source_revision TEXT NOT NULL DEFAULT '',
                    config_digest TEXT NOT NULL DEFAULT '',
                    asset_digest TEXT NOT NULL DEFAULT '',
                    payload_digest TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    parent_run_id TEXT NOT NULL DEFAULT '',
                    parent_checkpoint TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    registered_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_training_runs_product
                    ON training_runs(product_id, product_version);
                CREATE INDEX IF NOT EXISTS idx_training_runs_task
                    ON training_runs(task_id, status);
                CREATE INDEX IF NOT EXISTS idx_training_runs_parent
                    ON training_runs(parent_run_id);
                """
            )

    @staticmethod
    def canonical_run_dir(root: str | Path, record: TrainingRunRecord) -> Path:
        return Path(root) / "products" / record.product_id / record.task_id / record.run_id

    def register(self, record: TrainingRunRecord, *, write_manifest: bool = False) -> TrainingRunRecord:
        metadata_json = json.dumps(record.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        immutable = (
            "product_id", "product_version", "task_id", "run_dir", "source_revision",
            "config_digest", "asset_digest", "payload_digest", "parent_run_id", "parent_checkpoint",
        )
        with self._connect() as conn:
            existing = conn.execute("SELECT * FROM training_runs WHERE run_id = ?", (record.run_id,)).fetchone()
            if existing is not None:
                for field_name in immutable:
                    if str(existing[field_name]) != str(getattr(record, field_name)):
                        raise ValueError(f"run {record.run_id!r} changes immutable field {field_name}")
                conn.execute(
                    """UPDATE training_runs SET status=?, started_at=?, finished_at=?, metadata_json=?
                       WHERE run_id=?""",
                    (record.status, record.started_at, record.finished_at, metadata_json, record.run_id),
                )
            else:
                conn.execute(
                    """INSERT INTO training_runs (
                        run_id, product_id, product_version, task_id, status, run_dir,
                        source_revision, config_digest, asset_digest, payload_digest,
                        started_at, finished_at, parent_run_id, parent_checkpoint,
                        metadata_json, registered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.run_id, record.product_id, record.product_version, record.task_id,
                        record.status, record.run_dir, record.source_revision, record.config_digest,
                        record.asset_digest, record.payload_digest, record.started_at, record.finished_at,
                        record.parent_run_id, record.parent_checkpoint, metadata_json,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        if write_manifest:
            path = Path(record.run_dir) / "run.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record.to_mapping(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return record

    def register_manifest(self, path: str | Path, *, write_manifest: bool = False) -> TrainingRunRecord:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(f"run manifest must be an object: {path}")
        data = dict(data)
        run = data.get("run") if isinstance(data.get("run"), Mapping) else {}
        product = data.get("product") if isinstance(data.get("product"), Mapping) else {}
        contract = data.get("resolved_contract") if isinstance(data.get("resolved_contract"), Mapping) else {}
        payload = data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
        data.setdefault("run_id", run.get("run_id", ""))
        data.setdefault("run_dir", run.get("run_dir", str(path.parent)))
        data.setdefault("task_id", run.get("task", data.get("task", "")))
        data.setdefault("product_id", product.get("id", ""))
        data.setdefault("product_version", product.get("version", ""))
        data.setdefault("config_digest", contract.get("config_digest", ""))
        data.setdefault("asset_digest", contract.get("asset_digest", ""))
        data.setdefault(
            "payload_digest",
            payload.get("digest") or contract.get("payload_digest") or contract.get("digest", ""),
        )
        data.setdefault("parent_checkpoint", data.get("resume_checkpoint", run.get("resume_checkpoint", "")))
        data.setdefault("status", "unknown")
        record = TrainingRunRecord.from_mapping(data)
        return self.register(record, write_manifest=write_manifest)

    def register_runtime_manifest(self, path: str | Path, *, write_manifest: bool = False) -> TrainingRunRecord:
        """登记 payload 运行时 manifest，兼容嵌套的运行与合同字段。"""
        return self.register_manifest(path, write_manifest=write_manifest)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TrainingRunRecord:
        return TrainingRunRecord(
            run_id=row["run_id"], product_id=row["product_id"], product_version=row["product_version"],
            task_id=row["task_id"], status=row["status"], run_dir=row["run_dir"],
            source_revision=row["source_revision"], config_digest=row["config_digest"],
            asset_digest=row["asset_digest"], payload_digest=row["payload_digest"],
            started_at=row["started_at"], finished_at=row["finished_at"],
            parent_run_id=row["parent_run_id"], parent_checkpoint=row["parent_checkpoint"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def get(self, run_id: str) -> TrainingRunRecord:
        _safe_id(run_id, "run_id")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM training_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown training run: {run_id}")
        return self._row_to_record(row)

    def search(
        self,
        *,
        product_id: str | None = None,
        task_id: str | None = None,
        status: str | None = None,
        text: str | None = None,
        limit: int = 100,
    ) -> list[TrainingRunRecord]:
        limit = max(1, min(int(limit), 1000))
        clauses: list[str] = []
        args: list[str | int] = []
        for column, value in (("product_id", product_id), ("task_id", task_id), ("status", status)):
            if value:
                clauses.append(f"{column} = ?")
                args.append(value)
        if text:
            clauses.append("(run_id LIKE ? OR run_dir LIKE ? OR metadata_json LIKE ?)")
            needle = f"%{text}%"
            args.extend((needle, needle, needle))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM training_runs{where} ORDER BY registered_at DESC LIMIT ?",
                (*args, limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def lineage(self, run_id: str) -> list[TrainingRunRecord]:
        chain: list[TrainingRunRecord] = []
        seen: set[str] = set()
        current = run_id
        while current:
            if current in seen:
                raise ValueError(f"resume lineage cycle detected at {current}")
            seen.add(current)
            record = self.get(current)
            chain.append(record)
            current = record.parent_run_id
        return chain

    def products(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT product_id, product_version, COUNT(*) AS runs,
                          MAX(registered_at) AS latest_registration
                   FROM training_runs GROUP BY product_id, product_version
                   ORDER BY product_id, product_version"""
            ).fetchall()
        return [dict(row) for row in rows]


def default_archive() -> TrainingArchive:
    return TrainingArchive(os.environ.get("LOCOMOTION_TRAINING_ARCHIVE_ROOT", DEFAULT_ARCHIVE_ROOT))


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="查询训练运行归档")
    parser.add_argument("--root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    list_cmd = sub.add_parser("list")
    list_cmd.add_argument("--product")
    list_cmd.add_argument("--task")
    list_cmd.add_argument("--status")
    list_cmd.add_argument("--text")
    list_cmd.add_argument("--limit", type=int, default=50)
    line_cmd = sub.add_parser("lineage")
    line_cmd.add_argument("run_id")
    register_cmd = sub.add_parser("register")
    register_cmd.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    archive = TrainingArchive(args.root or os.environ.get("LOCOMOTION_TRAINING_ARCHIVE_ROOT", DEFAULT_ARCHIVE_ROOT))
    if args.command == "list":
        rows = archive.search(product_id=args.product, task_id=args.task, status=args.status, text=args.text, limit=args.limit)
        print(json.dumps([row.to_mapping() for row in rows], ensure_ascii=False, indent=2))
    elif args.command == "lineage":
        print(json.dumps([row.to_mapping() for row in archive.lineage(args.run_id)], ensure_ascii=False, indent=2))
    else:
        archive.register_manifest(args.manifest)
        print(f"registered={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
