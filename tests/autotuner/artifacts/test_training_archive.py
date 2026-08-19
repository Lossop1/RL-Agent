from __future__ import annotations

import json

from autotuner.artifacts import TrainingArchive, TrainingRunRecord


def test_archive_registers_and_queries_resume_lineage(tmp_path) -> None:
    archive = TrainingArchive(tmp_path / "archive")
    parent = TrainingRunRecord(
        run_id="parent",
        product_id="taili",
        product_version="1",
        task_id="locomotion",
        status="complete",
        run_dir=str(tmp_path / "parent"),
        config_digest="cfg-parent",
    )
    child = TrainingRunRecord(
        run_id="child",
        product_id="taili",
        product_version="1",
        task_id="locomotion",
        status="running",
        run_dir=str(tmp_path / "child"),
        config_digest="cfg-child",
        parent_run_id="parent",
        parent_checkpoint="parent/checkpoints/agent.pt",
    )
    archive.register(parent)
    archive.register(child)

    assert [item.run_id for item in archive.lineage("child")] == ["child", "parent"]
    assert archive.search(product_id="taili")[0].run_id == "child"


def test_runtime_manifest_normalizes_nested_contract(tmp_path) -> None:
    path = tmp_path / "run" / "runtime_manifest.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "status": "running",
                "run": {
                    "run_id": "runtime-1",
                    "run_dir": str(path.parent),
                    "task": "blind_task",
                    "resume_checkpoint": "parent.pt",
                },
                "product": {"id": "alpha", "version": "2"},
                "resolved_contract": {
                    "digest": "payload-digest",
                    "config_digest": "config-digest",
                    "asset_digest": "asset-digest",
                },
            }
        ),
        encoding="utf-8",
    )

    record = TrainingArchive(tmp_path / "archive").register_runtime_manifest(path)

    assert record.product_id == "alpha"
    assert record.task_id == "blind_task"
    assert record.parent_checkpoint == "parent.pt"
    assert record.payload_digest == "payload-digest"
