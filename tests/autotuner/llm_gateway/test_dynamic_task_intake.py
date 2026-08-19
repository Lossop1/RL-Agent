from __future__ import annotations

from pathlib import Path

from autotuner.llm_gateway.client import LLMResponse
from autotuner.llm_gateway import task_intake


def _response(parsed):
    return LLMResponse(
        parsed=parsed,
        raw_text="{}",
        model="test-model",
        elapsed_s=0.01,
        attempt=0,
    )


def test_dynamic_intake_compiles_against_selected_product(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        task_intake,
        "call_llm_with_schema",
        lambda **_kwargs: _response(
            {
                "objective": "得到可部署的稳定运动策略",
                "goals": [{"id": "flat", "objective": "平地运动满足验收"}],
                "constraints": {"deployment_requires_privileged_terrain": False},
                "protected_capabilities": ["flat"],
                "training": {"experiment": {"mode": "evidence_driven"}},
                "telemetry": {},
                "diagnostics": {},
                "simulation": {"sim2sim": {"required": True}},
                "deployment": {},
                "clarifications": [],
                "confidence": 0.93,
            }
        ),
    )

    result = task_intake.translate_dynamic(
        "训练 Taili 完成稳定盲态运动",
        product_id="taili",
        output_root=tmp_path,
    )

    assert result.ready
    assert result.bundle is not None
    assert result.bundle.contract.product_id == "taili"
    assert result.bundle.contract.status == "draft"
    assert result.bundle.contract.constraints["actor_observation"] == "proprioceptive_only"
    assert (tmp_path / "task_contract.json").is_file()
    assert result.contract_ref
    assert Path(result.materialization_manifest).is_file()


def test_dynamic_intake_never_accepts_llm_approval(monkeypatch) -> None:
    monkeypatch.setattr(
        task_intake,
        "call_llm_with_schema",
        lambda **_kwargs: _response(
            {
                "objective": "测试",
                "goals": [],
                "constraints": {},
                "approved": True,
                "clarifications": [],
                "confidence": 1.0,
            }
        ),
    )

    result = task_intake.translate_dynamic("测试任务", product_id="taili")

    assert result.bundle is not None
    assert result.bundle.contract.status == "draft"


def test_dynamic_intake_rejects_product_owned_override(monkeypatch) -> None:
    monkeypatch.setattr(
        task_intake,
        "call_llm_with_schema",
        lambda **_kwargs: _response(
            {
                "objective": "测试",
                "goals": [],
                "constraints": {},
                "deployment": {"remote_root": "/tmp/escape"},
                "clarifications": [],
                "confidence": 0.8,
            }
        ),
    )

    result = task_intake.translate_dynamic("测试任务", product_id="taili")

    assert not result.ready
    assert result.bundle is None
    assert "product-owned" in result.error


def test_dynamic_intake_failure_requests_clarification_without_fake_task(monkeypatch) -> None:
    monkeypatch.setattr(
        task_intake,
        "call_llm_with_schema",
        lambda **_kwargs: LLMResponse(None, "", "test", 0.01, 0, error="offline"),
    )

    result = task_intake.translate_dynamic("训练一个新任务", product_id="taili")

    assert result.request is None
    assert result.bundle is None
    assert result.clarifications
