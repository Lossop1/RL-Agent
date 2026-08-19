from __future__ import annotations

import json

from autotuner.blind_locomotion.telemetry_emit import TrainingTelemetryEmitter, _clean_float_map


class _Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, value, step))


def test_dynamic_metrics_remain_structured_in_jsonl_and_tensorboard(tmp_path):
    cleaned = _clean_float_map({
        "total": 1.0,
        "dynamic_metrics": {"metric:stability": 0.75},
    })
    assert cleaned["dynamic_metrics"] == {"metric:stability": 0.75}

    writer = _Writer()
    path = tmp_path / "train.telemetry.jsonl"
    emitter = TrainingTelemetryEmitter(jsonl_path=str(path), writer=writer)
    emitter.emit(
        step=1,
        total_steps=1,
        reward=cleaned,
        curriculum={"dynamic_gates": {"gate:stability": {"ready": True, "value": 0.75}}},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["reward"]["dynamic_metrics"]["metric:stability"] == 0.75
    assert (
        "Telemetry/reward/dynamic_metrics/metric:stability",
        0.75,
        1,
    ) in writer.scalars
    assert (
        "Telemetry/curriculum/dynamic_gates/gate:stability/value",
        0.75,
        1,
    ) in writer.scalars
