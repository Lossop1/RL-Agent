"""Execution tests for generated reward, telemetry, and gate semantics."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from autotuner.mechanisms.mechanism_artifacts import MechanismArtifactCompiler, verify_mechanism_artifact
from autotuner.mechanisms.mechanism_runtime import MechanismRuntime, MechanismRuntimeError, load_mechanism_bundle
from autotuner.mechanisms.mechanism_specs import (
    Expression,
    GateSpec,
    MechanismBundle,
    MetricSpec,
    ParameterSpec,
    RewardTermSpec,
    SignalSpec,
)


def _runtime_bundle() -> MechanismBundle:
    error = Expression(op="abs", args=(Expression.signal("tracking.error"),))
    quality = Expression(op="laplace", args=(error, Expression.param("tracking.scale")))
    return MechanismBundle(
        id="bundle:runtime",
        status="approved",
        contract_ref="contract:taili",
        signals=(SignalSpec(
            name="tracking.error", description="command-frame velocity error",
            unit="m/s", frame="command", source_ref="input.tracking_error",
        ),),
        parameters=(ParameterSpec(
            name="tracking.scale", value=0.5, lower_bound=0.05, upper_bound=2.0, unit="m/s",
        ),),
        rewards=(RewardTermSpec(
            id="reward:tracking-live", name="live tracking", role="positive_drive",
            expression=quality, weight=2.0, intended_effect="track command",
            failure_region="error is high", success_region="error is zero",
        ),),
        metrics=(MetricSpec(
            id="metric:tracking-error", name="tracking error", expression=error,
            aggregation="mean", unit="m/s", window_steps=3, intended_reading="lower is better",
        ),),
        gates=(GateSpec(
            id="gate:tracking", name="tracking acceptance", metric_ref="metric:tracking-error",
            comparator="le", threshold=0.2, min_samples=2, consecutive_windows=2,
            hysteresis=0.02, action="advance", scope="curriculum", rationale="advance only after sustained tracking",
        ),),
        evaluator_refs=("evaluator:flat",),
    )


def test_runtime_executes_reward_metric_and_sustained_gate():
    runtime = MechanismRuntime(_runtime_bundle())
    first = runtime.evaluate_step(inputs=SimpleNamespace(tracking_error=torch.tensor([0.1, 0.2])))
    assert torch.allclose(first.reward_components["dynamic/reward:tracking-live"], 2.0 * torch.exp(-torch.tensor([0.2, 0.4])))
    assert first.metric_values["metric:tracking-error"] == pytest.approx(0.15)
    assert first.gate_results["gate:tracking"]["ready"] is False

    second = runtime.evaluate_step(inputs=SimpleNamespace(tracking_error=torch.tensor([0.1, 0.1])))
    assert second.gate_results["gate:tracking"]["ready"] is True
    assert second.gate_results["gate:tracking"]["passed"] is False
    third = runtime.evaluate_step(inputs=SimpleNamespace(tracking_error=torch.tensor([0.1, 0.1])))
    assert third.gate_results["gate:tracking"]["passed"] is True
    assert third.gate_results["gate:tracking"]["action"] == "advance"


def test_runtime_rejects_vector_reward_without_explicit_reduction():
    bundle = _runtime_bundle()
    bad = bundle.rewards[0].model_copy(update={"expression": Expression.signal("tracking.error")})
    bundle = bundle.model_copy(update={"rewards": (bad,)})
    with pytest.raises(MechanismRuntimeError, match="one scalar per environment"):
        MechanismRuntime(bundle).evaluate_step(
            signals={"tracking.error": torch.ones(2, 3)}, inputs=SimpleNamespace(),
        )


def test_compiler_emits_immutable_verified_artifact(tmp_path):
    bundle = _runtime_bundle()
    path, manifest = MechanismArtifactCompiler().compile(output_root=tmp_path, baseline=bundle)
    assert manifest.bundle_fingerprint == bundle.fingerprint()
    assert load_mechanism_bundle(path / "mechanisms.json").id == bundle.id
    assert verify_mechanism_artifact(path).artifact_id == manifest.artifact_id

    (path / "test_vectors.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity failure"):
        verify_mechanism_artifact(path)


def test_remote_payload_contains_safe_mechanism_runtime():
    from products.taili.payload.payload_manifest import iter_payload_files

    destinations = {destination for _, destination in iter_payload_files()}
    assert "taili_blind_runtime/taili_core/mechanism_specs.py" in destinations
    assert "taili_blind_runtime/taili_core/mechanism_runtime.py" in destinations
