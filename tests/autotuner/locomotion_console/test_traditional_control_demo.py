"""Contract and lifecycle tests for the simulator-independent demo service."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time

import pytest

from autotuner.product import ProductRegistry
from autotuner.locomotion_console.config import LocomotionConsoleSettings
from autotuner.locomotion_console.schemas import (
    DiagnosticPlayback,
    DiagnosticPlaybackFrame,
    DiagnosticPlaybackRobot,
    TraditionalControlControllerInfo,
    TraditionalControlDataSourceInfo,
    TraditionalControlProductInfo,
    TraditionalControlRunRequest,
    TraditionalControlSceneInfo,
    TraditionalControlMetricInfo,
    TraditionalControlResult,
)
from autotuner.locomotion_console.traditional_control_demo.contracts import (
    DemoRunCancelled,
    ProviderCatalog,
    ProviderRunResult,
)
from autotuner.locomotion_console.traditional_control_demo.registry import (
    TraditionalControlProviderRegistry,
)
from autotuner.locomotion_console.traditional_control_demo.service import (
    TraditionalControlDemoService,
)


class _Provider:
    provider_id = "fake_provider"

    def catalog(self) -> ProviderCatalog:
        product = TraditionalControlProductInfo(
            id="fake_product",
            label="Fake product",
            robot=DiagnosticPlaybackRobot(
                id="fake_robot",
                label="Fake robot",
                urdf_url="/robot/fake.urdf",
                joint_order=["joint_a"],
                leg_order=["leg_a"],
            ),
            controllers=[TraditionalControlControllerInfo(id="fake_controller", label="Fake controller")],
            scenes=[TraditionalControlSceneInfo(id="flat", label="Flat", duration_s=1.0)],
        )
        return ProviderCatalog(
            product=product,
            data_sources=(
                TraditionalControlDataSourceInfo(id="local_headless", label="Run", kind="headless"),
                TraditionalControlDataSourceInfo(id="local_replay", label="Replay", kind="replay"),
            ),
        )

    def _playback(self, output_dir: Path) -> DiagnosticPlayback:
        return DiagnosticPlayback(
            available=True,
            source="local",
            output_dir=str(output_dir),
            robot=self.catalog().product.robot,
            joint_order=["joint_a"],
            leg_order=["leg_a"],
            frames=[
                DiagnosticPlaybackFrame(
                    t=index * 0.02,
                    stage="flat",
                    case_id=0,
                    segment_id=0,
                    command_mode="stand",
                    base_position=[index * 0.01, 0.0, 0.5],
                    base_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
                    joints=[0.0],
                )
                for index in range(10)
            ],
            source_rows=10,
        )

    def run(self, *, controller_id, scene_id, data_source_id, duration_s, output_dir, progress, cancelled):
        assert controller_id == "fake_controller"
        assert scene_id == "flat"
        assert data_source_id == "local_headless"
        progress(0.5, "fake run")
        trace = output_dir / "control_trace.jsonl"
        trace.write_text("{}\n", encoding="utf-8")
        assert cancelled() is False
        result = TraditionalControlResult(
            verdict="passed",
            summary="fake run",
            metrics=[TraditionalControlMetricInfo(id="steps", label="Steps", value=1)],
            raw={"steps": 1, "requested_steps": 5, "elapsed_s": 0.02},
        )
        return ProviderRunResult(result=result, playback=self._playback(output_dir), raw_artifacts={"trace": trace})

    def load_playback(self, output_dir: Path, max_frames: int = 900) -> DiagnosticPlayback:
        return self._playback(output_dir)


def _service(tmp_path: Path) -> TraditionalControlDemoService:
    registry = TraditionalControlProviderRegistry()
    registry.register(_Provider())
    settings = LocomotionConsoleSettings(
        source="fake",
        local_state_root=str(tmp_path / "state"),
        traditional_control_output_root=str(tmp_path / "runs"),
    )
    return TraditionalControlDemoService(settings, registry=registry)


def test_registry_exposes_provider_catalog_without_product_branching() -> None:
    registry = TraditionalControlProviderRegistry()
    registry.register(_Provider())
    catalog = registry.catalogs()[0]
    provider, _, source = registry.resolve(
        product_id="fake_product",
        controller_id="fake_controller",
        scene_id="flat",
        data_source_id="local_headless",
    )
    assert catalog.product.id == "fake_product"
    assert provider.provider_id == "fake_provider"
    assert source.kind == "headless"


def test_catalog_marks_replay_unavailable_until_matching_artifact_exists(tmp_path: Path) -> None:
    service = _service(tmp_path)
    catalog = asyncio.run(service.catalog())
    replay = next(item for item in catalog.products[0].data_sources if item.kind == "replay")
    assert replay.available is False

    request = TraditionalControlRunRequest(
        product_id="fake_product",
        controller_id="fake_controller",
        scene_id="flat",
        duration_s=0.1,
        data_source_id="local_headless",
    )

    async def run_once() -> None:
        await service.start(request)
        for _ in range(30):
            if (await service.status()).state == "complete":
                return
            await asyncio.sleep(0.01)
        pytest.fail("demo did not complete")

    asyncio.run(run_once())
    refreshed = asyncio.run(service.catalog())
    replay = next(item for item in refreshed.products[0].data_sources if item.kind == "replay")
    assert replay.available is True


def test_taili_replay_conversion_does_not_build_simulator_controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from products.taili.traditional_control import demo_provider as provider_module
    from products.taili.traditional_control.demo_provider import build_provider

    product = ProductRegistry().get("taili")
    provider = build_provider(product=product, workspace_root=Path.cwd())
    run_dir = tmp_path / "replay"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "request": {"scene_id": "flat", "product_id": "taili", "controller_id": "taili_mpc_wbc"},
        }),
        encoding="utf-8",
    )
    row = {
        "time_s": 0.0,
        "state": {
            "base_position_world": [0.0, 0.0, 0.55],
            "base_rpy": [0.0, 0.0, 0.0],
            "q": [0.0] * 12,
            "foot_positions_body": [[0.2, 0.1, -0.5], [0.2, -0.1, -0.5], [-0.2, 0.1, -0.5], [-0.2, -0.1, -0.5]],
            "contacts": {"in_contact": [True, True, True, True], "normal_force": [10.0] * 4},
        },
    }
    (run_dir / "control_trace.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        provider_module,
        "build_taili_controller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("simulator controller must not be built")),
    )
    playback = provider.load_playback(run_dir)
    assert playback.available is True
    assert playback.robot.id == "robot.taili"
    assert playback.command == [0.0, 0.0, 0.0]
    assert playback.scene.terrain_primitives


def test_service_writes_manifest_and_can_reload_as_replay(tmp_path: Path) -> None:
    service = _service(tmp_path)
    request = TraditionalControlRunRequest(
        product_id="fake_product",
        controller_id="fake_controller",
        scene_id="flat",
        duration_s=0.1,
        data_source_id="local_headless",
    )

    async def scenario() -> tuple[str, Path, DiagnosticPlayback]:
        started = await service.start(request)
        assert started.state == "starting"
        for _ in range(20):
            status = await service.status()
            if status.state == "complete":
                break
            await asyncio.sleep(0.01)
        assert status.state == "complete"
        playback = await service.playback()
        return status.run_id or "", Path(status.output_dir or ""), playback

    run_id, output_dir, playback = asyncio.run(scenario())
    status = asyncio.run(service.status(run_id=run_id))
    assert status.requested_duration_s == pytest.approx(0.1)
    assert status.simulated_duration_s == pytest.approx(0.02)
    assert status.completed_steps == 1
    assert status.requested_steps == 5
    assert status.verdict == "passed"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "traditional_control_demo_manifest_v1"
    assert manifest["provider_id"] == "fake_provider"
    assert manifest["artifacts"]["trace"]["available"] is True
    assert len(manifest["artifacts"]["trace"]["sha256"]) == 64
    assert playback.available and playback.robot.urdf_url == "/robot/fake.urdf"
    assert playback.command == []

    limited = asyncio.run(service.playback(max_frames=3, run_id=run_id))
    expanded = asyncio.run(service.playback(max_frames=8, run_id=run_id))
    assert len(limited.frames) == 3
    assert len(expanded.frames) == 8
    assert limited.frames[-1].t == playback.frames[-1].t

    replay_values = request.model_dump()
    replay_values.update(data_source_id="local_replay", replay_id=run_id)
    replay = TraditionalControlRunRequest(**replay_values)
    replay_status = asyncio.run(service.start(replay))
    assert replay_status.state == "complete"
    assert asyncio.run(service.playback(run_id=run_id)).available

    elapsed = asyncio.run(service.status(run_id=run_id)).elapsed_s
    time.sleep(0.02)
    assert asyncio.run(service.status(run_id=run_id)).elapsed_s == pytest.approx(elapsed)


def test_replay_path_cannot_escape_output_root(tmp_path: Path) -> None:
    service = _service(tmp_path)
    request = TraditionalControlRunRequest(
        product_id="fake_product",
        controller_id="fake_controller",
        scene_id="flat",
        duration_s=0.1,
        data_source_id="local_replay",
        replay_id="..\\outside",
    )
    with pytest.raises(Exception, match="inside|output root"):
        asyncio.run(service.start(request))


def test_history_exposes_early_termination_reason(tmp_path: Path) -> None:
    service = _service(tmp_path)
    run_dir = service.output_root / "20260825_000000_failed"
    run_dir.mkdir()
    request = TraditionalControlRunRequest(
        product_id="fake_product",
        controller_id="fake_controller",
        scene_id="flat",
        duration_s=3.0,
        data_source_id="local_headless",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "state": "complete",
            "provider_id": "fake_provider",
            "request": request.model_dump(mode="json"),
            "progress": 1.0,
        }),
        encoding="utf-8",
    )
    (run_dir / "playback.json").write_text(
        json.dumps(_Provider()._playback(run_dir).model_dump(mode="json")),
        encoding="utf-8",
    )
    (run_dir / "result.json").write_text(
        json.dumps({
            "verdict": "failed",
            "failure_reasons": ["controller_failure:qp_unsolved"],
            "raw": {
                "steps": 88,
                "requested_steps": 150,
                "elapsed_s": 1.76,
                "runtime_failure": {
                    "code": "qp_unsolved",
                    "message": "floating-base WBC QP failed",
                },
            },
        }),
        encoding="utf-8",
    )

    status = asyncio.run(service.status(run_id=run_dir.name))
    assert status.simulated_duration_s == pytest.approx(1.76)
    assert status.completed_steps == 88
    assert status.requested_steps == 150
    assert status.termination_reason == "qp_unsolved: floating-base WBC QP failed"


def test_catalog_merges_compatible_providers_without_product_duplicates(tmp_path: Path) -> None:
    class SecondProvider(_Provider):
        provider_id = "second_provider"

        def catalog(self) -> ProviderCatalog:
            base = super().catalog()
            return ProviderCatalog(
                product=base.product,
                data_sources=(
                    TraditionalControlDataSourceInfo(
                        id="second_headless", label="Second run", kind="headless"
                    ),
                ),
            )

    registry = TraditionalControlProviderRegistry()
    registry.register(_Provider())
    registry.register(SecondProvider())
    settings = LocomotionConsoleSettings(
        source="fake",
        local_state_root=str(tmp_path / "state"),
        traditional_control_output_root=str(tmp_path / "runs"),
    )
    service = TraditionalControlDemoService(settings, registry=registry)
    catalog = asyncio.run(service.catalog())
    assert len(catalog.products) == 1
    assert sorted(catalog.products[0].controllers[0].provider_ids) == [
        "fake_provider",
        "second_provider",
    ]
    assert {item.id for item in catalog.products[0].data_sources} == {
        "local_headless",
        "local_replay",
        "second_headless",
    }


def test_second_start_is_rejected_while_a_job_is_active(tmp_path: Path) -> None:
    class SlowProvider(_Provider):
        def run(self, **kwargs):
            time.sleep(0.1)
            return super().run(**kwargs)

    registry = TraditionalControlProviderRegistry()
    registry.register(SlowProvider())
    settings = LocomotionConsoleSettings(
        source="fake",
        local_state_root=str(tmp_path / "state"),
        traditional_control_output_root=str(tmp_path / "runs"),
    )
    service = TraditionalControlDemoService(settings, registry=registry)
    request = TraditionalControlRunRequest(
        product_id="fake_product",
        controller_id="fake_controller",
        scene_id="flat",
        duration_s=0.1,
        data_source_id="local_headless",
    )

    async def scenario() -> None:
        await service.start(request)
        with pytest.raises(RuntimeError, match="already running"):
            await service.start(request)
        for _ in range(30):
            if (await service.status()).state == "complete":
                return
            await asyncio.sleep(0.01)
        pytest.fail("slow demo did not complete")

    asyncio.run(scenario())


def test_service_marks_unfinished_manifest_interrupted_after_restart(tmp_path: Path) -> None:
    service = _service(tmp_path)
    request = TraditionalControlRunRequest(
        product_id="fake_product",
        controller_id="fake_controller",
        scene_id="flat",
        duration_s=0.1,
        data_source_id="local_headless",
    )

    async def run_once() -> Path:
        started = await service.start(request)
        for _ in range(30):
            status = await service.status()
            if status.state == "complete":
                return Path(status.manifest_path or "")
            await asyncio.sleep(0.01)
        pytest.fail(f"demo did not complete: {started.run_id}")

    manifest_path = asyncio.run(run_once())
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["state"] = "running"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    restarted = _service(tmp_path)
    status = asyncio.run(restarted.status())
    assert status.state == "error"
    assert "interrupted" in status.message


def test_service_cooperatively_cancels_active_provider(tmp_path: Path) -> None:
    class CancellableProvider(_Provider):
        def run(self, **kwargs):
            for _ in range(50):
                if kwargs["cancelled"]():
                    raise DemoRunCancelled("cancelled")
                time.sleep(0.005)
            return super().run(**kwargs)

    registry = TraditionalControlProviderRegistry()
    registry.register(CancellableProvider())
    settings = LocomotionConsoleSettings(
        source="fake",
        local_state_root=str(tmp_path / "state"),
        traditional_control_output_root=str(tmp_path / "runs"),
    )
    service = TraditionalControlDemoService(settings, registry=registry)
    request = TraditionalControlRunRequest(
        product_id="fake_product",
        controller_id="fake_controller",
        scene_id="flat",
        duration_s=0.1,
        data_source_id="local_headless",
    )

    async def scenario() -> None:
        await service.start(request)
        await asyncio.sleep(0.02)
        requested = await service.cancel()
        assert requested.message == "Cancellation requested"
        for _ in range(30):
            status = await service.status()
            if status.state == "cancelled":
                assert status.error == ""
                return
            await asyncio.sleep(0.01)
        pytest.fail("cancel request did not stop provider")

    asyncio.run(scenario())
