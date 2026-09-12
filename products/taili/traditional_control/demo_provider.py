"""Taili adapter for the generic traditional-control console demo service.

The adapter owns only product data conversion and invokes the existing nominal
MPC/WBC + MuJoCo evaluation chain.  The console never imports this module
directly; ``config/products/taili.yaml`` declares the plugin entrypoint.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from autotuner.locomotion_console.schemas import (
    DiagnosticPlayback,
    DiagnosticPlaybackFoot,
    DiagnosticPlaybackPrimitive,
    DiagnosticPlaybackRobot,
    DiagnosticPlaybackScene,
    DiagnosticPlaybackFrame,
    TraditionalControlControllerInfo,
    TraditionalControlDataSourceInfo,
    TraditionalControlProductInfo,
    TraditionalControlSceneInfo,
    TraditionalControlMetricInfo,
    TraditionalControlResult,
)
from autotuner.product.manifest import ProductManifest

from .backends import (
    nominal_terrain_primitives_from_config,
    terrain_model_from_config,
)
from .config import build_taili_controller, load_taili_control_config
from .evaluation import run_mujoco_episode
from .kinematics import TailiKinematics
from .profile import load_taili_profile

from autotuner.locomotion_console.traditional_control_demo.contracts import (
    ProgressCallback,
    CancellationCallback,
    DemoRunCancelled,
    ProviderCatalog,
    ProviderRunResult,
)


def _finite(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rpy_to_quaternion(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = (_finite(item) for item in (rpy + [0.0, 0.0, 0.0])[:3])
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _quaternion_matrix_wxyz(quaternion: list[float]) -> np.ndarray:
    w, x, y, z = (_finite(item) for item in (quaternion + [1.0, 0.0, 0.0, 0.0])[:4])
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _scene_config(config: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    scenes = config.get("scenarios")
    value = scenes.get(scene_id) if isinstance(scenes, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError(f"unknown Taili traditional-control scene: {scene_id}")
    return value


def _command_mode(command: list[float]) -> str:
    vx, vy, wz = (list(command) + [0.0, 0.0, 0.0])[:3]
    if abs(wz) > 1e-9:
        return "yaw_left" if wz > 0 else "yaw_right"
    if abs(vx) > 1e-9:
        return "forward" if vx > 0 else "backward"
    if abs(vy) > 1e-9:
        return "left" if vy > 0 else "right"
    return "stand"


@dataclass
class TailiMujocoProvider:
    product: ProductManifest
    workspace_root: Path
    config_path: Path
    urdf_url: str

    @property
    def provider_id(self) -> str:
        config = load_taili_control_config(self.config_path)
        return str(config.get("demo", {}).get("provider_id") or "").strip()

    def _config(self) -> dict[str, Any]:
        return load_taili_control_config(self.config_path)

    def _robot(self, config: Mapping[str, Any]) -> DiagnosticPlaybackRobot:
        return DiagnosticPlaybackRobot(
            id=self.product.robot.id,
            label=self.product.robot.label,
            urdf_url=self.urdf_url,
            base_link=self.product.robot.base_link,
            joint_order=list(self.product.robot.joint_order),
            leg_order=list(self.product.robot.leg_order),
        )

    def catalog(self) -> ProviderCatalog:
        config = self._config()
        demo = config.get("demo") if isinstance(config.get("demo"), Mapping) else {}
        controller_cfg = demo.get("controller") if isinstance(demo.get("controller"), Mapping) else {}
        controller = TraditionalControlControllerInfo(
            id=str(controller_cfg.get("id") or self.provider_id),
            label=str(controller_cfg.get("label") or controller_cfg.get("id") or self.provider_id),
            description=str(controller_cfg.get("description") or ""),
            provider_id=self.provider_id,
        )
        evaluation = config.get("evaluation") if isinstance(config.get("evaluation"), Mapping) else {}
        durations = evaluation.get("duration_s") if isinstance(evaluation.get("duration_s"), Mapping) else {}
        scenes: list[TraditionalControlSceneInfo] = []
        scenario_values = config.get("scenarios") if isinstance(config.get("scenarios"), Mapping) else {}
        for scene_id, raw in scenario_values.items():
            if not isinstance(raw, Mapping):
                continue
            command = [float(value) for value in raw.get("command", [0.0, 0.0, 0.0])]
            terrain = raw.get("terrain") if isinstance(raw.get("terrain"), Mapping) else {}
            scenes.append(
                TraditionalControlSceneInfo(
                    id=str(scene_id),
                    label=str(raw.get("label") or scene_id),
                    description=str(raw.get("description") or ""),
                    terrain_type=str(terrain.get("type") or ""),
                    command=command,
                    duration_s=float(durations.get(str(scene_id), durations.get("default", 3.0))),
                )
            )
        product = TraditionalControlProductInfo(
            id=self.product.product_id,
            label=self.product.label,
            robot=self._robot(config),
            controllers=[controller],
            scenes=scenes,
        )
        mujoco_available = importlib.util.find_spec("mujoco") is not None
        raw_sources = demo.get("data_sources") if isinstance(demo.get("data_sources"), list) else []
        data_sources = tuple(
            TraditionalControlDataSourceInfo(
                id=str(item.get("id") or ""),
                label=str(item.get("label") or ""),
                kind=str(item.get("kind") or ""),
                available=(mujoco_available if item.get("kind") == "headless" else True),
                description=str(item.get("description") or ""),
            )
            for item in raw_sources
            if isinstance(item, Mapping)
        )
        return ProviderCatalog(product=product, data_sources=data_sources)

    def run(
        self,
        *,
        controller_id: str,
        scene_id: str,
        data_source_id: str,
        duration_s: float,
        output_dir: Path,
        progress: ProgressCallback,
        cancelled: CancellationCallback,
    ) -> ProviderRunResult:
        catalog = self.catalog()
        controller = next((item for item in catalog.product.controllers if item.id == controller_id), None)
        if controller is None:
            raise ValueError(f"unknown Taili traditional-control controller: {controller_id}")
        scene = next((item for item in catalog.product.scenes if item.id == scene_id), None)
        if scene is None:
            raise ValueError(f"unknown Taili traditional-control scene: {scene_id}")
        source = next((item for item in catalog.data_sources if item.id == data_source_id), None)
        if source is None:
            raise ValueError(f"unknown Taili traditional-control data source: {data_source_id}")
        if source.kind == "replay":
            raise ValueError("Taili replay data source must be loaded by the service")
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / "control_trace.jsonl"
        result_path = output_dir / "result.json"
        progress(0.05, "building nominal MuJoCo scene")
        try:
            result = run_mujoco_episode(
                scene_id,
                duration_s=duration_s,
                config_path=self.config_path,
                trace_path=trace_path,
                progress=lambda value: progress(0.05 + value * 0.85, "running headless MuJoCo episode"),
                cancelled=cancelled,
            )
        except InterruptedError as exc:
            raise DemoRunCancelled(str(exc)) from exc
        progress(0.95, "converting control trace to playback")
        playback = self._playback_from_trace(
            trace_path,
            scene_id=scene_id,
            output_dir=output_dir,
            result_path=result_path,
            max_frames=4000,
        )
        progress(1.0, "demo artifacts written")
        result_values = result.as_dict()
        generic_result = TraditionalControlResult(
            available=True,
            verdict="passed" if result.passed else "failed",
            summary=(
                f"{scene.label}: {result.steps}/{result.requested_steps} control steps completed"
            ),
            metrics=[
                TraditionalControlMetricInfo(id="steps", label="控制步数", value=result.steps),
                TraditionalControlMetricInfo(
                    id="elapsed_s", label="仿真时长", value=result.elapsed_s, unit="s"
                ),
                TraditionalControlMetricInfo(
                    id="displacement_x", label="X 位移", value=result.displacement_world[0], unit="m"
                ),
                TraditionalControlMetricInfo(
                    id="max_tilt", label="最大倾角", value=result.max_tilt_rad, unit="rad"
                ),
            ],
            failure_reasons=list(result.failure_reasons),
            raw=result_values,
        )
        return ProviderRunResult(
            result=generic_result,
            playback=playback,
            raw_artifacts={"trace": trace_path, "result": result_path},
            provenance={
                "provider": "products.taili.traditional_control.demo_provider",
                "product_manifest_digest": self.product.digest(),
                "traditional_control_config": str(self.config_path),
                "traditional_control_config_sha256": _sha256(self.config_path),
                "trace_schema": "traditional_control_trace_v1",
                "data_source_id": data_source_id,
            },
        )

    def load_playback(self, output_dir: Path, max_frames: int = 900) -> DiagnosticPlayback:
        manifest_path = output_dir / "manifest.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        request = raw.get("request") if isinstance(raw.get("request"), Mapping) else {}
        scene_id = str(request.get("scene_id") or raw.get("scene_id") or "")
        return self._playback_from_trace(
            output_dir / "control_trace.jsonl",
            scene_id=scene_id,
            output_dir=output_dir,
            result_path=output_dir / "result.json",
            max_frames=max_frames,
            manifest_path=manifest_path,
        )

    def _playback_from_trace(
        self,
        trace_path: Path,
        *,
        scene_id: str,
        output_dir: Path,
        result_path: Path,
        max_frames: int = 900,
        manifest_path: Path | None = None,
    ) -> DiagnosticPlayback:
        config = self._config()
        catalog = self.catalog()
        robot = catalog.product.robot
        scene_cfg = _scene_config(config, scene_id)
        # Playback reconstruction uses only the tracked URDF and declarative
        # terrain model.  It must remain usable when MuJoCo is unavailable.
        profile = load_taili_profile(config)
        kinematics = TailiKinematics(profile)
        terrain = terrain_model_from_config(config, scene_id)
        terrain_primitives = [
            DiagnosticPlaybackPrimitive(
                id=primitive.id,
                type=primitive.type,
                center=list(primitive.center),
                size=list(primitive.size),
            )
            for primitive in nominal_terrain_primitives_from_config(
                config,
                scene_id,
                profile=profile,
                kinematics=kinematics,
                terrain=terrain,
            )
        ]
        scene = DiagnosticPlaybackScene(
            id=scene_id,
            label=str(scene_cfg.get("label") or scene_id),
            terrain_primitives=terrain_primitives,
        )
        frames: list[DiagnosticPlaybackFrame] = []
        command = [float(value) for value in scene_cfg.get("command", [0.0, 0.0, 0.0])]
        command_mode = _command_mode(command)
        if not trace_path.exists():
            return DiagnosticPlayback(
                available=False,
                message=f"control trace not found: {trace_path}",
                source="local",
                command=command,
                output_dir=str(output_dir),
                manifest_path=str(manifest_path or output_dir / "manifest.json"),
                result_path=str(result_path),
                joint_order=robot.joint_order,
                leg_order=robot.leg_order,
                robot=robot,
                scene=scene,
            )
        rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            state = row.get("state") if isinstance(row.get("state"), Mapping) else {}
            base = [_finite(value) for value in state.get("base_position_world", [0.0, 0.0, 0.0])]
            rpy = [_finite(value) for value in state.get("base_rpy", [0.0, 0.0, 0.0])]
            quaternion = _rpy_to_quaternion(rpy)
            rotation = _quaternion_matrix_wxyz(quaternion)
            body_feet = state.get("foot_positions_body")
            contacts = state.get("contacts") if isinstance(state.get("contacts"), Mapping) else {}
            contact_values = list(contacts.get("in_contact", []))
            normal_values = list(contacts.get("normal_force", []))
            feet: dict[str, DiagnosticPlaybackFoot] = {}
            for index, leg in enumerate(robot.leg_order):
                if not isinstance(body_feet, list) or index >= len(body_feet):
                    continue
                body_position = np.asarray(body_feet[index], dtype=np.float64)
                if body_position.shape != (3,) or not np.isfinite(body_position).all():
                    continue
                world_position = np.asarray(base, dtype=np.float64) + rotation @ body_position
                normal_force = _finite(normal_values[index]) if index < len(normal_values) else 0.0
                contact = bool(contact_values[index]) if index < len(contact_values) else normal_force > 0.0
                feet[leg] = DiagnosticPlaybackFoot(
                    position=[float(value) for value in world_position],
                    contact=contact,
                    normal_force=normal_force,
                    force_norm=normal_force,
                    force_w_z=normal_force,
                )
            terrain_height = _finite(terrain.height_at(base[0], base[1]))
            # Terrain models express height relative to their configured base;
            # the scene primitive carries the physical sole/world offset.
            if terrain_primitives:
                terrain_cfg = scene_cfg.get("terrain") if isinstance(scene_cfg.get("terrain"), Mapping) else {}
                terrain_base = _finite(terrain_cfg.get("base_height", terrain_cfg.get("height", 0.0)))
                terrain_height += float(terrain_primitives[0].center[2]) - terrain_base
            joints = state.get("q") if isinstance(state.get("q"), list) else []
            frames.append(
                DiagnosticPlaybackFrame(
                    t=_finite(row.get("time_s")),
                    stage=scene_id,
                    case_id=0,
                    env_id=0,
                    segment_id=0,
                    command_mode=command_mode,
                    base_position=[float(value) for value in base[:3]],
                    base_quaternion_wxyz=quaternion,
                    joints=[_finite(value) for value in joints],
                    feet=feet,
                    done=row.get("kind") == "control_failure",
                    terrain_height=terrain_height,
                    terrain=str(scene_cfg.get("terrain", {}).get("type") or ""),
                )
            )
        if len(frames) > max_frames:
            stride = int(math.ceil(len(frames) / max_frames))
            frames = frames[::stride]
        else:
            stride = 1
        fps = 50.0
        if len(frames) > 1:
            deltas = [b.t - a.t for a, b in zip(frames, frames[1:]) if b.t > a.t]
            if deltas:
                fps = max(1.0, min(120.0, 1.0 / (sum(deltas) / len(deltas))))
        return DiagnosticPlayback(
            available=bool(frames),
            message="本机无头传统控制 trace 回放" if frames else "control trace 不包含可回放状态帧",
            source="local",
            command=command,
            output_dir=str(output_dir),
            manifest_path=str(manifest_path or output_dir / "manifest.json"),
            result_path=str(result_path),
            fps=fps,
            joint_order=robot.joint_order,
            leg_order=robot.leg_order,
            robot=robot,
            scene=scene,
            frames=frames,
            source_rows=len(rows),
            stride=stride,
            selected_env_id=0,
            available_env_ids=[0],
        )


def build_provider(*, product: ProductManifest, workspace_root: Path) -> TailiMujocoProvider:
    """Product-plugin factory used by the generic registry."""

    simulation = product.to_dict().get("simulation", {})
    worlds = simulation.get("worlds", []) if isinstance(simulation, Mapping) else []
    declaration = next(
        (
            item.get("traditional_control")
            for item in worlds
            if isinstance(item, Mapping)
            and str(item.get("backend") or "") == "mujoco"
            and isinstance(item.get("traditional_control"), Mapping)
        ),
        None,
    )
    if not isinstance(declaration, Mapping):
        raise ValueError("product does not declare a MuJoCo traditional-control adapter")
    config_value = str(declaration.get("config") or "").strip()
    if not config_value:
        raise ValueError("traditional-control adapter config is empty")
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = workspace_root / config_path
    playback = declaration.get("playback") if isinstance(declaration.get("playback"), Mapping) else {}
    robot = playback.get("robot") if isinstance(playback.get("robot"), Mapping) else {}
    urdf_url = str(robot.get("urdf_url") or "").strip()
    if not urdf_url:
        raise ValueError("traditional-control playback robot.urdf_url is empty")
    return TailiMujocoProvider(product, workspace_root, config_path.resolve(), urdf_url)


__all__ = ["TailiMujocoProvider", "build_provider"]
