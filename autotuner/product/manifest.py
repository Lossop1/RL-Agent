"""与机器人型号无关的产品清单数据模型。

清单描述“用哪些输入生成一个产品”，不实现训练算法，也不假设 DoF、腿数或
具体仿真框架。所有路径都相对于仓库根目录，便于计算来源哈希和复现实验。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RELATIVE_PATH_RE = re.compile(r"^[^\\/].*$")


class ProductManifestError(ValueError):
    """产品清单格式或边界不满足契约。"""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProductManifestError(f"{name} must be a mapping")
    return value


def _string(value: Any, name: str, *, required: bool = True) -> str:
    result = "" if value is None else str(value).strip()
    if required and not result:
        raise ProductManifestError(f"{name} is required")
    return result


def _tuple_strings(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ProductManifestError(f"{name} must be a list")
    return tuple(_string(item, f"{name}[]") for item in value)


@dataclass(frozen=True)
class RobotProfile:
    """产品声明的机器人形态，不对机器人类别做硬编码假设。"""

    id: str
    label: str
    status: str
    dof: int
    base_link: str
    joint_order: tuple[str, ...]
    leg_order: tuple[str, ...] = ()
    foot_links: tuple[str, ...] = ()
    diagnostic_spec: str = ""
    capabilities: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class AssetSpec:
    """产品生成所需的一个资产或资产目录。"""

    id: str
    kind: str
    path: str
    required: bool = True
    sha256: str = ""
    note: str = ""

    def resolve(self, root: Path) -> Path:
        path = Path(self.path)
        if path.is_absolute() or not _RELATIVE_PATH_RE.match(self.path) or ".." in path.parts:
            raise ProductManifestError(f"asset path must stay under the workspace: {self.path!r}")
        return (root / path).resolve()


@dataclass(frozen=True)
class ProductManifest:
    """可生成产品的完整输入描述。"""

    schema_version: int
    product_id: str
    version: str
    label: str
    status: str
    robot: RobotProfile
    task_id: str
    task_family: str
    framework_id: str
    train_entrypoint: str
    diagnose_entrypoint: str
    config_path: str
    source_roots: tuple[str, ...] = ()
    sources: Mapping[str, str] = field(default_factory=dict)
    assets: tuple[AssetSpec, ...] = ()
    payload_builder: str = ""
    payload_package: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Logical task identity is product-owned. Runtime registration IDs are
    # kept separately because an IsaacLab/Gym process may expose aliases.
    runtime_task_ids: tuple[str, ...] = ()
    task_requirements: Mapping[str, Any] = field(default_factory=dict)
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    deployment: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    compatibility: Mapping[str, Any] = field(default_factory=dict)
    knowledge: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProductManifest":
        product = _mapping(data.get("product"), "product")
        robot_data = _mapping(data.get("robot"), "robot")
        task = _mapping(data.get("task"), "task")
        build = _mapping(data.get("build"), "build")
        sources = _mapping(data.get("sources"), "sources")
        runtime = _mapping(data.get("runtime"), "runtime")

        robot_id = _string(robot_data.get("id"), "robot.id")
        robot = RobotProfile(
            id=robot_id,
            label=_string(robot_data.get("label"), "robot.label"),
            status=_string(robot_data.get("status", product.get("status", "draft")), "robot.status"),
            dof=int(robot_data.get("dof", 0)),
            base_link=_string(robot_data.get("base_link"), "robot.base_link"),
            joint_order=_tuple_strings(robot_data.get("joint_order"), "robot.joint_order"),
            leg_order=_tuple_strings(robot_data.get("leg_order"), "robot.leg_order"),
            foot_links=_tuple_strings(robot_data.get("foot_links"), "robot.foot_links"),
            diagnostic_spec=_string(robot_data.get("diagnostic_spec", ""), "robot.diagnostic_spec", required=False),
            capabilities=_tuple_strings(robot_data.get("capabilities"), "robot.capabilities"),
            note=_string(robot_data.get("note", ""), "robot.note", required=False),
        )

        raw_assets = data.get("assets", ())
        if not isinstance(raw_assets, (list, tuple)):
            raise ProductManifestError("assets must be a list")
        assets = tuple(
            AssetSpec(
                id=_string(_mapping(item, "assets[]").get("id"), "assets[].id"),
                kind=_string(_mapping(item, "assets[]").get("kind"), "assets[].kind"),
                path=_string(_mapping(item, "assets[]").get("path"), "assets[].path"),
                required=bool(_mapping(item, "assets[]").get("required", True)),
                sha256=_string(_mapping(item, "assets[]").get("sha256", ""), "assets[].sha256", required=False),
                note=_string(_mapping(item, "assets[]").get("note", ""), "assets[].note", required=False),
            )
            for item in raw_assets
        )

        manifest = cls(
            schema_version=int(data.get("schema_version", 0)),
            product_id=_string(product.get("id"), "product.id"),
            version=_string(product.get("version"), "product.version"),
            label=_string(product.get("label"), "product.label"),
            status=_string(product.get("status", "draft"), "product.status"),
            robot=robot,
            task_id=_string(task.get("id"), "task.id"),
            runtime_task_ids=_tuple_strings(runtime.get("task_ids"), "runtime.task_ids"),
            task_family=_string(task.get("family"), "task.family"),
            framework_id=_string(task.get("framework_id"), "task.framework_id"),
            train_entrypoint=_string(task.get("train_entrypoint"), "task.train_entrypoint"),
            diagnose_entrypoint=_string(task.get("diagnose_entrypoint"), "task.diagnose_entrypoint"),
            config_path=_string(sources.get("config"), "sources.config"),
            source_roots=_tuple_strings(sources.get("roots"), "sources.roots"),
            sources={str(key): _string(value, f"sources.{key}") for key, value in sources.items() if key != "roots"},
            task_requirements=dict(_mapping(task.get("requirements"), "task.requirements")),
            assets=assets,
            payload_builder=_string(build.get("payload_builder", ""), "build.payload_builder", required=False),
            payload_package=_string(build.get("payload_package", ""), "build.payload_package", required=False),
            telemetry=dict(_mapping(data.get("telemetry"), "telemetry")),
            diagnostics=dict(_mapping(data.get("diagnostics"), "diagnostics")),
            deployment=dict(_mapping(data.get("deployment"), "deployment")),
            runtime=dict(runtime),
            compatibility=dict(_mapping(data.get("compatibility"), "compatibility")),
            knowledge=dict(_mapping(data.get("knowledge"), "knowledge")),
            metadata=dict(_mapping(data.get("metadata"), "metadata")),
        )
        return manifest

    def robot_profile(self) -> RobotProfile:
        return self.robot

    def validate(self, root: Path | None = None, *, check_files: bool = True) -> list[str]:
        issues: list[str] = []
        if self.schema_version != 1:
            issues.append(f"schema_version must be 1, got {self.schema_version}")
        for name, value in (
            ("product_id", self.product_id),
            ("robot.id", self.robot.id),
            ("task_id", self.task_id),
            ("framework_id", self.framework_id),
        ):
            if not _ID_RE.fullmatch(value):
                issues.append(f"{name} has unsafe id: {value!r}")
        for runtime_task_id in self.runtime_task_ids:
            if not _ID_RE.fullmatch(runtime_task_id):
                issues.append(f"runtime_task_id has unsafe id: {runtime_task_id!r}")
        if self.robot.dof <= 0:
            issues.append("robot.dof must be positive")
        if self.robot.dof != len(self.robot.joint_order):
            issues.append(f"robot.dof={self.robot.dof} but joint_order has {len(self.robot.joint_order)} joints")
        if len(set(self.robot.joint_order)) != len(self.robot.joint_order):
            issues.append("robot.joint_order contains duplicates")
        if not self.train_entrypoint or ":" not in self.train_entrypoint:
            issues.append("task.train_entrypoint must use module:callable form")
        if not self.diagnose_entrypoint or ":" not in self.diagnose_entrypoint:
            issues.append("task.diagnose_entrypoint must use module:callable form")
        paths = [self.config_path, *self.source_roots, *self.sources.values(), *(asset.path for asset in self.assets)]
        for value in paths:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                issues.append(f"path escapes workspace: {value!r}")
        if len({asset.id for asset in self.assets}) != len(self.assets):
            issues.append("assets contain duplicate ids")
        if root is not None and check_files:
            for asset in self.assets:
                try:
                    path = asset.resolve(root)
                except ProductManifestError as exc:
                    issues.append(str(exc))
                    continue
                if asset.required and not path.exists():
                    issues.append(f"missing required asset {asset.id}: {asset.path}")
            for name, value in (("config", self.config_path), *self.sources.items()):
                path = root / value
                if not path.exists():
                    issues.append(f"missing source {name}: {value}")
        return issues

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_product_manifest(path: str | Path) -> ProductManifest:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"product manifest does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ProductManifestError("product manifest root must be a mapping")
    return ProductManifest.from_mapping(raw)
