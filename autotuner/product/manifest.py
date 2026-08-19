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
_ENTRYPOINT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")


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
    # 资产复用策略是产品声明，不由执行层根据目录名称猜测。
    asset_reuse: Mapping[str, Any] = field(default_factory=dict)
    payload_builder: str = ""
    payload_package: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Logical task identity is product-owned. Runtime registration IDs are
    # kept separately because an IsaacLab/Gym process may expose aliases.
    runtime_task_ids: tuple[str, ...] = ()
    task_requirements: Mapping[str, Any] = field(default_factory=dict)
    # LLM 任务接收只能使用产品明确声明的词表，不能从系统代码猜测。
    task_vocabulary: Mapping[str, Any] = field(default_factory=dict)
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    # 仿真世界与 sim2sim 约束属于产品输入，而不是执行层的隐式默认值。
    simulation: Mapping[str, Any] = field(default_factory=dict)
    deployment: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    compatibility: Mapping[str, Any] = field(default_factory=dict)
    framework: Mapping[str, Any] = field(default_factory=dict)
    knowledge: Mapping[str, Any] = field(default_factory=dict)
    # 适配策略由产品声明；系统适配器只执行其中的通用算法和插件入口。
    adaptation: Mapping[str, Any] = field(default_factory=dict)
    # 产品专用能力通过角色化入口接入；系统层只依赖角色，不依赖产品包名。
    plugins: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

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
            task_vocabulary=dict(_mapping(task.get("intake"), "task.intake")),
            assets=assets,
            asset_reuse=dict(_mapping(data.get("asset_reuse"), "asset_reuse")),
            payload_builder=_string(build.get("payload_builder", ""), "build.payload_builder", required=False),
            payload_package=_string(build.get("payload_package", ""), "build.payload_package", required=False),
            telemetry=dict(_mapping(data.get("telemetry"), "telemetry")),
            diagnostics=dict(_mapping(data.get("diagnostics"), "diagnostics")),
            simulation=dict(_mapping(data.get("simulation"), "simulation")),
            deployment=dict(_mapping(data.get("deployment"), "deployment")),
            runtime=dict(runtime),
            compatibility=dict(_mapping(data.get("compatibility"), "compatibility")),
            framework=dict(_mapping(data.get("framework"), "framework")),
            knowledge=dict(_mapping(data.get("knowledge"), "knowledge")),
            adaptation=dict(_mapping(data.get("adaptation"), "adaptation")),
            plugins={
                str(role): {
                    str(operation): _string(reference, f"plugins.{role}.{operation}")
                    for operation, reference in _mapping(value, f"plugins.{role}").items()
                }
                for role, value in _mapping(data.get("plugins"), "plugins").items()
            },
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

        # 这些字段会直接驱动后续的解析、诊断和运行期探针，不能等到执行时
        # 才因类型错误失败；这里只校验结构，不替产品决定具体语义。
        if not isinstance(self.task_vocabulary, Mapping):
            issues.append("task.intake must be a mapping")
        else:
            for name, values in self.task_vocabulary.items():
                if not _ID_RE.fullmatch(str(name)):
                    issues.append(f"task.intake has unsafe field: {name!r}")
                if not isinstance(values, (list, tuple)) or not values or any(not str(item).strip() for item in values):
                    issues.append(f"task.intake.{name} must be a non-empty list of strings")

        profiles = self.framework.get("profiles")
        if profiles is not None and not isinstance(profiles, Mapping):
            issues.append("framework.profiles must be a mapping")
        elif isinstance(profiles, Mapping):
            default_profile = str(self.framework.get("default_profile") or "").strip()
            if default_profile and default_profile not in profiles:
                issues.append(f"framework.default_profile is not declared: {default_profile!r}")
            for identifier, declaration in profiles.items():
                name = str(identifier)
                if not _ID_RE.fullmatch(name):
                    issues.append(f"framework profile has unsafe id: {name!r}")
                    continue
                if not isinstance(declaration, Mapping):
                    issues.append(f"framework.profiles.{name} must be a mapping")
                    continue
                for field_name in ("run_globs", "checkpoint_roots"):
                    value = declaration.get(field_name)
                    if value is not None and (not isinstance(value, (list, tuple)) or any(not str(item).strip() for item in value)):
                        issues.append(f"framework.profiles.{name}.{field_name} must be a list of strings")
                commands = declaration.get("commands")
                if commands is not None and not isinstance(commands, Mapping):
                    issues.append(f"framework.profiles.{name}.commands must be a mapping")

        payload = self.diagnostics.get("payload")
        if payload is not None and not isinstance(payload, Mapping):
            issues.append("diagnostics.payload must be a mapping")
        elif isinstance(payload, Mapping):
            for field_name in ("payload_roots", "payload_globs", "required_files"):
                value = payload.get(field_name)
                if value is not None and (not isinstance(value, (list, tuple)) or any(not str(item).strip() for item in value)):
                    issues.append(f"diagnostics.payload.{field_name} must be a list of strings")
        presets = self.diagnostics.get("presets")
        if presets is not None:
            if not isinstance(presets, (list, tuple)):
                issues.append("diagnostics.presets must be a list")
            else:
                preset_ids: list[str] = []
                for index, preset in enumerate(presets):
                    if not isinstance(preset, Mapping):
                        issues.append(f"diagnostics.presets[{index}] must be a mapping")
                        continue
                    preset_id = str(preset.get("id") or "").strip()
                    if not _ID_RE.fullmatch(preset_id):
                        issues.append(f"diagnostics.presets[{index}].id is unsafe or missing")
                    preset_ids.append(preset_id)
                    stages = preset.get("stages")
                    if stages is not None and (not isinstance(stages, (list, tuple)) or not stages):
                        issues.append(f"diagnostics.presets[{index}].stages must be a non-empty list")
                if len(set(preset_ids)) != len(preset_ids):
                    issues.append("diagnostics.presets contains duplicate ids")

        artifact_names = self.deployment.get("configuration_artifacts")
        if artifact_names is not None:
            if not isinstance(artifact_names, Mapping):
                issues.append("deployment.configuration_artifacts must be a mapping")
            else:
                for key, value in artifact_names.items():
                    if not _ID_RE.fullmatch(str(key)) or not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                        issues.append(f"deployment.configuration_artifacts.{key} must be a safe file name")
        for role, operations in self.plugins.items():
            if not _ID_RE.fullmatch(role) or not isinstance(operations, Mapping):
                issues.append(f"plugins.{role} must be a mapping")
                continue
            for operation, reference in operations.items():
                if not _ID_RE.fullmatch(operation):
                    issues.append(f"plugins.{role} has unsafe operation id: {operation!r}")
                if not _ENTRYPOINT_RE.fullmatch(str(reference)):
                    issues.append(
                        f"plugins.{role}.{operation} must use module:callable form: {reference!r}"
                    )
        compositions = self.framework.get("compositions")
        if compositions is not None and not isinstance(compositions, Mapping):
            issues.append("framework.compositions must be a mapping")
        elif isinstance(compositions, Mapping):
            for identifier, declaration in compositions.items():
                name = str(identifier)
                if not _ID_RE.fullmatch(name):
                    issues.append(f"framework composition has unsafe id: {name!r}")
                    continue
                if not isinstance(declaration, Mapping):
                    issues.append(f"framework.compositions.{name} must be a mapping")
                    continue
                component_ids = declaration.get("component_ids")
                if not isinstance(component_ids, (list, tuple)) or not component_ids:
                    issues.append(
                        f"framework.compositions.{name}.component_ids must be a non-empty list"
                    )
                    continue
                normalized_ids = [str(item).strip() for item in component_ids]
                if any(not item for item in normalized_ids):
                    issues.append(
                        f"framework.compositions.{name}.component_ids contains an empty id"
                    )
                if len(set(normalized_ids)) != len(normalized_ids):
                    issues.append(
                        f"framework.compositions.{name}.component_ids contains duplicates"
                    )
        if not isinstance(self.adaptation, Mapping):
            issues.append("adaptation must be a mapping")
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
