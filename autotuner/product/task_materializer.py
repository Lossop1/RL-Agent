"""把任务合同物料化为五类可消费、可校验的配置 artifact。

默认物料化只生成合同中的声明式 spec，不猜测某个机器人框架的字段。
产品若需要生成 IsaacLab、MuJoCo 或部署专用文件，必须在产品清单中声明
materializer.task_bundle 插件；插件只负责产品语义，artifact 身份和
完整性仍由本模块统一封装。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any, Mapping

from .plugins import ProductPluginError, load_product_plugin
from .task_contract import TaskContractBundle, TaskContractError


TASK_ARTIFACT_SCHEMA = "rl-agent.task-artifact/v1"
MATERIALIZER_VERSION = "task-bundle-materializer/1"
ARTIFACT_KINDS = ("training", "telemetry", "diagnostics", "simulation", "deployment")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MaterializedArtifact:
    """单个五类配置产物及其内容校验信息。"""

    kind: str
    path: Path
    content_digest: str
    file_digest: str
    source: str

    def to_dict(self, root: Path | None = None) -> dict[str, Any]:
        path = self.path
        if root is not None:
            try:
                path = path.relative_to(root)
            except ValueError:
                pass
        return {
            "kind": self.kind,
            "path": path.as_posix(),
            "content_digest": self.content_digest,
            "file_digest": self.file_digest,
            "source": self.source,
        }


@dataclass(frozen=True)
class MaterializedFile:
    """产品插件生成的实际运行文件及其校验信息。"""

    path: Path
    file_digest: str
    source: str

    def to_dict(self, root: Path | None = None) -> dict[str, Any]:
        path = self.path
        if root is not None:
            try:
                path = path.relative_to(root)
            except ValueError:
                pass
        return {
            "path": path.as_posix(),
            "file_digest": self.file_digest,
            "source": self.source,
        }


@dataclass(frozen=True)
class MaterializedTaskBundle:
    """一次物料化的完整结果。"""

    bundle_ref: str
    bundle_digest: str
    contract_digest: str
    root: Path
    artifacts: tuple[MaterializedArtifact, ...]
    generated_files: tuple[MaterializedFile, ...]
    manifest: Path
    manifest_digest: str

    @property
    def manifest_ref(self) -> str:
        """返回不依赖本机绝对路径的 artifact 清单引用。"""
        return f"task-artifact-manifest:{self.bundle_ref}:{self.manifest_digest}"

    def artifact_ref(self, kind: str) -> str:
        """为单个 artifact 生成稳定的逻辑引用。"""
        for artifact in self.artifacts:
            if artifact.kind == kind:
                return f"task-artifact:{self.bundle_ref}:{kind}:{artifact.file_digest}"
        raise TaskContractError(f"unknown materialized artifact kind: {kind}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TASK_ARTIFACT_SCHEMA,
            "bundle_ref": self.bundle_ref,
            "bundle_digest": self.bundle_digest,
            "contract_digest": self.contract_digest,
            "root": str(self.root),
            "manifest": str(self.manifest),
            "manifest_ref": self.manifest_ref,
            "manifest_digest": self.manifest_digest,
            "artifacts": [
                {
                    **item.to_dict(self.root),
                    "ref": self.artifact_ref(item.kind),
                }
                for item in self.artifacts
            ],
            "generated_files": [item.to_dict(self.root) for item in self.generated_files],
        }


class TaskBundleMaterializer:
    """统一生成任务合同的五类运行配置。"""

    def __init__(self, *, version: str = MATERIALIZER_VERSION) -> None:
        self.version = version

    @staticmethod
    def _plugin(product: Any) -> Any | None:
        if product is None:
            return None
        try:
            return load_product_plugin(product, "materializer", "task_bundle")
        except ProductPluginError:
            # 没有产品插件不是错误；此时交付通用 spec，禁止猜产品字段。
            return None

    def materialize(
        self,
        bundle: TaskContractBundle,
        output_root: str | os.PathLike[str],
        *,
        product: Any | None = None,
        bundle_ref: str = "",
    ) -> MaterializedTaskBundle:
        if set(bundle.specs) != set(ARTIFACT_KINDS):
            missing = sorted(set(ARTIFACT_KINDS) - set(bundle.specs))
            extra = sorted(set(bundle.specs) - set(ARTIFACT_KINDS))
            raise TaskContractError(f"task bundle kinds invalid; missing={missing}, extra={extra}")
        target = Path(output_root)
        existing = target / "materialization_manifest.json"
        if existing.is_file():
            current = json.loads(existing.read_text(encoding="utf-8"))
            if current.get("bundle_digest") != bundle.bundle_digest:
                raise TaskContractError(f"materialization output already belongs to another bundle: {target}")
            return self._load_result(target, current)

        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = parent / f".{target.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            plugin = self._plugin(product)
            plugin_specs: Mapping[str, Any] = {}
            plugin_files: Mapping[str, Any] = {}
            source = "generic_contract_spec"
            if plugin is not None:
                parameters = inspect.signature(plugin).parameters
                kwargs: dict[str, Any] = {"bundle": bundle, "output_dir": staging}
                if product is not None and (
                    "product" in parameters
                    or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
                ):
                    kwargs["product"] = product
                value = plugin(**kwargs)
                if value is not None:
                    if not isinstance(value, Mapping):
                        raise TaskContractError("task bundle materializer plugin must return a mapping")
                    plugin_specs = value.get("artifacts", value)
                    if not isinstance(plugin_specs, Mapping):
                        raise TaskContractError("materializer plugin artifacts must be a mapping")
                    raw_files = value.get("files", {})
                    if raw_files is not None and not isinstance(raw_files, Mapping):
                        raise TaskContractError("materializer plugin files must be a mapping")
                    plugin_files = raw_files or {}
                    source = "product_materializer"

            artifact_dir = staging / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifacts: list[MaterializedArtifact] = []
            for kind in ARTIFACT_KINDS:
                spec = plugin_specs.get(kind, bundle.specs[kind])
                if not isinstance(spec, Mapping):
                    raise TaskContractError(f"materialized {kind} spec must be a mapping")
                artifact = {
                    "schema_version": TASK_ARTIFACT_SCHEMA,
                    "kind": kind,
                    "contract_id": bundle.contract.contract_id,
                    "contract_version": bundle.contract.contract_version,
                    "contract_digest": bundle.contract.contract_digest,
                    "bundle_digest": bundle.bundle_digest,
                    "generator": self.version,
                    "source": source,
                    "content_digest": _digest(spec),
                    "spec": dict(spec),
                }
                path = artifact_dir / f"{kind}.json"
                path.write_text(_canonical(artifact) + "\n", encoding="utf-8", newline="\n")
                artifacts.append(
                    MaterializedArtifact(
                        kind, path, artifact["content_digest"], _file_digest(path), source
                    )
                )

            generated_files: list[MaterializedFile] = []
            files_root = staging / "files"
            for raw_path, raw_content in sorted(plugin_files.items(), key=lambda item: str(item[0])):
                relative = Path(str(raw_path))
                if relative.is_absolute() or ".." in relative.parts or not str(relative):
                    raise TaskContractError(f"generated file path escapes materialization root: {raw_path!r}")
                if isinstance(raw_content, bytes):
                    content = raw_content
                elif isinstance(raw_content, str):
                    content = raw_content.encode("utf-8")
                else:
                    raise TaskContractError(f"generated file content must be text or bytes: {raw_path!r}")
                if len(content) > 20 * 1024 * 1024:
                    raise TaskContractError(f"generated file is too large: {raw_path!r}")
                path = files_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                generated_files.append(MaterializedFile(path, _file_digest(path), source))

            manifest_data = {
                "schema_version": TASK_ARTIFACT_SCHEMA,
                "materializer_version": self.version,
                "bundle_ref": bundle_ref or f"{bundle.contract.contract_id}@{bundle.contract.contract_version}",
                "bundle_digest": bundle.bundle_digest,
                "contract_digest": bundle.contract.contract_digest,
                "artifacts": [item.to_dict(staging) for item in artifacts],
                "generated_files": [item.to_dict(staging) for item in generated_files],
            }
            manifest_path = staging / "materialization_manifest.json"
            manifest_path.write_text(_canonical(manifest_data) + "\n", encoding="utf-8", newline="\n")
            if target.exists():
                raise TaskContractError(f"materialization target appeared during generation: {target}")
            os.replace(staging, target)
            final_manifest = target / "materialization_manifest.json"
            return self._load_result(target, json.loads(final_manifest.read_text(encoding="utf-8")))
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _load_result(root: Path, data: Mapping[str, Any]) -> MaterializedTaskBundle:
        artifacts: list[MaterializedArtifact] = []
        for raw in data.get("artifacts", ()):
            if not isinstance(raw, Mapping):
                raise TaskContractError("materialization manifest contains an invalid artifact")
            relative = Path(str(raw.get("path") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise TaskContractError("materialization artifact path escapes its root")
            path = root / relative
            if not path.is_file() or _file_digest(path) != raw.get("file_digest"):
                raise TaskContractError(f"materialized artifact is missing or changed: {path}")
            artifact_data = json.loads(path.read_text(encoding="utf-8"))
            if artifact_data.get("content_digest") != raw.get("content_digest"):
                raise TaskContractError(f"materialized artifact content digest changed: {path}")
            artifacts.append(
                MaterializedArtifact(
                    kind=str(raw.get("kind") or ""),
                    path=path,
                    content_digest=str(raw.get("content_digest") or ""),
                    file_digest=str(raw.get("file_digest") or ""),
                    source=str(raw.get("source") or ""),
                )
            )
        generated_files: list[MaterializedFile] = []
        for raw in data.get("generated_files", ()):
            if not isinstance(raw, Mapping):
                raise TaskContractError("materialization manifest contains an invalid generated file")
            relative = Path(str(raw.get("path") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise TaskContractError("generated file path escapes its root")
            path = root / relative
            if not path.is_file() or _file_digest(path) != raw.get("file_digest"):
                raise TaskContractError(f"generated file is missing or changed: {path}")
            generated_files.append(
                MaterializedFile(
                    path=path,
                    file_digest=str(raw.get("file_digest") or ""),
                    source=str(raw.get("source") or ""),
                )
            )
        manifest = root / "materialization_manifest.json"
        return MaterializedTaskBundle(
            bundle_ref=str(data.get("bundle_ref") or ""),
            bundle_digest=str(data.get("bundle_digest") or ""),
            contract_digest=str(data.get("contract_digest") or ""),
            root=root,
            artifacts=tuple(artifacts),
            generated_files=tuple(generated_files),
            manifest=manifest,
            manifest_digest=_file_digest(manifest),
        )


def materialize_task_bundle(
    bundle: TaskContractBundle,
    output_root: str | os.PathLike[str],
    *,
    product: Any | None = None,
    bundle_ref: str = "",
) -> MaterializedTaskBundle:
    return TaskBundleMaterializer().materialize(bundle, output_root, product=product, bundle_ref=bundle_ref)


__all__ = [
    "ARTIFACT_KINDS",
    "MATERIALIZER_VERSION",
    "MaterializedArtifact",
    "MaterializedFile",
    "MaterializedTaskBundle",
    "TASK_ARTIFACT_SCHEMA",
    "TaskBundleMaterializer",
    "materialize_task_bundle",
]
