"""产品清单解析后的统一运行合同。

合同是产品输入与系统内部模块之间的稳定边界。训练、遥测、诊断和部署
模块只消费这份解析结果，不直接读取某个机器人目录或猜测配置。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from autotuner.execution.runtime import resolve_runtime_identity

from .manifest import ProductManifest, ProductManifestError
from .registry import DEFAULT_PRODUCT_ROOT, PROJECT_ROOT, ProductRegistry


_IGNORED_HASH_DIRECTORIES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
)
_IGNORED_HASH_SUFFIXES = frozenset({".pyc", ".pyo"})


class ContractResolutionError(ProductManifestError):
    """产品合同无法从清单和资产中解析时抛出。"""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_path(path: Path) -> tuple[str, int]:
    """对文件或目录生成稳定摘要，不把 mtime 纳入合同。"""
    if path.is_file():
        return _hash_file(path), 1
    if not path.is_dir():
        return "", 0
    digest = hashlib.sha256()
    count = 0
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        relative_path = item.relative_to(path)
        if any(part in _IGNORED_HASH_DIRECTORIES for part in relative_path.parts):
            continue
        if item.suffix.lower() in _IGNORED_HASH_SUFFIXES:
            continue
        relative = relative_path.as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(_hash_file(item)))
        count += 1
    return digest.hexdigest(), count


def _safe_resolve(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ContractResolutionError(f"path escapes workspace: {relative!r}") from exc
    return candidate


@dataclass(frozen=True)
class ResolvedAsset:
    id: str
    kind: str
    declared_path: str
    resolved_path: str
    required: bool
    exists: bool
    sha256: str = ""
    declared_sha256: str = ""
    file_count: int = 0


@dataclass(frozen=True)
class ResolvedProductContract:
    """绑定产品、资产、配置和运行接口的不可变合同。"""

    schema_version: int
    product_id: str
    product_version: str
    product_digest: str
    asset_digest: str
    source_digest: str
    config_digest: str
    robot: Mapping[str, Any]
    training: Mapping[str, Any]
    telemetry: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    deployment: Mapping[str, Any]
    runtime: Mapping[str, Any]
    compatibility: Mapping[str, Any]
    simulation: Mapping[str, Any] = field(default_factory=dict)
    asset_reuse: Mapping[str, Any] = field(default_factory=dict)
    framework: Mapping[str, Any] = field(default_factory=dict)
    adaptation: Mapping[str, Any] = field(default_factory=dict)
    plugins: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    assets: tuple[ResolvedAsset, ...] = ()
    issues: tuple[str, ...] = ()
    contract_digest: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["assets"] = [asdict(asset) for asset in self.assets]
        return data

    def digest(self) -> str:
        data = self.to_dict()
        data.pop("contract_digest", None)
        return _sha256_bytes(_canonical_json(data).encode("utf-8"))

    def with_digest(self) -> "ResolvedProductContract":
        return ResolvedProductContract(**{**self.to_dict(), "assets": self.assets, "contract_digest": self.digest()})

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_canonical_json(self.to_dict()) + "\n", encoding="utf-8")
        return target


def _source_paths(manifest: ProductManifest) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = [("config", manifest.config_path)]
    values.extend((f"root:{index}", value) for index, value in enumerate(manifest.source_roots))
    values.extend((f"source:{key}", value) for key, value in manifest.sources.items())
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for label, value in values:
        if value and value not in seen:
            result.append((label, value))
            seen.add(value)
    return result


def _section_with_defaults(values: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(values)
    for key, value in defaults.items():
        result.setdefault(key, value)
    return result


def resolve_product_contract(
    product: ProductManifest | str | None = None,
    *,
    root: str | Path = PROJECT_ROOT,
    registry: ProductRegistry | None = None,
    check_files: bool = True,
) -> ResolvedProductContract:
    """从产品清单解析一份可供各子系统消费的合同。"""
    workspace = Path(root).resolve()
    if isinstance(product, ProductManifest):
        manifest = product
    else:
        selected_registry = registry or ProductRegistry(
            workspace / "config" / "products" if registry is None else DEFAULT_PRODUCT_ROOT,
            workspace_root=workspace,
        )
        manifest = selected_registry.get(product)

    issues = list(manifest.validate(workspace, check_files=check_files))
    resolved_assets: list[ResolvedAsset] = []
    for asset in manifest.assets:
        path = _safe_resolve(workspace, asset.path)
        exists = path.exists()
        digest, file_count = _hash_path(path) if exists else ("", 0)
        resolved_assets.append(
            ResolvedAsset(
                id=asset.id,
                kind=asset.kind,
                declared_path=asset.path,
                resolved_path=path.relative_to(workspace).as_posix() if exists else asset.path,
                required=asset.required,
                exists=exists,
                sha256=digest,
                declared_sha256=asset.sha256,
                file_count=file_count,
            )
        )
        if asset.sha256 and digest and asset.sha256.lower() != digest.lower():
            issues.append(f"asset {asset.id} sha256 does not match declared digest")

    source_records: list[dict[str, Any]] = []
    for label, relative in _source_paths(manifest):
        path = _safe_resolve(workspace, relative)
        digest, file_count = _hash_path(path) if path.exists() else ("", 0)
        source_records.append(
            {"label": label, "path": relative, "exists": path.exists(), "sha256": digest, "file_count": file_count}
        )
    config_digest = next((item["sha256"] for item in source_records if item["label"] == "config"), "")
    asset_digest = _sha256_bytes(_canonical_json([asdict(item) for item in resolved_assets]).encode("utf-8"))
    source_digest = _sha256_bytes(_canonical_json(source_records).encode("utf-8"))

    telemetry = _section_with_defaults(
        manifest.telemetry,
        {"source": manifest.sources.get("telemetry", ""), "contract": manifest.sources.get("telemetry_contract", "")},
    )
    diagnostics = _section_with_defaults(
        manifest.diagnostics,
        {"entrypoint": manifest.diagnose_entrypoint, "spec": manifest.robot.diagnostic_spec},
    )
    simulation = _section_with_defaults(
        manifest.simulation,
        {"worlds": [], "sim2sim": {"required": False}},
    )
    deployment = _section_with_defaults(
        manifest.deployment,
        {"payload_builder": manifest.payload_builder, "payload_package": manifest.payload_package},
    )
    runtime_identity = resolve_runtime_identity(manifest.runtime, root=workspace)
    if runtime_identity.issues:
        issues.extend(f"runtime: {issue}" for issue in runtime_identity.issues)
    runtime = runtime_identity.to_dict()
    compatibility = dict(manifest.compatibility)
    training = {
        "task_id": manifest.task_id,
        "runtime_task_ids": list(manifest.runtime_task_ids),
        "task_family": manifest.task_family,
        "framework_id": manifest.framework_id,
        "train_entrypoint": manifest.train_entrypoint,
        "config_path": manifest.config_path,
        "source_roots": list(manifest.source_roots),
        "sources": dict(manifest.sources),
        "requirements": dict(manifest.task_requirements),
        "intake": dict(manifest.task_vocabulary),
        "knowledge": dict(manifest.knowledge),
    }
    contract = ResolvedProductContract(
        schema_version=1,
        product_id=manifest.product_id,
        product_version=manifest.version,
        product_digest=manifest.digest(),
        asset_digest=asset_digest,
        source_digest=source_digest,
        config_digest=config_digest,
        robot=asdict(manifest.robot),
        training=training,
        telemetry=telemetry,
        diagnostics=diagnostics,
        deployment=deployment,
        runtime=runtime,
        compatibility=compatibility,
        simulation=simulation,
        asset_reuse=dict(manifest.asset_reuse),
        framework=dict(manifest.framework),
        adaptation=dict(manifest.adaptation),
        plugins={role: dict(operations) for role, operations in manifest.plugins.items()},
        assets=tuple(resolved_assets),
        issues=tuple(issues),
    )
    if issues and check_files:
        raise ContractResolutionError("product contract is invalid:\n" + "\n".join(issues))
    return contract.with_digest()
