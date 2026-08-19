"""产品 payload 适配边界。

产品适配器负责把已解析合同变成可验证的归档；本模块只负责按合同加载
构建入口、统一返回值并创建执行层的部署规格。它不包含任何具体产品或
IsaacLab 专用分支，因此新机器人只需提供自己的清单和构建器。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import importlib
import inspect
from pathlib import Path
from typing import Any, Mapping

from autotuner.execution import load_payload_manifest, verify_payload_archive

from .contracts import ResolvedProductContract


class ProductPayloadError(ValueError):
    """产品 payload 构建结果与合同不一致时抛出。"""


@dataclass(frozen=True)
class ProductPayload:
    """产品构建器输出的统一、可验证结果。"""

    archive: Path
    manifest: Path
    payload_digest: str
    file_count: int = 0
    root_name: str = ""
    task_contract_ref: str = ""
    task_bundle_digest: str = ""
    task_artifact_manifest_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["archive"] = str(self.archive)
        data["manifest"] = str(self.manifest)
        return data


def _contract_data(contract: ResolvedProductContract | Mapping[str, Any]) -> Mapping[str, Any]:
    if hasattr(contract, "to_dict") and callable(contract.to_dict):
        contract = contract.to_dict()
    if not isinstance(contract, Mapping):
        raise ProductPayloadError("product contract must be a mapping")
    return contract


def _load_entrypoint(value: str) -> Any:
    module_name, separator, attribute = str(value or "").partition(":")
    if not separator or not module_name or not attribute:
        raise ProductPayloadError(f"payload builder must use module:callable form: {value!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ProductPayloadError(f"payload builder is missing: {value!r}") from exc


def _normalise_result(value: Any) -> ProductPayload:
    if isinstance(value, ProductPayload):
        return value
    if isinstance(value, Mapping):
        data = value
        archive = data.get("archive")
        manifest = data.get("manifest")
        digest = data.get("payload_digest") or data.get("digest")
        count = data.get("file_count", 0)
        root_name = data.get("root_name", "")
    else:
        archive = getattr(value, "archive", None)
        manifest = getattr(value, "manifest", None)
        digest = getattr(value, "payload_digest", None) or getattr(value, "digest", None)
        count = getattr(value, "file_count", 0)
        root_name = getattr(value, "root_name", "")
    if not archive or not manifest or not digest:
        raise ProductPayloadError("payload builder must return archive, manifest and payload_digest")
    return ProductPayload(
        archive=Path(archive).resolve(),
        manifest=Path(manifest).resolve(),
        payload_digest=str(digest),
        file_count=int(count or 0),
        root_name=str(root_name or ""),
    )


def build_product_payload(
    contract: ResolvedProductContract | Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    task_bundle: Any | None = None,
    task_artifacts: Any | None = None,
) -> ProductPayload:
    """调用产品清单声明的构建器，并在返回前完成本地归档校验。"""
    data = _contract_data(contract)
    deployment = data.get("deployment") if isinstance(data.get("deployment"), Mapping) else {}
    builder_name = str(deployment.get("payload_builder") or "").strip()
    builder = _load_entrypoint(builder_name)
    kwargs: dict[str, Any] = {}
    parameters = inspect.signature(builder).parameters
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if "contract" in parameters or accepts_kwargs:
        kwargs["contract"] = contract
    else:
        raise ProductPayloadError(f"payload builder must accept the resolved product contract: {builder_name!r}")
    if output_dir is not None and ("output_dir" in parameters or accepts_kwargs):
        kwargs["output_dir"] = Path(output_dir)
    if task_bundle is not None and ("task_bundle" in parameters or accepts_kwargs):
        kwargs["task_bundle"] = task_bundle
    if task_artifacts is not None and ("task_artifacts" in parameters or accepts_kwargs):
        kwargs["task_artifacts"] = task_artifacts
    result = _normalise_result(builder(**kwargs))
    payload_manifest = load_payload_manifest(result.manifest)
    payload_manifest.validate(require_digest=True)
    errors = verify_payload_archive(result.archive, payload_manifest)
    if errors:
        raise ProductPayloadError("payload archive failed verification:\n" + "\n".join(errors))
    expected_product = str(data.get("product_id") or "")
    if payload_manifest.product_id != expected_product:
        raise ProductPayloadError(
            f"payload product mismatch: {payload_manifest.product_id!r} != {expected_product!r}"
        )
    if payload_manifest.product_version != str(data.get("product_version") or ""):
        raise ProductPayloadError("payload product version does not match the resolved contract")
    if payload_manifest.contract_digest != str(data.get("contract_digest") or ""):
        raise ProductPayloadError("payload was built from a different product contract")
    runtime = data.get("runtime") if isinstance(data.get("runtime"), Mapping) else {}
    if payload_manifest.runtime_digest != str(runtime.get("digest") or ""):
        raise ProductPayloadError("payload runtime identity does not match the resolved contract")
    if payload_manifest.payload_digest != result.payload_digest:
        raise ProductPayloadError("builder result digest does not match its manifest")
    if task_bundle is not None:
        bundle_data = task_bundle.to_dict() if hasattr(task_bundle, "to_dict") else task_bundle
        if not isinstance(bundle_data, Mapping):
            raise ProductPayloadError("task bundle must expose a mapping representation")
        contract_id = str(bundle_data.get("contract_id") or bundle_data.get("contract_ref") or "")
        version = bundle_data.get("contract_version")
        result = replace(
            result,
            task_contract_ref=contract_id + (f"@{version}" if version is not None else ""),
            task_bundle_digest=str(bundle_data.get("bundle_digest") or ""),
        )
    if task_artifacts is not None:
        artifact_data = task_artifacts.to_dict() if hasattr(task_artifacts, "to_dict") else task_artifacts
        if not isinstance(artifact_data, Mapping):
            raise ProductPayloadError("task artifacts must expose a mapping representation")
        result = replace(
            result,
            task_artifact_manifest_ref=str(
                artifact_data.get("manifest_ref")
                or artifact_data.get("manifest")
                or artifact_data.get("manifest_path")
                or ""
            ),
        )
    return result


def make_deployment_spec(
    contract: ResolvedProductContract | Mapping[str, Any],
    payload: ProductPayload,
    *,
    run_id: str,
    run_manifest: Mapping[str, Any] | str | Path,
    task_bundle: Any | None = None,
    task_artifacts: Any | None = None,
) -> Any:
    """将产品产物绑定到无产品知识的执行层。"""
    from autotuner.execution import DeploymentSpec

    data = _contract_data(contract)
    runtime = data.get("runtime") if isinstance(data.get("runtime"), Mapping) else {}
    runtime_digest = str(runtime.get("digest") or "")
    if not runtime_digest:
        raise ProductPayloadError("resolved product contract has no runtime digest")
    task_contract_ref = payload.task_contract_ref
    task_bundle_digest = payload.task_bundle_digest
    task_artifact_manifest_ref = payload.task_artifact_manifest_ref
    if task_bundle is not None:
        bundle_data = task_bundle.to_dict() if hasattr(task_bundle, "to_dict") else task_bundle
        if isinstance(bundle_data, Mapping):
            contract_id = str(bundle_data.get("contract_id") or bundle_data.get("contract_ref") or "")
            version = bundle_data.get("contract_version")
            task_contract_ref = contract_id + (f"@{version}" if version is not None else "")
            task_bundle_digest = str(bundle_data.get("bundle_digest") or "")
    if task_artifacts is not None:
        artifact_data = task_artifacts.to_dict() if hasattr(task_artifacts, "to_dict") else task_artifacts
        if isinstance(artifact_data, Mapping):
            task_artifact_manifest_ref = str(
                artifact_data.get("manifest_ref")
                or artifact_data.get("manifest")
                or artifact_data.get("manifest_path")
                or ""
            )
    return DeploymentSpec(
        runtime=runtime,
        runtime_digest=runtime_digest,
        payload_archive=payload.archive,
        payload_manifest=payload.manifest,
        run_id=run_id,
        run_manifest=run_manifest,
        payload_digest=payload.payload_digest,
        task_contract_ref=task_contract_ref,
        task_bundle_digest=task_bundle_digest,
        task_artifact_manifest_ref=task_artifact_manifest_ref,
    )


__all__ = [
    "ProductPayload",
    "ProductPayloadError",
    "build_product_payload",
    "make_deployment_spec",
]
