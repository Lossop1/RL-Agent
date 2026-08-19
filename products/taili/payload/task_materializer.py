"""Taili 任务 artifact 物料化插件。

产品插件只负责把 Taili 已声明的配置来源和任务覆盖转换为实际运行文件。
合同校验、digest、原子写入和谱系由 autotuner.product 统一负责。
插件不修改仓库中的源 YAML，只返回由系统统一写入 artifact 根目录的文件。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml


_KINDS = ("training", "telemetry", "diagnostics", "simulation", "deployment")


def _spec(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and isinstance(value.get("spec"), Mapping):
        return dict(value["spec"])
    return dict(value) if isinstance(value, Mapping) else {}


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """按声明式 overlay 合并 YAML；列表整体替换，避免隐含的拼接语义。"""
    result = {str(key): value for key, value in base.items()}
    for raw_key, value in overlay.items():
        key = str(raw_key)
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _product_data(product: Any | None) -> Mapping[str, Any]:
    if product is None:
        return {}
    value = product.to_dict() if hasattr(product, "to_dict") else product
    return value if isinstance(value, Mapping) else {}


def _config_path(contract: Any) -> Path:
    raw = str(contract.training.get("config_path") or "").strip()
    if not raw:
        raise ValueError("Taili product contract has no training.config_path")
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path]
    # 从任意工作目录构建时，回到仓库根目录解析产品源文件。
    candidates.append(Path(__file__).resolve().parents[3] / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Taili task config source not found: {raw}")


def _allowed_overlay_roots(product: Any | None) -> frozenset[str]:
    data = _product_data(product)
    adaptation = data.get("adaptation") if isinstance(data.get("adaptation"), Mapping) else {}
    materialization = (
        adaptation.get("materialization")
        if isinstance(adaptation.get("materialization"), Mapping)
        else {}
    )
    raw = materialization.get("config_overlay_allowlist", ())
    return frozenset(str(item).strip() for item in raw if str(item).strip())


def _validate_overlay(overlay: Mapping[str, Any], allowed_roots: frozenset[str]) -> None:
    unknown = sorted(str(key) for key in overlay if str(key) not in allowed_roots)
    if unknown:
        raise ValueError(
            "Taili task config overlay contains undeclared roots: " + ", ".join(unknown)
        )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def materialize_task_bundle(
    *,
    bundle: Any,
    output_dir: Any,
    product: Any | None = None,
) -> Mapping[str, Any]:
    """生成 Taili 五类声明和实际运行文件，不执行训练或修改源配置。"""
    del output_dir  # 实际文件由通用 materializer 统一写入和计算 digest。
    contract = bundle.contract
    product_data = _product_data(product)
    product_identity = {
        "id": contract.product_id,
        "version": str(contract.provenance.get("product_version") or ""),
        "contract_digest": contract.product_contract_digest,
    }
    sections = {
        "training": contract.training,
        "telemetry": contract.telemetry,
        "diagnostics": contract.diagnostics,
        "simulation": contract.simulation,
        "deployment": contract.deployment,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for kind in _KINDS:
        artifacts[kind] = {
            "product": product_identity,
            "task_contract_ref": f"{contract.contract_id}@{contract.contract_version}",
            "task_spec": _spec(bundle.specs[kind]),
            "product_section": dict(sections[kind]),
        }

    source_path = _config_path(contract)
    source_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_config, Mapping):
        raise ValueError(f"Taili source config must be a mapping: {source_path}")
    overlay = contract.training.get("config_overlay", {})
    if not isinstance(overlay, Mapping):
        raise ValueError("Taili training.config_overlay must be a mapping")
    _validate_overlay(overlay, _allowed_overlay_roots(product_data))
    effective_config = _deep_merge(source_config, overlay)

    artifacts["training"].update(
        {
            "config_source": str(contract.training.get("config_path") or ""),
            "framework_id": str(contract.training.get("framework_id") or ""),
            "effective_config": "files/training/taili_blind_config.yaml",
            "mechanism_manifest": "files/training/mechanisms.json",
        }
    )
    artifacts["diagnostics"]["entrypoint"] = str(
        contract.diagnostics.get("entrypoint") or ""
    )
    artifacts["simulation"]["sim2sim"] = contract.simulation.get("sim2sim", {})
    artifacts["deployment"]["payload_package"] = str(
        contract.deployment.get("payload_package") or ""
    )

    return {
        "artifacts": artifacts,
        "files": {
            "training/taili_blind_config.yaml": yaml.safe_dump(
                effective_config,
                allow_unicode=True,
                sort_keys=False,
            ),
            "training/mechanisms.json": _json(
                {
                    "schema_version": "taili.task-mechanisms/v1",
                    "task_contract_ref": f"{contract.contract_id}@{contract.contract_version}",
                    "mechanisms": contract.training.get("mechanism_proposals", []),
                }
            ),
            "telemetry/monitoring.json": _json(
                {
                    "schema_version": "taili.task-monitoring/v1",
                    "task_contract_ref": f"{contract.contract_id}@{contract.contract_version}",
                    "monitoring": contract.telemetry.get("monitoring_proposals", []),
                }
            ),
            "simulation/simulation.json": _json(dict(contract.simulation)),
            "deployment/deployment.json": _json(dict(contract.deployment)),
        },
    }


__all__ = ["materialize_task_bundle"]
