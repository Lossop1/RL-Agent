"""产品参考生成插件的兼容入口。"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from autotuner.product import call_product_plugin, resolve_product_adapter, resolve_product_contract


def regenerate(
    urdf_path: str,
    out_dir: str,
    which: str = "all",
    *,
    contract: Any | None = None,
    product_id: str | None = None,
) -> tuple[list[str], Mapping[str, Any]]:
    """按产品合同调用参考生成器，不在系统包中导入产品实现。"""
    resolved = contract or resolve_product_contract(product_id)
    adapter = resolve_product_adapter(resolved)
    settings = adapter.adaptation.get("reference_geometry")
    if not isinstance(settings, Mapping):
        raise ValueError("adaptation.reference_geometry must be a mapping")
    result = call_product_plugin(
        resolved,
        "adaptation",
        "regenerate_reference",
        urdf_path,
        out_dir,
        settings,
        which=which,
    )
    if not isinstance(result, Mapping) or not isinstance(result.get("clips"), (list, tuple)):
        raise ValueError("reference plugin must return clips and applied mappings")
    applied = result.get("applied")
    if not isinstance(applied, Mapping):
        applied = {}
    return [str(path) for path in result["clips"]], applied


__all__ = ["regenerate"]
