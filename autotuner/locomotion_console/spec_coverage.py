"""产品 spec coverage 的兼容转发。

验收语义属于产品，不属于控制台。旧 import 保留是为了兼容已有客户端；
实际实现由当前产品合同的 acceptance.spec_coverage 插件提供。
"""
from __future__ import annotations

from typing import Any

from autotuner.product import load_product_plugin, resolve_product_contract


def build_spec_coverage_report(*, product_id: str | None = None, **kwargs: Any):
    contract = resolve_product_contract(product_id, check_files=False)
    plugin = load_product_plugin(contract, "acceptance", "spec_coverage")
    return plugin(**kwargs)


__all__ = ["build_spec_coverage_report"]
