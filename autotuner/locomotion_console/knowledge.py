"""产品知识插件的只读控制台接口。"""
from __future__ import annotations

from typing import Any

from autotuner.product import load_product_plugin, resolve_product_contract


def _plugin(operation: str, product_id: str | None = None):
    return load_product_plugin(resolve_product_contract(product_id), "knowledge", operation)


def search(query: str = "", *, product_id: str | None = None) -> dict[str, Any]:
    return _plugin("search", product_id)(query)


def build_context_pack(
    query: str = "",
    include_docs: bool = True,
    *,
    product_id: str | None = None,
) -> dict[str, Any]:
    return _plugin("context_pack", product_id)(query=query, include_docs=include_docs)


def build_taili_context_pack(
    query: str = "",
    include_docs: bool = True,
    *,
    product_id: str | None = None,
) -> dict[str, Any]:
    """旧工具名兼容入口；实际实现由当前产品插件选择。"""
    return build_context_pack(query=query, include_docs=include_docs, product_id=product_id)


__all__ = ["build_context_pack", "build_taili_context_pack", "search"]
