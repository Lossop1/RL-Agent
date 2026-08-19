"""产品代码知识插件的控制台边界。

控制台只提供稳定的只读 API。具体源码白名单和任务语义由当前产品插件
提供，因此新增机器人不会要求修改控制台源码。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from autotuner.product import load_product_plugin, resolve_product_contract


_ROOT = Path(__file__).resolve().parents[2]


def _plugin(operation: str, product_id: str | None = None) -> Any:
    return load_product_plugin(resolve_product_contract(product_id), "code_knowledge", operation)


def get_allowlist_files(product_id: str | None = None) -> tuple[str, ...]:
    return tuple(str(item) for item in _plugin("allowlist_files", product_id))


# 兼容只读消费者；新代码应调用 get_allowlist_files，以支持运行时切换产品。
_ALLOWLIST_FILES = get_allowlist_files()


def _read_allowlisted(relative: str, product_id: str | None = None) -> str | None:
    return _plugin("read_allowlisted", product_id)(relative)


def build_signal_map(query: str = "", limit: int = 12, *, product_id: str | None = None) -> dict[str, Any]:
    return _plugin("build_signal_map", product_id)(query=query, limit=limit)


def search_code_knowledge(
    query: str = "",
    max_snippets: int = 16,
    *,
    product_id: str | None = None,
) -> dict[str, Any]:
    return _plugin("search", product_id)(query=query, max_snippets=max_snippets)


def source_inventory(*, product_id: str | None = None) -> dict[str, Any]:
    return _plugin("source_inventory", product_id)()


__all__ = [
    "_ALLOWLIST_FILES",
    "_ROOT",
    "_read_allowlisted",
    "build_signal_map",
    "get_allowlist_files",
    "search_code_knowledge",
    "source_inventory",
]
