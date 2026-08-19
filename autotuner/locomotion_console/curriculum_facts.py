"""产品课程事实卡的兼容转发。"""
from __future__ import annotations

from typing import Any

from autotuner.product import load_product_plugin, resolve_product_contract


def build_facts_card(*args: Any, product_id: str | None = None, **kwargs: Any):
    contract = resolve_product_contract(product_id, check_files=False)
    plugin = load_product_plugin(contract, "acceptance", "curriculum_facts")
    return plugin(*args, **kwargs)


__all__ = ["build_facts_card"]
