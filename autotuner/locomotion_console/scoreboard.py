"""产品记分牌的通用转发层。

指标族、底线和拆台关系由产品插件声明；模块级常量只为旧的只读导入保留，
通过 ``__getattr__`` 在访问时解析当前默认合同，不在系统源码中导入具体产品。
"""
from __future__ import annotations

from typing import Any

from autotuner.product import load_product_plugin, resolve_product_contract


# 仅声明兼容接口的类型，不在导入时绑定具体产品；实际值仍由 __getattr__ 动态解析。
FAMILY_ORDER: list[str]
TENSIONS: list[tuple[str, str, str]]


def _metadata(product_id: str | None = None) -> dict[str, Any]:
    contract = resolve_product_contract(product_id, check_files=False)
    plugin = load_product_plugin(contract, "acceptance", "metadata")
    value = plugin()
    return value if isinstance(value, dict) else {}


def build_scoreboard(
    telemetry,
    effective_config_text: str = "",
    *,
    product_id: str | None = None,
    **kwargs: Any,
):
    contract = resolve_product_contract(product_id, check_files=False)
    plugin = load_product_plugin(contract, "acceptance", "scoreboard")
    return plugin(telemetry, effective_config_text, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "FAMILY_ORDER":
        return _metadata().get("family_order", [])
    if name == "TENSIONS":
        return _metadata().get("tensions", [])
    raise AttributeError(name)


__all__ = ["FAMILY_ORDER", "TENSIONS", "build_scoreboard"]
