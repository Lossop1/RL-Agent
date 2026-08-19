"""产品插件的统一加载边界。

产品清单声明能力角色，例如验收解析或策略编辑；系统只按角色取入口，
不在业务模块中导入某个具体机器人。插件本身可以依赖产品资产，但不能
反向依赖控制台请求对象。
"""
from __future__ import annotations

from typing import Any, Mapping

from .adapter import ProductAdapterError, load_plugin_entrypoint


class ProductPluginError(ProductAdapterError):
    """产品插件缺失、格式错误或无法加载。"""


def _data(contract: Any) -> Mapping[str, Any]:
    value = contract.to_dict() if hasattr(contract, "to_dict") else contract
    if not isinstance(value, Mapping):
        raise ProductPluginError("product contract must be a mapping")
    return value


def plugin_reference(contract: Any, role: str, operation: str) -> str:
    """返回产品声明的插件入口；缺失时明确失败，不回退到其他产品。"""
    plugins = _data(contract).get("plugins", {})
    role_map = plugins.get(role) if isinstance(plugins, Mapping) else None
    reference = role_map.get(operation) if isinstance(role_map, Mapping) else None
    if not reference:
        raise ProductPluginError(f"product plugin is not declared: {role}.{operation}")
    return str(reference)


def load_product_plugin(contract: Any, role: str, operation: str) -> Any:
    """加载一个产品插件入口，但不隐式执行。"""
    return load_plugin_entrypoint(plugin_reference(contract, role, operation), role=f"{role}.{operation}")


def call_product_plugin(contract: Any, role: str, operation: str, *args: Any, **kwargs: Any) -> Any:
    """调用产品插件；参数协议由该角色定义，系统只负责边界转发。"""
    plugin = load_product_plugin(contract, role, operation)
    return plugin(*args, **kwargs)


__all__ = ["ProductPluginError", "call_product_plugin", "load_product_plugin", "plugin_reference"]
