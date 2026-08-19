"""产品运行投影。

产品清单是输入合同，训练、诊断和控制台不应各自解释 YAML。这个模块把
合同投影成只读的运行视图，供执行层消费；它不实现任何具体训练逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import ResolvedProductContract, resolve_product_contract


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class ProductRuntimeView:
    """同一份产品合同在运行侧的稳定只读视图。"""

    contract: ResolvedProductContract

    @property
    def product_id(self) -> str:
        return self.contract.product_id

    @property
    def contract_digest(self) -> str:
        return self.contract.contract_digest

    @property
    def robot(self) -> Mapping[str, Any]:
        return _mapping(self.contract.robot)

    @property
    def training(self) -> Mapping[str, Any]:
        return _mapping(self.contract.training)

    @property
    def framework(self) -> Mapping[str, Any]:
        return _mapping(self.contract.framework)

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        return _mapping(self.contract.diagnostics)

    @property
    def deployment(self) -> Mapping[str, Any]:
        return _mapping(self.contract.deployment)

    @property
    def simulation(self) -> Mapping[str, Any]:
        return _mapping(self.contract.simulation)

    @property
    def runtime(self) -> Mapping[str, Any]:
        return _mapping(self.contract.runtime)

    def framework_profiles(self) -> Mapping[str, Mapping[str, Any]]:
        """返回产品声明的框架档案，不提供系统级默认档案。"""
        profiles = _mapping(self.framework.get("profiles"))
        return {
            str(identifier): dict(_mapping(value))
            for identifier, value in profiles.items()
            if isinstance(value, Mapping)
        }

    def default_framework_id(self) -> str:
        value = str(self.framework.get("default_profile") or "").strip()
        if value:
            return value
        profiles = self.framework_profiles()
        if len(profiles) == 1:
            return next(iter(profiles))
        return ""

    def diagnostic_payload(self) -> Mapping[str, Any]:
        """返回诊断 payload 约定；缺失时由调用方报告配置错误。"""
        return _mapping(self.diagnostics.get("payload"))

    def diagnostic_presets(self) -> list[Mapping[str, Any]]:
        values = self.diagnostics.get("presets", ())
        if not isinstance(values, (list, tuple)):
            return []
        return [dict(_mapping(value)) for value in values if isinstance(value, Mapping)]


def resolve_product_runtime(
    product_id: str | None = None,
    *,
    check_files: bool = False,
) -> ProductRuntimeView:
    """解析当前产品的运行投影。

    控制台初始化阶段允许源文件暂缺，以便先显示合同错误；真正执行前仍由
    ``resolve_product_contract(..., check_files=True)`` 负责硬性阻断。
    """
    return ProductRuntimeView(resolve_product_contract(product_id, check_files=check_files))


__all__ = ["ProductRuntimeView", "resolve_product_runtime"]
