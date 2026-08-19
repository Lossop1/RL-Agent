"""机器人产品清单与注册表。

这里是系统与具体机器人产品之间的边界。系统只依赖通用清单契约；
Taili、Taishan 等型号通过各自的配置清单接入。
"""

from .manifest import AssetSpec, ProductManifest, ProductManifestError, RobotProfile, load_product_manifest
from .contracts import ContractResolutionError, ResolvedAsset, ResolvedProductContract, resolve_product_contract
from autotuner.execution.runtime import RuntimeIdentity, resolve_runtime_identity
from .registry import ProductRegistry, get_product, get_robot_profile, validate_product

__all__ = [
    "AssetSpec",
    "ContractResolutionError",
    "ProductManifest",
    "ProductManifestError",
    "ProductRegistry",
    "RobotProfile",
    "ResolvedAsset",
    "ResolvedProductContract",
    "RuntimeIdentity",
    "get_product",
    "get_robot_profile",
    "load_product_manifest",
    "resolve_product_contract",
    "resolve_runtime_identity",
    "validate_product",
]
