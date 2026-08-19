"""机器人产品清单与注册表。

这里是系统与具体机器人产品之间的边界。系统只依赖通用清单契约；
Taili、Taishan 等型号通过各自的配置清单接入。
"""

from .manifest import AssetSpec, ProductManifest, ProductManifestError, RobotProfile, load_product_manifest
from .registry import ProductRegistry, get_product, get_robot_profile, validate_product

__all__ = [
    "AssetSpec",
    "ProductManifest",
    "ProductManifestError",
    "ProductRegistry",
    "RobotProfile",
    "get_product",
    "get_robot_profile",
    "load_product_manifest",
    "validate_product",
]
