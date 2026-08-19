"""产品清单注册与选择。

注册表不提供 Taili 默认分支。只有在目录中恰好存在一个产品时才允许隐式选择；
当 Taishan 等第二个产品加入后，调用者必须显式指定 product_id。
"""
from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable

from .manifest import ProductManifest, ProductManifestError, RobotProfile, load_product_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRODUCT_ROOT = PROJECT_ROOT / "config" / "products"


class ProductRegistry:
    def __init__(self, root: str | Path = DEFAULT_PRODUCT_ROOT):
        self.root = Path(root)

    def paths(self) -> Iterable[Path]:
        if not self.root.is_dir():
            return ()
        return sorted((*self.root.glob("*.yaml"), *self.root.glob("*.yml"), *self.root.glob("*.json")))

    def list(self, *, valid_only: bool = False) -> list[ProductManifest]:
        products: list[ProductManifest] = []
        for path in self.paths():
            try:
                product = load_product_manifest(path)
            except (OSError, ValueError, ProductManifestError):
                if not valid_only:
                    continue
                raise
            issues = product.validate(PROJECT_ROOT)
            if valid_only and issues:
                raise ProductManifestError(f"{path}: " + "; ".join(issues))
            products.append(product)
        return products

    def get(self, product_id: str | None = None) -> ProductManifest:
        requested = (product_id or "").strip()
        products = self.list()
        if requested:
            for product in products:
                if product.product_id == requested or product.robot.id == requested:
                    return product
            known = ", ".join(sorted(p.product_id for p in products)) or "<none>"
            raise ProductManifestError(f"unknown product {requested!r}; known products: {known}")
        if len(products) == 1:
            return products[0]
        if not products:
            raise ProductManifestError(f"no product manifests found under {self.root}")
        known = ", ".join(sorted(p.product_id for p in products))
        raise ProductManifestError(f"product_id is required when multiple products are registered: {known}")


def product_registry(root: str | Path | None = None) -> ProductRegistry:
    configured = root or os.environ.get("LOCOMOTION_PRODUCT_ROOT")
    return ProductRegistry(configured or DEFAULT_PRODUCT_ROOT)


def get_product(product_id: str | None = None) -> ProductManifest:
    return product_registry().get(product_id or os.environ.get("LOCOMOTION_PRODUCT_ID"))


def get_robot_profile(robot_id: str | None = None) -> RobotProfile:
    return get_product(robot_id).robot_profile()


def validate_product(product_id: str | None = None) -> list[str]:
    product = get_product(product_id)
    return product.validate(PROJECT_ROOT)
