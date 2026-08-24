"""Configuration-discovered registry for traditional-control demo providers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from autotuner.product import ProductRegistry, load_product_plugin

from ..schemas import TraditionalControlDataSourceInfo
from .contracts import ProviderCatalog, TraditionalControlDemoProvider


_PLUGIN_ROLE = "traditional_control_demo"
_PLUGIN_OPERATION = "provider"


class ProviderRegistryError(RuntimeError):
    """The provider catalog is ambiguous or references an unknown target."""


class TraditionalControlProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, TraditionalControlDemoProvider] = {}
        self._catalogs: dict[str, ProviderCatalog] = {}
        self.load_errors: list[str] = []

    def register(self, provider: TraditionalControlDemoProvider) -> None:
        provider_id = provider.provider_id.strip()
        if not provider_id:
            raise ProviderRegistryError("traditional-control provider id is empty")
        if provider_id in self._providers:
            raise ProviderRegistryError(f"duplicate traditional-control provider: {provider_id}")
        catalog = provider.catalog()
        catalog.product.data_sources = list(catalog.data_sources)
        for controller in catalog.product.controllers:
            if controller.provider_id and controller.provider_id != provider_id:
                raise ProviderRegistryError(
                    f"controller {controller.id!r} declares provider {controller.provider_id!r}, "
                    f"but was registered by {provider_id!r}"
                )
            controller.provider_id = provider_id
            controller.provider_ids = [provider_id]
        controller_ids = [item.id for item in catalog.product.controllers]
        for scene in catalog.product.scenes:
            scene.provider_id = provider_id
            scene.provider_ids = [provider_id]
            if not scene.controller_ids:
                scene.controller_ids = list(controller_ids)
        for source in catalog.data_sources:
            source.provider_id = provider_id
        self._providers[provider_id] = provider
        self._catalogs[provider_id] = catalog

    def providers(self) -> tuple[TraditionalControlDemoProvider, ...]:
        return tuple(self._providers[key] for key in sorted(self._providers))

    def catalogs(self) -> tuple[ProviderCatalog, ...]:
        return tuple(self._catalogs[key] for key in sorted(self._catalogs))

    def provider(self, provider_id: str) -> TraditionalControlDemoProvider:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise ProviderRegistryError(f"unknown traditional-control provider: {provider_id}") from exc

    def resolve(
        self,
        *,
        product_id: str,
        controller_id: str,
        scene_id: str,
        data_source_id: str,
    ) -> tuple[TraditionalControlDemoProvider, ProviderCatalog, TraditionalControlDataSourceInfo]:
        matches: list[tuple[TraditionalControlDemoProvider, ProviderCatalog, TraditionalControlDataSourceInfo]] = []
        for provider_id, catalog in self._catalogs.items():
            if catalog.product.id != product_id:
                continue
            if not any(item.id == controller_id for item in catalog.product.controllers):
                continue
            scene = next((item for item in catalog.product.scenes if item.id == scene_id), None)
            if scene is None or (scene.controller_ids and controller_id not in scene.controller_ids):
                continue
            source = next((item for item in catalog.data_sources if item.id == data_source_id), None)
            if source is not None:
                matches.append((self._providers[provider_id], catalog, source))
        if not matches:
            raise ProviderRegistryError(
                "no provider matches the selected product, controller, scene, and data source"
            )
        if len(matches) > 1:
            providers = ", ".join(item[0].provider_id for item in matches)
            raise ProviderRegistryError(f"demo selection is ambiguous across providers: {providers}")
        return matches[0]


def discover_traditional_control_providers(
    *,
    product_registry: ProductRegistry | None = None,
    workspace_root: str | Path | None = None,
) -> TraditionalControlProviderRegistry:
    """Load only product-declared providers; the console has no product imports."""

    products = product_registry or ProductRegistry()
    root = Path(workspace_root or products.workspace_root).resolve()
    registry = TraditionalControlProviderRegistry()
    for product in products.list():
        role = product.plugins.get(_PLUGIN_ROLE, {})
        if _PLUGIN_OPERATION not in role:
            continue
        try:
            factory = load_product_plugin(product, _PLUGIN_ROLE, _PLUGIN_OPERATION)
            provider = factory(product=product, workspace_root=root)
            registry.register(provider)
        except Exception as exc:  # noqa: BLE001 - one bad product must remain visible in catalog
            registry.load_errors.append(
                f"{product.product_id}: {type(exc).__name__}: {exc}"
            )
    return registry


__all__ = [
    "ProviderRegistryError",
    "TraditionalControlProviderRegistry",
    "discover_traditional_control_providers",
]
