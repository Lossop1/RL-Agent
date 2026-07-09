"""Console presentation layer for the Framework Library ② (autotuner.framework_library).

Surfaces the mechanism-component catalog (M1-M9 + blind-TP) and the named compositions to the
front-end, so the UI's ConfigSet view can show/select which mechanisms make up a framework and what
the Adapter does per component (invariant/derive/regenerate/scale). Read-only; no SSH.

Note: this is distinct from framework_profile.py (run-level identity: task id / run globs). This is
the framework's *mechanism content* (§4②).
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel

from autotuner.framework_library import CATALOG, COMPOSITIONS, adapt_plan


class FrameworkComponentInfo(BaseModel):
    id: str
    label: str
    role: str
    adapt_kind: str
    applies_to: List[str]
    module: str
    version: str
    requires: List[str]
    note: str


class FrameworkCompositionInfo(BaseModel):
    id: str
    label: str
    status: str
    component_ids: List[str]
    adapt_plan: Dict[str, List[str]]
    note: str


class FrameworkCatalogInfo(BaseModel):
    components: List[FrameworkComponentInfo]
    compositions: List[FrameworkCompositionInfo]


def build_framework_catalog() -> FrameworkCatalogInfo:
    components = [
        FrameworkComponentInfo(
            id=c.id, label=c.label, role=c.role, adapt_kind=c.adapt_kind,
            applies_to=list(c.applies_to), module=c.module, version=c.version,
            requires=list(c.requires), note=c.note,
        )
        for c in CATALOG.values()
    ]
    compositions = [
        FrameworkCompositionInfo(
            id=k.id, label=k.label, status=k.status,
            component_ids=list(k.component_ids), adapt_plan=adapt_plan(k), note=k.note,
        )
        for k in COMPOSITIONS.values()
    ]
    return FrameworkCatalogInfo(components=components, compositions=compositions)
