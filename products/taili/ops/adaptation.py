"""Taili 产品的 URDF 参考几何与 AMP 参考生成插件。"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from autotuner.adapter.ref_geom import extract
from products.taili.blind_locomotion import gen_taili_gaits as generator


def reference_geometry(urdf_path: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    """按 Taili 关节链提取参考生成器所需的几何。"""
    leg = str(settings.get("leg") or "").strip()
    if not leg:
        raise ValueError("adaptation.reference_geometry.leg is required")
    geometry = extract(urdf_path, leg=leg, style=settings)
    return {
        "generator": geometry.as_globals(),
        "total_leg": geometry.total_leg,
        "base_height": geometry.base_z,
        "swing_clearance": geometry.clearance,
    }


def regenerate_reference(
    urdf_path: str,
    out_dir: str,
    settings: Mapping[str, Any],
    *,
    which: str = "all",
) -> dict[str, Any]:
    """使用 Taili 参考库重生 clip；输出格式由产品插件负责。"""
    geometry = reference_geometry(urdf_path, settings)
    applied = generator.apply_geometry(geometry["generator"])
    if which == "multispeed":
        library = generator.flat_multispeed_library()
    elif which == "all":
        library = {
            **generator.gait_library(),
            **generator.flat_multispeed_library(),
        }
    elif which == "flat6":
        library = generator.gait_library()
    else:
        raise ValueError(f"unsupported Taili reference group: {which!r}")

    produced = [generator.generate(config, out_dir) for config in library.values()]
    return {"clips": produced, "applied": applied}


__all__ = ["reference_geometry", "regenerate_reference"]
