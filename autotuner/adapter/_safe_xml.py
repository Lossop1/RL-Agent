"""Safe URDF parsing helpers.

Two independent hardening concerns for the robot-import path, which parses a *new, untrusted*
URDF supplied via the console (`GET /config/robot/import?urdf=...`):

1. XXE / entity-expansion: stdlib ``xml.etree.ElementTree`` (expat) still expands internal DTD
   entities (billion-laughs / quadratic blow-up). A URDF never legitimately carries a DTD, so we
   reject any ``<!DOCTYPE`` / ``<!ENTITY`` declaration outright and cap the input size — a
   dependency-free block that does not rely on defusedxml being installed.
2. Path traversal: the console must not turn an arbitrary local path into an XML-parse /
   file-existence oracle (``?urdf=/etc/passwd``). ``resolve_within_roots`` confines the path to the
   known robot-asset directories.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

_MAX_URDF_BYTES = 8_000_000


def _reject_dtd(data: bytes) -> None:
    low = data.lower()
    if b"<!doctype" in low or b"<!entity" in low:
        raise ValueError("URDF contains a DTD/ENTITY declaration; refused (XXE / entity-expansion protection)")


def safe_parse_urdf(urdf_path: str):
    """Parse a URDF file and return its root element, with size + DTD/entity guards.

    Drop-in for ``ET.parse(path).getroot()`` but safe against entity-expansion DoS."""
    data = Path(urdf_path).read_bytes()
    if len(data) > _MAX_URDF_BYTES:
        raise ValueError(f"URDF too large ({len(data)} bytes > {_MAX_URDF_BYTES})")
    _reject_dtd(data)
    return ET.fromstring(data)


def _is_within(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def allowed_urdf_roots() -> set[Path]:
    """The directories a console-supplied URDF path is allowed to resolve inside."""
    roots: set[Path] = set()
    try:
        from autotuner.adapter.__main__ import PRESETS
        for preset in PRESETS.values():
            u = preset.get("urdf")
            if u:
                roots.add(Path(u).resolve().parent)
    except Exception:
        pass
    repo = Path(__file__).resolve().parents[2]
    for extra in ("autotuner/blind_locomotion/assets", "frontend/dist/robot",
                  "autotuner/training_payloads"):
        d = repo / extra
        if d.exists():
            roots.add(d.resolve())
    return roots


def resolve_within_roots(urdf_path: str, roots: Iterable[Path] | None = None) -> str:
    """Resolve `urdf_path` and confirm it lies within an allowed asset root, else raise.

    Blocks `?urdf=/etc/passwd` and `..`-traversal from the console import endpoint."""
    allowed = list(roots) if roots is not None else list(allowed_urdf_roots())
    p = Path(urdf_path).resolve()
    if not any(_is_within(p, r) for r in allowed):
        raise ValueError("URDF path is outside the allowed robot-asset roots")
    return str(p)
