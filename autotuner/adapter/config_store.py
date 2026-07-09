"""ConfigSet persistence ① — assemble / store / version the operational unit (SYSTEM_ARCHITECTURE §4①).

A ConfigSet (autotuner.adapter.pipeline.ConfigSet) is the system's operational unit: robot URDF +
local sources + remote destinations + framework composition + launch. This store saves/loads/lists
them by id as JSON, and keeps a timestamped version snapshot on every save (§4① 版本化), so a
workspace can hold many robots/tasks and roll a ConfigSet back to an earlier revision.

Pure disk + dataclass (de)serialization — no SSH, no GPU. JSON turns tuples into lists, so load()
restores tuple fields (robot_capabilities) for round-trip fidelity.
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, List, Optional

from autotuner.adapter.pipeline import ConfigSet

_TUPLE_FIELDS = {f.name for f in fields(ConfigSet) if isinstance(f.default, tuple)}


def _to_configset(d: dict) -> ConfigSet:
    known = {f.name for f in fields(ConfigSet)}
    data = {k: v for k, v in d.items() if k in known}
    for tf in _TUPLE_FIELDS:                       # JSON list → tuple (round-trip fidelity)
        if tf in data and isinstance(data[tf], list):
            data[tf] = tuple(data[tf])
    return ConfigSet(**data)


class ConfigSetStore:
    """Filesystem store: <root>/<id>.json (latest) + <root>/<id>.history/<stamp>.json (versions)."""

    def __init__(self, root: str = "config/config_sets"):
        self.root = Path(root)

    def _path(self, cid: str) -> Path:
        if not cid or "/" in cid or "\\" in cid or cid.startswith("."):
            raise ValueError(f"unsafe config-set id: {cid!r}")
        return self.root / f"{cid}.json"

    def _history_dir(self, cid: str) -> Path:
        return self.root / f"{cid}.history"

    def save(self, cid: str, cs: ConfigSet, stamp: str, note: str = "") -> str:
        """Write the latest record + a version snapshot. `stamp` supplied by caller (no wall-clock)."""
        self.root.mkdir(parents=True, exist_ok=True)
        record = {"id": cid, "updated": stamp, "note": note, "config_set": asdict(cs)}
        path = self._path(cid)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        hist = self._history_dir(cid)
        hist.mkdir(parents=True, exist_ok=True)
        (hist / f"{stamp}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def load(self, cid: str) -> ConfigSet:
        path = self._path(cid)
        if not path.exists():
            raise FileNotFoundError(f"config-set {cid!r} not found under {self.root}")
        record = json.loads(path.read_text(encoding="utf-8"))
        return _to_configset(record["config_set"])

    def load_record(self, cid: str) -> dict:
        path = self._path(cid)
        if not path.exists():
            raise FileNotFoundError(f"config-set {cid!r} not found under {self.root}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> List[Dict[str, str]]:
        if not self.root.exists():
            return []
        out = []
        for p in sorted(self.root.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                cs = rec.get("config_set", {})
                out.append({"id": rec.get("id", p.stem), "updated": rec.get("updated", ""),
                            "robot_urdf": cs.get("urdf", ""),
                            "framework": cs.get("framework_composition", ""),
                            "note": rec.get("note", "")})
            except Exception:
                continue
        return out

    def versions(self, cid: str) -> List[str]:
        hist = self._history_dir(cid)
        if not hist.exists():
            return []
        return sorted(p.stem for p in hist.glob("*.json"))

    def load_version(self, cid: str, stamp: str) -> ConfigSet:
        p = self._history_dir(cid) / f"{stamp}.json"
        if not p.exists():
            raise FileNotFoundError(f"version {stamp!r} of {cid!r} not found")
        return _to_configset(json.loads(p.read_text(encoding="utf-8"))["config_set"])

    def rollback(self, cid: str, stamp: str, new_stamp: str) -> str:
        """Restore an earlier version as the new latest (itself snapshotted under new_stamp)."""
        cs = self.load_version(cid, stamp)
        return self.save(cid, cs, new_stamp, note=f"rollback to {stamp}")

    def delete(self, cid: str, *, keep_history: bool = True) -> bool:
        path = self._path(cid)
        existed = path.exists()
        if existed:
            path.unlink()
        if not keep_history:
            hist = self._history_dir(cid)
            if hist.exists():
                for p in hist.glob("*.json"):
                    p.unlink()
                hist.rmdir()
        return existed
