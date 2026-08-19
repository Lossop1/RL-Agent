"""Compile a validated mechanism candidate into immutable runtime artifacts."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .mechanism_specs import MechanismBundle, MechanismPatch, apply_mechanism_patch, content_fingerprint


ARTIFACT_SCHEMA_VERSION = "rl-agent.mechanism-artifact/v1"


class MechanismArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ARTIFACT_SCHEMA_VERSION
    artifact_id: str
    bundle_id: str
    bundle_fingerprint: str
    patch_id: str = ""
    files: dict[str, str] = Field(default_factory=dict)
    protected_capabilities: tuple[str, ...] = ()
    required_evaluators: tuple[str, ...] = ()
    status: str = "compiled"


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _file_hash(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MechanismArtifactCompiler:
    """Materialize a bundle without executing generated source."""

    def compile(
        self,
        *,
        output_root: str | os.PathLike[str],
        baseline: MechanismBundle,
        patch: MechanismPatch | None = None,
        test_vectors: list[dict[str, Any]] | None = None,
    ) -> tuple[Path, MechanismArtifactManifest]:
        bundle = apply_mechanism_patch(baseline, patch) if patch else baseline
        fingerprint = bundle.fingerprint()
        artifact_id = f"mechanism-{fingerprint[:16]}"
        root = Path(output_root)
        target = root / artifact_id
        root.mkdir(parents=True, exist_ok=True)
        if target.is_dir():
            manifest = MechanismArtifactManifest.model_validate_json((target / "manifest.json").read_text(encoding="utf-8"))
            if manifest.bundle_fingerprint != fingerprint:
                raise ValueError(f"artifact id collision at {target}")
            return target, manifest

        temp = root / f".{artifact_id}.{uuid.uuid4().hex}.tmp"
        temp.mkdir(parents=False)
        try:
            bundle_path = temp / "mechanisms.json"
            _write(bundle_path, json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            if patch is not None:
                _write(temp / "patch.json", json.dumps(patch.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            _write(temp / "test_vectors.json", json.dumps(test_vectors or [], ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            _write(
                temp / "runtime_loader.py",
                "\"\"\"Generated loader for an immutable mechanism bundle.\"\"\"\n"
                "from pathlib import Path\n"
                "from autotuner.mechanisms.mechanism_runtime import load_mechanism_bundle\n\n"
                f"BUNDLE_FINGERPRINT = {fingerprint!r}\n\n"
                "def load():\n"
                "    bundle = load_mechanism_bundle(Path(__file__).with_name('mechanisms.json'))\n"
                "    if bundle.fingerprint() != BUNDLE_FINGERPRINT:\n"
                "        raise RuntimeError('mechanism bundle fingerprint mismatch')\n"
                "    return bundle\n",
            )
            _write(
                temp / "test_candidate_contract.py",
                "\"\"\"Generated artifact integrity test.\"\"\"\n"
                "from runtime_loader import BUNDLE_FINGERPRINT, load\n\n"
                "def test_generated_bundle_integrity():\n"
                "    assert load().fingerprint() == BUNDLE_FINGERPRINT\n",
            )
            tracked = ["mechanisms.json", "test_vectors.json", "runtime_loader.py", "test_candidate_contract.py"]
            if patch is not None:
                tracked.append("patch.json")
            manifest = MechanismArtifactManifest(
                artifact_id=artifact_id,
                bundle_id=bundle.id,
                bundle_fingerprint=fingerprint,
                patch_id=patch.id if patch else "",
                files={name: _file_hash(temp / name) for name in tracked},
                protected_capabilities=patch.protected_capabilities if patch else (),
                required_evaluators=patch.required_evaluators if patch else bundle.evaluator_refs,
            )
            _write(temp / "manifest.json", json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            os.replace(temp, target)
            return target, manifest
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise


def verify_mechanism_artifact(path: str | os.PathLike[str]) -> MechanismArtifactManifest:
    root = Path(path)
    manifest = MechanismArtifactManifest.model_validate_json((root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.files.items():
        candidate = root / name
        if not candidate.is_file() or _file_hash(candidate) != expected:
            raise ValueError(f"mechanism artifact integrity failure: {name}")
    bundle = MechanismBundle.model_validate_json((root / "mechanisms.json").read_text(encoding="utf-8"))
    if bundle.fingerprint() != manifest.bundle_fingerprint:
        raise ValueError("mechanism artifact bundle fingerprint mismatch")
    return manifest
