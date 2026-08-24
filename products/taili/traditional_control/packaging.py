"""构建按内容寻址、可逐字节复现的 Taili 传统控制 bundle。"""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import tarfile

from autotuner.control.manifest import (
    build_manifest,
    build_run_manifest,
    canonical_json_bytes,
    sha256_file,
    write_manifest,
)


RUNTIME_DEPENDENCIES = ("mujoco", "numpy", "osqp", "PyYAML", "scipy")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_files(root: Path | None = None) -> tuple[Path, ...]:
    base = root or project_root()
    files = [
        *sorted((base / "autotuner" / "control").glob("*.py")),
        *sorted((base / "products" / "taili" / "traditional_control").rglob("*.py")),
        base / "config" / "traditional_control" / "taili_nominal.yaml",
        base / "locomotion-console-ui" / "public" / "robot" / "taili_dog_description" / "urdf" / "robot.urdf",
        base / "requirements-traditional-control.txt",
    ]
    return tuple(path for path in files if path.is_file())


def _tar_info(path: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(path)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    return info


def _archive_bytes(
    root: Path,
    files: tuple[Path, ...],
    artifact_manifest: dict[str, object],
) -> bytes:
    raw_tar = io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w", format=tarfile.PAX_FORMAT) as tar:
        manifest_bytes = canonical_json_bytes(artifact_manifest) + b"\n"
        tar.addfile(_tar_info("MANIFEST.json", len(manifest_bytes)), io.BytesIO(manifest_bytes))
        for path in files:
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            tar.addfile(_tar_info(relative, len(data)), io.BytesIO(data))
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=compressed, mode="wb", mtime=0) as stream:
        stream.write(raw_tar.getvalue())
    return compressed.getvalue()


def _inside_workspace(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("traditional-control output must stay inside the workspace") from exc
    return resolved


def build_bundle(output_root: str | Path | None = None) -> dict[str, object]:
    root = project_root().resolve()
    files = source_files(root)
    artifact_manifest = build_manifest(
        root,
        files,
        artifact_type="taili_traditional_control_bundle",
        metadata={
            "controller": "centroidal_mpc_floating_base_wbc",
            "scopes": ["nominal_flat", "nominal_stairs"],
        },
    )
    digest = str(artifact_manifest["artifact_digest"])
    destination = _inside_workspace(
        root,
        Path(output_root or root / "output" / "traditional_control" / "archive"),
    )
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"taili_traditional_control_{digest[:16]}"
    archive = destination / f"{stem}.tar.gz"
    artifact_manifest_path = destination / f"{stem}.manifest.json"

    expected_archive = _archive_bytes(root, files, artifact_manifest)
    if archive.exists() and archive.read_bytes() != expected_archive:
        raise RuntimeError(f"content-addressed archive does not match its inputs: {archive}")
    if not archive.exists():
        temporary = archive.with_suffix(archive.suffix + ".tmp")
        temporary.write_bytes(expected_archive)
        temporary.replace(archive)
    write_manifest(artifact_manifest_path, artifact_manifest)

    archive_sha256 = sha256_file(archive)
    now = datetime.now(timezone.utc)
    run_manifest = build_run_manifest(
        root,
        artifact_manifest,
        archive_path=archive,
        archive_sha256=archive_sha256,
        operation="build_bundle",
        dependencies=RUNTIME_DEPENDENCIES,
        generated_at=now,
    )
    run_name = now.strftime("%Y%m%dT%H%M%S.%fZ")
    run_manifest_path = destination / "runs" / f"{run_name}_{str(run_manifest['run_digest'])[:12]}.json"
    write_manifest(run_manifest_path, run_manifest)
    return {
        **artifact_manifest,
        "archive": archive,
        "archive_sha256": archive_sha256,
        "manifest": artifact_manifest_path,
        "run_manifest": run_manifest_path,
    }
