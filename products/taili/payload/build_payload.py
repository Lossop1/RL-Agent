"""构建 Taili 盲态运行 payload 的 tar.gz 包。"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import argparse
import shutil
import tarfile
import time
from typing import Any, Mapping

from .payload_manifest import (
    GENERATED_FILES,
    PAYLOAD_DIR,
    ROOT,
    RUNTIME_PACKAGE,
    iter_payload_files,
    validate_manifest,
)
from autotuner.execution.payload import build_payload_manifest, verify_payload_archive


@dataclass(frozen=True)
class BuildResult:
    archive: Path
    build_dir: Path
    root_name: str
    file_count: int
    payload_digest: str
    manifest: Path


def build_payload(
    *,
    contract: Any,
    output_dir: Path | None = None,
    stamp: str | None = None,
    keep_build_dir: bool = False,
) -> BuildResult:
    contract_data = contract.to_dict() if hasattr(contract, "to_dict") else contract
    if not isinstance(contract_data, Mapping):
        raise TypeError("contract must be a resolved product contract")
    deployment = contract_data.get("deployment")
    deployment = deployment if isinstance(deployment, Mapping) else {}
    if deployment.get("payload_package") != RUNTIME_PACKAGE:
        raise ValueError("resolved contract does not target this payload package")

    report = validate_manifest(ROOT)
    if not report.ok:
        raise RuntimeError("payload manifest invalid:\n" + "\n".join(report.errors))

    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or (PAYLOAD_DIR / "dist")
    build_root = PAYLOAD_DIR / ".build" / f"{RUNTIME_PACKAGE}_{stamp}"
    archive = output_dir / f"{RUNTIME_PACKAGE}_{stamp}.tar.gz"

    if build_root.exists():
        shutil.rmtree(build_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_root.mkdir(parents=True)

    count = 0
    for dst, text in GENERATED_FILES.items():
        target = build_root / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        count += 1

    contract_target = build_root / RUNTIME_PACKAGE / "product_contract.json"
    contract_target.parent.mkdir(parents=True, exist_ok=True)
    contract.write(contract_target)
    count += 1
    runtime_target = build_root / RUNTIME_PACKAGE / "runtime_identity.json"
    runtime_target.write_text(json.dumps(contract.runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    count += 1

    for src, dst in iter_payload_files(ROOT):
        target = build_root / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        count += 1

    payload_manifest = build_payload_manifest(build_root, contract=contract)
    embedded_manifest = build_root / "payload_manifest.json"
    payload_manifest.write(embedded_manifest)
    count += 1
    archive = output_dir / f"{RUNTIME_PACKAGE}_{stamp}_{payload_manifest.payload_digest[:12]}.tar.gz"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(build_root.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(build_root))

    archive_errors = verify_payload_archive(archive, payload_manifest)
    if archive_errors:
        archive.unlink(missing_ok=True)
        shutil.rmtree(build_root, ignore_errors=True)
        raise RuntimeError("payload archive failed post-build verification:\n" + "\n".join(archive_errors))

    manifest_path = output_dir / f"{RUNTIME_PACKAGE}_{stamp}_{payload_manifest.payload_digest[:12]}.manifest.json"
    payload_manifest.write(manifest_path)

    result = BuildResult(
        archive=archive,
        build_dir=build_root,
        root_name=archive.name[:-7],
        file_count=count,
        payload_digest=payload_manifest.payload_digest,
        manifest=manifest_path,
    )
    if not keep_build_dir:
        shutil.rmtree(build_root, ignore_errors=True)
    return result


def main() -> None:
    from autotuner.product import resolve_product_contract

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--keep-build-dir", action="store_true")
    parser.add_argument("--product-id", default=None)
    args = parser.parse_args()
    # 命令行入口通过产品注册表选择合同；构建器本身不再猜测产品身份。
    contract = resolve_product_contract(args.product_id, root=ROOT)
    result = build_payload(
        contract=contract,
        output_dir=args.out,
        stamp=args.stamp,
        keep_build_dir=args.keep_build_dir,
    )
    print(result.archive)
    print(f"files={result.file_count}")


if __name__ == "__main__":
    main()
