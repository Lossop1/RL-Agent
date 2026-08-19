"""检查源码/产物边界、包依赖和 payload 源映射。

这是仓库维护门，不是训练验收器。它只做静态检查和 payload 清单校验，
不会启动 IsaacLab、加载 checkpoint、连接 SSH 或执行前端开发服务器。
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable


TEMP_DIR_NAMES = {
    ".pytest_cache",
    ".pytest-tmp",
    ".pytest_tmp",
    ".pt",
    "__pycache__",
    "node_modules",
}
MOJIBAKE_MARKERS = ("�", "鈹", "閿", "Ã", "Â")


@dataclass
class CheckReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _layout(root: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - Python 3.11 is required by pyproject
        raise RuntimeError("Python 3.11+ is required for repository_structure.toml") from exc
    path = root / "config" / "repository_structure.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _under(path: Path, root: Path, relative_roots: Iterable[str]) -> bool:
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    return any(rel == item or rel.startswith(item.rstrip("/") + "/") for item in relative_roots)


def _python_files(root: Path) -> Iterable[Path]:
    for base in (root / "autotuner", root / "tools" / "isaaclab_quad_diag_observation" / "isaaclab_quad_diag"):
        if not base.is_dir():
            continue
        yield from (p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def _module_name(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root.resolve()).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _absolute_imports(tree: ast.AST) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def _is_compatibility(module: str, layout: dict) -> bool:
    return module in set(layout.get("compatibility", {}).get("modules", []))


def check_encoding(root: Path, report: CheckReport) -> None:
    """拒绝 BOM 和常见乱码，避免注释成为错误的第二套事实源。"""
    candidates = list(_python_files(root))
    candidates.extend((root / "docs").rglob("*.md") if (root / "docs").is_dir() else [])
    candidates.extend((root / "config").glob("*.toml"))
    candidates.extend((root / "config").glob("*.yaml"))
    candidates.extend((root / "config").glob("*.yml"))
    for path in candidates:
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            report.error(f"非 UTF-8 文件: {path}: {exc}")
            continue
        if raw.startswith(b"\xef\xbb\xbf"):
            report.error(f"文件含 UTF-8 BOM: {path}")
        for marker in MOJIBAKE_MARKERS:
            if marker in text:
                report.error(f"疑似乱码标记 {marker!r}: {path}")


def check_source_boundaries(root: Path, layout: dict, report: CheckReport) -> None:
    source_roots = layout.get("source", {}).get("roots", [])
    for path in _python_files(root):
        rel = path.resolve().relative_to(root.resolve()).as_posix()
        if any(part in TEMP_DIR_NAMES for part in path.relative_to(root).parts):
            report.warning(f"生成缓存出现在源码扫描范围，提交前应清理: {rel}")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            report.error(f"无法解析 Python 源码 {rel}: {exc}")
            continue
        owner = next((prefix for prefix in layout.get("dependency_rules", {}) if _module_name(path, root).startswith(prefix)), None)
        if not owner or _is_compatibility(_module_name(path, root), layout):
            continue
        forbidden = layout["dependency_rules"][owner]
        for imported in _absolute_imports(tree):
            for target in forbidden:
                if imported == target or imported.startswith(target + "."):
                    report.error(f"非法依赖: {_module_name(path, root)} -> {imported} (禁止依赖 {target})")

    for root_name in ("autotuner", "tools"):
        base = root / root_name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_dir():
                continue
            if path.name in TEMP_DIR_NAMES:
                report.warning(f"源码树包含本地生成目录，提交前应清理: {path.relative_to(root).as_posix()}")


def check_compatibility_shims(root: Path, layout: dict, report: CheckReport) -> None:
    for module in layout.get("compatibility", {}).get("modules", []):
        path = root / Path(*module.split("."))
        path = path.with_suffix(".py")
        if not path.is_file():
            report.error(f"兼容入口缺失: {path.relative_to(root).as_posix()}")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            report.error(f"兼容入口无法解析: {path}: {exc}")
            continue
        definitions = [node for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        unexpected = [node.name for node in definitions if node.name != "__getattr__"]
        if unexpected:
            report.error(f"兼容入口包含业务实现 {path.relative_to(root)}: {unexpected}")


def check_required_paths(root: Path, layout: dict, report: CheckReport) -> None:
    required = [
        "docs/README.md",
        "docs/repository_architecture.md",
        "docs/maintenance_standard.md",
        "autotuner/mechanisms/__init__.py",
        "autotuner/research/__init__.py",
        "autotuner/infrastructure/__init__.py",
        "autotuner/taili_ops/__init__.py",
    ]
    required.extend(layout.get("entrypoints", {}).values())
    # entrypoint value 是命令而非路径；只检查文档中列出的模块/文件。
    required = [item for item in required if "/" in item and not item.startswith("python ")]
    for item in required:
        if not (root / item).exists():
            report.error(f"结构契约要求的路径缺失: {item}")


def check_payload_manifest(root: Path, report: CheckReport) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from autotuner.training_payloads.taili_blind_runtime.payload_manifest import validate_manifest
        result = validate_manifest(root)
    except Exception as exc:  # pragma: no cover - dependency/environment failure is reported clearly
        report.error(f"payload 清单无法加载: {type(exc).__name__}: {exc}")
        return
    for item in result.errors:
        report.error(f"payload: {item}")
    for item in result.warnings:
        report.warning(f"payload: {item}")


def run(root: Path) -> CheckReport:
    layout = _layout(root)
    report = CheckReport()
    check_required_paths(root, layout, report)
    check_encoding(root, report)
    check_source_boundaries(root, layout, report)
    check_compatibility_shims(root, layout, report)
    check_payload_manifest(root, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", help="以机器可读 JSON 输出")
    args = parser.parse_args(argv)
    report = run(args.root.resolve())
    if args.json:
        print(json.dumps({"ok": report.ok, "errors": report.errors, "warnings": report.warnings}, ensure_ascii=False, indent=2))
    else:
        for item in report.errors:
            print(f"ERROR {item}")
        for item in report.warnings:
            print(f"WARN  {item}")
        print(f"repository structure: {'OK' if report.ok else 'FAILED'}; errors={len(report.errors)} warnings={len(report.warnings)}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
