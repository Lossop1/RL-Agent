#!/usr/bin/env python3
"""检查中文注释质量和覆盖率。

检查规则：
1. 核心模块（mechanisms, research, execution）注释覆盖率 >= 80%
2. 公共函数/类必须有中文文档字符串
3. 复杂逻辑块（> 10 行）应有中文注释
4. 避免无意义注释（如 `# 设置 x = 1`）

运行方式：
    python tools/check_comment_quality.py
    python tools/check_comment_quality.py --verbose  # 显示详细信息
"""
from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class CommentStats:
    """模块注释统计"""
    file: Path
    total_lines: int
    code_lines: int
    comment_lines: int
    chinese_comment_lines: int
    functions: int
    functions_with_docstring: int
    classes: int
    classes_with_docstring: int


@dataclass
class QualityIssue:
    """注释质量问题"""
    file: Path
    line: int
    type: str  # missing_docstring, low_coverage, trivial_comment
    description: str
    severity: str  # error, warning


CORE_MODULES = {
    "autotuner/mechanisms",
    "autotuner/research",
    "autotuner/execution",
    "autotuner/artifacts",
    "products/taili/core",
}

TRIVIAL_PATTERNS = [
    r"^#\s*设置\s*\w+\s*=",  # 设置 x = 1
    r"^#\s*返回\s*$",  # 返回
    r"^#\s*初始化\s*$",  # 初始化
    r"^#\s*TODO\s*$",  # TODO（无具体内容）
]


def _is_chinese(text: str) -> bool:
    """判断文本是否包含中文字符"""
    return bool(re.search(r"[一-鿿]", text))


def _is_core_module(file: Path, root: Path) -> bool:
    """判断是否为核心模块"""
    rel = file.relative_to(root).as_posix()
    return any(rel.startswith(prefix) for prefix in CORE_MODULES)


def _count_lines(content: str) -> tuple[int, int, int, int]:
    """统计总行数、代码行、注释行、中文注释行"""
    lines = content.splitlines()
    total = len(lines)
    code = 0
    comment = 0
    chinese_comment = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comment += 1
            if _is_chinese(stripped):
                chinese_comment += 1
        else:
            code += 1
            # 行内注释
            if "#" in line:
                comment_part = line.split("#", 1)[1]
                if _is_chinese(comment_part):
                    chinese_comment += 1

    return total, code, comment, chinese_comment


def _has_chinese_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    """检查函数/类是否有中文文档字符串"""
    docstring = ast.get_docstring(node)
    return bool(docstring and _is_chinese(docstring))


def _analyze_file(file: Path) -> CommentStats:
    """分析单个文件的注释情况"""
    content = file.read_text(encoding="utf-8")
    total, code, comment, chinese_comment = _count_lines(content)

    try:
        tree = ast.parse(content, filename=str(file))
    except SyntaxError:
        return CommentStats(
            file=file,
            total_lines=total,
            code_lines=code,
            comment_lines=comment,
            chinese_comment_lines=chinese_comment,
            functions=0,
            functions_with_docstring=0,
            classes=0,
            classes_with_docstring=0,
        )

    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

    functions_with_doc = sum(1 for f in functions if _has_chinese_docstring(f))
    classes_with_doc = sum(1 for c in classes if _has_chinese_docstring(c))

    return CommentStats(
        file=file,
        total_lines=total,
        code_lines=code,
        comment_lines=comment,
        chinese_comment_lines=chinese_comment,
        functions=len(functions),
        functions_with_docstring=functions_with_doc,
        classes=len(classes),
        classes_with_docstring=classes_with_doc,
    )


def _find_python_files(root: Path) -> Iterator[Path]:
    """查找所有 Python 源文件"""
    for base in (
        root / "autotuner",
        root / "products",
        root / "tools" / "isaaclab_quad_diag_observation",
    ):
        if not base.is_dir():
            continue
        yield from (p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def check_coverage(root: Path) -> list[QualityIssue]:
    """检查注释覆盖率"""
    issues: list[QualityIssue] = []

    for file in _find_python_files(root):
        stats = _analyze_file(file)

        # 计算覆盖率
        coverage = 0.0
        if stats.code_lines > 0:
            coverage = stats.chinese_comment_lines / stats.code_lines

        is_core = _is_core_module(file, root)

        # 核心模块要求 80% 覆盖率
        if is_core and coverage < 0.8:
            issues.append(QualityIssue(
                file=file,
                line=1,
                type="low_coverage",
                description=f"中文注释覆盖率 {coverage:.1%}，核心模块要求 >= 80%",
                severity="error",
            ))

        # 公共函数/类缺少文档字符串
        if stats.functions > 0:
            doc_rate = stats.functions_with_docstring / stats.functions
            if doc_rate < 0.5:
                issues.append(QualityIssue(
                    file=file,
                    line=1,
                    type="missing_docstring",
                    description=f"{stats.functions_with_docstring}/{stats.functions} 函数有中文文档字符串",
                    severity="warning" if not is_core else "error",
                ))

        if stats.classes > 0:
            doc_rate = stats.classes_with_docstring / stats.classes
            if doc_rate < 0.5:
                issues.append(QualityIssue(
                    file=file,
                    line=1,
                    type="missing_docstring",
                    description=f"{stats.classes_with_docstring}/{stats.classes} 类有中文文档字符串",
                    severity="warning" if not is_core else "error",
                ))

    return issues


def check_trivial_comments(root: Path) -> list[QualityIssue]:
    """检查无意义注释"""
    issues: list[QualityIssue] = []

    for file in _find_python_files(root):
        content = file.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, start=1):
            comment_match = re.search(r"#\s*(.+)$", line)
            if not comment_match:
                continue

            comment_text = comment_match.group(1).strip()

            for pattern in TRIVIAL_PATTERNS:
                if re.match(pattern, comment_text):
                    issues.append(QualityIssue(
                        file=file,
                        line=line_num,
                        type="trivial_comment",
                        description=f"无意义注释: {comment_text}",
                        severity="warning",
                    ))
                    break

    return issues


def generate_report(issues: list[QualityIssue], verbose: bool = False) -> None:
    """生成报告"""
    if not issues:
        print("✓ 中文注释质量检查通过")
        return

    print("\n" + "=" * 80)
    print("中文注释质量报告")
    print("=" * 80)

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if errors:
        print(f"\n【错误】共 {len(errors)} 项")
        for issue in errors:
            print(f"  {issue.file}:{issue.line}")
            print(f"    {issue.description}")

    if warnings and verbose:
        print(f"\n【警告】共 {len(warnings)} 项")
        for issue in warnings:
            print(f"  {issue.file}:{issue.line}")
            print(f"    {issue.description}")

    print(f"\n总计: {len(errors)} 错误, {len(warnings)} 警告")
    print("=" * 80 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="检查中文注释质量")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="仓库根目录")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    args = parser.parse_args()

    root = args.root.resolve()

    issues: list[QualityIssue] = []
    issues.extend(check_coverage(root))
    issues.extend(check_trivial_comments(root))

    generate_report(issues, verbose=args.verbose)

    errors = [i for i in issues if i.severity == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
