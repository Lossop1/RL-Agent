#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中文注释质量检查工具：扫描 Python 代码中的注释覆盖率和质量。

检查规则：
1. 模块级文档字符串必须包含中文
2. 公共函数/类必须有中文文档字符串
3. 复杂逻辑（if/for/while 嵌套 > 2 层）必须有行内中文注释
4. 注释密度：关键模块注释行数 / 代码行数 > 10%

用法：
    python tools/check_comment_quality.py
    python tools/check_comment_quality.py --path autotuner/
    python tools/check_comment_quality.py --strict  # 严格模式
"""
import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence


CHINESE_PATTERN = re.compile(r"[一-鿿]+")


@dataclass
class CommentIssue:
    """注释质量问题记录"""
    severity: str  # "warning" | "error"
    file: Path
    line: int
    reason: str


def has_chinese(text: str) -> bool:
    """检查文本是否包含中文字符"""
    return bool(CHINESE_PATTERN.search(text))


def count_nesting_depth(node: ast.AST) -> int:
    """计算 AST 节点的嵌套深度"""
    max_depth = 0
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
            depth = 1
            current = child
            for parent in ast.walk(node):
                for field, value in ast.iter_fields(parent):
                    if value is current or (isinstance(value, list) and current in value):
                        if isinstance(parent, (ast.If, ast.For, ast.While, ast.With)):
                            depth += 1
            max_depth = max(max_depth, depth)
    return max_depth


def check_module_docstring(file: Path, tree: ast.Module) -> list[CommentIssue]:
    """检查模块级文档字符串"""
    issues = []
    docstring = ast.get_docstring(tree)

    if not docstring:
        issues.append(CommentIssue(
            severity="error",
            file=file,
            line=1,
            reason="模块缺少文档字符串"
        ))
    elif not has_chinese(docstring):
        issues.append(CommentIssue(
            severity="error",
            file=file,
            line=1,
            reason="模块文档字符串必须包含中文说明"
        ))

    return issues


def check_function_docstring(file: Path, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[CommentIssue]:
    """检查函数文档字符串"""
    issues = []

    # 跳过私有函数（以 _ 开头）
    if node.name.startswith("_") and not node.name.startswith("__"):
        return issues

    docstring = ast.get_docstring(node)

    if not docstring:
        issues.append(CommentIssue(
            severity="warning",
            file=file,
            line=node.lineno,
            reason=f"公共函数 '{node.name}' 缺少文档字符串"
        ))
    elif not has_chinese(docstring):
        issues.append(CommentIssue(
            severity="warning",
            file=file,
            line=node.lineno,
            reason=f"函数 '{node.name}' 文档字符串应包含中文说明"
        ))

    return issues


def check_class_docstring(file: Path, node: ast.ClassDef) -> list[CommentIssue]:
    """检查类文档字符串"""
    issues = []

    # 跳过私有类
    if node.name.startswith("_"):
        return issues

    docstring = ast.get_docstring(node)

    if not docstring:
        issues.append(CommentIssue(
            severity="error",
            file=file,
            line=node.lineno,
            reason=f"公共类 '{node.name}' 缺少文档字符串"
        ))
    elif not has_chinese(docstring):
        issues.append(CommentIssue(
            severity="warning",
            file=file,
            line=node.lineno,
            reason=f"类 '{node.name}' 文档字符串应包含中文说明"
        ))

    return issues


def check_complex_logic_comments(file: Path, source_lines: list[str], tree: ast.Module) -> list[CommentIssue]:
    """检查复杂逻辑是否有注释"""
    issues = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            depth = count_nesting_depth(node)
            if depth > 2:  # 嵌套深度 > 2
                # 检查函数体附近是否有中文注释
                start_line = node.lineno
                end_line = node.end_lineno or start_line

                has_comment = False
                for line_num in range(max(0, start_line - 1), min(len(source_lines), end_line)):
                    line = source_lines[line_num]
                    if "#" in line and has_chinese(line):
                        has_comment = True
                        break

                if not has_comment:
                    issues.append(CommentIssue(
                        severity="warning",
                        file=file,
                        line=start_line,
                        reason=f"函数 '{node.name}' 包含复杂嵌套逻辑（深度 {depth}）但缺少中文注释"
                    ))

    return issues


def check_comment_density(file: Path, source_lines: list[str], strict: bool) -> list[CommentIssue]:
    """检查注释密度"""
    issues = []

    total_lines = len(source_lines)
    code_lines = sum(1 for line in source_lines if line.strip() and not line.strip().startswith("#"))
    comment_lines = sum(1 for line in source_lines if "#" in line and has_chinese(line))

    if code_lines == 0:
        return issues

    density = comment_lines / code_lines
    threshold = 0.15 if strict else 0.10

    if density < threshold:
        issues.append(CommentIssue(
            severity="warning",
            file=file,
            line=1,
            reason=f"中文注释密度过低（{density:.1%}），建议 > {threshold:.0%}"
        ))

    return issues


def check_file(file: Path, strict: bool) -> list[CommentIssue]:
    """检查单个 Python 文件"""
    try:
        source = file.read_text(encoding="utf-8")
        source_lines = source.splitlines()
        tree = ast.parse(source, filename=str(file))
    except Exception as e:
        return [CommentIssue(
            severity="error",
            file=file,
            line=1,
            reason=f"解析失败: {e}"
        )]

    issues: list[CommentIssue] = []

    # 检查模块文档字符串
    issues.extend(check_module_docstring(file, tree))

    # 检查函数和类文档字符串
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            issues.extend(check_function_docstring(file, node))
        elif isinstance(node, ast.ClassDef):
            issues.extend(check_class_docstring(file, node))

    # 检查复杂逻辑注释
    issues.extend(check_complex_logic_comments(file, source_lines, tree))

    # 检查注释密度
    issues.extend(check_comment_density(file, source_lines, strict))

    return issues


def main():
    parser = argparse.ArgumentParser(description="检查中文注释质量")
    parser.add_argument("--path", type=str, default="autotuner", help="检查路径")
    parser.add_argument("--strict", action="store_true", help="严格模式（更高的注释密度要求）")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"错误: 路径不存在: {root}")
        return 1

    # 收集所有 Python 文件
    if root.is_file():
        files = [root]
    else:
        files = list(root.rglob("*.py"))

    # 排除测试文件和第三方代码
    files = [
        f for f in files
        if "test" not in f.parts and "venv" not in f.parts and ".venv" not in f.parts
    ]

    print(f"检查 {len(files)} 个 Python 文件...")

    all_issues: list[CommentIssue] = []
    for file in files:
        issues = check_file(file, args.strict)
        all_issues.extend(issues)

    # 按文件和行号排序
    all_issues.sort(key=lambda x: (str(x.file), x.line))

    # 统计
    error_count = sum(1 for i in all_issues if i.severity == "error")
    warning_count = sum(1 for i in all_issues if i.severity == "warning")

    if not all_issues:
        print("✓ 所有文件的中文注释质量符合规范")
        return 0

    print(f"\n发现 {error_count} 个错误，{warning_count} 个警告：\n")

    current_file = None
    for issue in all_issues:
        if issue.file != current_file:
            current_file = issue.file
            print(f"\n{issue.file.relative_to(Path.cwd())}:")

        prefix = "  ERROR" if issue.severity == "error" else "  WARNING"
        print(f"{prefix} 第 {issue.line} 行: {issue.reason}")

    print(f"\n总计: {len(files)} 个文件, {error_count} 个错误, {warning_count} 个警告")

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    exit(main())
