#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文档冗余检测工具：扫描 docs/ 目录中的重复内容。

检测规则：
1. 文件名相似度 > 80%（编辑距离）
2. 内容重复度 > 60%（行级去重后的重叠率）
3. 同一概念在多个文件中重复定义
4. 过时文档未归档到 docs/archive/

用法：
    python tools/check_doc_redundancy.py
    python tools/check_doc_redundancy.py --threshold 0.7
    python tools/check_doc_redundancy.py --fix  # 自动移动到 archive
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence


@dataclass
class RedundancyIssue:
    """文档冗余问题记录"""
    severity: str  # "warning" | "error"
    file1: Path
    file2: Path | None
    reason: str
    similarity: float


def levenshtein_distance(s1: str, s2: str) -> int:
    """计算编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def filename_similarity(name1: str, name2: str) -> float:
    """计算文件名相似度（0-1）"""
    distance = levenshtein_distance(name1.lower(), name2.lower())
    max_len = max(len(name1), len(name2))
    return 1 - (distance / max_len) if max_len > 0 else 0


def content_similarity(file1: Path, file2: Path) -> float:
    """计算内容相似度（0-1）：去重后的行级重叠率"""
    try:
        lines1 = set(
            line.strip()
            for line in file1.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        lines2 = set(
            line.strip()
            for line in file2.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        if not lines1 or not lines2:
            return 0.0
        overlap = len(lines1 & lines2)
        total = len(lines1 | lines2)
        return overlap / total if total > 0 else 0.0
    except Exception:
        return 0.0


def extract_concepts(file: Path) -> set[str]:
    """提取文档中的关键概念（标题、术语）"""
    concepts = set()
    try:
        content = file.read_text(encoding="utf-8")
        # 提取 Markdown 标题
        for match in re.finditer(r"^#+\s+(.+)$", content, re.MULTILINE):
            concepts.add(match.group(1).strip().lower())
        # 提取加粗术语
        for match in re.finditer(r"\*\*(.+?)\*\*", content):
            term = match.group(1).strip().lower()
            if len(term) > 3 and len(term) < 50:
                concepts.add(term)
    except Exception:
        pass
    return concepts


def check_filename_redundancy(files: Sequence[Path], threshold: float) -> list[RedundancyIssue]:
    """检测文件名冗余"""
    issues = []
    for i, f1 in enumerate(files):
        for f2 in files[i + 1:]:
            sim = filename_similarity(f1.stem, f2.stem)
            if sim > threshold:
                issues.append(RedundancyIssue(
                    severity="warning",
                    file1=f1,
                    file2=f2,
                    reason=f"文件名高度相似（{sim:.1%}）：是否为重复文档？",
                    similarity=sim
                ))
    return issues


def check_content_redundancy(files: Sequence[Path], threshold: float) -> list[RedundancyIssue]:
    """检测内容冗余"""
    issues = []
    for i, f1 in enumerate(files):
        for f2 in files[i + 1:]:
            sim = content_similarity(f1, f2)
            if sim > threshold:
                issues.append(RedundancyIssue(
                    severity="error",
                    file1=f1,
                    file2=f2,
                    reason=f"内容高度重复（{sim:.1%}）：应合并或归档其中一个",
                    similarity=sim
                ))
    return issues


def check_concept_duplication(files: Sequence[Path]) -> list[RedundancyIssue]:
    """检测概念在多个文档中重复定义"""
    concept_locations: dict[str, list[Path]] = defaultdict(list)
    for file in files:
        concepts = extract_concepts(file)
        for concept in concepts:
            concept_locations[concept].append(file)

    issues = []
    for concept, locations in concept_locations.items():
        if len(locations) > 2:  # 同一概念出现在 3+ 个文档中
            issues.append(RedundancyIssue(
                severity="warning",
                file1=locations[0],
                file2=None,
                reason=f"概念 '{concept}' 在 {len(locations)} 个文档中重复定义：{', '.join(f.name for f in locations[:3])}",
                similarity=1.0
            ))
    return issues


def check_outdated_docs(docs_root: Path) -> list[RedundancyIssue]:
    """检测可能过时但未归档的文档"""
    issues = []
    archive_path = docs_root / "archive"

    # 查找包含 "旧"、"废弃"、"已弃用" 等关键词的文档
    outdated_keywords = ["旧", "废弃", "已弃用", "过时", "old", "deprecated", "obsolete", "legacy"]

    for file in docs_root.rglob("*.md"):
        if archive_path in file.parents:
            continue  # 已在归档目录中

        try:
            content = file.read_text(encoding="utf-8").lower()
            for keyword in outdated_keywords:
                if keyword in content or keyword in file.stem.lower():
                    issues.append(RedundancyIssue(
                        severity="warning",
                        file1=file,
                        file2=None,
                        reason=f"文档可能过时（包含关键词 '{keyword}'）但未归档",
                        similarity=0.0
                    ))
                    break
        except Exception:
            pass

    return issues


def main():
    parser = argparse.ArgumentParser(description="检测文档冗余")
    parser.add_argument("--threshold", type=float, default=0.6, help="相似度阈值（0-1）")
    parser.add_argument("--fix", action="store_true", help="自动将疑似过时文档移动到 archive")
    args = parser.parse_args()

    docs_root = Path("docs")
    if not docs_root.exists():
        print("错误: docs/ 目录不存在")
        return 1

    # 收集所有 Markdown 文档（排除 archive 目录）
    archive_path = docs_root / "archive"
    all_docs = [
        f for f in docs_root.rglob("*.md")
        if archive_path not in f.parents
    ]

    print(f"扫描 {len(all_docs)} 个文档...")

    issues: list[RedundancyIssue] = []

    # 检测各类冗余
    issues.extend(check_filename_redundancy(all_docs, threshold=0.8))
    issues.extend(check_content_redundancy(all_docs, threshold=args.threshold))
    issues.extend(check_concept_duplication(all_docs))
    issues.extend(check_outdated_docs(docs_root))

    # 按严重程度排序
    issues.sort(key=lambda x: (x.severity != "error", -x.similarity))

    # 输出结果
    error_count = sum(1 for i in issues if i.severity == "error")
    warning_count = sum(1 for i in issues if i.severity == "warning")

    if not issues:
        print("✓ 未发现文档冗余问题")
        return 0

    print(f"\n发现 {error_count} 个错误，{warning_count} 个警告：\n")

    for issue in issues:
        prefix = "ERROR" if issue.severity == "error" else "WARNING"
        try:
            file1_path = issue.file1.relative_to(Path.cwd())
        except ValueError:
            file1_path = issue.file1
        print(f"{prefix}: {file1_path}")
        if issue.file2:
            try:
                file2_path = issue.file2.relative_to(Path.cwd())
            except ValueError:
                file2_path = issue.file2
            print(f"       vs {file2_path}")
        print(f"       {issue.reason}\n")

    # 自动修复（移动到 archive）
    if args.fix:
        archive_path.mkdir(parents=True, exist_ok=True)
        for issue in issues:
            if "过时" in issue.reason or "obsolete" in issue.reason:
                target = archive_path / issue.file1.name
                print(f"移动到归档: {issue.file1} -> {target}")
                issue.file1.rename(target)

    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    exit(main())
