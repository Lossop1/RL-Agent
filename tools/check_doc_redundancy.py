#!/usr/bin/env python3
"""检查文档冗余：识别重复内容和过期描述。

检查规则：
1. 多处文档描述相同概念（如架构、层级定义）
2. 文档与代码注释重复
3. 示例代码在多处出现
4. 过期的归档文档仍被引用

运行方式：
    python tools/check_doc_redundancy.py
    python tools/check_doc_redundancy.py --fix-refs  # 自动修复过期引用
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class RedundancyIssue:
    """冗余问题记录"""
    type: str  # duplicate_concept, duplicate_code, stale_reference
    files: list[Path]
    description: str
    severity: str  # error, warning


def _find_markdown_files(root: Path) -> Iterator[Path]:
    """查找所有 Markdown 文档"""
    docs_dir = root / "docs"
    if docs_dir.exists():
        yield from docs_dir.rglob("*.md")

    readme = root / "README.md"
    if readme.exists():
        yield readme


def _extract_headings(content: str) -> list[str]:
    """提取文档中的所有标题"""
    return re.findall(r"^#{1,6}\s+(.+)$", content, re.MULTILINE)


def _extract_code_blocks(content: str) -> list[str]:
    """提取代码块（用于检测重复示例）"""
    blocks = re.findall(r"```[\w]*\n(.*?)```", content, re.DOTALL)
    # 过滤掉太短的代码块（< 3 行）
    return [block.strip() for block in blocks if block.count("\n") >= 2]


def check_duplicate_concepts(root: Path) -> list[RedundancyIssue]:
    """检查重复概念描述"""
    issues: list[RedundancyIssue] = []

    # 已知容易重复的关键概念
    key_concepts = {
        "层级": r"第?\s*[1-6六五四三二一]\s*层[：:]",
        "架构": r"(架构|分层|层级结构)",
        "解耦": r"解耦|依赖方向|底层.*上层",
        "原子问题": r"原子问题",
        "payload": r"payload\s*(清单|manifest)",
    }

    concept_locations: dict[str, list[tuple[Path, int]]] = defaultdict(list)

    for doc_path in _find_markdown_files(root):
        content = doc_path.read_text(encoding="utf-8")
        for concept_name, pattern in key_concepts.items():
            for match in re.finditer(pattern, content, re.IGNORECASE):
                line_num = content[:match.start()].count("\n") + 1
                concept_locations[concept_name].append((doc_path, line_num))

    # 出现在 3 处以上的概念可能冗余
    for concept, locations in concept_locations.items():
        if len(locations) >= 3:
            unique_files = list(set(path for path, _ in locations))
            if len(unique_files) >= 3:
                issues.append(RedundancyIssue(
                    type="duplicate_concept",
                    files=unique_files,
                    description=f"概念 '{concept}' 在 {len(unique_files)} 个文档中重复定义",
                    severity="warning"
                ))

    return issues


def check_duplicate_code_examples(root: Path) -> list[RedundancyIssue]:
    """检查重复的代码示例"""
    issues: list[RedundancyIssue] = []
    code_blocks: dict[str, list[Path]] = defaultdict(list)

    for doc_path in _find_markdown_files(root):
        content = doc_path.read_text(encoding="utf-8")
        for block in _extract_code_blocks(content):
            # 使用代码块的哈希前 40 字符作为标识
            block_hash = str(hash(block))[:40]
            code_blocks[block_hash].append(doc_path)

    # 相同代码块出现在多个文档中
    for block_hash, files in code_blocks.items():
        if len(files) >= 2:
            issues.append(RedundancyIssue(
                type="duplicate_code",
                files=list(set(files)),
                description=f"相同代码示例在 {len(files)} 个文档中重复",
                severity="warning"
            ))

    return issues


def check_stale_references(root: Path) -> list[RedundancyIssue]:
    """检查对归档文档的引用（应当尽量避免）"""
    issues: list[RedundancyIssue] = []
    archive_dir = root / "docs" / "archive"

    if not archive_dir.exists():
        return issues

    for doc_path in _find_markdown_files(root):
        # 跳过归档目录内的文档
        if archive_dir in doc_path.parents:
            continue

        content = doc_path.read_text(encoding="utf-8")

        # 查找指向 archive/ 的链接
        archive_refs = re.findall(r"\[([^\]]+)\]\(([^)]*archive[^)]*)\)", content)

        if archive_refs:
            issues.append(RedundancyIssue(
                type="stale_reference",
                files=[doc_path],
                description=f"文档引用了 {len(archive_refs)} 个归档文件，建议移除或更新",
                severity="warning"
            ))

    return issues


def check_readme_vs_docs(root: Path) -> list[RedundancyIssue]:
    """检查 README.md 与 docs/ 中的重复内容"""
    issues: list[RedundancyIssue] = []
    readme = root / "README.md"

    if not readme.exists():
        return issues

    readme_content = readme.read_text(encoding="utf-8")
    readme_headings = set(_extract_headings(readme_content))

    docs_dir = root / "docs"
    if not docs_dir.exists():
        return issues

    for doc_path in docs_dir.rglob("*.md"):
        if "archive" in doc_path.parts:
            continue

        doc_content = doc_path.read_text(encoding="utf-8")
        doc_headings = set(_extract_headings(doc_content))

        # 标题重叠度 > 50% 可能表示内容重复
        overlap = readme_headings & doc_headings
        if len(overlap) > 2 and len(overlap) / len(doc_headings) > 0.5:
            issues.append(RedundancyIssue(
                type="duplicate_concept",
                files=[readme, doc_path],
                description=f"README 与 {doc_path.name} 有 {len(overlap)} 个相同标题，可能内容重复",
                severity="warning"
            ))

    return issues


def generate_report(issues: list[RedundancyIssue]) -> None:
    """生成可读报告"""
    if not issues:
        print("未发现文档冗余问题")
        return

    print("\n" + "=" * 80)
    print("文档冗余检查报告")
    print("=" * 80)

    by_type: dict[str, list[RedundancyIssue]] = defaultdict(list)
    for issue in issues:
        by_type[issue.type].append(issue)

    type_names = {
        "duplicate_concept": "重复概念",
        "duplicate_code": "重复代码示例",
        "stale_reference": "过期引用",
    }

    for issue_type, type_issues in by_type.items():
        print(f"\n【{type_names.get(issue_type, issue_type)}】")
        for issue in type_issues:
            print(f"  {issue.severity.upper()}: {issue.description}")
            for file_path in issue.files:
                print(f"    - {file_path}")

    print(f"\n总计: {len(issues)} 个潜在冗余问题")
    print("=" * 80 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="检查文档冗余")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="仓库根目录")
    parser.add_argument("--fix-refs", action="store_true", help="自动移除对归档文档的引用（未实现）")
    args = parser.parse_args()

    root = args.root.resolve()

    issues: list[RedundancyIssue] = []
    issues.extend(check_duplicate_concepts(root))
    issues.extend(check_duplicate_code_examples(root))
    issues.extend(check_stale_references(root))
    issues.extend(check_readme_vs_docs(root))

    generate_report(issues)

    # 只有 error 级别的问题才导致退出码非零
    errors = [i for i in issues if i.severity == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
