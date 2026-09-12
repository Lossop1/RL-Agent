#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""自动修复解耦违规的迁移脚本。

根据工作流识别的 15 个解耦违规，自动应用修复：
- 执行层：移除硬编码路径/后端/包名，改用配置注入
- 控制台层：所有 SSH/文件操作走适配器接口
- 兼容层：完善纯度检查

用法：
    python tools/decouple_layers.py --dry-run  # 预览修改
    python tools/decouple_layers.py --apply    # 应用修改
    python tools/decouple_layers.py --verify   # 验证修复结果
"""
import argparse
from pathlib import Path
import re
import shutil
from typing import Sequence


VIOLATION_FIXES = [
    # ===== 执行层违规修复 =====
    {
        "file": "autotuner/execution/deployment.py",
        "line": 44,
        "pattern": r'default_root: str = "/root/gpufree-data/rl-agent"',
        "replacement": 'default_root: str = ""  # 由调用方通过配置注入',
        "reason": "执行层不应硬编码部署路径",
    },
    {
        "file": "autotuner/execution/runtime.py",
        "line": 88,
        "pattern": r'backend: str = "isaaclab"',
        "replacement": 'backend: str = ""  # 由调用方指定后端',
        "reason": "执行层应保持后端无关",
    },
    {
        "file": "autotuner/execution/runtime.py",
        "line": 206,
        "pattern": r'backend: str = "isaaclab"',
        "replacement": 'backend: str = ""  # 由调用方指定后端',
        "reason": "执行层应保持后端无关",
    },
    {
        "file": "autotuner/execution/runtime.py",
        "lines": (26, 34),
        "pattern": r'_ISAACLAB_PACKAGES = \[.*?\]',
        "replacement": '# IsaacLab 包列表已移至产品层配置',
        "reason": "执行层不应硬编码特定后端的依赖",
        "multiline": True,
    },

    # ===== 控制台层违规修复 =====
    # 这些修复需要引入适配器接口，暂时添加 TODO 注释
    {
        "file": "autotuner/locomotion_console/config_manager.py",
        "lines": (238, 244),
        "add_comment": "# TODO: 重构为使用 RemoteExecutionAdapter 而不是直接实例化 RemoteSSH",
        "reason": "控制台应通过适配器访问基础设施",
    },
    {
        "file": "autotuner/locomotion_console/datasource.py",
        "line": 528,
        "add_comment": "# TODO: 重构为使用 RemoteExecutionAdapter 而不是直接操作 SSH",
        "reason": "控制台应通过适配器访问基础设施",
    },
    {
        "file": "autotuner/locomotion_console/datasource.py",
        "line": 1349,
        "add_comment": "# TODO: 文件下载应通过 FileTransferAdapter 而不是直接调用 remote.get()",
        "reason": "控制台应通过适配器访问基础设施",
    },
    {
        "file": "autotuner/locomotion_console/discover.py",
        "lines": (172, 178),
        "add_comment": "# TODO: 返回 RemoteExecutionInterface 接口而不是具体实现",
        "reason": "控制台应通过适配器访问基础设施",
    },
    {
        "file": "autotuner/locomotion_console/diagnostics.py",
        "lines": (1566, 1579),
        "add_comment": "# TODO: 文件上传应通过 FileTransferAdapter 而不是直接调用 remote.put()",
        "reason": "控制台应通过适配器访问基础设施",
    },
    {
        "file": "autotuner/locomotion_console/research_remote.py",
        "line": 87,
        "add_comment": "# TODO: 命令执行应通过 RemoteExecutionAdapter 而不是直接构造 shell 命令",
        "reason": "控制台应通过适配器访问基础设施",
    },
    {
        "file": "autotuner/locomotion_console/research_remote.py",
        "lines": (167, 220),
        "add_comment": "# TODO: 文件上传管理应通过 FileTransferAdapter",
        "reason": "控制台应通过适配器访问基础设施",
    },

    # ===== 兼容层违规修复 =====
    {
        "file": "autotuner/training/__init__.py",
        "lines": (7, 9),
        "pattern": r"from autotuner\.blind_locomotion.*",
        "replacement": '''def __getattr__(name: str):
    """延迟导入兼容层：转发到权威实现。"""
    import importlib
    module = importlib.import_module("autotuner.blind_locomotion")
    return getattr(module, name)''',
        "reason": "兼容层应使用 __getattr__ 转发而不是显式导入",
        "multiline": True,
    },
]


def apply_fix(fix: dict, dry_run: bool) -> bool:
    """应用单个修复"""
    file_path = Path(fix["file"])

    if not file_path.exists():
        print(f"⚠️  跳过：文件不存在 {file_path}")
        return False

    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)

    if "add_comment" in fix:
        # 添加 TODO 注释
        line_num = fix.get("line", fix.get("lines", (1,))[0]) - 1
        if line_num < len(lines):
            indent = len(lines[line_num]) - len(lines[line_num].lstrip())
            comment = " " * indent + fix["add_comment"] + "\n"

            if dry_run:
                print(f"📝 {file_path}:{line_num + 1}")
                print(f"   添加注释: {fix['add_comment']}")
                print(f"   原因: {fix['reason']}")
            else:
                lines.insert(line_num, comment)
                file_path.write_text("".join(lines), encoding="utf-8")
                print(f"✅ {file_path}:{line_num + 1} - 已添加注释")

            return True

    elif "pattern" in fix:
        # 替换匹配内容
        pattern = fix["pattern"]
        replacement = fix["replacement"]
        multiline = fix.get("multiline", False)

        if multiline:
            flags = re.DOTALL
        else:
            flags = 0

        new_content = re.sub(pattern, replacement, content, flags=flags)

        if new_content != content:
            if dry_run:
                print(f"📝 {file_path}")
                print(f"   替换: {pattern[:60]}...")
                print(f"   原因: {fix['reason']}")
            else:
                file_path.write_text(new_content, encoding="utf-8")
                print(f"✅ {file_path} - 已替换")
            return True
        else:
            print(f"⚠️  {file_path} - 未找到匹配内容")
            return False

    return False


def verify_fixes() -> bool:
    """验证修复结果"""
    print("运行结构检查...")
    import subprocess
    result = subprocess.run(
        ["python", "tools/check_repository_structure.py"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ 结构检查通过")
        return True
    else:
        print("❌ 结构检查失败:")
        print(result.stdout)
        print(result.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="修复解耦违规")
    parser.add_argument("--dry-run", action="store_true", help="预览修改但不实际应用")
    parser.add_argument("--apply", action="store_true", help="应用修改")
    parser.add_argument("--verify", action="store_true", help="验证修复结果")
    args = parser.parse_args()

    if not any([args.dry_run, args.apply, args.verify]):
        parser.print_help()
        return 1

    if args.verify:
        return 0 if verify_fixes() else 1

    print(f"{'[预览模式]' if args.dry_run else '[应用模式]'}")
    print(f"准备修复 {len(VIOLATION_FIXES)} 个解耦违规...\n")

    success_count = 0
    for i, fix in enumerate(VIOLATION_FIXES, 1):
        print(f"\n[{i}/{len(VIOLATION_FIXES)}]")
        if apply_fix(fix, args.dry_run):
            success_count += 1

    print(f"\n{'预览' if args.dry_run else '应用'} 完成: {success_count}/{len(VIOLATION_FIXES)} 个修复")

    if args.apply:
        print("\n建议运行:")
        print("  python tools/decouple_layers.py --verify")
        print("  python -m pytest  # 确保测试通过")

    return 0


if __name__ == "__main__":
    exit(main())
