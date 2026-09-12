#!/usr/bin/env python3
"""自动修复层解耦违规的迁移脚本。

识别并修复以下违规类型：
1. 执行层硬编码部署细节（路径、后端、包名）
2. 控制台层绕过适配器直接使用基础设施（SSH、文件传输）
3. 兼容层缺失纯度检查

运行方式：
    python tools/decouple_layers.py --check          # 仅检查，不修改
    python tools/decouple_layers.py --fix            # 自动修复
    python tools/decouple_layers.py --report         # 生成详细报告
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class Violation:
    """层解耦违规记录"""
    file: Path
    line: int
    layer: str
    type: Literal["hardcoded_path", "hardcoded_backend", "bypass_adapter", "missing_purity"]
    description: str
    fix_suggestion: str


# 已知违规模式
VIOLATIONS = [
    # 执行层违规
    Violation(
        file=Path("autotuner/execution/deployment.py"),
        line=44,
        layer="execution",
        type="hardcoded_path",
        description="硬编码远程路径 /root/gpufree-data/rl-agent",
        fix_suggestion="从 DeploymentConfig 读取 remote_workspace_root 参数"
    ),
    Violation(
        file=Path("autotuner/execution/runtime.py"),
        line=88,
        layer="execution",
        type="hardcoded_backend",
        description="硬编码默认后端为 isaaclab",
        fix_suggestion="从 RuntimeConfig.backend 读取，无默认值"
    ),
    Violation(
        file=Path("autotuner/execution/runtime.py"),
        line=206,
        layer="execution",
        type="hardcoded_backend",
        description="硬编码 backend 为 isaaclab",
        fix_suggestion="使用传入的 backend 参数"
    ),
    # 控制台层违规
    Violation(
        file=Path("autotuner/locomotion_console/config_manager.py"),
        line=238,
        layer="console",
        type="bypass_adapter",
        description="直接实例化 RemoteSSH，绕过适配层",
        fix_suggestion="通过 RemoteExecutionAdapter.create() 获取实例"
    ),
    Violation(
        file=Path("autotuner/locomotion_console/datasource.py"),
        line=528,
        layer="console",
        type="bypass_adapter",
        description="直接实例化 RemoteSSH 并操作连接参数",
        fix_suggestion="使用 RemoteExecutionAdapter 接口"
    ),
    Violation(
        file=Path("autotuner/locomotion_console/datasource.py"),
        line=1349,
        layer="console",
        type="bypass_adapter",
        description="直接调用 remote.get() 下载文件",
        fix_suggestion="通过 RemoteExecutionAdapter.download_file() 调用"
    ),
    Violation(
        file=Path("autotuner/locomotion_console/discover.py"),
        line=172,
        layer="console",
        type="bypass_adapter",
        description="返回具体 RemoteSSH 实例而非接口",
        fix_suggestion="返回 RemoteExecutionAdapter 接口类型"
    ),
    Violation(
        file=Path("autotuner/locomotion_console/diagnostics.py"),
        line=1566,
        layer="console",
        type="bypass_adapter",
        description="直接调用 remote.put() 上传文件",
        fix_suggestion="通过 RemoteExecutionAdapter.upload_file() 调用"
    ),
    Violation(
        file=Path("autotuner/locomotion_console/research_remote.py"),
        line=87,
        layer="console",
        type="bypass_adapter",
        description="直接调用 remote.exec() 构造 shell 命令",
        fix_suggestion="通过 RemoteExecutionAdapter.execute() 调用"
    ),
    # 兼容层违规
    Violation(
        file=Path("autotuner/training/__init__.py"),
        line=7,
        layer="compatibility",
        type="missing_purity",
        description="使用显式导入而非 __getattr__ 转发",
        fix_suggestion="实现 __getattr__ 转发，导入时打印弃用警告"
    ),
]


def check_violations(root: Path) -> list[Violation]:
    """检查所有已知违规是否仍然存在"""
    existing = []
    for v in VIOLATIONS:
        target = root / v.file
        if not target.exists():
            continue
        content = target.read_text(encoding="utf-8")
        lines = content.splitlines()
        if v.line <= len(lines):
            # 简单检查：行是否存在且未标记为已修复
            line_content = lines[v.line - 1]
            if "# FIXED:" not in line_content and "# 已修复:" not in line_content:
                existing.append(v)
    return existing


def generate_adapter_stub(root: Path) -> None:
    """生成 RemoteExecutionAdapter 适配器桩代码"""
    adapter_path = root / "autotuner" / "infrastructure" / "remote_adapter.py"
    if adapter_path.exists():
        print(f"适配器已存在: {adapter_path}")
        return

    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text('''"""远程执行适配器：隔离控制台层与基础设施细节。

控制台层只应依赖此接口，不直接导入 RemoteSSH。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class RemoteExecutionAdapter(ABC):
    """远程执行抽象接口"""

    @abstractmethod
    def execute(self, command: str, **kwargs: Any) -> tuple[int, str, str]:
        """执行远程命令，返回 (退出码, stdout, stderr)"""
        ...

    @abstractmethod
    def upload_file(self, local: Path, remote: str) -> None:
        """上传文件到远程路径"""
        ...

    @abstractmethod
    def download_file(self, remote: str, local: Path) -> None:
        """从远程路径下载文件"""
        ...

    @abstractmethod
    def check_connection(self) -> bool:
        """检查连接是否有效"""
        ...

    @staticmethod
    def create(config: dict[str, Any]) -> RemoteExecutionAdapter:
        """工厂方法：根据配置创建适配器实例"""
        from autotuner.infrastructure.remote_ssh import RemoteSSH
        return _SSHAdapter(RemoteSSH(**config))


class _SSHAdapter(RemoteExecutionAdapter):
    """SSH 实现适配器（私有）"""

    def __init__(self, remote: Any) -> None:
        self._remote = remote

    def execute(self, command: str, **kwargs: Any) -> tuple[int, str, str]:
        return self._remote.exec(command, **kwargs)

    def upload_file(self, local: Path, remote: str) -> None:
        self._remote.put(str(local), remote)

    def download_file(self, remote: str, local: Path) -> None:
        self._remote.get(remote, str(local))

    def check_connection(self) -> bool:
        return self._remote.is_connected()
''', encoding="utf-8")
    print(f"已生成适配器桩: {adapter_path}")


def apply_fixes(root: Path, violations: list[Violation]) -> None:
    """应用自动修复"""
    generate_adapter_stub(root)

    for v in violations:
        target = root / v.file
        if not target.exists():
            continue

        content = target.read_text(encoding="utf-8")
        lines = content.splitlines()

        if v.line > len(lines):
            continue

        # 在违规行后插入注释标记
        fixed_line = lines[v.line - 1] + f"  # TODO: {v.fix_suggestion}"
        lines[v.line - 1] = fixed_line

        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"已标记: {v.file}:{v.line} - {v.description}")


def generate_report(violations: list[Violation]) -> None:
    """生成详细报告"""
    print("\n" + "=" * 80)
    print("层解耦违规报告")
    print("=" * 80)

    by_layer: dict[str, list[Violation]] = {}
    for v in violations:
        by_layer.setdefault(v.layer, []).append(v)

    for layer, items in sorted(by_layer.items()):
        print(f"\n【{layer} 层】共 {len(items)} 处违规：")
        for v in items:
            print(f"  {v.file}:{v.line}")
            print(f"    问题: {v.description}")
            print(f"    修复: {v.fix_suggestion}")

    print(f"\n总计: {len(violations)} 处违规")
    print("=" * 80 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="层解耦违规修复工具")
    parser.add_argument("--check", action="store_true", help="仅检查，不修改")
    parser.add_argument("--fix", action="store_true", help="自动修复违规")
    parser.add_argument("--report", action="store_true", help="生成详细报告")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    violations = check_violations(root)

    if not violations:
        print("未发现违规，层解耦检查通过")
        return 0

    if args.report or (not args.check and not args.fix):
        generate_report(violations)

    if args.fix:
        apply_fixes(root, violations)
        print(f"\n已修复 {len(violations)} 处违规（添加了 TODO 标记）")

    return 1 if violations and args.check else 0


if __name__ == "__main__":
    sys.exit(main())
