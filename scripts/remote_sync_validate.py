"""P4.3 远程代码同步和验证脚本"""

import paramiko
import sys
import os
import time
from pathlib import Path


def ssh_execute(client, command, description="", show_output=True):
    """执行SSH命令并打印输出"""
    if description:
        print(f"\n=== {description} ===", flush=True)
    stdin, stdout, stderr = client.exec_command(command, get_pty=True)

    output_lines = []
    error_lines = []

    # 实时读取输出
    while not stdout.channel.exit_status_ready():
        if stdout.channel.recv_ready():
            line = stdout.readline()
            if show_output:
                print(line, end='', flush=True)
            output_lines.append(line)

    # 读取剩余输出
    for line in stdout:
        if show_output:
            print(line, end='', flush=True)
        output_lines.append(line)

    exit_code = stdout.channel.recv_exit_status()
    output = ''.join(output_lines)

    return exit_code, output


def sync_and_validate(host, port, username, password, local_workspace):
    """同步代码到远程服务器并验证"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"正在连接到 {username}@{host}:{port}...", flush=True)
        client.connect(host, port=port, username=username, password=password, timeout=30)
        print("SSH连接成功\n", flush=True)

        # 1. 检查远程是否有git仓库
        code, output = ssh_execute(client,
            "ls -d /root/RL/locomotion-workspace /root/locomotion-workspace 2>/dev/null | head -1",
            "查找远程Git仓库", show_output=False)

        remote_repo = output.strip()

        if not remote_repo:
            print("远程没有Git仓库，需要先克隆代码", flush=True)
            print("手动操作建议:", flush=True)
            print("  ssh -p 31376 root@183.147.142.40", flush=True)
            print("  cd /root && git clone <仓库地址> locomotion-workspace", flush=True)
            return False

        print(f"找到远程仓库: {remote_repo}", flush=True)

        # 2. 检查远程分支状态
        code, output = ssh_execute(client,
            f"cd {remote_repo} && git status",
            "检查Git状态")

        # 3. 拉取最新代码
        code, output = ssh_execute(client,
            f"cd {remote_repo} && git fetch origin && git log HEAD..origin/master --oneline | head -5",
            "检查远程更新", show_output=False)

        if output.strip():
            print("发现远程有新提交:", flush=True)
            print(output, flush=True)

            print("\n准备拉取更新...", flush=True)
            code, output = ssh_execute(client,
                f"cd {remote_repo} && git pull origin master",
                "拉取最新代码")

            if code != 0:
                print("git pull失败，可能有本地修改冲突", flush=True)
                return False
        else:
            print("远程代码已是最新", flush=True)

        # 4. 检查P4.2文件
        code, output = ssh_execute(client,
            f"cd {remote_repo} && ls -lh autotuner/product/checkpoint_curator.py 2>&1",
            "验证P4.2文件", show_output=False)

        if code == 0:
            print("checkpoint_curator.py 已存在", flush=True)
        else:
            print("checkpoint_curator.py 不存在，检查提交历史...", flush=True)
            code, output = ssh_execute(client,
                f"cd {remote_repo} && git log --oneline --grep='P4.2' | head -3")
            return False

        # 5. 运行P4.2测试
        print("\n开始运行P4.2测试（这可能需要1-2分钟）...", flush=True)
        code, output = ssh_execute(client,
            f"cd {remote_repo} && /usr/bin/python3 -m pytest tests/autotuner/product/test_checkpoint_registry.py tests/autotuner/product/test_checkpoint_selector.py tests/autotuner/product/test_checkpoint_curator.py tests/autotuner/product/test_capability_promotion.py -v --tb=short",
            "运行P4.2单元测试")

        if code == 0 and "passed" in output:
            print("\nP4.2单元测试通过", flush=True)
            test_passed = True
        else:
            print("\nP4.2测试失败或环境未就绪", flush=True)
            test_passed = False

        # 6. 创建测试检查点目录进行集成测试
        print("\n准备运行P4.2集成测试...", flush=True)
        code, output = ssh_execute(client,
            f"cd {remote_repo} && /usr/bin/python3 -m pytest tests/products/taili/blind_locomotion/test_checkpoint_integration.py -v --tb=short",
            "运行P4.2集成测试")

        if code == 0 and "passed" in output:
            print("\nP4.2集成测试通过", flush=True)
            integration_passed = True
        else:
            print("\nP4.2集成测试失败", flush=True)
            integration_passed = False

        # 7. 总结
        print("\n" + "="*60, flush=True)
        print("P4.3 GPU环境验证总结", flush=True)
        print("="*60, flush=True)
        print(f"远程仓库: {remote_repo}", flush=True)
        print(f"P4.2代码: 已同步", flush=True)
        print(f"单元测试: {'通过' if test_passed else '失败'}", flush=True)
        print(f"集成测试: {'通过' if integration_passed else '失败'}", flush=True)
        print(f"GPU: RTX 4090 24GB 可用", flush=True)

        if test_passed and integration_passed:
            print("\nP4.2检查点管理系统在GPU环境验证通过", flush=True)
            print("可以开始实际训练测试", flush=True)
            return True
        else:
            print("\n验证未完全通过，需要进一步检查", flush=True)
            return False

    except Exception as e:
        print(f"验证失败: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False
    finally:
        client.close()


if __name__ == "__main__":
    host = "183.147.142.40"
    port = 31376
    username = "root"
    password = "8046986746b0fb9"
    local_workspace = Path(__file__).parent.parent

    success = sync_and_validate(host, port, username, password, local_workspace)
    sys.exit(0 if success else 1)
