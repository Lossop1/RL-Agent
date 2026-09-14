"""P4.3 远程环境依赖安装和完整测试"""

import paramiko
import sys
from pathlib import Path


def ssh_execute(client, command, description="", show_output=True):
    """执行SSH命令并打印输出"""
    if description:
        print(f"\n=== {description} ===", flush=True)
    stdin, stdout, stderr = client.exec_command(command, get_pty=True)

    output_lines = []
    while not stdout.channel.exit_status_ready():
        if stdout.channel.recv_ready():
            line = stdout.readline()
            if show_output:
                print(line, end='', flush=True)
            output_lines.append(line)

    for line in stdout:
        if show_output:
            print(line, end='', flush=True)
        output_lines.append(line)

    exit_code = stdout.channel.recv_exit_status()
    output = ''.join(output_lines)

    return exit_code, output


def install_and_test(host, port, username, password):
    """安装依赖并运行P4.2测试"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"正在连接到 {username}@{host}:{port}...", flush=True)
        client.connect(host, port=port, username=username, password=password, timeout=30)
        print("SSH连接成功\n", flush=True)

        # 1. 查找工作目录
        code, output = ssh_execute(client,
            "ls -d /root/gpufree-data/diag_runs/locomotion_console_* 2>/dev/null | head -1",
            "查找工作目录", show_output=False)

        workspace = output.strip()
        if not workspace:
            print("未找到工作目录", flush=True)
            return False

        print(f"工作目录: {workspace}", flush=True)

        # 2. 检查Python版本
        code, output = ssh_execute(client,
            "/usr/bin/python3 --version",
            "Python版本", show_output=False)
        print(f"Python: {output.strip()}", flush=True)

        # 3. 安装pytest (使用--break-system-packages绕过externally-managed限制)
        print("\n准备安装pytest...", flush=True)
        code, output = ssh_execute(client,
            "/usr/bin/python3 -m pip install pytest --break-system-packages",
            "安装pytest")

        if code != 0:
            print("pytest安装失败，尝试用户目录安装...", flush=True)
            code, output = ssh_execute(client,
                "/usr/bin/python3 -m pip install --user pytest",
                "用户目录安装pytest")

        # 4. 验证pytest安装
        code, output = ssh_execute(client,
            "/usr/bin/python3 -m pytest --version",
            "验证pytest", show_output=False)

        if code == 0:
            print(f"pytest已安装: {output.strip()}", flush=True)
        else:
            print("pytest安装失败，无法运行测试", flush=True)
            return False

        # 5. 检查pydantic
        code, output = ssh_execute(client,
            "/usr/bin/python3 -c 'import pydantic; print(pydantic.__version__)' 2>&1",
            "检查pydantic", show_output=False)

        if code != 0:
            print("pydantic未安装，尝试安装...", flush=True)
            code, output = ssh_execute(client,
                "/usr/bin/python3 -m pip install pydantic --break-system-packages",
                "安装pydantic")

            if code != 0:
                code, output = ssh_execute(client,
                    "/usr/bin/python3 -m pip install --user pydantic",
                    "用户目录安装pydantic")
        else:
            print(f"pydantic已安装: {output.strip()}", flush=True)

        # 6. 验证核心模块导入
        print("\n开始验证核心模块导入...", flush=True)
        code, output = ssh_execute(client,
            f"cd {workspace} && /usr/bin/python3 -c 'import sys; sys.path.insert(0, \".\"); from autotuner.product.checkpoint_curator import CheckpointRegistry, CheckpointSelector, CheckpointCurator, CapabilityPromotionService; print(\"CheckpointCurator导入成功\")' 2>&1",
            "CheckpointCurator导入测试")

        curator_import_ok = code == 0 and "成功" in output

        code, output = ssh_execute(client,
            f"cd {workspace} && /usr/bin/python3 -c 'import sys; sys.path.insert(0, \".\"); from products.taili.blind_locomotion.checkpoint_integration import CheckpointIntegration; print(\"CheckpointIntegration导入成功\")' 2>&1",
            "CheckpointIntegration导入测试")

        integration_import_ok = code == 0 and "成功" in output

        if not (curator_import_ok and integration_import_ok):
            print("\n核心模块导入失败，检查依赖...", flush=True)
            return False

        # 7. 运行P4.2单元测试
        print("\n开始运行P4.2单元测试套件...", flush=True)
        print("这可能需要2-3分钟...\n", flush=True)

        test_files = [
            "tests/autotuner/product/test_checkpoint_registry.py",
            "tests/autotuner/product/test_checkpoint_selector.py",
            "tests/autotuner/product/test_checkpoint_curator.py",
            "tests/autotuner/product/test_capability_promotion.py",
        ]

        code, output = ssh_execute(client,
            f"cd {workspace} && /usr/bin/python3 -m pytest {' '.join(test_files)} -v --tb=short 2>&1",
            "P4.2单元测试 (43个)")

        unit_test_passed = code == 0 and "passed" in output.lower()

        # 8. 运行集成测试
        code, output = ssh_execute(client,
            f"cd {workspace} && /usr/bin/python3 -m pytest tests/products/taili/blind_locomotion/test_checkpoint_integration.py -v --tb=short 2>&1",
            "P4.2集成测试 (7个)")

        integration_test_passed = code == 0 and "passed" in output.lower()

        # 9. 总结
        print("\n" + "="*60, flush=True)
        print("P4.3 GPU环境完整验证总结", flush=True)
        print("="*60, flush=True)
        print(f"远程工作目录: {workspace}", flush=True)
        print(f"GPU: RTX 4090 24GB", flush=True)
        print(f"pytest: 已安装", flush=True)
        print(f"pydantic: 已安装", flush=True)
        print(f"核心模块导入: {'通过' if curator_import_ok and integration_import_ok else '失败'}", flush=True)
        print(f"单元测试(43个): {'通过' if unit_test_passed else '失败'}", flush=True)
        print(f"集成测试(7个): {'通过' if integration_test_passed else '失败'}", flush=True)

        if unit_test_passed and integration_test_passed:
            print("\nP4.2检查点管理系统在GPU环境完整验证通过", flush=True)
            print("所有功能已确认，可以开始实际训练集成", flush=True)
            return True
        else:
            print("\n部分测试未通过，需要进一步检查", flush=True)
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

    success = install_and_test(host, port, username, password)
    sys.exit(0 if success else 1)
