"""P4.3 GPU环境远程验证脚本"""

import paramiko
import sys
import time


def ssh_connect_and_test(host, port, username, password):
    """连接SSH并执行GPU环境测试"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"正在连接到 {username}@{host}:{port}...", flush=True)
        client.connect(host, port=port, username=username, password=password, timeout=30)
        print("SSH连接成功", flush=True)

        # 测试基本环境
        print("\n=== 基本环境检查 ===", flush=True)
        stdin, stdout, stderr = client.exec_command("pwd && whoami && python --version 2>&1")
        output = stdout.read().decode('utf-8', errors='ignore')
        error = stderr.read().decode('utf-8', errors='ignore')
        print(output, flush=True)
        if error:
            print(f"stderr: {error}", flush=True)

        # 测试GPU
        print("\n=== GPU检查 ===", flush=True)
        stdin, stdout, stderr = client.exec_command("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>&1")
        output = stdout.read().decode('utf-8', errors='ignore')
        print(output, flush=True)

        # 检查工作目录结构
        print("\n=== 工作目录检查 ===", flush=True)
        stdin, stdout, stderr = client.exec_command("ls -lah /root/ 2>&1 | head -20")
        output = stdout.read().decode('utf-8', errors='ignore')
        print(output, flush=True)

        # 检查是否有现有的训练运行目录
        print("\n=== 训练运行目录检查 ===", flush=True)
        stdin, stdout, stderr = client.exec_command("find /root -maxdepth 3 -name 'runtime_manifest.json' 2>/dev/null | head -5")
        output = stdout.read().decode('utf-8', errors='ignore')
        if output.strip():
            print(f"发现现有运行目录:\n{output}", flush=True)
        else:
            print("未发现现有运行目录", flush=True)

        # 检查Python环境和依赖
        print("\n=== Python环境检查 ===", flush=True)
        stdin, stdout, stderr = client.exec_command("python -c 'import torch; import skrl; print(f\"torch={torch.__version__}, skrl={skrl.__version__}\")' 2>&1")
        output = stdout.read().decode('utf-8', errors='ignore')
        print(output, flush=True)

        print("\n=== 连接测试完成 ===", flush=True)
        return True

    except paramiko.AuthenticationException:
        print("认证失败: 用户名或密码错误", flush=True)
        return False
    except paramiko.SSHException as e:
        print(f"SSH错误: {e}", flush=True)
        return False
    except Exception as e:
        print(f"连接失败: {e}", flush=True)
        return False
    finally:
        client.close()


if __name__ == "__main__":
    host = "183.147.142.40"
    port = 31376
    username = "root"
    password = "8046986746b0fb9"

    success = ssh_connect_and_test(host, port, username, password)
    sys.exit(0 if success else 1)
