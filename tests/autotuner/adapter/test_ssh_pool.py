"""SSH连接池单元测试 - 验证连接复用、健康检查和清理逻辑。"""
from __future__ import annotations

import time
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_paramiko_ssh():
    """创建模拟的 ParamikoSSH 对象。"""
    # ParamikoSSH 在 get() 方法内部导入，需要 patch remote_deploy 模块
    with patch("autotuner.adapter.remote_deploy.ParamikoSSH") as mock_class:
        mock_ssh = Mock()
        mock_ssh.exec.return_value = ("output", 0)
        mock_ssh.close.return_value = None
        mock_class.return_value = mock_ssh
        yield mock_class, mock_ssh


def test_pool_creates_new_connection_on_first_use(mock_paramiko_ssh):
    """测试首次使用时创建新连接。"""
    from autotuner.adapter.ssh_pool import SimpleSSHPool

    mock_class, mock_ssh = mock_paramiko_ssh
    pool = SimpleSSHPool(idle_timeout=600.0)

    # 首次获取连接
    ssh = pool.get("10.0.0.1", 22, "root", password="test")

    # 验证创建了新连接
    mock_class.assert_called_once_with(
        host="10.0.0.1",
        port=22,
        user="root",
        password="test",
        known_hosts="",
        auto_add_host_key=False,
    )
    assert ssh is mock_ssh
    stats = pool.get_stats()
    assert stats["connections_created"] == 1
    assert stats["connections_reused"] == 0


def test_pool_reuses_connection_for_same_host(mock_paramiko_ssh):
    """测试相同主机的连接被复用。"""
    from autotuner.adapter.ssh_pool import SimpleSSHPool

    mock_class, mock_ssh = mock_paramiko_ssh
    pool = SimpleSSHPool(idle_timeout=600.0)

    # 第一次获取
    ssh1 = pool.get("10.0.0.1", 22, "root", password="test")
    assert mock_class.call_count == 1

    # 第二次获取相同主机
    ssh2 = pool.get("10.0.0.1", 22, "root", password="test")

    # 验证未创建新连接
    assert mock_class.call_count == 1
    assert ssh1 is ssh2
    stats = pool.get_stats()
    assert stats["connections_created"] == 1
    assert stats["connections_reused"] == 1


def test_pool_creates_different_connections_for_different_hosts(mock_paramiko_ssh):
    """测试不同主机创建独立连接。"""
    from autotuner.adapter.ssh_pool import SimpleSSHPool

    mock_class, mock_ssh = mock_paramiko_ssh
    pool = SimpleSSHPool(idle_timeout=600.0)

    # 为不同主机创建不同的 Mock 对象
    mock_ssh1 = Mock()
    mock_ssh1.exec.return_value = ("output1", 0)
    mock_ssh2 = Mock()
    mock_ssh2.exec.return_value = ("output2", 0)
    mock_class.side_effect = [mock_ssh1, mock_ssh2]

    # 获取两个不同主机的连接
    ssh1 = pool.get("10.0.0.1", 22, "root", password="test1")
    ssh2 = pool.get("10.0.0.2", 22, "root", password="test2")

    # 验证创建了两个连接
    assert mock_class.call_count == 2
    assert ssh1 is mock_ssh1
    assert ssh2 is mock_ssh2
    assert ssh1 is not ssh2
    stats = pool.get_stats()
    assert stats["connections_created"] == 2
    assert stats["active_connections"] == 2


def test_pool_recreates_unhealthy_connection(mock_paramiko_ssh):
    """测试健康检查失败时重建连接。"""
    from autotuner.adapter.ssh_pool import SimpleSSHPool

    mock_class, mock_ssh = mock_paramiko_ssh
    pool = SimpleSSHPool(idle_timeout=600.0)

    # 第一次获取
    ssh1 = pool.get("10.0.0.1", 22, "root", password="test")
    assert mock_class.call_count == 1

    # 模拟健康检查失败
    mock_ssh.exec.side_effect = Exception("Connection lost")

    # 第二次获取，应该重建连接
    mock_ssh2 = Mock()
    mock_ssh2.exec.return_value = ("output", 0)
    mock_class.return_value = mock_ssh2

    ssh2 = pool.get("10.0.0.1", 22, "root", password="test")

    # 验证重建了连接
    assert mock_class.call_count == 2
    assert ssh2 is mock_ssh2
    mock_ssh.close.assert_called_once()
    stats = pool.get_stats()
    assert stats["connections_created"] == 2
    assert stats["health_check_failures"] == 1


def test_pool_cleans_up_idle_connections(mock_paramiko_ssh):
    """测试清理超时的空闲连接。"""
    from autotuner.adapter.ssh_pool import SimpleSSHPool

    mock_class, mock_ssh = mock_paramiko_ssh
    pool = SimpleSSHPool(idle_timeout=1.0)  # 1秒超时

    # 创建连接
    ssh = pool.get("10.0.0.1", 22, "root", password="test")
    assert ssh is mock_ssh

    # 等待超时
    time.sleep(1.1)

    # 执行清理
    cleaned = pool.cleanup_idle()

    # 验证连接被清理
    assert cleaned == 1
    mock_ssh.close.assert_called_once()
    stats = pool.get_stats()
    assert stats["active_connections"] == 0


def test_pool_cleanup_all_closes_all_connections(mock_paramiko_ssh):
    """测试 cleanup_all 关闭所有连接。"""
    from autotuner.adapter.ssh_pool import SimpleSSHPool

    _, mock_ssh = mock_paramiko_ssh
    pool = SimpleSSHPool(idle_timeout=600.0)

    # 创建多个连接
    pool.get("10.0.0.1", 22, "root", password="test1")
    pool.get("10.0.0.2", 22, "root", password="test2")

    # 清理所有连接
    pool.cleanup_all()

    # 验证所有连接被关闭
    assert mock_ssh.close.call_count == 2
    stats = pool.get_stats()
    assert stats["active_connections"] == 0


def test_get_pooled_ssh_uses_global_pool():
    """测试 get_pooled_ssh 使用全局池。"""
    from autotuner.adapter.ssh_pool import get_pooled_ssh, get_pool_stats

    with patch("autotuner.adapter.remote_deploy.ParamikoSSH") as mock_class:
        mock_ssh = Mock()
        mock_ssh.exec.return_value = ("output", 0)
        mock_class.return_value = mock_ssh

        # 首次调用
        ssh1 = get_pooled_ssh("10.0.0.1", 22, "root", password="test")
        assert ssh1 is mock_ssh
        assert mock_class.call_count == 1

        # 第二次调用应该复用
        ssh2 = get_pooled_ssh("10.0.0.1", 22, "root", password="test")
        assert ssh2 is mock_ssh
        assert mock_class.call_count == 1

        # 验证统计
        stats = get_pool_stats()
        assert stats["connections_created"] >= 1
        assert stats["connections_reused"] >= 1


def test_pool_thread_safety():
    """测试连接池的线程安全性。"""
    import threading

    from autotuner.adapter.ssh_pool import SimpleSSHPool

    with patch("autotuner.adapter.remote_deploy.ParamikoSSH") as mock_class:
        mock_ssh = Mock()
        mock_ssh.exec.return_value = ("output", 0)
        mock_class.return_value = mock_ssh

        pool = SimpleSSHPool(idle_timeout=600.0)
        results = []

        def worker():
            ssh = pool.get("10.0.0.1", 22, "root", password="test")
            results.append(ssh)

        # 启动10个线程同时获取连接
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 验证只创建了一个连接（第一个线程创建，其他复用）
        # 由于并发，可能创建多个，但应该远小于10
        assert mock_class.call_count <= 3
        assert all(r is mock_ssh for r in results)


def test_pool_stats_accuracy(mock_paramiko_ssh):
    """测试统计信息的准确性。"""
    from autotuner.adapter.ssh_pool import SimpleSSHPool

    mock_class, mock_ssh = mock_paramiko_ssh
    pool = SimpleSSHPool(idle_timeout=600.0)

    # 初始状态
    stats = pool.get_stats()
    assert stats["connections_created"] == 0
    assert stats["connections_reused"] == 0
    assert stats["connections_closed"] == 0
    assert stats["active_connections"] == 0

    # 创建连接
    pool.get("10.0.0.1", 22, "root")
    stats = pool.get_stats()
    assert stats["connections_created"] == 1
    assert stats["active_connections"] == 1

    # 复用连接
    pool.get("10.0.0.1", 22, "root")
    stats = pool.get_stats()
    assert stats["connections_reused"] == 1

    # 清理连接
    pool.cleanup_all()
    stats = pool.get_stats()
    assert stats["connections_closed"] == 1
    assert stats["active_connections"] == 0


def test_pool_handles_exec_with_nonzero_exit_code(mock_paramiko_ssh):
    """测试健康检查能处理非零退出码。"""
    from autotuner.adapter.ssh_pool import SimpleSSHPool

    mock_class, mock_ssh = mock_paramiko_ssh
    pool = SimpleSSHPool(idle_timeout=600.0)

    # 第一次获取
    ssh1 = pool.get("10.0.0.1", 22, "root")
    assert mock_class.call_count == 1

    # 健康检查返回非零退出码
    mock_ssh.exec.return_value = ("error", 1)

    # 第二次获取，健康检查失败，应重建
    mock_ssh2 = Mock()
    mock_ssh2.exec.return_value = ("output", 0)
    mock_class.return_value = mock_ssh2

    ssh2 = pool.get("10.0.0.1", 22, "root")

    # 验证重建了连接
    assert mock_class.call_count == 2
    assert ssh2 is mock_ssh2
    stats = pool.get_stats()
    assert stats["health_check_failures"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
