"""旧训练命令的兼容命名空间。

新代码应使用 ``autotuner.taili_ops`` 和 ``autotuner.infrastructure``；本包只用于
保持既有 CLI 与外部导入可用。
"""

from autotuner.infrastructure.remote import RemoteSSH, get_default_remote

__all__ = ["RemoteSSH", "get_default_remote"]

__all__ = ["RemoteSSH", "get_default_remote"]
