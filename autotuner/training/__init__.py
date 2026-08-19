"""旧训练命令的兼容命名空间。

产品训练命令位于 ``products/<product>/ops``，通用远程能力位于
``autotuner.infrastructure``；本包只用于保持既有 CLI 与外部导入可用。
"""

from autotuner.infrastructure.remote import RemoteSSH, get_default_remote

__all__ = ["RemoteSSH", "get_default_remote"]
