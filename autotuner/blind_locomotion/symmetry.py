"""对称增强的兼容导出。

环境入口固定从本模块导入；具体实现放在 ``_symmetry_local``，避免改变旧 payload
中的模块路径，同时保证启用镜像增强时任务可以正常构造。
"""
from ._symmetry_local import (  # noqa: F401
    set_active,
    set_mirrors,
    patch_memory_sample_all,
    build_obs_mirror,
    build_action_mirror,
    mirror_obs,
    mirror_action,
)
