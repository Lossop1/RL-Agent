"""产品无关的可复用机制目录与组合校验接口。

目录只描述机制的作用、依赖和适配方式。具体机器人的实现由
``product://`` 角色定位，组合由产品清单声明，不在系统包中内置。
"""
from autotuner.framework_library.catalog import (  # noqa: F401
    FrameworkComponent,
    FrameworkComposition,
    CATALOG,
    COMPOSITIONS,
    get_component,
    get_composition,
    get_compositions,
    validate_composition,
    adapt_plan,
)
