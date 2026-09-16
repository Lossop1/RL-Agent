"""交给 skrl 的必须是原生 IsaacLab 环境，不是 SimulatorBackend 适配器。

背景（2026-09-13 的 commit c781480 引入的回归）：那次把 4 个训练/评估入口里的

    env = gym.make(args.task, cfg=cfg, render_mode=None)

换成了

    env = create_backend(args.task, cfg, backend_type="isaaclab")

但下一行 `env = SkrlVecEnvWrapper(env, ml_framework="torch")` 原样留着。create_backend 返回的是
SimulatorBackend 适配器（IsaacLabAdapter），而 isaaclab_rl.skrl.SkrlVecEnvWrapper 要的是原生
IsaacLab 环境：它按 `env.unwrapped` 判断底层类型，并把 step/reset 直接发给传进来的那个对象。
适配器的 reset 只返回观测、step 返回四元组，与 skrl 期望的 `(obs, info)` / 五元组不同。

结果：这 4 个入口从 2026-09-13 起一个都跑不起来，直到 2026-09-16 在 3060 机器上第一次端到端
跑训练才暴露（报错点是 isaaclab_rl/skrl.py:66，`'IsaacLabAdapter' object has no attribute
'unwrapped'`）。当时这一步的"验证"是 simulation 层 38 个测试全绿——它们只测适配器自己，
从没测过这条接线。

这个测试补的就是那一段：静态检查每个入口交给 SkrlVecEnvWrapper 的第一个参数，要么是
`gym.make(...)` 的结果，要么是 `create_backend(...)` 那个变量身上的 `.unwrapped`。
两种写法都在下面的表里记着，将来谁把某个入口迁到一半就会在这里变红。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
ENTRY_DIR = ROOT / "products" / "taili" / "blind_locomotion"

# 会交给 skrl 的全部入口。该用哪种写法由文件自己决定，见下面的判断：
#   用了 create_backend 的 —— 必须取 .unwrapped（适配器不是 skrl 要的那种对象）
#   没用 create_backend 的 —— 保持原来的 gym.make，变量传进去即可
# calibrate_taili_gates.py 属于后者：P6.1 没有迁移它，它一直是原样。
ENTRIES = [
    "train_taili.py",
    "physeval_blind.py",
    "physeval_blind_e.py",
    "diagnose_taili.py",
    "calibrate_taili_gates.py",
]


def _dotted_name(node: ast.AST) -> str | None:
    """把 a.b.c 这种名字/属性链还原成 "a.b.c"；不是这种形状就返回 None。"""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _call_target(node: ast.AST) -> str | None:
    """被调用者的点号名字，例如 create_backend(...) 的 "create_backend"。"""
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return None


def _vars_assigned_from(tree: ast.AST, call_name: str) -> set[str]:
    """从某个调用赋值出来的变量名，例如全部 `x = create_backend(...)` 里的 x。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if _call_target(node.value) != call_name:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


@pytest.mark.parametrize("filename", ENTRIES)
def test_skrl_wrapper_receives_the_raw_environment(filename: str) -> None:
    tree = ast.parse((ENTRY_DIR / filename).read_text(encoding="utf-8"))

    wrappers = [node for node in ast.walk(tree) if _call_target(node) == "SkrlVecEnvWrapper"]
    assert wrappers, f"{filename} 里找不到 SkrlVecEnvWrapper(...) 调用"

    # 这个入口是否已经迁到后端抽象上。迁了就必须取 .unwrapped 交出去；
    # 没迁的保持 gym.make，直接把变量交给 skrl 就是对的。
    backends = _vars_assigned_from(tree, "create_backend")
    if backends:
        expected = "create_backend(...) 返回值上的 .unwrapped（适配器不是 skrl 要的那种对象）"

        def ok(arg: ast.AST) -> bool:
            return (
                isinstance(arg, ast.Attribute)
                and arg.attr == "unwrapped"
                and isinstance(arg.value, ast.Name)
                and arg.value.id in backends
            )
    else:
        expected = "gym.make(...) 出来的环境"
        gym_vars = _vars_assigned_from(tree, "gym.make")

        def ok(arg: ast.AST) -> bool:
            if _call_target(arg) == "gym.make":
                return True
            return isinstance(arg, ast.Name) and arg.id in gym_vars

    for call in wrappers:
        assert call.args, f"{filename}:{call.lineno} SkrlVecEnvWrapper 没有位置参数"
        assert ok(call.args[0]), (
            f"{filename}:{call.lineno} 交给 skrl 的是 {ast.unparse(call.args[0])}，"
            f"应当是{expected}"
        )
