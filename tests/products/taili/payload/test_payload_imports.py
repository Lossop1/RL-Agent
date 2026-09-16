"""载荷的导入自足性用例。

远端只把**载荷目录**放进 PYTHONPATH（`products/taili/ops/tune_orchestrator.py:242`
的 `export PYTHONPATH={payload}`），所以载荷里任何一个模块 `import autotuner.*`，
在远端都必然 `ModuleNotFoundError`——除非这个 import 有兜底、或者对应的模块也进了载荷。

2026-09-16 就是这样崩的：`train_taili.py` 里那句 `from autotuner.simulation.backend_factory
import create_backend` 没有任何兜底，容器里 app 起来了、torchvision 也修好了，
结果卡在 `ModuleNotFoundError: No module named 'autotuner'`，训练一步没跑。

这里用 AST 把这一类钉住：载荷出货的每个 .py，凡是引用 `autotuner` 的 import，
必须写在能接住 `ImportError`（或更宽的 `Exception`、裸 `except`）的 try 里。
"""
from __future__ import annotations

import ast
import importlib
import shutil
import sys
from pathlib import Path

import pytest

from products.taili.payload.payload_manifest import RUNTIME_PACKAGE, iter_payload_files

ROOT = Path(__file__).resolve().parents[4]
SIM_SOURCES = (
    "autotuner/simulation/simulator_protocol.py",
    "autotuner/simulation/isaaclab_adapter.py",
    "autotuner/simulation/backend_factory.py",
)

# 载荷里能在远端解析到的顶层包（其余顶层 import 要么是 IsaacLab/skrl/torch 这类运行时依赖，
# 要么是标准库）。这份清单只用来解释这一条用例盯的是什么，不参与判定。
EXTERNAL_TOP_LEVEL = {"autotuner"}


def _imports_autotuner(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] in EXTERNAL_TOP_LEVEL
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] in EXTERNAL_TOP_LEVEL for alias in node.names)
    return False


def _handles_import_error(handler: ast.ExceptHandler) -> bool:
    """接得住 `ImportError` 才算兜底；`Exception` 和裸 `except` 更宽，也算。"""
    if handler.type is None:
        return True
    names = []
    if isinstance(handler.type, ast.Tuple):
        names = [elt for elt in handler.type.elts]
    else:
        names = [handler.type]
    for name in names:
        if isinstance(name, ast.Name) and name.id in {"ImportError", "ModuleNotFoundError", "Exception"}:
            return True
    return False


def _is_packaged_import(node: ast.AST) -> bool:
    """是不是"从载荷内部拿"的 import：相对导入，或绝对导入 `taili_blind_runtime.*`。"""
    return (
        isinstance(node, ast.ImportFrom)
        and (node.level > 0 or (node.module or "").split(".")[0] == RUNTIME_PACKAGE)
    )


def _fallback_tries(tree: ast.AST) -> list[ast.Try]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try) and any(_handles_import_error(h) for h in node.handlers)
    ]


def _imports_naming(statements: list[ast.stmt], predicate) -> bool:
    return any(predicate(inner) for statement in statements for inner in ast.walk(statement))


def _import_violations(text: str, label: str) -> list[str]:
    """找出"远端一定会 ModuleNotFoundError"的那种 autotuner import。

    两条规矩，对应仓库里既有的那套写法（`try: 载荷内的 / except ImportError: autotuner 的`）：

    1. autotuner import 必须待在一个能接住 ImportError 的 try 里——要么在主干（说明它有兜底），
       要么在 `except ImportError` 分支里（它自己就是兜底）。裸在模块里或函数里的，判违规。
    2. 如果**兜底分支**用了 autotuner，那 try 主干就必须真的从载荷里拿东西
       （相对导入或 `taili_blind_runtime.*`）。不然这只是把崩溃往后挪了一行。
    """
    tree = ast.parse(text, filename=label)
    relative = label
    violations: list[str] = []

    guarded: set[int] = set()
    for node in _fallback_tries(tree):
        for statement in node.body:
            for inner in ast.walk(statement):
                guarded.add(id(inner))
        for handler in node.handlers:
            if not _handles_import_error(handler):
                continue
            for inner in ast.walk(handler):
                guarded.add(id(inner))

        uses_autotuner_as_fallback = any(
            _imports_autotuner(inner)
            for handler in node.handlers
            if _handles_import_error(handler)
            for inner in ast.walk(handler)
        )
        if uses_autotuner_as_fallback and not _imports_naming(node.body, _is_packaged_import):
            violations.append(
                f"{relative}:{node.lineno}: 兜底分支引用了 autotuner，"
                "但 try 主干没有从载荷里导入对应模块"
            )

    for node in ast.walk(tree):
        if _imports_autotuner(node) and id(node) not in guarded:
            violations.append(f"{relative}:{node.lineno}: {ast.unparse(node)}")
    return violations


def _payload_import_violations(source: Path) -> list[str]:
    return _import_violations(
        source.read_text(encoding="utf-8", errors="replace"),
        source.relative_to(ROOT).as_posix(),
    )


def _payload_python_sources() -> list[Path]:
    return [src for src, _ in iter_payload_files(ROOT) if src.suffix == ".py"]


def test_the_payload_ships_python_modules():
    """前提检查：清单真的列到了 .py，否则下面那条用例会在空集合上变绿。"""
    sources = _payload_python_sources()
    assert len(sources) > 20, f"载荷清单里的 .py 只有 {len(sources)} 个，像是清单读错了"


def test_every_autotuner_import_in_the_payload_is_guarded():
    violations = [
        violation for source in _payload_python_sources() for violation in _payload_import_violations(source)
    ]
    assert not violations, (
        "载荷出货的模块里有没兜底的 autotuner import——远端只有载荷在 PYTHONPATH 上，"
        "这些 import 一定会 ModuleNotFoundError：\n  " + "\n  ".join(violations)
    )


@pytest.mark.parametrize(
    "text",
    [
        # 就是 2026-09-16 崩在容器里的那一行
        "from autotuner.simulation.backend_factory import create_backend\n",
        # 挪进函数里也一样：远端照样 ModuleNotFoundError
        "def main():\n    from autotuner.simulation.backend_factory import create_backend\n",
        # 有 try，但接的是别的异常——ImportError 会漏出去
        "try:\n    from autotuner.x import y\nexcept ValueError:\n    y = None\n",
        # 兜底分支用 autotuner，主干却没从载荷里拿东西：等于把崩溃往后挪一行
        "try:\n    from elsewhere import y\nexcept ImportError:\n    from autotuner.x import y\n",
    ],
)
def test_the_guard_catches_unguarded_imports(text):
    """守着自己的守卫：没有这条，检查逻辑写错时上面那条用例会安静地变绿。"""
    assert _import_violations(text, "<合成模块>")


def test_the_guard_accepts_the_documented_fallback_pattern():
    """反过来：仓库里既有的"载荷优先、源码树兜底"写法必须被判为合规。"""
    for name in ("runtime_manifest.py", "train_taili.py", "physeval_blind.py", "diagnose_taili.py"):
        source = ROOT / "products" / "taili" / "blind_locomotion" / name
        violations = _payload_import_violations(source)
        assert not violations, f"{name} 仍被判违规：{violations}"


def test_the_payload_manifest_validates_clean():
    """清单自身过得去：任何一条 ERROR 都会让 build_payload 直接抛错。"""
    from products.taili.payload.payload_manifest import validate_manifest

    report = validate_manifest(ROOT)
    assert report.ok, "载荷清单校验不过：\n  " + "\n  ".join(report.errors)


def test_the_checkpoint_management_modules_ship_inside_the_payload():
    """P4.2 的四个模块都得在载荷里。

    2026-09-16 之前它们一个都没进清单：train_taili 的 `from .checkpoint_hook import ...`
    抛 ImportError 后被 `except Exception` 吞成一行 "checkpoint hook install failed"，
    于是记为已完成的 P4.2 在远端其实一次都没生效（smoke12 日志实录）。
    checkpoint_curator -> research_ledger 是它的传递依赖，漏了同样白搭。
    """
    destinations = {dst for _, dst in iter_payload_files(ROOT)}
    for module in ("checkpoint_hook", "checkpoint_integration", "checkpoint_curator", "research_ledger"):
        assert f"{RUNTIME_PACKAGE}/{module}.py" in destinations, (
            f"{module}.py 没进载荷清单；远端只有载荷在 PYTHONPATH 上，检查点管理会静默失效"
        )


@pytest.mark.parametrize(
    "text, expected_members",
    [
        # 双布局：兜底分支自己也在导入 -> 两支成一组，只要一支解析得出来就算数
        (
            "try:\n    from .hashing import x\nexcept ImportError:\n    from .execution_hashing import x\n",
            2,
        ),
        # 兜底分支只有日志 -> 不成组。这正是 checkpoint_hook 之前的样子：报出来才对
        (
            "try:\n    from .checkpoint_hook import x\nexcept Exception as e:\n    print(e)\n",
            0,
        ),
        # 裸的首方绝对导入，没有任何 try -> 不成组
        ("from autotuner.simulation.backend_factory import create_backend\n", 0),
    ],
)
def test_only_a_fallback_that_imports_counts_as_a_dual_layout_pair(text, expected_members):
    """守着"成组"的判据：兜底分支光打印日志不算兜底，否则 P4.2 那种静默降级会被放过。"""
    from products.taili.payload.payload_manifest import _dual_layout_groups

    groups = _dual_layout_groups(ast.parse(text), frozenset({"autotuner"}))
    members = max((len(group) for group in groups.values()), default=0)
    assert members == expected_members


def test_the_backend_factory_ships_inside_the_payload():
    destinations = {dst for _, dst in iter_payload_files(ROOT)}
    for module in ("simulator_protocol", "isaaclab_adapter", "backend_factory"):
        assert f"{RUNTIME_PACKAGE}/taili_sim/{module}.py" in destinations, (
            f"{module}.py 没有进载荷清单；训练入口要 create_backend，远端只有载荷在 PYTHONPATH 上"
        )


def test_backend_factory_uses_relative_imports_within_its_package():
    """三个后端模块之间必须是相对导入，否则搬进载荷后互相找不到。"""
    for relative in SIM_SOURCES:
        source = ROOT / relative
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        absolute = [
            f"{relative}:{node.lineno}: {ast.unparse(node)}"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("autotuner.")
        ]
        assert not absolute, "包内互相引用写成了绝对路径，搬进载荷就断了：\n  " + "\n  ".join(absolute)


def test_the_backend_trio_can_be_relocated_into_another_package(tmp_path):
    """把三个模块原样搬进另一个包目录后仍然导得进来——这正是载荷里发生的事。

    这条用真实的模块文件、真实地换一个包名导入，不是字符串比对。
    """
    pytest.importorskip("torch")
    pytest.importorskip("numpy")

    target = tmp_path / "relocated_sim"
    target.mkdir()
    (target / "__init__.py").write_text('"""relocated"""\n', encoding="utf-8")
    for relative in SIM_SOURCES:
        shutil.copyfile(ROOT / relative, target / Path(relative).name)

    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module("relocated_sim.backend_factory")
        assert callable(module.create_backend)
        assert module.IsaacLabAdapter.__module__ == "relocated_sim.isaaclab_adapter"
        assert module.SimulatorBackend.__module__ == "relocated_sim.simulator_protocol"
    finally:
        sys.path.remove(str(tmp_path))
        for name in list(sys.modules):
            if name == "relocated_sim" or name.startswith("relocated_sim."):
                del sys.modules[name]
