"""`_torchvision_pair` 的用例。

这个模块修的是一个**导入顺序**缺陷：Isaac Sim 容器里有两份 CUDA 构建不同的 torchvision，
谁先被导入谁定终身；只要 skrl 先把 torch 导进来、而 torchvision 留到 app 启动时才导，
就会拿到不同源的那份，报 `torchvision::nms does not exist`。
所以这里除了测"挑目录"的纯逻辑，还有一条钉住调用位置的用例（见 `test_pin_runs_before_skrl_import`）：
顺序反了这个修复就失效，而失效的表现要到真实容器里才看得见，本地没有别的东西能拦住。
"""
from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path

import pytest

from products.taili.blind_locomotion import _torchvision_pair

PACKAGE_INIT = Path(_torchvision_pair.__file__).resolve().parent / "__init__.py"


def _make_sibling(tmp_path: Path, *, with_init: bool = True) -> Path:
    """造一个 `<root>/torch/__init__.py` + `<root>/torchvision/`，返回 torch 的 __file__。"""
    torch_dir = tmp_path / "torch"
    torch_dir.mkdir(parents=True)
    torch_init = torch_dir / "__init__.py"
    torch_init.write_text("", encoding="utf-8")
    vision_dir = tmp_path / "torchvision"
    vision_dir.mkdir(parents=True)
    if with_init:
        (vision_dir / "__init__.py").write_text("PAIR_MARKER = True\n", encoding="utf-8")
    return torch_init


def test_no_file_gives_nothing():
    assert _torchvision_pair.torchvision_alongside(None) is None
    assert _torchvision_pair.torchvision_alongside("") is None


def test_sibling_directory_is_found(tmp_path):
    torch_init = _make_sibling(tmp_path)
    assert _torchvision_pair.torchvision_alongside(str(torch_init)) == tmp_path / "torchvision"


def test_directory_without_init_is_not_a_package(tmp_path):
    """只有目录、没有 `__init__.py`，是命名空间包的碎块，不算数。"""
    torch_init = _make_sibling(tmp_path, with_init=False)
    assert _torchvision_pair.torchvision_alongside(str(torch_init)) is None


def test_missing_sibling_gives_nothing(tmp_path):
    torch_dir = tmp_path / "torch"
    torch_dir.mkdir()
    torch_init = torch_dir / "__init__.py"
    torch_init.write_text("", encoding="utf-8")
    assert _torchvision_pair.torchvision_alongside(str(torch_init)) is None


def test_parent_directory_is_not_searched(tmp_path):
    """只认兄弟目录。祖辈目录里那个同名目录不能被抓来用。"""
    nested_torch = _make_sibling(tmp_path)
    torch_init = tmp_path / "sub" / "torch" / "__init__.py"
    (tmp_path / "sub" / "torch").mkdir(parents=True)
    torch_init.write_text("", encoding="utf-8")

    assert _torchvision_pair.torchvision_alongside(str(nested_torch)) == tmp_path / "torchvision"
    assert _torchvision_pair.torchvision_alongside(str(torch_init)) is None


def test_already_imported_torchvision_is_left_alone(tmp_path, monkeypatch):
    """已经导入过就什么都不做——即便旁边真有一份，也不换掉已经在 sys.modules 里的那个。

    旁边放一份真的（不是空目录），否则这条用例在"早返回被删掉"时也会绿：
    删掉早返回后，没有兄弟目录一样返回 None。
    """
    torch_init = _make_sibling(tmp_path)
    sentinel = types.ModuleType("torchvision")
    monkeypatch.setitem(sys.modules, "torchvision", sentinel)
    before = list(sys.path)

    assert _torchvision_pair.pin_torchvision(torch_module=types.SimpleNamespace(__file__=str(torch_init))) is None
    assert sys.modules["torchvision"] is sentinel
    assert sys.path == before


def test_pin_imports_the_sibling_and_restores_sys_path(tmp_path, monkeypatch):
    """钉住之后：用的是兄弟目录那份，且临时插进 sys.path 的那一项要拿掉。"""
    torch_init = _make_sibling(tmp_path)
    monkeypatch.delitem(sys.modules, "torchvision", raising=False)
    before = list(sys.path)

    fake_torch = types.SimpleNamespace(__file__=str(torch_init))
    pinned = _torchvision_pair.pin_torchvision(torch_module=fake_torch)

    assert pinned == tmp_path / "torchvision"
    assert sys.modules["torchvision"].__file__ == str(tmp_path / "torchvision" / "__init__.py")
    assert getattr(sys.modules["torchvision"], "PAIR_MARKER", False) is True
    assert sys.path == before


def test_pin_without_sibling_does_not_import_anything(tmp_path, monkeypatch):
    """旁边没有 torchvision 时返回 None，且不留下任何已导入的 torchvision。"""
    torch_dir = tmp_path / "torch"
    torch_dir.mkdir()
    torch_init = torch_dir / "__init__.py"
    torch_init.write_text("", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "torchvision", raising=False)

    assert _torchvision_pair.pin_torchvision(torch_module=types.SimpleNamespace(__file__=str(torch_init))) is None
    assert "torchvision" not in sys.modules


def test_module_is_importable_on_its_own():
    """不依赖包内其它模块（本地 CPU 环境没有 torch/skrl，也应当能导）。"""
    module = importlib.import_module("products.taili.blind_locomotion._torchvision_pair")
    assert module is _torchvision_pair


def test_pin_runs_before_skrl_import():
    """`_register_blind_locomotion` 里，钉 torchvision 必须排在 `from skrl ...` 之前。

    顺序是这个修复的全部内容：skrl 先导 torch 就把配对机会让出去了。
    这里用 AST 钉住相对位置——只做行为测试的话，本地没有 Isaac Sim，
    顺序反了也测不出来（要真实容器才复现）。
    """
    tree = ast.parse(PACKAGE_INIT.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_register_blind_locomotion"
    )

    pin_lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pin_torchvision"
    ]
    skrl_lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("skrl")
    ]

    assert pin_lines, "_register_blind_locomotion 里没有再调用 pin_torchvision"
    assert skrl_lines, "找不到 skrl 的导入语句，这条用例的前提没了"
    assert min(pin_lines) < min(skrl_lines), (
        f"pin_torchvision 在 {min(pin_lines)} 行，skrl 导入在 {min(skrl_lines)} 行——顺序反了，配对就失效了"
    )


@pytest.mark.parametrize("name", ["pin_torchvision", "torchvision_alongside"])
def test_public_names_exist(name):
    assert callable(getattr(_torchvision_pair, name))
