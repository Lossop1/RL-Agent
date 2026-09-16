"""把 torchvision 钉到 torch 所在的那个目录。

Isaac Sim 的镜像里有**两份** torchvision：kit 的 site-packages 一份、
`omni.isaac.ml_archive` 扩展的 `pip_prebundle` 一份，两者是**不同的 CUDA 构建**
（实测 `0.20.1+cu124` 与 `0.20.1+cu118`）。Isaac Sim 启动时会把 prebundle 那一份接进来，
于是谁先被导入就定终身：本包在导入阶段就会 `import skrl`，skrl 又会导入 torch，
等到 app 启动、`isaaclab_tasks` 这个 Kit 扩展去导 torchvision 时，拿到的就是 prebundle 那份，
和新版 torch 不同源，报 `operator torchvision::nms does not exist`
（再导一次则是 `partially initialized module 'torchvision' has no attribute 'extension'`）。

所以在导入 skrl 之前先把这一对定下来：**torch 从哪儿来，torchvision 就从哪儿来**。
只要先导入了同源的 torchvision，app 启动时就只会复用它，不会再从 prebundle 里拿。
"""
from __future__ import annotations

from pathlib import Path


def torchvision_alongside(torch_file: str | None) -> Path | None:
    """`torch_file` 旁边那份 torchvision 的目录；旁边没有就返回 None。

    只认一种布局：torch 在 `<dir>/torch/__init__.py`，torchvision 在 `<dir>/torchvision/__init__.py`。
    别的布局（zipimport、egg、被拆开的安装）一律返回 None，交给调用方按原样兜底，
    不在这里猜。
    """
    if not torch_file:
        return None
    directory = Path(torch_file).resolve().parent.parent / "torchvision"
    if (directory / "__init__.py").is_file():
        return directory
    return None


def pin_torchvision(*, torch_module=None) -> Path | None:
    """导入与 torch 同源的 torchvision，返回实际用的目录；无需干预时返回 None。

    已经导入过 torchvision 就什么都不做——那说明它已经定下来了，再改只会把 sys.modules
    里的模块和盘上的文件对不上。torch 旁边没有 torchvision 时也不做，返回 None。
    """
    import importlib
    import sys

    if "torchvision" in sys.modules:
        return None
    if torch_module is None:
        torch_module = importlib.import_module("torch")
    directory = torchvision_alongside(getattr(torch_module, "__file__", None))
    if directory is None:
        return None

    parent = str(directory.parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        importlib.import_module("torchvision")
    finally:
        if added:
            try:
                sys.path.remove(parent)
            except ValueError:  # pragma: no cover - 只可能被别的线程改过
                pass
    return directory
