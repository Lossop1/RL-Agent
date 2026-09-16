"""Taili 盲态运行 payload 的清单和静态检查。"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
import posixpath
import re
import tempfile
from typing import Iterable

from autotuner.product import ContractResolutionError, get_product, resolve_product_contract
from products.taili.core import taili_geometry


RUNTIME_PACKAGE = "taili_blind_runtime"
PRODUCT = get_product("taili")
PRODUCT_ID = PRODUCT.product_id
TASK_IDS = PRODUCT.runtime_task_ids

ROOT = Path(__file__).resolve().parents[3]
PAYLOAD_DIR = Path(__file__).resolve().parent
ROBOT_ASSET_DIR = ROOT / "locomotion-console-ui" / "public" / "robot" / "taili_dog_description"
ROBOT_URDF_SOURCE = ROBOT_ASSET_DIR / "urdf" / "robot.urdf"
ROBOT_MESH_DIR = ROBOT_ASSET_DIR / "meshes"
SANITIZED_ROBOT_URDF = Path(tempfile.gettempdir()) / "taili_blind_payload_robot.urdf"

STATIC_FILES: tuple[tuple[str, str], ...] = (
    ("products/taili/blind_locomotion/__init__.py", f"{RUNTIME_PACKAGE}/__init__.py"),
    ("products/taili/blind_locomotion/_torchvision_pair.py", f"{RUNTIME_PACKAGE}/_torchvision_pair.py"),
    # 后端抽象（P6.1）随载荷发布：训练/诊断/验收入口都要 `create_backend`，
    # 而远端只有载荷在 PYTHONPATH 上（products/taili/ops/tune_orchestrator.py:242），
    # 少一个模块就 `ModuleNotFoundError: No module named 'autotuner'`，训练一步都跑不起来。
    # 这三个模块内部改成相对导入，所以搬进哪个包都能用。
    ("autotuner/simulation/simulator_protocol.py", f"{RUNTIME_PACKAGE}/taili_sim/simulator_protocol.py"),
    ("autotuner/simulation/isaaclab_adapter.py", f"{RUNTIME_PACKAGE}/taili_sim/isaaclab_adapter.py"),
    ("autotuner/simulation/backend_factory.py", f"{RUNTIME_PACKAGE}/taili_sim/backend_factory.py"),
    ("products/taili/blind_locomotion/blind_tp_env.py", f"{RUNTIME_PACKAGE}/blind_tp_env.py"),
    ("products/taili/blind_locomotion/blind_tp_env_cfg.py", f"{RUNTIME_PACKAGE}/blind_tp_env_cfg.py"),
    ("products/taili/blind_locomotion/taili_blind_env_cfg.py", f"{RUNTIME_PACKAGE}/taili_blind_env_cfg.py"),
    ("products/taili/blind_locomotion/taili_amp_env.py", f"{RUNTIME_PACKAGE}/taili_amp_env.py"),
    ("products/taili/blind_locomotion/taili_amp_env_cfg.py", f"{RUNTIME_PACKAGE}/taili_amp_env_cfg.py"),
    ("products/taili/blind_locomotion/assets/__init__.py", f"{RUNTIME_PACKAGE}/assets/__init__.py"),
    ("products/taili/blind_locomotion/assets/taili.py", f"{RUNTIME_PACKAGE}/assets/taili.py"),
    ("products/taili/blind_locomotion/terrain_perceiver_policy.py", f"{RUNTIME_PACKAGE}/terrain_perceiver_policy.py"),
    ("products/taili/blind_locomotion/terrain_perceiver_aux_patch.py", f"{RUNTIME_PACKAGE}/terrain_perceiver_aux_patch.py"),
    ("products/taili/blind_locomotion/telemetry_emit.py", f"{RUNTIME_PACKAGE}/telemetry_emit.py"),
    ("products/taili/blind_locomotion/telemetry_payloads.py", f"{RUNTIME_PACKAGE}/telemetry_payloads.py"),
    ("products/taili/blind_locomotion/runtime_manifest.py", f"{RUNTIME_PACKAGE}/runtime_manifest.py"),
    ("autotuner/execution/runtime.py", f"{RUNTIME_PACKAGE}/runtime_identity.py"),
    ("autotuner/execution/hashing.py", f"{RUNTIME_PACKAGE}/execution_hashing.py"),
    ("autotuner/execution/compatibility.py", f"{RUNTIME_PACKAGE}/resume_compatibility.py"),
    # P4.2 检查点管理。此前只有源码树在跑：训练入口 import 不到 checkpoint_hook，
    # 被 except 吞成一行 "checkpoint hook install failed"，于是"已完成"的 P4.2
    # 在远端其实一次都没生效（2026-09-16 smoke12 实测）。
    # checkpoint_curator 依赖 research_ledger，后者要 pydantic（载荷构建用宿主
    # python 有，训练容器是否有一并在 smoke 里验）。
    ("products/taili/blind_locomotion/checkpoint_hook.py", f"{RUNTIME_PACKAGE}/checkpoint_hook.py"),
    ("products/taili/blind_locomotion/checkpoint_integration.py", f"{RUNTIME_PACKAGE}/checkpoint_integration.py"),
    ("autotuner/product/checkpoint_curator.py", f"{RUNTIME_PACKAGE}/checkpoint_curator.py"),
    ("autotuner/research/research_ledger.py", f"{RUNTIME_PACKAGE}/research_ledger.py"),
    ("products/taili/blind_locomotion/launch_taili_train.py", f"{RUNTIME_PACKAGE}/launch_taili_train.py"),
    ("products/taili/blind_locomotion/train_taili.py", f"{RUNTIME_PACKAGE}/train_taili.py"),
    ("products/taili/blind_locomotion/calibrate_taili_gates.py", f"{RUNTIME_PACKAGE}/calibrate_taili_gates.py"),
    ("products/taili/blind_locomotion/diagnose_taili.py", f"{RUNTIME_PACKAGE}/diagnose_taili.py"),
    ("products/taili/blind_locomotion/diagnose_taili_cases.py", f"{RUNTIME_PACKAGE}/diagnose_taili_cases.py"),
    ("products/taili/blind_locomotion/stair_validation.py", f"{RUNTIME_PACKAGE}/stair_validation.py"),
    ("products/taili/blind_locomotion/taili_blind_config.py", f"{RUNTIME_PACKAGE}/taili_blind_config.py"),
    ("products/taili/blind_locomotion/taili_blind_config.yaml", f"{RUNTIME_PACKAGE}/taili_blind_config.yaml"),
    ("products/taili/blind_locomotion/agents/__init__.py", f"{RUNTIME_PACKAGE}/agents/__init__.py"),
    # 研究系统拥有机制合成与验证；训练 payload 只接收声明模型和安全解释器。
    ("autotuner/mechanisms/mechanism_specs.py", f"{RUNTIME_PACKAGE}/taili_core/mechanism_specs.py"),
    ("autotuner/mechanisms/mechanism_runtime.py", f"{RUNTIME_PACKAGE}/taili_core/mechanism_runtime.py"),
    ("products/taili/blind_locomotion/multi_motion_loader.py", f"{RUNTIME_PACKAGE}/multi_motion_loader.py"),
    ("products/taili/blind_locomotion/parametric_ref.py", f"{RUNTIME_PACKAGE}/parametric_ref.py"),
    ("products/taili/blind_locomotion/motions.py", f"{RUNTIME_PACKAGE}/motions.py"),
    ("products/taili/blind_locomotion/symmetry.py", f"{RUNTIME_PACKAGE}/symmetry.py"),
    ("products/taili/blind_locomotion/_symmetry_local.py", f"{RUNTIME_PACKAGE}/_symmetry_local.py"),
    # physeval 验收框架和纯评分器：payload 需要能在远端按规格自测。
    # physeval_blind 会直接 import acceptance_score，因此评分器既放包内，
    # 也放 payload 根目录；payload 根目录会进入 PYTHONPATH。
    ("products/taili/blind_locomotion/physeval_blind.py", f"{RUNTIME_PACKAGE}/physeval_blind.py"),
    ("products/taili/blind_locomotion/physeval_blind_e.py", f"{RUNTIME_PACKAGE}/physeval_blind_e.py"),
    ("products/taili/blind_locomotion/physeval_suite.py", f"{RUNTIME_PACKAGE}/physeval_suite.py"),
    ("products/taili/blind_locomotion/acceptance_score.py", f"{RUNTIME_PACKAGE}/acceptance_score.py"),
    ("products/taili/blind_locomotion/acceptance_score.py", "acceptance_score.py"),
    ("products/taili/blind_locomotion/acceptance_aggregate.py", f"{RUNTIME_PACKAGE}/acceptance_aggregate.py"),
    ("products/taili/blind_locomotion/acceptance_aggregate.py", "acceptance_aggregate.py"),
    ("tools/isaaclab_quad_diag_observation/isaaclab_quad_diag/__init__.py", f"{RUNTIME_PACKAGE}/isaaclab_quad_diag/__init__.py"),
    ("tools/isaaclab_quad_diag_observation/isaaclab_quad_diag/events.py", f"{RUNTIME_PACKAGE}/isaaclab_quad_diag/events.py"),
    ("tools/isaaclab_quad_diag_observation/isaaclab_quad_diag/metrics.py", f"{RUNTIME_PACKAGE}/isaaclab_quad_diag/metrics.py"),
    ("tools/isaaclab_quad_diag_observation/isaaclab_quad_diag/notes.py", f"{RUNTIME_PACKAGE}/isaaclab_quad_diag/notes.py"),
    ("tools/isaaclab_quad_diag_observation/isaaclab_quad_diag/schema.py", f"{RUNTIME_PACKAGE}/isaaclab_quad_diag/schema.py"),
    ("tools/isaaclab_quad_diag_observation/isaaclab_quad_diag/slices.py", f"{RUNTIME_PACKAGE}/isaaclab_quad_diag/slices.py"),
    ("tools/isaaclab_quad_diag_observation/isaaclab_quad_diag/util.py", f"{RUNTIME_PACKAGE}/isaaclab_quad_diag/util.py"),
    ("tools/isaaclab_quad_diag_observation/specs/taili.yaml", f"{RUNTIME_PACKAGE}/diagnostic_specs/taili.yaml"),
    ("tools/isaaclab_quad_diag_observation/suites/remote_probe.yaml", f"{RUNTIME_PACKAGE}/diagnostic_suites/remote_probe.yaml"),
    ("tools/isaaclab_quad_diag_observation/suites/direction_forward.yaml", f"{RUNTIME_PACKAGE}/diagnostic_suites/direction_forward.yaml"),
    ("tools/isaaclab_quad_diag_observation/suites/direction_backward.yaml", f"{RUNTIME_PACKAGE}/diagnostic_suites/direction_backward.yaml"),
    ("tools/isaaclab_quad_diag_observation/suites/direction_lateral.yaml", f"{RUNTIME_PACKAGE}/diagnostic_suites/direction_lateral.yaml"),
    ("tools/isaaclab_quad_diag_observation/suites/direction_yaw.yaml", f"{RUNTIME_PACKAGE}/diagnostic_suites/direction_yaw.yaml"),
    ("tools/isaaclab_quad_diag_observation/suites/terrain_probe.yaml", f"{RUNTIME_PACKAGE}/diagnostic_suites/terrain_probe.yaml"),
    ("tools/isaaclab_quad_diag_observation/suites/dr_probe.yaml", f"{RUNTIME_PACKAGE}/diagnostic_suites/dr_probe.yaml"),
    ("tools/isaaclab_quad_diag_observation/suites/push_probe.yaml", f"{RUNTIME_PACKAGE}/diagnostic_suites/push_probe.yaml"),
)

OPTIONAL_STATIC_PREFIXES = (
    "tools/isaaclab_quad_diag_observation/",
)

GENERATED_FILES = {
    # 源码树里 `autotuner/simulation/` 没有 __init__.py（靠 PEP 420 命名空间包），
    # 搬进载荷后是一个普通包目录，得自己带一个。
    f"{RUNTIME_PACKAGE}/taili_sim/__init__.py": (
        '"""打包进载荷的仿真器后端，源在 autotuner/simulation/。"""\n'
    ),
    "sitecustomize.py": (
        '"""当 payload 位于 PYTHONPATH 时自动注册 Taili 盲态任务。"""\n'
        "try:\n"
        f"    import {RUNTIME_PACKAGE}  # noqa: F401\n"
        "except Exception as exc:\n"
        "    print(f'[taili_blind_runtime] auto-import failed: {type(exc).__name__}: {exc}', flush=True)\n"
    ),
}

FORBIDDEN_TEXT = (
    "robot_lab.",
    "robot_lab.tasks.direct.taili_amp",
    "robot_lab.tasks.direct.taili_amp_blind",
    "/root/robot_lab/source/robot_lab/robot_lab/tasks/direct",
)


@dataclass
class ValidationReport:
    files: list[tuple[Path, str]] = field(default_factory=list)
    generated: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def iter_payload_files(root: Path = ROOT) -> Iterable[tuple[Path, str]]:
    for src, dst in STATIC_FILES:
        source = root / src
        if source.is_file() or not any(src.startswith(prefix) for prefix in OPTIONAL_STATIC_PREFIXES):
            yield source, dst

    taili_core = root / "products" / "taili" / "core"
    for src in sorted(taili_core.glob("*.py")):
        if src.name.startswith("test_") or src.name == "__pycache__":
            continue
        yield src, f"{RUNTIME_PACKAGE}/taili_core/{src.name}"

    clips = root / "products" / "taili" / "blind_locomotion" / "motions" / "clips"
    for src in sorted(clips.glob("*.npz")):
        yield src, f"{RUNTIME_PACKAGE}/motions/clips/{src.name}"

    if ROBOT_URDF_SOURCE.is_file():
        text = ROBOT_URDF_SOURCE.read_text(encoding="utf-8", errors="replace")
        text = text.replace('filename="../meshes/', 'filename="meshes/')
        SANITIZED_ROBOT_URDF.write_text(text, encoding="utf-8", newline="\n")
    yield SANITIZED_ROBOT_URDF, f"{RUNTIME_PACKAGE}/assets/robots/taili-dog/robot.urdf"
    for src in sorted(ROBOT_MESH_DIR.glob("*")):
        if src.is_file():
            yield src, f"{RUNTIME_PACKAGE}/assets/robots/taili-dog/meshes/{src.name}"


def validate_manifest(root: Path = ROOT) -> ValidationReport:
    report = ValidationReport(generated=dict(GENERATED_FILES))
    report.generated[f"{RUNTIME_PACKAGE}/product_contract.json"] = "resolved from product manifest"
    seen_dest: set[str] = set()

    try:
        resolve_product_contract(PRODUCT, root=root)
    except (ContractResolutionError, OSError, ValueError) as exc:
        report.errors.append(f"product contract invalid: {exc}")

    for src, dst in iter_payload_files(root):
        report.files.append((src, dst))
        if dst in seen_dest:
            report.errors.append(f"duplicate destination: {dst}")
        seen_dest.add(dst)
        if not src.is_file():
            rel = src.relative_to(root).as_posix() if src.is_absolute() and root in src.parents else src.as_posix()
            if any(rel.startswith(prefix) for prefix in OPTIONAL_STATIC_PREFIXES):
                report.warnings.append(f"optional diagnostic source missing: {src}")
            else:
                report.errors.append(f"missing source: {src}")
            continue
        if src.suffix in {".py", ".yaml", ".yml", ".md"}:
            text = src.read_text(encoding="utf-8", errors="replace")
            for needle in FORBIDDEN_TEXT:
                if needle in text and not (src.name == "taili_multicritic.py" and needle == "robot_lab."):
                    report.errors.append(f"forbidden text {needle!r} in {src}")
            if "robot_lab." in text:
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if "robot_lab." in line and not (src.name == "taili_multicritic.py" and "packaged payload" in line):
                        report.errors.append(f"unexpected robot_lab dependency in {src}:{line_no}: {line.strip()}")

    _validate_policy_contract(root, report)
    _validate_yaml_contract(root, report)
    _validate_registration_contract(root, report)
    _validate_robot_asset_contract(root, report)
    _validate_import_closure(root, report)
    return report


def _dual_layout_groups(
    tree: ast.AST, first_party: frozenset[str]
) -> dict[int, list[ast.ImportFrom]]:
    """把 ``try/except`` 里成对的导入归成组（导入节点 id -> 同组全部成员）。

    ``try: from .hashing import x / except ImportError: from .execution_hashing import x``
    是有意的双布局写法——源码树一套、载荷另一套（载荷会给模块改名），两支里只有一支
    解析得出来。判据是**兜底分支自己有没有导入**：只有日志的兜底不成组，那正是 P4.2
    在远端静默降级的样子，必须报出来。

    载荷内包名（``taili_blind_runtime.*``）与首方绝对导入（``from autotuner... import``）
    也算组员：前者是载荷那一支，后者是源码树那一支。
    """
    groups: dict[int, list[ast.ImportFrom]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(
            isinstance(inner, (ast.Import, ast.ImportFrom))
            for handler in node.handlers
            for inner in ast.walk(handler)
        ):
            continue
        members = [
            inner
            for stmt in [*node.body, *node.handlers]
            for inner in ast.walk(stmt)
            if isinstance(inner, ast.ImportFrom)
            and (
                inner.level
                or (inner.module or "").split(".")[0] in first_party
                or (inner.module or "").split(".")[0] == RUNTIME_PACKAGE
            )
        ]
        for member in members:
            groups[id(member)] = members
    return groups


def _validate_import_closure(root: Path, report: ValidationReport) -> None:
    """载荷文件引用的模块必须也在载荷里。

    STATIC_FILES 是手工白名单，漏一个模块既不会在构建时报错，也不会在导入时报错：
    调用点通常把 ImportError 吞成一行日志，于是功能在远端静默消失。已经咬过两次——
    P4.7 的 ``autotuner.simulation``（训练一步都起不来）、以及 P4.2 的
    ``checkpoint_hook``（``train_taili`` 的检查点管理只打印一句 install failed）。

    查三类：相对导入（``from .x import``）、首方包的裸绝对导入（载荷里没有
    ``autotuner``/``products`` 这些名字，``sitecustomize`` 只注册 ``taili_blind_runtime``，
    所以裸用必然在远端 ImportError）、以及漏打的包内模块。第三方导入（torch、skrl、
    isaaclab）不管，它们由运行环境提供。

    解析在**目标空间**做，不是源码树空间：有些模块进载荷时会改名
    （``autotuner/execution/runtime.py`` -> ``runtime_identity.py``），只按源码树
    解析会把 ``from .hashing import ...`` 这种注定在远端失败的写法判成通过。
    """
    dest_of: dict[str, Path] = {}
    for src, dst in report.files:
        dest_of[dst] = src
    dest_files = set(dest_of)
    dest_dirs = {posixpath.dirname(dst) for dst in dest_files}
    first_party = frozenset({"autotuner", "products", "tools"})

    def resolves(dest: str) -> bool:
        return dest in dest_files or dest.rstrip("/") in dest_dirs

    for dst in sorted(dest_files):
        src = dest_of[dst]
        if not dst.endswith(".py") or not src.is_file():
            continue
        try:
            tree = ast.parse(src.read_text(encoding="utf-8", errors="replace"), filename=str(src))
        except SyntaxError as exc:
            report.errors.append(f"cannot parse {src}: {exc}")
            continue
        groups = _dual_layout_groups(tree, first_party)
        base = posixpath.dirname(dst)

        def candidates(node: ast.ImportFrom) -> list[str]:
            """这条导入在目标空间里可能指向哪些模块路径（不含扩展名）。"""
            if not node.level:
                # `taili_blind_runtime.x.y` 在载荷里就是 `taili_blind_runtime/x/y.py`；
                # 首方名（autotuner/products）在载荷里根本不存在，必然解析不到。
                return [(node.module or "").replace(".", "/")]
            anchor = base
            for _ in range(node.level - 1):
                anchor = posixpath.dirname(anchor)
            if node.module is not None:
                modules = [node.module]
            else:
                # `from . import x`：x 可能是子模块，也可能是包 __init__ 里的名字。
                # 只有它确实对应一个模块文件时才当作模块来查，避免误报。
                modules = [
                    alias.name
                    for alias in node.names
                    if (src.parent / f"{alias.name}.py").is_file()
                    or (src.parent / alias.name / "__init__.py").is_file()
                ]
            out = []
            for module in modules:
                rel = module.replace(".", "/")
                if anchor:
                    rel = posixpath.join(anchor, rel)
                out.append(rel)
            return out

        def ok(node: ast.ImportFrom) -> bool:
            # 无候选 = 这条 `from . import` 说的不是模块，无从检查，不算缺口。
            return all(
                resolves(f"{rel}.py") or resolves(posixpath.join(rel, "__init__.py"))
                for rel in candidates(node)
            )

        checked: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or id(node) in checked:
                continue
            if node.level == 0:
                root = (node.module or "").split(".")[0]
                if root not in first_party and root != RUNTIME_PACKAGE:
                    continue  # 第三方，运行环境提供
                group = groups.get(id(node), [node])
                checked.update(id(member) for member in group)
                if any(ok(member) for member in group):
                    continue
                if root in first_party:
                    report.errors.append(
                        f"first-party absolute import unusable in payload: {dst} imports "
                        f"{node.module!r}; 载荷里没有 {node.module.split('.')[0]!r} 这个包名"
                        f"（sitecustomize 只注册 {RUNTIME_PACKAGE}），且没有兜底分支"
                    )
                else:
                    report.errors.append(
                        f"payload-internal import not packaged: {dst} imports {node.module!r} "
                        f"-> {', '.join(candidates(node))}.py is not in the payload"
                    )
                continue
            group = groups.get(id(node), [node])
            checked.update(id(member) for member in group)
            if any(ok(member) for member in group):
                continue
            if len(group) == 1:
                report.errors.append(
                    f"relative import not packaged: {dst} imports {node.module!r} "
                    f"-> {', '.join(candidates(node)) or '<no module target>'}.py "
                    f"is not in the payload"
                )
            else:
                branches = "; ".join(f"{m.module!r}->{', '.join(candidates(m))}" for m in group)
                report.errors.append(
                    f"relative import not packaged: {dst} has no resolvable branch "
                    f"({branches})"
                )


def _validate_policy_contract(root: Path, report: ValidationReport) -> None:
    policy = root / "products" / "taili" / "blind_locomotion" / "terrain_perceiver_policy.py"
    text = policy.read_text(encoding="utf-8", errors="replace")
    required = {
        "BODY_DIM = 57": "actor body must be 57",
        "HIST_LEN = 25": "history length must be 25",
        "TICK_DIM = 54": "history tick dim must be 54",
        "Z_DIM = 32": "terrain latent must be 32",
        "BODY_DIM + HIST_FLAT": "policy must slice body+history, not privileged obs",
    }
    for needle, msg in required.items():
        if needle not in text:
            report.errors.append(f"policy contract violation: {msg}")


def _validate_yaml_contract(root: Path, report: ValidationReport) -> None:
    cfg = root / "products" / "taili" / "blind_locomotion" / "taili_blind_config.yaml"
    text = cfg.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^\s*style_reward_weight:\s*([0-9.]+)", text, flags=re.MULTILINE)
    if not m:
        report.errors.append("style_reward_weight missing from taili_blind_config.yaml")
        return
    value = float(m.group(1))
    # 这里只做 sanity range 检查。早期策略书中的 0.3..0.5 只是草案起点，
    # 不是当前权威范围；当前权重按 physeval / taili_spec 验收目标调过。
    # 上限从 2.0 放宽到 4.0 后，AMP 是主要风格塑形项，因此 style_reward_weight
    # 可以合理超过 2.0；该保护只拒绝关闭通道或明显荒谬的数值。
    if not 0.0 < value <= 4.0:
        report.errors.append(f"style_reward_weight={value} outside sanity range (0, 4.0]")
    for key, expected in (
        ("nominal_base_h", taili_geometry.NOMINAL_BASE_HEIGHT),
        ("flat_move_height_target", taili_geometry.NOMINAL_BASE_HEIGHT),
    ):
        match = re.search(rf"^\s*{key}:\s*([0-9.]+)", text, flags=re.MULTILINE)
        if not match or abs(float(match.group(1)) - expected) > 1e-6:
            report.errors.append(f"{key} must match Taili geometry ({expected:.10f})")
    trace = re.search(
        r"^\s*terrain_collision_trace_decay_time:\s*([0-9.]+)",
        text,
        flags=re.MULTILINE,
    )
    if not trace or not 0.35 <= float(trace.group(1)) <= 0.45:
        report.errors.append("terrain_collision_trace_decay_time must cover about half a gait cycle")


def _validate_registration_contract(root: Path, report: ValidationReport) -> None:
    init_py = root / "products" / "taili" / "blind_locomotion" / "__init__.py"
    text = init_py.read_text(encoding="utf-8", errors="replace")
    if not TASK_IDS:
        report.errors.append("product manifest must declare at least one runtime task ID")
    else:
        try:
            tree = ast.parse(text, filename=str(init_py))
            source_task_ids: tuple[str, ...] | None = None
            for node in tree.body:
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                    value = node.value
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                    value = node.value
                else:
                    continue
                if any(isinstance(target, ast.Name) and target.id == "TASK_IDS" for target in targets):
                    literal = ast.literal_eval(value)
                    source_task_ids = tuple(str(item) for item in literal)
                    break
            if source_task_ids != tuple(TASK_IDS):
                report.errors.append(
                    "product runtime.task_ids do not match the adapter's registered TASK_IDS"
                )
        except (SyntaxError, ValueError, TypeError) as exc:
            report.errors.append(f"cannot validate adapter TASK_IDS: {type(exc).__name__}: {exc}")
    if ".blind_tp_env:TailiBlindTPEnv" not in text:
        report.errors.append("task entry point must resolve to package-local blind_tp_env:TailiBlindTPEnv")
    if ".taili_blind_env_cfg:TailiBlindEnvCfg" not in text:
        report.errors.append("env cfg entry point must resolve to package-local taili_blind_env_cfg")
    if "skrl_taili_blind_cfg.yaml" in text:
        report.errors.append("registration must use taili_blind_config.yaml as the source config")

    env_py = root / "products" / "taili" / "blind_locomotion" / "blind_tp_env.py"
    env_text = env_py.read_text(encoding="utf-8", errors="replace")
    required_terrain_calls = (
        "terrain_curriculum.loaded_support_height(",
        "terrain_curriculum.terrain_motion_credit(",
    )
    for call in required_terrain_calls:
        if call not in env_text:
            report.errors.append(f"blind_tp_env missing continuous terrain contract: {call}")
    forbidden_state_machine = (
        "compute_stair_event_progress(",
        "_stair_event_stage_memory",
        "_stair_event_lead_foot",
        "terrain_curriculum.continuous_terrain_height_drive(",
        "terrain_curriculum.continuous_terrain_layer_hold(",
        "terrain_curriculum.support_layer_split_quality(",
    )
    for marker in forbidden_state_machine:
        if marker in env_text:
            report.errors.append(f"blind_tp_env still contains stair state-machine marker: {marker}")
    if "0.014" in env_text:
        report.errors.append("blind_tp_env contains the obsolete 0.014m foot radius")
    for marker in (
        "taili_geometry.sole_clearance(",
        "taili_reward.transition_style_weight(",
        'self.extras["amp_style_scale"]',
    ):
        if marker not in env_text:
            report.errors.append(f"blind_tp_env missing geometry/transition contract: {marker}")


def _validate_robot_asset_contract(root: Path, report: ValidationReport) -> None:
    if not ROBOT_URDF_SOURCE.is_file():
        report.errors.append(f"missing robot URDF source: {ROBOT_URDF_SOURCE}")
        return
    if not ROBOT_MESH_DIR.is_dir():
        report.errors.append(f"missing robot mesh source directory: {ROBOT_MESH_DIR}")
        return
    text = ROBOT_URDF_SOURCE.read_text(encoding="utf-8", errors="replace")
    text = text.replace('filename="../meshes/', 'filename="meshes/')
    if "../meshes/" in text:
        report.errors.append("robot.urdf mesh paths must be payload-local meshes/... paths, not ../meshes/...")
    if 'filename="meshes/base_link.STL"' not in text:
        report.errors.append("robot.urdf must reference payload-local meshes/base_link.STL")
    radii = [float(value) for value in re.findall(r"<sphere\s+radius=\"([0-9.]+)\"", text)]
    if len(radii) != 4 or any(abs(value - taili_geometry.FOOT_RADIUS) > 1e-9 for value in radii):
        report.errors.append(
            f"robot.urdf foot spheres must all use radius {taili_geometry.FOOT_RADIUS:.3f}m"
        )


def manifest_summary(root: Path = ROOT) -> str:
    report = validate_manifest(root)
    lines = [
        f"runtime_package={RUNTIME_PACKAGE}",
        f"files={len(report.files)} generated={len(report.generated)}",
        f"errors={len(report.errors)} warnings={len(report.warnings)}",
    ]
    lines.extend(f"ERROR: {e}" for e in report.errors)
    lines.extend(f"WARN: {w}" for w in report.warnings)
    return "\n".join(lines)


if __name__ == "__main__":
    print(manifest_summary())
    raise SystemExit(0 if validate_manifest().ok else 1)
