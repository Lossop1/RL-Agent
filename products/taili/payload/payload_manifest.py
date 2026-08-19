"""Taili 盲态运行 payload 的清单和静态检查。"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
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
    return report


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
