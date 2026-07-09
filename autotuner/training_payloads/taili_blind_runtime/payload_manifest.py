"""Manifest and static checks for the Taili blind runtime payload."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import tempfile
from typing import Iterable


RUNTIME_PACKAGE = "taili_blind_runtime"
TASK_IDS = (
    "RobotLab-Isaac-Taili-Blind-Direct-v0",
    "RobotLab-Isaac-Taili-AMP-Blind-Direct-v0",
)

ROOT = Path(__file__).resolve().parents[3]
PAYLOAD_DIR = Path(__file__).resolve().parent
ROBOT_ASSET_DIR = ROOT / "locomotion-console-ui" / "public" / "robot" / "taili_dog_description"
ROBOT_URDF_SOURCE = ROBOT_ASSET_DIR / "urdf" / "robot.urdf"
ROBOT_MESH_DIR = ROBOT_ASSET_DIR / "meshes"
SANITIZED_ROBOT_URDF = Path(tempfile.gettempdir()) / "taili_blind_payload_robot.urdf"

STATIC_FILES: tuple[tuple[str, str], ...] = (
    ("autotuner/blind_locomotion/__init__.py", f"{RUNTIME_PACKAGE}/__init__.py"),
    ("autotuner/blind_locomotion/blind_tp_env.py", f"{RUNTIME_PACKAGE}/blind_tp_env.py"),
    ("autotuner/blind_locomotion/blind_tp_env_cfg.py", f"{RUNTIME_PACKAGE}/blind_tp_env_cfg.py"),
    ("autotuner/blind_locomotion/taili_blind_env_cfg.py", f"{RUNTIME_PACKAGE}/taili_blind_env_cfg.py"),
    ("autotuner/blind_locomotion/taili_amp_env.py", f"{RUNTIME_PACKAGE}/taili_amp_env.py"),
    ("autotuner/blind_locomotion/taili_amp_env_cfg.py", f"{RUNTIME_PACKAGE}/taili_amp_env_cfg.py"),
    ("autotuner/blind_locomotion/assets/__init__.py", f"{RUNTIME_PACKAGE}/assets/__init__.py"),
    ("autotuner/blind_locomotion/assets/taili.py", f"{RUNTIME_PACKAGE}/assets/taili.py"),
    ("autotuner/blind_locomotion/terrain_perceiver_policy.py", f"{RUNTIME_PACKAGE}/terrain_perceiver_policy.py"),
    ("autotuner/blind_locomotion/terrain_perceiver_aux_patch.py", f"{RUNTIME_PACKAGE}/terrain_perceiver_aux_patch.py"),
    ("autotuner/blind_locomotion/telemetry_emit.py", f"{RUNTIME_PACKAGE}/telemetry_emit.py"),
    ("autotuner/blind_locomotion/launch_taili_train.py", f"{RUNTIME_PACKAGE}/launch_taili_train.py"),
    ("autotuner/blind_locomotion/train_taili.py", f"{RUNTIME_PACKAGE}/train_taili.py"),
    ("autotuner/blind_locomotion/diagnose_taili.py", f"{RUNTIME_PACKAGE}/diagnose_taili.py"),
    ("autotuner/blind_locomotion/diagnose_taili_cases.py", f"{RUNTIME_PACKAGE}/diagnose_taili_cases.py"),
    ("autotuner/blind_locomotion/taili_blind_config.py", f"{RUNTIME_PACKAGE}/taili_blind_config.py"),
    ("autotuner/blind_locomotion/taili_blind_config.yaml", f"{RUNTIME_PACKAGE}/taili_blind_config.yaml"),
    ("autotuner/blind_locomotion/agents/__init__.py", f"{RUNTIME_PACKAGE}/agents/__init__.py"),
    ("autotuner/blind_locomotion/multi_motion_loader.py", f"{RUNTIME_PACKAGE}/multi_motion_loader.py"),
    ("autotuner/blind_locomotion/parametric_ref.py", f"{RUNTIME_PACKAGE}/parametric_ref.py"),
    ("autotuner/blind_locomotion/motions.py", f"{RUNTIME_PACKAGE}/motions.py"),
    ("autotuner/blind_locomotion/symmetry.py", f"{RUNTIME_PACKAGE}/symmetry.py"),
    ("autotuner/blind_locomotion/_symmetry_local.py", f"{RUNTIME_PACKAGE}/_symmetry_local.py"),
    # physeval acceptance harness + pure scorers — so the payload can SELF-EVALUATE against the spec
    # (physeval_blind does `import acceptance_score`, so the scorers ship at BOTH the package dir and
    # the payload root, which is on PYTHONPATH). These were missing, so remote physeval failed.
    ("autotuner/blind_locomotion/physeval_blind.py", f"{RUNTIME_PACKAGE}/physeval_blind.py"),
    ("autotuner/blind_locomotion/physeval_blind_e.py", f"{RUNTIME_PACKAGE}/physeval_blind_e.py"),
    ("autotuner/blind_locomotion/physeval_suite.py", f"{RUNTIME_PACKAGE}/physeval_suite.py"),
    ("autotuner/blind_locomotion/acceptance_score.py", f"{RUNTIME_PACKAGE}/acceptance_score.py"),
    ("autotuner/blind_locomotion/acceptance_score.py", "acceptance_score.py"),
    ("autotuner/blind_locomotion/acceptance_aggregate.py", f"{RUNTIME_PACKAGE}/acceptance_aggregate.py"),
    ("autotuner/blind_locomotion/acceptance_aggregate.py", "acceptance_aggregate.py"),
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
        '"""Auto-register Taili blind runtime tasks when this payload is on PYTHONPATH."""\n'
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

    taili_core = root / "autotuner" / "taili_core"
    for src in sorted(taili_core.glob("*.py")):
        if src.name.startswith("test_") or src.name == "__pycache__":
            continue
        yield src, f"{RUNTIME_PACKAGE}/taili_core/{src.name}"

    clips = root / "autotuner" / "blind_locomotion" / "motions" / "clips"
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
    seen_dest: set[str] = set()

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
    policy = root / "autotuner" / "blind_locomotion" / "terrain_perceiver_policy.py"
    text = policy.read_text(encoding="utf-8", errors="replace")
    required = {
        "BODY_DIM = 53": "actor body must be 53",
        "HIST_LEN = 25": "history length must be 25",
        "TICK_DIM = 54": "history tick dim must be 54",
        "Z_DIM = 32": "terrain latent must be 32",
        "BODY_DIM + HIST_FLAT": "policy must slice body+history, not privileged obs",
    }
    for needle, msg in required.items():
        if needle not in text:
            report.errors.append(f"policy contract violation: {msg}")


def _validate_yaml_contract(root: Path, report: ValidationReport) -> None:
    cfg = root / "autotuner" / "blind_locomotion" / "taili_blind_config.yaml"
    text = cfg.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^\s*style_reward_weight:\s*([0-9.]+)", text, flags=re.MULTILINE)
    if not m:
        report.errors.append("style_reward_weight missing from taili_blind_config.yaml")
        return
    value = float(m.group(1))
    # Sanity range only. The strategy book's 0.3..0.5 is a DRAFT starting band, not authority
    # (owner directive 2026-07-02): weights are physeval-tuned toward the taili_spec acceptance
    # bars. Upper bound raised 2.0 -> 4.0 (owner directive 2026-07-04, "OPTION B"): AMP is now the
    # DOMINANT style shaper (hand-designed gait terms cut), so style_reward_weight legitimately
    # exceeds 2.0. Guard still rejects a dead (<=0) or absurd (>4) style channel.
    if not 0.0 < value <= 4.0:
        report.errors.append(f"style_reward_weight={value} outside sanity range (0, 4.0]")


def _validate_registration_contract(root: Path, report: ValidationReport) -> None:
    init_py = root / "autotuner" / "blind_locomotion" / "__init__.py"
    text = init_py.read_text(encoding="utf-8", errors="replace")
    if ".blind_tp_env:TailiBlindTPEnv" not in text:
        report.errors.append("task entry point must resolve to package-local blind_tp_env:TailiBlindTPEnv")
    if ".taili_blind_env_cfg:TailiBlindEnvCfg" not in text:
        report.errors.append("env cfg entry point must resolve to package-local taili_blind_env_cfg")
    if "skrl_taili_blind_cfg.yaml" in text:
        report.errors.append("registration must use taili_blind_config.yaml as the source config")


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
