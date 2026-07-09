"""Framework registry for the locomotion console.

The framework is intentionally a profile, not hardcoded behavior. The current
candidate framework is not the teacher task; teacher remains a reference profile
that must be selected explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


FrameworkStatus = Literal["draft", "validated", "reference", "legacy"]


@dataclass(frozen=True)
class FrameworkProfile:
    id: str
    label: str
    status: FrameworkStatus
    experiment: str
    task_id: str
    diagnostic_task: str
    run_globs: tuple[str, ...]
    checkpoint_roots: tuple[str, ...]
    note: str = ""
    required_robot_capabilities: tuple[str, ...] = field(default_factory=tuple)
    # per-framework action commands (override BoxProfile when set); {checkpoint} substituted at run time.
    physeval_cmd: str = ""
    physeval_log: str = ""
    resume_cmd: str = ""
    train_running_probe: str = ""
    kill_cmd: str = ""


_FRAMEWORKS: dict[str, FrameworkProfile] = {
    "taili_amp_blind": FrameworkProfile(
        id="taili_amp_blind",
        label="Taili AMP Blind",
        status="draft",
        experiment="taili_amp_blind",
        task_id="RobotLab-Isaac-Taili-AMP-Blind-Direct-v0",
        diagnostic_task="RobotLab-Isaac-Taili-AMP-Blind-Direct-v0",
        run_globs=(
            # Runtime payload launches create one data-disk run directory containing
            # train.log, train.telemetry.jsonl, console.log, checkpoints/, and tfevents.
            "/root/gpufree-data/taili_runs/*/",
            # Legacy fallback for runs created before the self-contained payload launcher.
            "/root/tp_runs/*/",
            "/root/gpufree-data/robot_lab/logs/skrl/taili_amp_blind/*/",
            "/root/gpufree-data/logs/skrl/taili_amp_blind/*/",
            "/root/robot_lab/logs/skrl/taili_amp_blind/*/",
        ),
        checkpoint_roots=(
            "/root/gpufree-data/taili_runs",
            "/root/tp_runs",
            "/root/gpufree-data/robot_lab/logs/skrl/taili_amp_blind",
            "/root/gpufree-data/logs/skrl/taili_amp_blind",
            "/root/robot_lab/logs/skrl/taili_amp_blind",
        ),
        required_robot_capabilities=("quadruped_12dof", "blind_actor", "amp_reference"),
        note="Current draft framework target. It is the active default and is not the teacher task.",
        # blind acceptance uses physeval_blind.py (deployment-口径 mean-action), NOT the old physeval.py
        physeval_cmd="",
        physeval_log="/root/gpufree-data/diag_runs/physeval_blind.log",
        train_running_probe=(
            "pgrep -fa 'taili_blind_runtime\\.train_taili|launch_taili_train|RobotLab-Isaac-Taili.*Blind' "
            "| grep -v pgrep"
        ),
        kill_cmd="pkill -f 'taili_blind_runtime\\.train_taili|launch_taili_train|RobotLab-Isaac-Taili.*Blind'",
        # §3 D 运维: warm-start resume from a checkpoint (the box auto-shut-down killed v1; resume
        # instead of restarting from 0). tp_resume.py = agent.load(ckpt) then continue.
        resume_cmd="",
    ),
    "taili_amp_teacher_reference": FrameworkProfile(
        id="taili_amp_teacher_reference",
        label="Taili AMP Teacher Reference",
        status="reference",
        experiment="taili_amp_teacher",
        task_id="RobotLab-Isaac-Taili-AMP-Teacher-Direct-v0",
        diagnostic_task="RobotLab-Isaac-Taili-AMP-Teacher-Direct-v0",
        run_globs=(
            "/root/gpufree-data/robot_lab/logs/skrl/taili_amp_teacher/*/",
            "/root/gpufree-data/logs/skrl/taili_amp_teacher/*/",
            "/root/robot_lab/logs/skrl/taili_amp_teacher/*/",
        ),
        checkpoint_roots=(
            "/root/gpufree-data/robot_lab/logs/skrl/taili_amp_teacher",
            "/root/gpufree-data/logs/skrl/taili_amp_teacher",
            "/root/robot_lab/logs/skrl/taili_amp_teacher",
        ),
        required_robot_capabilities=("quadruped_12dof", "privileged_teacher"),
        note="Reference-only profile for already generated teacher checkpoints; never the default.",
    ),
}


DEFAULT_FRAMEWORK_ID = "taili_amp_blind"


def list_framework_profiles() -> list[FrameworkProfile]:
    return list(_FRAMEWORKS.values())


def get_framework_profile(framework_id: str | None = None) -> FrameworkProfile:
    key = framework_id or DEFAULT_FRAMEWORK_ID
    try:
        return _FRAMEWORKS[key]
    except KeyError as exc:
        known = ", ".join(sorted(_FRAMEWORKS))
        raise ValueError(f"Unknown framework profile: {key}. Known: {known}") from exc


def merge_box_profile(profile, framework: FrameworkProfile):
    """Return a BoxProfile aligned to the active framework.

    BoxProfile still carries discovered remote details, but framework identity,
    run globs, and task id come from the selected FrameworkProfile so the console
    cannot accidentally show teacher state while the active framework is blind.
    """
    update = {
        "experiment": framework.experiment,
        "runs_glob": framework.run_globs[0] if framework.run_globs else profile.runs_glob,
        "task_id": framework.task_id,
    }
    # per-framework action commands override the box defaults (the blind framework's physeval is
    # physeval_blind.py, not the box's old physeval.py / task).
    if framework.physeval_cmd:
        update["physeval_cmd"] = framework.physeval_cmd
    if framework.physeval_log:
        update["physeval_log"] = framework.physeval_log
    if framework.resume_cmd:
        update["resume_cmd"] = framework.resume_cmd
    if framework.train_running_probe:
        update["train_running_probe"] = framework.train_running_probe
    if framework.kill_cmd:
        update["kill_cmd"] = framework.kill_cmd
    return profile.model_copy(update=update)
