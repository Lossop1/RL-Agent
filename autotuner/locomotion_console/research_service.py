"""Shared construction and path policy for the research loop."""
from __future__ import annotations

import os
from pathlib import Path

from .config import LocomotionConsoleSettings, PROJECT_ROOT, get_settings
from autotuner.research.research_cycle import ResearchCycleManager
from autotuner.research.research_ledger import ResearchLedgerStore
from autotuner.research.research_state import ResearchStateStore
from autotuner.research.research_supervisor import CommandExperimentBackend, ExperimentBackend, ResearchSupervisor


DEFAULT_RESEARCH_ROOT = "output/research"


def resolve_research_root(configured: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the research root while keeping all mutable state in the repository."""
    base = PROJECT_ROOT.resolve()
    raw = str(configured) if configured is not None else os.environ.get(
        "LOCOMOTION_RESEARCH_ROOT", DEFAULT_RESEARCH_ROOT
    )
    raw = raw.strip() or DEFAULT_RESEARCH_ROOT
    candidate = (base / raw).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("research root must remain inside the local repository") from exc
    return candidate


def build_experiment_backend(
    settings: LocomotionConsoleSettings | None = None,
) -> ExperimentBackend:
    """Build the explicitly configured local or SSH execution boundary."""
    active_settings = settings or get_settings()
    default_mode = "ssh" if active_settings.source == "real" else "local"
    mode = os.environ.get("LOCOMOTION_RESEARCH_BACKEND", default_mode).strip().lower()
    if mode == "local":
        return CommandExperimentBackend()
    if mode == "ssh":
        if active_settings.source != "real":
            raise RuntimeError("SSH research backend requires LOCOMOTION_CONSOLE_SOURCE=real")
        from .config_manager import effective_remote_config
        from .research_remote import SSHExperimentBackend

        config = effective_remote_config(active_settings)
        if not str(config.get("ssh_host") or "").strip():
            raise RuntimeError("SSH research backend requires a configured ssh_host")
        remote_root = os.environ.get(
            "LOCOMOTION_RESEARCH_REMOTE_ROOT",
            "/root/gpufree-data/research_experiments",
        )
        return SSHExperimentBackend(config, remote_root=remote_root)
    raise ValueError("LOCOMOTION_RESEARCH_BACKEND must be 'local' or 'ssh'")


def build_research_cycle_manager(
    *,
    settings: LocomotionConsoleSettings | None = None,
    root: str | os.PathLike[str] | None = None,
    backend: ExperimentBackend | None = None,
) -> ResearchCycleManager:
    research_root = resolve_research_root(root)
    supervisor = ResearchSupervisor(
        research_root / "experiments",
        backend=backend or build_experiment_backend(settings),
    )
    return ResearchCycleManager(
        research_root / "cycles",
        state_store=ResearchStateStore(research_root / "state"),
        ledger=ResearchLedgerStore(research_root / "ledger"),
        supervisor=supervisor,
    )
