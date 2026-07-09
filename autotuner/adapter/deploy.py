"""Adapter ④ deploy — SYSTEM_ARCHITECTURE §14.2 (materialize ✅ → deploy → launch).

Takes the materialized work_dir (the env_cfg / asset COPIES that `materialize.py` wrote, plus any
regenerated reference clips) and a remote_map {local_filename → remote_path}, and:

  1. build_plan()  — PURE, no SSH. Resolves each file → remote dest + a unique backup name,
                     checksums the local copy, and warns if references were not regenerated
                     (committed flat clip has drifted, §6: regenerate before deploy). Default mode.
  2. render_plan() — human-readable plan for review / audit.
  3. execute()     — GATED behind confirm=True. For each item: backup remote original
                     (backup-then-put, never overwrite blind), SFTP put the copy, then verify by
                     sha256 readback. Optionally launch training in tmux. OUTWARD-FACING / hard to
                     reverse → refuses without confirm; restore_cmds() gives the rollback.

Iron rules honoured: only deploy COPIES; back up every original first; verify by readback;
deterministic plan + provenance; the LLM is nowhere in this control path (§10 free).
"""
from __future__ import annotations

import hashlib
import shlex
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable


# ── SSH interface (dependency-injected so the planner stays pure + execute is testable) ──
@runtime_checkable
class SSH(Protocol):
    def exec(self, cmd: str, timeout: int = 30) -> Tuple[str, int]:
        """Run a remote command; return (stdout, returncode)."""
        ...

    def put(self, local: str, remote: str) -> None:
        """SFTP upload local → remote."""
        ...


@dataclass
class DeployItem:
    local: str                  # absolute path of the materialized copy in work_dir
    remote: str                 # remote destination path it replaces
    role: str                   # "env_cfg" | "asset" | "reference_clip" | "other"
    sha256: str
    size: int


@dataclass
class DeployPlan:
    items: List[DeployItem]
    backup_suffix: str          # appended to each remote original, e.g. ".autobak-20260629-231500"
    work_dir: str
    launch_cmd: Optional[str] = None
    launch_session: str = "tpdeploy"
    warnings: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)   # in remote_map but absent from work_dir

    def to_dict(self) -> dict:
        return asdict(self)


_ROLE_BY_HINT = (
    ("env_cfg", "env_cfg"), ("_cfg", "env_cfg"),
    ("asset", "asset"), ("taili.py", "asset"),
    (".npz", "reference_clip"), ("clip", "reference_clip"),
)


def _role_of(name: str) -> str:
    low = name.lower()
    for hint, role in _ROLE_BY_HINT:
        if hint in low:
            return role
    return "other"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_plan(work_dir: str, remote_map: Dict[str, str], *,
               launch_cmd: Optional[str] = None, launch_session: str = "tpdeploy",
               stamp: Optional[str] = None, require_refs: bool = True) -> DeployPlan:
    """PURE. work_dir = materialize output; remote_map = {filename: remote_dest_path}.

    stamp must be supplied by the caller (deterministic; no wall-clock here so the plan is
    reproducible/testable). require_refs warns when no reference clip is being shipped (§6).
    """
    wd = Path(work_dir)
    suffix = f".autobak-{stamp}" if stamp else ".autobak"
    items: List[DeployItem] = []
    missing: List[str] = []
    warnings: List[str] = []

    for name, remote in remote_map.items():
        f = wd / name
        if not f.exists():
            missing.append(name)
            continue
        items.append(DeployItem(local=str(f), remote=remote, role=_role_of(name),
                                sha256=_sha256(f), size=f.stat().st_size))

    if missing:
        warnings.append(f"{len(missing)} mapped file(s) absent from work_dir: {', '.join(missing)} "
                        "→ run materialize / regenerate first")
    if require_refs and not any(it.role == "reference_clip" for it in items):
        warnings.append("no reference_clip in this deploy — §6 says regenerate the full reference set "
                        "before deploy (committed flat clip has drifted 0.2 rad)")
    if launch_cmd is None:
        warnings.append("no launch_cmd — files will deploy but no training will start")

    return DeployPlan(items=items, backup_suffix=suffix, work_dir=str(wd),
                      launch_cmd=launch_cmd, launch_session=launch_session,
                      warnings=warnings, missing=missing)


def render_plan(plan: DeployPlan) -> str:
    lines = [f"DEPLOY PLAN  (work_dir={plan.work_dir})",
             f"  backup originals → <remote>{plan.backup_suffix}  (backup-then-put)",
             f"  {len(plan.items)} file(s):"]
    for it in plan.items:
        lines.append(f"    [{it.role:14}] {Path(it.local).name:30} → {it.remote}")
        lines.append(f"        {it.size:>9} B  sha256={it.sha256[:16]}…")
    if plan.launch_cmd:
        lines.append(f"  launch: tmux '{plan.launch_session}' ← {plan.launch_cmd}")
    else:
        lines.append("  launch: (none)")
    for w in plan.warnings:
        lines.append(f"  ! {w}")
    return "\n".join(lines)


def restore_cmds(plan: DeployPlan) -> List[str]:
    """Rollback recipe: move each backup back over the deployed file."""
    return [f"mv -f {shlex.quote(it.remote + plan.backup_suffix)} {shlex.quote(it.remote)}"
            for it in plan.items]


@dataclass
class ItemResult:
    remote: str
    backed_up: bool
    backup_path: str
    put_ok: bool
    verified: bool
    detail: str = ""


@dataclass
class DeployResult:
    ok: bool
    items: List[ItemResult]
    launched: bool = False
    launch_detail: str = ""


def execute(plan: DeployPlan, ssh: SSH, *, confirm: bool = False, do_launch: bool = False,
            logger: Callable[[str], None] = print) -> DeployResult:
    """GATED. Backs up each remote original, SFTP-puts the copy, verifies by sha256 readback.

    OUTWARD-FACING (writes remote files, optionally starts a GPU job). Refuses unless confirm=True.
    Stops at the first item whose backup or verification fails — earlier items keep their backups
    (use restore_cmds() to roll back).
    """
    if not confirm:
        raise PermissionError("deploy.execute refused: pass confirm=True to write remote files "
                              "(outward-facing, hard to reverse). Review render_plan() first.")
    results: List[ItemResult] = []
    all_ok = True
    for it in plan.items:
        backup = it.remote + plan.backup_suffix
        backed_up = False
        # 1. backup-then-put: copy the original aside only if it exists.
        # Every remote path is shlex.quote'd — it.remote flows from a ConfigSet the operator sets
        # and must never be able to inject shell metacharacters into these root-level commands.
        out, rc = ssh.exec(f"test -f {shlex.quote(it.remote)} && echo EXISTS || echo ABSENT")
        if "EXISTS" in out:
            _, rc = ssh.exec(f"cp -p {shlex.quote(it.remote)} {shlex.quote(backup)}")
            if rc != 0:
                results.append(ItemResult(it.remote, False, backup, False, False,
                                          "backup cp failed — aborting before put"))
                all_ok = False
                logger(f"[deploy] ABORT {it.remote}: backup failed")
                break
            backed_up = True
        # 2. put the materialized copy
        try:
            ssh.put(it.local, it.remote)
            put_ok = True
        except Exception as e:                       # noqa: BLE001 — report any transport failure
            results.append(ItemResult(it.remote, backed_up, backup, False, False, f"put failed: {e}"))
            all_ok = False
            logger(f"[deploy] ABORT {it.remote}: put failed ({e})")
            break
        # 3. verify by sha256 readback
        out, rc = ssh.exec(f"sha256sum {shlex.quote(it.remote)}")
        got = out.strip().split()[0] if out.strip() else ""
        verified = (got == it.sha256)
        results.append(ItemResult(it.remote, backed_up, backup, put_ok, verified,
                                  "ok" if verified else f"sha mismatch got={got[:16]} want={it.sha256[:16]}"))
        logger(f"[deploy] {'OK ' if verified else 'BAD'} {it.remote}"
               + ("" if verified else "  (checksum mismatch!)"))
        if not verified:
            all_ok = False
            break

    res = DeployResult(ok=all_ok, items=results)
    if all_ok and do_launch and plan.launch_cmd:
        esc = plan.launch_cmd.replace("'", "'\\''")
        out, rc = ssh.exec(f"tmux new-session -d -s {shlex.quote(plan.launch_session)} '{esc}'")
        res.launched = (rc == 0)
        res.launch_detail = out.strip() or ("tmux session started" if rc == 0 else f"launch rc={rc}")
        logger(f"[deploy] launch {'OK' if res.launched else 'FAILED'}: {res.launch_detail}")
    elif do_launch and not plan.launch_cmd:
        res.launch_detail = "do_launch requested but plan has no launch_cmd"
    return res


if __name__ == "__main__":
    # DRY-RUN self-check: adapt → materialize → build_plan → render. Pure local, no SSH.
    import tempfile
    from autotuner.adapter.adapt import adapt, _DEFAULT_URDF
    from autotuner.adapter.materialize import materialize

    env_src = "autotuner/blind_locomotion/env_edit/taili_amp_env_cfg.py"
    asset_src = "autotuner/blind_locomotion/assets/taili.py"
    wd = tempfile.mkdtemp(prefix="adapter_deploy_")
    cfg = adapt(_DEFAULT_URDF)
    materialize(cfg, env_src, asset_src, wd)

    remote_base = ("/root/gpufree-data/robot_lab/source/robot_lab/robot_lab/tasks/"
                   "direct/taili_amp")
    remote_map = {
        Path(env_src).name:   f"{remote_base}/{Path(env_src).name}",
        Path(asset_src).name: "/root/gpufree-data/robot_lab/.../asset/taili.py",
    }
    plan = build_plan(wd, remote_map, stamp="20260629-231500",
                      launch_cmd="/opt/conda/envs/isaaclab/bin/python3 /root/tp_train_real.py",
                      require_refs=True)
    print(render_plan(plan))
    print("\nrollback recipe:")
    for c in restore_cmds(plan):
        print("  " + c)
    print("\n(DRY-RUN only — execute(confirm=True) is required to touch the remote.)")
