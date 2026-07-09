"""Discover a remote box's layout into a typed BoxProfile.

LLM-first, deterministic fallback. The system runs a battery of READ-ONLY probe commands
(the "tools"); the LLM interprets the *real* outputs into a BoxProfile. If the LLM is not
configured (no config/llm.json / no key) we fall back to a heuristic interpreter so the
locomotion console still runs today. Either way the profile carries the raw probe outputs as provenance.

LLM's role here is DISCOVERY / CONFIG (fuzzy, machine-varying) - not decision/actuation. It
reads tool output and fills a typed schema; it never executes free-form commands and never
touches training cfg.
"""
from __future__ import annotations

import json
import re
from typing import Dict

from .box_profile import BoxProfile


# -- read-only probe battery (the "tools" the discoverer sees) ----------------

def _probe_cmds(work_dir: str) -> Dict[str, str]:
    return {
        "host": "hostname; whoami",
        "gpu": "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu "
               "--format=csv,noheader 2>/dev/null | head -1",
        "train_procs": "pgrep -fa 'train.py' | grep -v pgrep | head -3",
        "tmux": "tmux ls 2>/dev/null",
        # framework/experiment layout: logs/<framework>/<experiment>/<run>/
        "log_tree": f"ls -d {work_dir}/logs/*/*/ 2>/dev/null | head -10",
        "newest_runs": f"ls -dt {work_dir}/logs/*/*/*/ 2>/dev/null | head -3",
        # contents of the newest run: reveals the tfevents filename + checkpoint names
        "newest_run_contents": (
            f'd=$(ls -dt {work_dir}/logs/*/*/*/ 2>/dev/null | head -1); '
            'echo "RUN=$d"; ls -la "$d" 2>/dev/null; '
            'echo "--- checkpoints ---"; ls "$d"checkpoints/ 2>/dev/null | head -8'
        ),
        # action commands: the launcher scripts + how training was actually invoked
        "scripts": f"find {work_dir} -name physeval.py -o -name train.py 2>/dev/null "
                   f"| grep -v __pycache__ | head",
        "launch_history": "grep -hE 'train.py|physeval' /root/.bash_history 2>/dev/null | tail -5",
        # config files (for read-only viewing): env cfg .py + skrl agent yaml + robot asset .py
        "config_files": f"find {work_dir} -name '*_env_cfg.py' 2>/dev/null | grep -v __pycache__ "
                        f"| head -20; echo '---yaml---'; "
                        f"find {work_dir} -name 'skrl_*.yaml' 2>/dev/null | head -20; "
                        f"echo '---assets---'; "
                        f"find {work_dir} -path '*/assets/*.py' 2>/dev/null | grep -v __pycache__ "
                        f"| head -20",
    }


def run_probes(remote, work_dir: str) -> Dict[str, str]:
    """Run the read-only battery, return {name: raw stdout}. Never mutates the box."""
    out: Dict[str, str] = {}
    for name, cmd in _probe_cmds(work_dir).items():
        try:
            out[name] = (remote.exec_out(cmd) or "").strip()
        except Exception as e:  # noqa: BLE001
            out[name] = f"<probe error: {e}>"
    return out


# -- deterministic interpreter (fallback) -------------------------------------

def _heuristic(probes: Dict[str, str], work_dir: str) -> BoxProfile:
    """Derive structure from the real probe outputs without an LLM.

    Reads the newest-run path to recover framework/experiment and confirm the run glob.
    """
    prof = BoxProfile(work_dir=work_dir, discovered_by="heuristic")
    host_line = probes.get("host", "").splitlines()
    if host_line:
        prof.hostname = host_line[0].strip()

    newest = probes.get("newest_runs", "").splitlines()
    if newest:
        run_path = newest[0].rstrip("/")
        # .../logs/<framework>/<experiment>/<run>
        m = re.search(r"/logs/([^/]+)/([^/]+)/[^/]+$", run_path)
        if m:
            prof.framework, prof.experiment = m.group(1), m.group(2)
            prof.runs_glob = f"{work_dir}/logs/{prof.framework}/{prof.experiment}/*/"
    if not prof.runs_glob:
        prof.runs_glob = f"{work_dir}/logs/{prof.framework}/{prof.experiment}/*/"
    prof.note = "structure derived heuristically from real probe outputs"
    prof.provenance = probes
    return prof


# -- LLM interpreter (soul) ---------------------------------------------------

_LLM_SYSTEM = """You are a remote-training-box discovery agent. You are given the raw stdout
of read-only probe commands run over SSH on a GPU training box. Interpret them into a typed
BoxProfile describing where training runs, tfevents, and checkpoints live, so a deterministic
program can read metrics without hardcoded paths.

Return STRICT JSON with keys: framework, experiment, runs_glob, tfevents_glob,
checkpoints_subdir, checkpoint_glob, hostname, physeval_cmd, resume_cmd, env_cfg_path,
agent_cfg_path, note.

env_cfg_path = the *_env_cfg.py whose path matches the experiment/task (e.g. contains the
experiment name). agent_cfg_path = the skrl agent yaml for that task. asset_cfg_path = the robot
asset .py under assets/ matching the robot/experiment (holds actuator stiffness/damping/effort).
Pick from the config_files probe; leave empty if none clearly matches. Base every value
on the probe outputs you were given. Do NOT invent paths that don't appear in the outputs.
runs_glob must be an absolute glob ending in '/*/' that matches the run directories.

physeval_cmd and resume_cmd are full shell commands to run ON THE BOX. Use the EXACT environment
setup shown in the launch history (e.g. 'cd <work_dir> && source /opt/conda/etc/profile.d/conda.sh
&& conda activate <env> && python ...'). Put a LITERAL {checkpoint} placeholder where a checkpoint
path goes:
- resume_cmd  = the training launch command from history + ' --checkpoint {checkpoint}'
- physeval_cmd = run the physeval.py script with the same env setup, '--checkpoint {checkpoint}
  --headless'
If the history doesn't show how training was launched, leave them as empty strings."""


def _llm(probes: Dict[str, str], work_dir: str):
    """Try the LLM. Returns a BoxProfile or None if the LLM is unavailable/invalid."""
    try:
        from autotuner.llm_gateway.client import call_llm_with_schema
    except Exception:
        return None

    user = "work_dir = %s\n\nProbe outputs:\n%s" % (
        work_dir,
        "\n".join(f"### {k}\n{v}" for k, v in probes.items()),
    )
    resp = call_llm_with_schema(_LLM_SYSTEM, user, schema_name="box_profile")
    if not resp or not resp.parsed:
        return None
    d = resp.parsed
    if not d.get("runs_glob"):
        return None
    prof = BoxProfile(
        work_dir=work_dir,
        discovered_by="llm",
        hostname=d.get("hostname", ""),
        framework=d.get("framework", "skrl"),
        experiment=d.get("experiment", ""),
        runs_glob=d["runs_glob"],
        tfevents_glob=d.get("tfevents_glob", "events.out.tfevents.*"),
        checkpoints_subdir=d.get("checkpoints_subdir", "checkpoints"),
        checkpoint_glob=d.get("checkpoint_glob", "agent_*.pt"),
        physeval_cmd=d.get("physeval_cmd", ""),
        resume_cmd=d.get("resume_cmd", ""),
        env_cfg_path=d.get("env_cfg_path", ""),
        agent_cfg_path=d.get("agent_cfg_path", ""),
        asset_cfg_path=d.get("asset_cfg_path", ""),
        note=d.get("note", "interpreted by LLM"),
        provenance=probes,
    )
    return prof


# -- entry point --------------------------------------------------------------

def discover(remote, work_dir: str, prefer_llm: bool = True) -> BoxProfile:
    """Probe the box, interpret into a BoxProfile (LLM-first, heuristic fallback)."""
    probes = run_probes(remote, work_dir)
    prof = _llm(probes, work_dir) if prefer_llm else None
    if prof is None:
        prof = _heuristic(probes, work_dir)
    return prof


def _connect_default():
    """Load config/ssh.json into runtime_state and return a RemoteSSH."""
    import json as _json
    from pathlib import Path
    import runtime_state
    from autotuner.training.remote import RemoteSSH

    if not runtime_state.get_config_path():
        cfg_path = Path("config/ssh.json")
        runtime_state.set_config(_json.loads(cfg_path.read_text(encoding="utf-8")),
                                 str(cfg_path))
    return RemoteSSH(runtime_state.get_config()), runtime_state.get_config()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Discover a box into a BoxProfile.")
    ap.add_argument("--no-llm", action="store_true", help="force heuristic interpreter")
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    remote, cfg = _connect_default()
    work_dir = args.work_dir or cfg.get("work_dir", "/root/robot_lab")
    prof = discover(remote, work_dir, prefer_llm=not args.no_llm)
    path = prof.save()
    print(f"discovered_by = {prof.discovered_by}")
    print(f"hostname      = {prof.hostname}")
    print(f"framework/exp = {prof.framework} / {prof.experiment}")
    print(f"runs_glob     = {prof.runs_glob}")
    print(f"reward_tag    = {prof.reward_tag}")
    print(f"note          = {prof.note}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
