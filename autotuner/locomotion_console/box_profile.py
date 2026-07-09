"""BoxProfile - the discovered, typed description of a remote training box.

The whole point (per the "LLM is the soul" steer): paths and layout are NOT hardcoded in the
locomotion console. They are *discovered* (LLM-first, deterministic fallback) into this typed profile,
which the deterministic core (`RealDataSource`) consumes. Change machine / robot / layout -> re-discover, don't edit code.

A profile holds the discovered STRUCTURE (globs, patterns, tags, commands). The runtime
resolves the current INSTANCE (newest run, latest checkpoint) from those globs each call - so
a profile keeps working as new runs appear.

Every profile carries `provenance`: the raw stdout of the probe commands it was derived from,
so a discovered path is always traceable to "this command printed this on this box" - not an
LLM hallucination.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

DEFAULT_PROFILE_PATH = Path("autotuner/locomotion_console/box_profile.json")


class BoxProfile(BaseModel):
    # identity
    ssh_config: str = "config/ssh.json"
    hostname: str = ""
    work_dir: str = "/root/robot_lab"

    # run layout (discovered structure; runtime resolves the newest instance)
    framework: str = "skrl"
    experiment: str = "taili_amp"
    runs_glob: str = ""                       # e.g. /root/robot_lab/logs/skrl/taili_amp/*/
    tfevents_glob: str = "events.out.tfevents.*"
    checkpoints_subdir: str = "checkpoints"
    checkpoint_glob: str = "agent_*.pt"

    # tfevents reading
    reader: str = "sftp_local_tbparse"        # how the locomotion console reads metrics
    reward_tag: str = "Reward / Total reward (mean)"
    extra_tags: List[str] = Field(default_factory=lambda: [
        "Policy / Standard deviation",
        "Episode / Total timesteps (mean)",
    ])

    # liveness + actions (commands run on the box)
    train_running_probe: str = "pgrep -fa 'train.*\\.py' | grep -v pgrep"
    kill_cmd: str = "pkill -9 -f train.py"
    task_id: str = "RobotLab-Isaac-Taili-AMP-Direct-v0"
    # full shell command templates discovered from the box (env setup + flags), with a literal
    # {checkpoint} placeholder the locomotion console substitutes at run time.
    physeval_cmd: str = ""
    resume_cmd: str = ""
    # where training stdout is tee'd (from the launch command) - for the live log view
    train_log: str = "/tmp/rl_train.log"
    physeval_log: str = "/tmp/locomotion_console_physeval.log"
    # config files (read-only viewing - the locomotion console NEVER writes these)
    env_cfg_path: str = ""
    agent_cfg_path: str = ""
    asset_cfg_path: str = ""   # the robot asset .py (actuator stiffness/damping/effort/mass)

    # how the profile was obtained
    discovered_by: str = "heuristic"          # "llm" | "heuristic"
    provenance: Dict[str, str] = Field(default_factory=dict)
    note: str = ""

    def save(self, path: Optional[Path] = None) -> Path:
        p = path or DEFAULT_PROFILE_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional["BoxProfile"]:
        p = path or DEFAULT_PROFILE_PATH
        if not p.exists():
            return None
        return cls.model_validate_json(p.read_text(encoding="utf-8"))
