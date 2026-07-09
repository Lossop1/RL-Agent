"""The mechanism-library knowledge base.

The hand-tuning experience is distilled in the repository docs. This exposes it as retrievable
sections so LLM explanations can be grounded in real project notes instead of free-form guesses.

Read-only. Local files (the locomotion console backend runs in the repo), no SSH.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

_DOCS = Path(__file__).resolve().parents[2] / "docs"
_ROOT = Path(__file__).resolve().parents[2]
_TAILI_YAML = _ROOT / "autotuner" / "blind_locomotion" / "taili_blind_config.yaml"

_ALLOWLIST_DOCS = {
    "taili_ops_runbook.md": "运营手册：已知故障签名(GPU挂起/OOM/SSH)、训练体制事实、系统自动化、当前基线。",
    "taili_tuning_followups.md": "调参实战全记录(D3a-D3p)：每个修复的证据链、度量伪影、结构性结论。",
    "taili_spec.md": "验收 spec：速度、姿态、地形、鲁棒和部署要求。",
    "taili_strategy_decisions.md": "策略决策书：AMP+地形感知器+课程门控+奖励结构的设计理由。",
    "SYSTEM_ARCHITECTURE.md": "系统架构：控制台、LLM 权限、数据来源、部署/诊断边界。",
    "amp_multigait_terrain_plan.md": "AMP 多步态与地形训练计划。",
    "start.md": "启动与操作说明。",
}

_QUERY_ALIASES = {
    "奖励": "reward tracking penalty gate",
    "步态": "gait trot diagonal duty air slip impact",
    "小跑": "trot diagonal duty gait",
    "地形": "terrain stairs slope rough curriculum",
    "显存": "gpu memory remote status",
    "远程": "remote ssh gpu tmux",
    "配置": "yaml config reward curriculum actuator",
    "高度": "height base_h posture termination",
    "策略": "strategy curriculum reward AMP perceiver",
    "感知器": "terrain perceiver privileged observation",
    "观察": "observation history actor perceiver",
    "网络": "actor critic hidden perceiver model",
    "诊断": "diagnostic playback report slip impact duty tracking",
    "卡住": "stall frozen step gpu kernel hang auto_drive restart 挂起 冻结",
    "停滞": "stall frozen step gpu hang auto restart 挂起",
    "挂起": "stall frozen gpu kernel hang restart checkpoint",
    "滑": "slip stance contact",
    "撞": "impact touchdown collision",
    "慢": "fps gpu memory rollout num_envs",
}


@lru_cache(maxsize=1)
def _sections() -> List[Dict[str, str]]:
    """Parse every doc into {source, title, body} sections split on level-2/3 headers."""
    out: List[Dict[str, str]] = []
    if not _DOCS.is_dir():
        return out
    for f in sorted(_DOCS.glob("*.md")):
        if f.name not in _ALLOWLIST_DOCS:
            continue
        text = f.read_text(encoding="utf-8")
        cur = {"source": f.stem, "title": f"{f.stem} (intro)", "body": ""}
        for line in text.splitlines():
            if re.match(r"^#{1,3}\s+\S", line):
                if cur["body"].strip():
                    out.append(cur)
                cur = {"source": f.stem, "title": line.lstrip("#").strip(), "body": ""}
            else:
                cur["body"] += line + "\n"
        if cur["body"].strip():
            out.append(cur)
    return out


def search(query: str = "") -> Dict:
    """No query -> the table of contents (so the soul can pick a section by its descriptive
    title, bridging Chinese question -> English doc). With a query -> full text of matching
    sections (matches title / body / source, case-insensitive)."""
    secs = _sections()
    if not secs:
        return {"error": "knowledge base not found", "docs_dir": str(_DOCS)}
    original = (query or "").strip()
    expanded = original
    for needle, alias in _QUERY_ALIASES.items():
        if needle in original:
            expanded += " " + alias
    q = expanded.lower().strip()
    if not q:
        return {"table_of_contents": [f"[{s['source']}] {s['title']}" for s in secs],
                "hint": "call search_knowledge('<keyword or section title>') to read a section"}
    terms = [item for item in re.split(r"\s+", q) if item]
    scored = []
    for section in secs:
        haystack = f"{section['source']} {section['title']} {section['body']}".lower()
        score = sum(1 for term in terms if term in haystack)
        if q in haystack:
            score += 4
        if score:
            scored.append((score, section))
    scored.sort(key=lambda item: item[0], reverse=True)
    hits = [section for _score, section in scored]
    return {"query": query,
            "expanded_query": expanded,
            "matches": [{"source": s["source"], "title": s["title"],
                         "text": s["body"].strip()[:1600]} for s in hits[:4]],
            "n_matches": len(hits)}


def _read_text(path: Path, limit: int = 18000) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"[unavailable: {type(exc).__name__}: {exc}]"
    return text[:limit]


def _load_yaml_config() -> dict[str, Any]:
    text = _read_text(_TAILI_YAML, limit=60000)
    if text.startswith("[unavailable:"):
        return {"available": False, "path": str(_TAILI_YAML), "error": text}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": str(_TAILI_YAML), "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "path": str(_TAILI_YAML), "data": data}


def _compact_yaml_summary(data: dict[str, Any]) -> dict[str, Any]:
    cfg = data.get("data") if isinstance(data.get("data"), dict) else {}
    recipe = cfg.get("training_recipe") if isinstance(cfg.get("training_recipe"), dict) else {}
    env = cfg.get("env") if isinstance(cfg.get("env"), dict) else {}
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    reward = cfg.get("reward") if isinstance(cfg.get("reward"), dict) else {}
    skrl = cfg.get("skrl") if isinstance(cfg.get("skrl"), dict) else {}
    phases = recipe.get("phases") if isinstance(recipe.get("phases"), dict) else {}
    curriculum = env.get("curriculum") if isinstance(env.get("curriculum"), dict) else {}
    return {
        "profile": cfg.get("profile"),
        "task_default_id": (cfg.get("task") or {}).get("default_id") if isinstance(cfg.get("task"), dict) else "",
        "recipe": {
            "id": recipe.get("id"),
            "purpose": str(recipe.get("purpose") or "").strip(),
            "command_mode": recipe.get("command_mode"),
            "init_phase": recipe.get("init_phase"),
            "touchdown_impact_only": recipe.get("touchdown_impact_only"),
        },
        "model": {
            "terrain_perceiver": model.get("terrain_perceiver"),
            "actor": model.get("actor"),
            "terrain_perceiver_aux": model.get("terrain_perceiver_aux"),
        },
        "observation_contract": env.get("observation_contract"),
        "amp": env.get("amp"),
        "control": env.get("control"),
        "actuator": env.get("actuator"),
        "commands": env.get("commands"),
        "curriculum_gates": {
            key: value
            for key, value in curriculum.items()
            if str(key).startswith("phase_gate") or key in {
                "terrain_start_phase",
                "dr_start_phase",
                "max_training_phase",
                "vel_cur_up",
                "vel_cur_down",
                "vel_cur_step",
            }
        },
        "phases": {
            str(key): {
                "command_mode": phase.get("command_mode"),
                "active_dirs": phase.get("active_dirs"),
                "stand_prob": phase.get("stand_prob"),
                "prob_fwd": phase.get("prob_fwd"),
                "prob_back": phase.get("prob_back"),
                "prob_lat": phase.get("prob_lat"),
                "prob_yaw": phase.get("prob_yaw"),
                "note": phase.get("note"),
            }
            for key, phase in phases.items()
            if isinstance(phase, dict)
        },
        "reward_keys": sorted(reward.keys()) if isinstance(reward, dict) else [],
        "reward_core": {
            key: reward.get(key)
            for key in [
                "nominal_base_h",
                "sigma_lin_abs",
                "sigma_yaw",
                "w_tracking_lin",
                "w_tracking_yaw",
                "w_stand",
                "w_gait_anchor",
                "w_feet_air_time",
                "w_stance_slip",
                "w_landing_impact",
                "w_terminal",
            ]
            if key in reward
        },
        "skrl": {
            "agent": skrl.get("agent") if isinstance(skrl.get("agent"), dict) else {},
            "trainer": skrl.get("trainer") if isinstance(skrl.get("trainer"), dict) else {},
        },
    }


def build_taili_context_pack(query: str = "", include_docs: bool = True) -> dict[str, Any]:
    """Return the allowlisted strategy/spec context the LLM is allowed to know.

    This is deliberately local and read-only. It does not scan arbitrary paths; it reads the
    Taili strategy YAML plus a fixed docs allowlist, then includes targeted search hits.
    """
    yaml_data = _load_yaml_config()
    docs = []
    if include_docs:
        for name, role in _ALLOWLIST_DOCS.items():
            path = _DOCS / name
            text = _read_text(path, limit=2600)
            docs.append({
                "name": name,
                "role": role,
                "available": path.is_file() and not text.startswith("[unavailable:"),
                "path": str(path),
                "preview": text[:1200],
            })
    return {
        "kind": "taili_controlled_context_pack",
        "permission_boundary": (
            "Local read-only allowlist: docs/*.md selected by name and "
            "autotuner/blind_locomotion/taili_blind_config.yaml. No arbitrary local scan, "
            "no remote path scan, no action execution."
        ),
        "sources": {
            "docs_allowlist": list(_ALLOWLIST_DOCS.keys()),
            "strategy_yaml": str(_TAILI_YAML),
        },
        "yaml": _compact_yaml_summary(yaml_data) if yaml_data.get("available") else yaml_data,
        "docs": docs,
        "search_hits": search(query) if query else search("策略 reward gait terrain diagnostic"),
        "operator_hint": (
            "Use this pack with get_training_telemetry/get_remote_status for current facts. "
            "Do not treat strategy docs as live telemetry; YAML is the editable strategy contract."
        ),
    }
