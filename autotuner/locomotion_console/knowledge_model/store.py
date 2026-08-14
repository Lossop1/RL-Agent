"""奖励 + 本体知识库:从**激活机器人**的源码推导 + 对齐 + 覆盖自审 + 装配。

不写死 taili:源文件与符号名从 robot_sources(按 robot_id)解析,推导逻辑通用。
只放稳定结构;当前权重值不入库,assemble 时从 effective_config 现读、经边拼上。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..code_knowledge import _ROOT, _read_allowlisted
from .reward_deriver import _sha, derive_reward_terms
from .robot_sources import RobotSources, active_robot_id, get_robot_sources
from .schema import (
    CurriculumModelResult, Edge, Entity, KnowledgeAudit, RewardModelResult, RewardTermView,
    RobotFact, RobotModelResult, SourceRef,
)

try:
    from ..scoreboard import TENSIONS as _SCOREBOARD_TENSIONS
except Exception:  # noqa: BLE001 — 拆台边是加分项,缺了不该拖垮库
    _SCOREBOARD_TENSIONS = []


def _mtime(rel: str) -> float:
    try:
        return (_ROOT / rel).stat().st_mtime
    except Exception:  # noqa: BLE001
        return 0.0


class KnowledgeStore:
    """按激活机器人构建的奖励知识库;源文件 mtime 或激活机器人变化时自动重建。"""

    def __init__(self, robot_id: Optional[str] = None) -> None:
        self._robot_id = robot_id
        self._cache: Optional[Dict[str, Any]] = None
        self._key: Any = None

    def _sources(self) -> RobotSources:
        return get_robot_sources(self._robot_id or active_robot_id())

    # ── 构建 / 缓存 ───────────────────────────────────────────────────────
    def _build(self, s: RobotSources, primary_text: Optional[str] = None) -> Dict[str, Any]:
        ptext = primary_text if primary_text is not None else (_read_allowlisted(s.reward_file) or "")
        derived = derive_reward_terms(
            ptext, s.reward_file, func_name=s.reward_func, cfg_class=s.reward_cfg_class,
            gate_const=s.reward_gate_const, group_const=s.reward_group_const, strict=True)

        entities: List[Entity] = list(derived["entities"])
        edges: List[Edge] = list(derived["edges"])
        group_of: Dict[str, str] = derived["group_of"]
        config_defaults: Dict[str, float] = derived["config_defaults"]
        derived_terms = {e.name for e in entities if e.type == "reward_term"}

        # 环境专属项:用代码自己的分组清单建 cited_only 实体(可见,不硬解结构)。
        stext = _read_allowlisted(s.env_reward_file) or ""
        for key in group_of:
            if key in derived_terms:
                continue
            src = None
            for idx, line in enumerate(stext.splitlines()):
                if f'comp["{key}"]' in line or f"comp['{key}']" in line:
                    src = SourceRef(file=s.env_reward_file, line_start=idx + 1, line_end=idx + 1,
                                    sha256=_sha(line), symbol=f'comp["{key}"]')
                    break
            entities.append(Entity(
                id=f"reward_term:{key}", type="reward_term", name=key,
                attrs={"sign": "unknown", "kernel_kind": "cited", "group": group_of.get(key),
                       "note": "环境专属项,Stage 1 只做覆盖+引用,结构留给后续层"},
                source=src, freshness="cited_only"))

        # 拆台边:复用记分牌 TENSIONS。端点是奖励项就连实体,否则连 signal: 外部引用。
        term_names = {e.name for e in entities if e.type == "reward_term"}
        for a, b, reason in _SCOREBOARD_TENSIONS:
            for x, y in ((a, b), (b, a)):
                if x in term_names:
                    dst = f"reward_term:{y}" if y in term_names else f"signal:{y}"
                    edges.append(Edge(src=f"reward_term:{x}", rel="trades_off_with", dst=dst, note=reason))

        return {
            "entities": entities, "edges": edges, "group_of": group_of,
            "config_defaults": config_defaults, "derived_terms": derived_terms,
            "errors": derived["errors"], "sources": s,
        }

    def _ensure(self) -> Dict[str, Any]:
        s = self._sources()
        key = (s.robot_id, _mtime(s.reward_file), _mtime(s.env_reward_file))
        if self._cache is None or self._key != key:
            self._cache = self._build(s)
            self._key = key
        return self._cache

    # ── 查询辅助 ─────────────────────────────────────────────────────────
    def _term_entity(self, model: Dict[str, Any], key: str) -> Optional[Entity]:
        return next((e for e in model["entities"] if e.type == "reward_term" and e.name == key), None)

    def _edges_of(self, model: Dict[str, Any], key: str, rel: str) -> List[Edge]:
        sid = f"reward_term:{key}"
        return [e for e in model["edges"] if e.src == sid and e.rel == rel]

    # ── 对齐("永不悄悄错")────────────────────────────────────────────────
    def check_alignment(self, primary_override: Optional[str] = None) -> Dict[str, List[str]]:
        model = self._ensure()
        s: RobotSources = model["sources"]
        cached = {e.name: (e.source.sha256 if e.source else "")
                  for e in model["entities"] if e.type == "reward_term" and e.freshness != "cited_only"}
        ctext = primary_override if primary_override is not None else (_read_allowlisted(s.reward_file) or "")
        cur = derive_reward_terms(
            ctext, s.reward_file, func_name=s.reward_func, cfg_class=s.reward_cfg_class,
            gate_const=s.reward_gate_const, group_const=s.reward_group_const, strict=True)
        cur_sha = {e.name: (e.source.sha256 if e.source else "")
                   for e in cur["entities"] if e.type == "reward_term"}
        fresh, changed, vanished = [], [], []
        for key, sha in cached.items():
            if key not in cur_sha:
                vanished.append(key)
            elif cur_sha[key] == sha:
                fresh.append(key)
            else:
                changed.append(key)
        return {"fresh": sorted(fresh), "changed": sorted(changed), "vanished": sorted(vanished)}

    # ── 覆盖 / 漂移自审 ───────────────────────────────────────────────────
    def reward_model_audit(self) -> KnowledgeAudit:
        model = self._ensure()
        entities = model["entities"]
        edges = model["edges"]
        group_of: Dict[str, str] = model["group_of"]
        config_defaults: Dict[str, float] = model["config_defaults"]
        derived_terms: set = model["derived_terms"]
        s: RobotSources = model["sources"]

        store_terms = {e.name for e in entities if e.type == "reward_term"}
        code_terms = set(derived_terms) | set(group_of.keys())
        missing = sorted(code_terms - store_terms)
        dangling = sorted(store_terms - code_terms)

        # 死权重:边没抓到、且代码文本里也没出现的 w_*;出现但没边的只算"未捕获"(推导器边界)。
        weighted = {e.dst.split(":", 1)[1] for e in edges if e.rel == "weighted_by"}
        w_fields = {k for k in config_defaults if k.startswith("w_")}
        unref = w_fields - weighted
        ptext = _read_allowlisted(s.reward_file) or ""
        dead_params = sorted(p for p in unref if p not in ptext)
        uncaptured = sorted(p for p in unref if p in ptext)

        ungrouped = sorted(derived_terms - set(group_of.keys()))
        notes: List[str] = []
        if uncaptured:
            notes.append(f"权重在代码中出现、但未被推成 weighted_by 边(推导器边界,非死): {uncaptured}")
        if ungrouped:
            notes.append(f"推出但不在分组清单的项(代码自身漏分组): {ungrouped}")
        cited = sorted(e.name for e in entities if e.type == "reward_term" and e.freshness == "cited_only")
        if cited:
            notes.append(f"环境专属项仅覆盖/引用(结构留后续层): {len(cited)} 项")
        if model["errors"]:
            notes.append(f"推导告警: {model['errors']}")

        align = self.check_alignment()
        return KnowledgeAudit(
            fresh=align["fresh"], changed=align["changed"],
            vanished=sorted(set(align["vanished"]) | set(dangling)),
            missing=missing, dead_params=dead_params, unmeasured=[], notes=notes,
        )

    # ── 装配(durable 结构 + live 权重经边现读)────────────────────────────
    def assemble_reward_term(self, key: str, effective_config_text: str = "") -> Optional[RewardTermView]:
        model = self._ensure()
        e = self._term_entity(model, key)
        if e is None:
            return None
        weight_param = None
        wedges = self._edges_of(model, key, "weighted_by")
        if wedges:
            weight_param = wedges[0].dst.split(":", 1)[1]
        gates = [ed.dst.split(":", 1)[1] for ed in self._edges_of(model, key, "gated_by")]
        tensions = [ed.dst.split(":", 1)[1] for ed in self._edges_of(model, key, "trades_off_with")]
        weight_default = e.attrs.get("weight_default")

        weight_value: Optional[float] = None
        weight_source = "unknown"
        if weight_param:
            live = _reward_weight_from_config(effective_config_text, weight_param)
            if live is not None:
                weight_value, weight_source = live, "effective_config"
            elif weight_default is not None:
                weight_value, weight_source = float(weight_default), "code_default"

        live_snippet, freshness = "", e.freshness
        if e.source and e.source.file:
            text = _read_allowlisted(e.source.file)
            if text:
                lines = text.splitlines()
                s0, t0 = max(0, e.source.line_start - 1), min(len(lines), e.source.line_end)
                live_snippet = "\n".join(f"{i + 1}: {lines[i]}" for i in range(s0, t0))[:1600]
                if e.freshness != "cited_only":
                    freshness = "fresh" if _sha("\n".join(lines[s0:t0])) == e.source.sha256 else "changed"

        note = " ; ".join(p for p in (
            e.attrs.get("note", ""),
            e.attrs.get("inline_default_note", ""),
            e.attrs.get("blended_weight_note", ""),
        ) if p)

        return RewardTermView(
            term=key,
            sign=e.attrs.get("sign", "unknown"),
            kernel_kind=e.attrs.get("kernel_kind", ""),
            weight_param=weight_param,
            weight_params=list(e.attrs.get("weight_params", [])),
            weight_value=weight_value,
            weight_source=weight_source,  # type: ignore[arg-type]
            weight_default=weight_default,
            gates=gates,
            measured_as=f"reward.{key}",
            trades_off_with=tensions,
            computed_in=e.source,
            live_snippet=live_snippet,
            freshness=freshness,  # type: ignore[arg-type]
            note=note,
        )

    def get_reward_model(self, query: str = "", term: str = "",
                         effective_config_text: str = "") -> RewardModelResult:
        model = self._ensure()
        audit = self.reward_model_audit()
        if term:
            view = self.assemble_reward_term(term, effective_config_text)
            terms = [view] if view else []
            notes = [] if view else [f"未找到奖励项 '{term}'"]
            return RewardModelResult(available=bool(view), terms=terms, audit=audit, notes=notes)

        names = [e.name for e in model["entities"] if e.type == "reward_term"]
        if query:
            toks = [t for t in query.lower().replace("_", " ").split() if len(t) >= 2]

            def score(n: str) -> int:
                nl = n.lower()
                return sum(1 for t in toks if t in nl)

            ranked = sorted(names, key=lambda n: (-score(n), n))
            ranked = [n for n in ranked if score(n) > 0] or names[:6]
        else:
            ranked = names[:6]
        views = [v for v in (self.assemble_reward_term(n, effective_config_text) for n in ranked[:8]) if v]
        return RewardModelResult(available=True, terms=views, audit=audit)


def _reward_weight_from_config(effective_config_text: str, param: str) -> Optional[float]:
    """从 effective_config.yaml 的 reward.<param> 现读当前权重值;缺则 None。"""
    if not effective_config_text.strip():
        return None
    try:
        import yaml
        raw = yaml.safe_load(effective_config_text) or {}
    except Exception:  # noqa: BLE001
        return None
    reward = raw.get("reward") if isinstance(raw, dict) else None
    if isinstance(reward, dict) and param in reward:
        try:
            return float(reward[param])
        except (TypeError, ValueError):
            return None
    return None


# 模块级单例 + 便捷函数(公共 API 从 __init__ 汇出)。
_STORE = KnowledgeStore()


def get_reward_model(query: str = "", term: str = "", effective_config_text: str = "") -> RewardModelResult:
    return _STORE.get_reward_model(query=query, term=term, effective_config_text=effective_config_text)


def assemble_reward_term(term: str, effective_config_text: str = "") -> Optional[RewardTermView]:
    return _STORE.assemble_reward_term(term, effective_config_text)


def reward_model_audit() -> KnowledgeAudit:
    return _STORE.reward_model_audit()


def check_alignment(primary_override: Optional[str] = None) -> Dict[str, List[str]]:
    return _STORE.check_alignment(primary_override)


# ── 机器人本体域:从激活机器人的资产推导 + 核对手写 robot_profile ─────────────
_robot_cache: Dict[str, Any] = {}


def _robot_result(d: Dict[str, Any], s: RobotSources) -> RobotModelResult:
    facts: List[RobotFact] = list(d["facts"])
    notes: List[str] = list(d.get("errors") or [])
    # 拿资产推断值去核那份手写的 robot_profile(现有自查抓不到的漂移)。
    try:
        from ..robot_profile import get_robot_profile
        prof = get_robot_profile(s.robot_id)
        facts.append(RobotFact(key="dof_profile", label="自由度(robot_profile 手写)", value=prof.dof,
                               note="来自 robot_profile.py 手写 profile;与上面资产推断值互核"))
        facts.append(RobotFact(key="joint_order_profile", label="关节顺序(profile)", value=list(prof.joint_order)))
        facts.append(RobotFact(key="foot_links_profile", label="足链(profile)", value=list(prof.foot_links)))
        import re as _re
        dof_fact = next((f for f in facts if f.key == "dof_inferred"), None)
        if dof_fact is not None:
            m = _re.match(r"\s*(\d+)", str(dof_fact.value))
            if m and int(m.group(1)) != prof.dof:
                notes.append(f"漂移:资产推断自由度={m.group(1)} ≠ robot_profile.dof={prof.dof}(手写 profile 可能过时)")
    except Exception:  # noqa: BLE001
        pass
    return RobotModelResult(available=bool(facts), facts=facts, notes=notes)


def get_robot_model(source_override: Optional[str] = None, robot_id: Optional[str] = None) -> RobotModelResult:
    """机器人本体知识:从激活机器人的资产 AST 推导 + 核对手写 profile。按 robot_id + mtime 缓存。"""
    from .robot_deriver import derive_robot_model
    s = get_robot_sources(robot_id or active_robot_id())
    constant_sources = {
        namespace: _read_allowlisted(path) or ""
        for namespace, path in s.asset_constant_files
    }
    if source_override is not None:
        return _robot_result(derive_robot_model(source_override, s.asset_file, constant_sources), s)
    key = (
        s.robot_id,
        _mtime(s.asset_file),
        tuple((path, _mtime(path)) for _namespace, path in s.asset_constant_files),
    )
    if _robot_cache.get("key") != key:
        text = _read_allowlisted(s.asset_file) or ""
        _robot_cache["model"] = _robot_result(derive_robot_model(text, s.asset_file, constant_sources), s)
        _robot_cache["key"] = key
    return _robot_cache["model"]


# ── 机制域:阶段门控阈值(durable 代码默认 + effective_config 现值)──────────────
_curr_cache: Dict[str, Any] = {}


def _gate_live_from_config(effective_config_text: str, name: str) -> Optional[float]:
    """从 effective_config 的 env.curriculum(或 curriculum).<name> 现读本次 run 的门控值。"""
    if not effective_config_text.strip():
        return None
    try:
        import yaml
        raw = yaml.safe_load(effective_config_text) or {}
    except Exception:  # noqa: BLE001
        return None
    cur = None
    if isinstance(raw, dict):
        env = raw.get("env")
        if isinstance(env, dict) and isinstance(env.get("curriculum"), dict):
            cur = env["curriculum"]
        elif isinstance(raw.get("curriculum"), dict):
            cur = raw["curriculum"]
    if isinstance(cur, dict) and name in cur:
        try:
            return float(cur[name])
        except (TypeError, ValueError):
            return None
    return None


def _curr_result(d: Dict[str, Any], effective_config_text: str) -> CurriculumModelResult:
    gates = []
    for g in d["gates"]:
        gg = g.model_copy()
        live = _gate_live_from_config(effective_config_text, g.name)
        if live is not None:
            gg.live_value, gg.live_source = live, "effective_config"
            if g.code_default is not None and live != g.code_default:
                gg.note = (gg.note + " ; " if gg.note else "") + f"本次 run 与代码默认 {g.code_default} 不同"
        elif g.code_default is not None:
            gg.live_value, gg.live_source = g.code_default, "code_default"
        gates.append(gg)
    return CurriculumModelResult(available=bool(gates), gates=gates, notes=list(d.get("errors") or []))


def get_curriculum_model(effective_config_text: str = "", robot_id: Optional[str] = None,
                         source_override: Optional[str] = None) -> CurriculumModelResult:
    """阶段门控阈值知识:代码默认从 taili_amp_env_cfg.py AST 推导,当前值从 effective_config 现读。"""
    from .curriculum_deriver import derive_gate_thresholds
    s = get_robot_sources(robot_id or active_robot_id())
    if source_override is not None:
        return _curr_result(derive_gate_thresholds(source_override, s.curriculum_file), effective_config_text)
    key = (s.robot_id, _mtime(s.curriculum_file))
    if _curr_cache.get("key") != key:
        text = _read_allowlisted(s.curriculum_file) or ""
        _curr_cache["derived"] = derive_gate_thresholds(text, s.curriculum_file)
        _curr_cache["key"] = key
    return _curr_result(_curr_cache["derived"], effective_config_text)
