"""奖励知识库(Stage 1)单测。

锁定"真设计"的每一条:从代码 AST 推导(非手抄)、对齐(改一行能测出)、覆盖自审
(库缺/幽灵能报)、只放稳定结构而现值现读、关系(拆台边)复用记分牌。全程纯本地、免 torch。
"""
import pytest

from autotuner.locomotion_console import agent
from autotuner.locomotion_console.code_knowledge import _read_allowlisted
from autotuner.locomotion_console.knowledge_model import (
    assemble_reward_term,
    check_alignment,
    get_code_facts,
    get_curriculum_model,
    get_reward_model,
    get_robot_model,
    reward_model_audit,
)
from autotuner.locomotion_console.knowledge_model.robot_sources import active_robot_id, get_robot_sources
from autotuner.locomotion_console.knowledge_model.reward_deriver import derive_reward_terms
from autotuner.locomotion_console.knowledge_model.schema import Entity
from autotuner.locomotion_console.knowledge_model.store import KnowledgeStore

_REL = "autotuner/taili_core/taili_reward.py"
_SRC = _read_allowlisted(_REL) or ""


def _derive():
    return derive_reward_terms(_SRC, _REL)


def _term(d, name):
    return next((e for e in d["entities"] if e.type == "reward_term" and e.name == name), None)


def _weights(d, name):
    return [e.dst.split(":", 1)[1] for e in d["edges"] if e.src == f"reward_term:{name}" and e.rel == "weighted_by"]


def _gates(d, name):
    return [e.dst.split(":", 1)[1] for e in d["edges"] if e.src == f"reward_term:{name}" and e.rel == "gated_by"]


# ① 推导抽出已知项的结构(权重/门控/符号/核/来源)。
def test_derivation_extracts_tracking_lin():
    d = _derive()
    t = _term(d, "tracking_lin")
    assert t is not None
    assert t.attrs["sign"] == "reward"
    assert t.attrs["kernel_kind"] == "exp_kernel"
    assert _weights(d, "tracking_lin") == ["w_tracking_lin"]
    # 主跟踪只由命令作用域、真实终止和速度大小退让控制；质量约束保持加性。
    gates = set(_gates(d, "tracking_lin"))
    assert {"moving_gate", "hard_survival_gate"} <= gates
    assert "capability_gate" not in gates
    assert "effective_tracking_gate" not in gates
    assert t.source is not None
    assert t.source.file.endswith("taili_reward.py")
    assert t.source.symbol == 'comp["tracking_lin"]'
    assert t.source.line_start > 0


# 门控键与 total 不被当奖励项;惩罚项符号正确。
def test_gate_keys_and_total_not_reward_terms():
    names = {e.name for e in _derive()["entities"] if e.type == "reward_term"}
    for g in ("stable_motion_gate", "moving_gate", "quality_gate", "validated_tracking_gate", "total"):
        assert g not in names


def test_penalty_sign():
    d = _derive()
    assert _term(d, "off_axis").attrs["sign"] == "penalty"
    assert _term(d, "action_rate").attrs["sign"] == "penalty"


# ⑥ 影子案例:tracking_lin_far 的 getattr 内联默认 0.6,但字段默认 1.0 → 以字段为准(证推导非手抄)。
def test_shadow_uses_field_default_not_inline():
    d = _derive()
    assert _term(d, "tracking_lin_far").attrs.get("inline_default") == 0.6
    v = assemble_reward_term("tracking_lin_far", "")
    assert v.weight_value == 1.0 and v.weight_source == "code_default"


# 混合权重(基 + 后期/尾部)要全抓到,不能只抓一个。
def test_blended_weights_all_captured():
    d = _derive()
    assert set(_weights(d, "stance_slip")) >= {"w_stance_slip", "w_stance_slip_late"}
    assert set(_weights(d, "landing_impact")) >= {"w_landing_impact", "w_landing_impact_late"}


# ⑥ 现读权重:有配置取 effective_config,无配置回退代码默认(带标注)。
def test_live_weight_from_effective_config():
    v = assemble_reward_term("tracking_lin", "reward:\n  w_tracking_lin: 3.5\n")
    assert v.weight_value == 3.5 and v.weight_source == "effective_config"
    assert v.computed_in.file.endswith("taili_reward.py") and v.live_snippet
    v2 = assemble_reward_term("tracking_lin", "")
    assert v2.weight_value == 2.0 and v2.weight_source == "code_default"


# ③ 对齐:改一行源码,重推能精确报出该项 changed、其余 fresh(不动文件)。
def test_alignment_flags_mutated_source():
    mut = _SRC.replace("cfg.w_tracking_lin * tracking_lin_raw", "cfg.w_stand * tracking_lin_raw", 1)
    assert mut != _SRC
    al = check_alignment(primary_override=mut)
    assert "tracking_lin" in al["changed"]
    assert "tracking_lin" not in al["fresh"]
    assert "off_axis" in al["fresh"]


# ④ 覆盖:库里丢了一项 → audit 报 missing(代码有、库里没有)。
def test_coverage_missing_when_entity_dropped():
    st = KnowledgeStore()
    st._ensure()
    st._cache["entities"] = [e for e in st._cache["entities"]
                             if not (e.type == "reward_term" and e.name == "tracking_lin")]
    assert "tracking_lin" in st.reward_model_audit().missing


# ⑤ 漂移:库里有代码没有的幽灵项 → audit 报 dangling(并入 vanished)。
def test_dangling_when_phantom_entity():
    st = KnowledgeStore()
    st._ensure()
    st._cache["entities"].append(Entity(id="reward_term:ghost_xyz", type="reward_term", name="ghost_xyz"))
    assert "ghost_xyz" in st.reward_model_audit().vanished


# 混合权重全抓到后,不应有死权重误报。
def test_no_false_dead_params():
    assert reward_model_audit().dead_params == []


# ⑦ 拆台边复用记分牌 TENSIONS:装配时能取到相互拆台的对端。
def test_tension_edges_from_scoreboard():
    assert "clearance_over" in assemble_reward_term("clearance_under", "").trades_off_with
    assert "progress_gate" in assemble_reward_term("stand", "").trades_off_with


# ⑧ 工具端到端:已注册、在文档、返回 durable 结构 + 现读权重 + 只读边界。
def test_tool_registered_and_returns_structure():
    assert "get_reward_model" in agent.TOOLS
    assert "get_reward_model(query" in agent._TOOLS_DOC
    r = get_reward_model(term="tracking_lin", effective_config_text="reward:\n  w_tracking_lin: 3.5\n")
    assert r.available and r.terms
    v = r.terms[0]
    assert v.term == "tracking_lin" and v.weight_value == 3.5
    assert v.computed_in.file.endswith("taili_reward.py")
    assert r.permission_boundary


# 覆盖自审会如实报出代码自身的漏分组(touchdown_slip 在代码里但不在 _REWARD_GROUP_OF)。
def test_audit_surfaces_code_ungrouped_term():
    notes = " ".join(reward_model_audit().notes)
    assert "touchdown_slip" in notes


# ── 去 taili 化:源从按 robot_id 的描述符解析,不写死 ──────────────────────────
def test_sources_resolved_from_descriptor_not_hardcoded():
    s = get_robot_sources()
    assert s.robot_id == active_robot_id()                 # 激活机器人从 robot_profile 解析
    assert s.reward_func == "compute_reward_components"     # 符号名来自描述符
    assert s.reward_file.endswith("taili_reward.py")        # 当前机器人恰是 taili,但经描述符


# ── 机器人本体域:从资产推导 + 核对手写 profile ──────────────────────────────
def _rf(rm, key):
    return next((f for f in rm.facts if f.key == key), None)


def test_robot_model_derives_from_asset():
    rm = get_robot_model()
    assert _rf(rm, "base_nominal_height").value == pytest.approx(0.5471961541175842)
    act = _rf(rm, "actuator_model").value
    assert act["model"] == "DCMotor" and set(act["joint_types"]) == {"hip", "thigh", "calf"}
    assert act["stiffness"] == 120.0 and act["damping"] == 10.0


def test_robot_model_verifies_profile_no_drift():
    rm = get_robot_model()
    assert "12" in str(_rf(rm, "dof_inferred").value)
    assert _rf(rm, "dof_profile").value == 12
    assert not any("漂移" in n for n in rm.notes)


def test_robot_model_flags_profile_drift():
    # 喂一个只有 2 类关节的资产 → 推断 8 自由度 ≠ 手写 profile 的 12 → 报漂移。
    mutant = (
        "from isaaclab.assets.articulation import ArticulationCfg\n"
        "from isaaclab.actuators import DCMotorCfg\n"
        "TAILI_DOG_CFG = ArticulationCfg(\n"
        "    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.52), joint_pos={'.*_hip_joint': 0.0}),\n"
        "    actuators={'legs': DCMotorCfg(joint_names_expr=['.*_hip_joint', '.*_thigh_joint'], stiffness=1.0, damping=1.0)},\n"
        ")\n"
    )
    rm = get_robot_model(source_override=mutant)
    assert any("漂移" in n for n in rm.notes)


def test_robot_tool_registered():
    assert "get_robot_model" in agent.TOOLS
    assert "get_robot_model()" in agent._TOOLS_DOC


# ── 机制域:阶段门控阈值(代码默认推导 + effective_config 现值)──────────────
def _gate(r, name):
    return next((g for g in r.gates if g.name == name), None)


def test_curriculum_derives_gate_defaults_with_provenance():
    r = get_curriculum_model()
    assert _gate(r, "phase_gate_prog_0").code_default == 0.60
    assert _gate(r, "phase_gate_terrain_2").code_default == 6.0
    assert _gate(r, "phase_gate_fall_2").code_default == 0.05
    assert _gate(r, "penalty_ramp_intervals").code_default == 25.0
    assert _gate(r, "terrain_levels").code_default == 10.0     # 来自 TerrainGeneratorCfg(num_rows=)
    g = _gate(r, "phase_gate_prog_0")
    assert g.source.file.endswith("taili_amp_env_cfg.py")
    assert g.source.symbol == "phase_gate_prog_0"
    assert g.source.line_start > 0


def test_curriculum_live_from_effective_config():
    cfg = "env:\n  curriculum:\n    phase_gate_prog_2: 0.62\n"
    r = get_curriculum_model(effective_config_text=cfg)
    g = _gate(r, "phase_gate_prog_2")
    assert g.live_value == 0.62 and g.live_source == "effective_config"
    assert "不同" in g.note                                     # 现值≠代码默认 0.5,标注出来
    r0 = get_curriculum_model(effective_config_text="")
    assert _gate(r0, "phase_gate_prog_2").live_source == "code_default"


def test_curriculum_tool_registered():
    assert "get_curriculum_model" in agent.TOOLS
    assert 'get_curriculum_model(query="")' in agent._TOOLS_DOC


# ── 通用代码取证:任意问题 → 结构化定义 + 出处;查不到就明说(知道自己不知道)──────
def _cf_names(r):
    return {d["name"] for d in r["definitions"]}


def test_code_facts_finds_network_classes():
    r = get_code_facts("actor 和 critic 是什么网络结构")
    assert _cf_names(r) & {"TCNEncoder", "EquivariantActor", "TerrainPerceiver"}
    assert all("file" in d and "line" in d for d in r["definitions"])   # 带出处


def test_code_facts_finds_scan_grid():
    r = get_code_facts("地形高度扫描 height scan 网格 多少射线")
    assert "n_height_scan" in _cf_names(r)                              # =17*11=187 条射线


def test_code_facts_reports_not_found():
    r = get_code_facts("qzwxqzv_nosuch_thing")
    assert "qzwxqzv_nosuch_thing" in r["not_found_symbols"]             # 知道自己没找到


def test_code_facts_falls_back_to_function_internal_source():
    r = get_code_facts("为什么 phase_count 不增加")
    snippets = r["source_context"]["snippets"]
    assert r["complete"] is True
    assert any("_phase_count" in item["text"] for item in snippets), [
        r["query"], r["tokens"], r["source_context"]["terms_used"], r["source_context"]["files_considered"],
        [(item["path"], item["line_start"], item["matched_terms"]) for item in snippets],
    ]
    assert any(item["path"].endswith("taili_amp_env.py") for item in snippets)


def test_code_facts_internal_search_generalizes_to_observation_state():
    r = get_code_facts("为什么 _tick_history 没变化")
    snippets = r["source_context"]["snippets"]
    assert any("_tick_history" in item["text"] for item in snippets)
    assert any(item["path"].endswith(("blind_tp_env.py", "taili_obs.py")) for item in snippets)


def test_code_facts_tool_registered():
    assert "get_code_facts" in agent.TOOLS
    assert "get_code_facts(query" in agent._TOOLS_DOC
