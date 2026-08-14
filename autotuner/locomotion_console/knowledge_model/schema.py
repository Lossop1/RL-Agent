"""知识模型的实体 / 边 / 视图 schema（Stage 1：奖励域）。

只放稳定结构:实体/边描述"奖励项的公式结构、权重参数名、门控、代码位置、相互拆台"。
当前权重**值**不在这里——它易变,答题时从 effective_config 现读、经 weighted_by 边拼上。
每条来源都带 SourceRef.sha256,用于对齐(源码段变了就能判过时,不会悄悄供旧)。

枚举预留了全部实体/边类型;Stage 1 只填 param/reward_term/gate/telemetry_field，
后续层(config-schema/机制/本体/spec)加推导器时不用改 schema。风格照 scoreboard.ScoreboardMetric。
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    """一条知识来自哪段代码:文件 + 行范围 + 该段源码的 sha256(对齐用)。"""
    file: str                       # allowlist 内的仓库相对路径
    line_start: int = 0
    line_end: int = 0
    sha256: str = ""                # sha256(ast.get_source_segment(...))（规约后）
    symbol: str = ""                # 如 'comp["tracking_lin"]' 或 'exp_kernel'


# 实体类型:Stage 1 只用 param/reward_term/gate/telemetry_field;其余预留给后续层。
EntityType = Literal[
    "param", "reward_term", "gate", "robot_part", "telemetry_field", "spec_item", "mechanism",
]
# 边类型:Stage 1 用 weighted_by/computed_in/measured_as/gated_by/trades_off_with/consumed_in;controls 预留。
EdgeType = Literal[
    "weighted_by", "computed_in", "measured_as", "consumed_in", "trades_off_with", "gated_by", "controls",
]
# 对齐状态:fresh=与源码一致;changed=源码段变了;vanished=源码里找不到;cited_only=没硬解、仅引用。
Freshness = Literal["fresh", "changed", "vanished", "cited_only"]


class Entity(BaseModel):
    id: str                         # 稳定自然键,如 "reward_term:tracking_lin" / "param:w_tracking_lin"
    type: EntityType
    name: str
    attrs: dict[str, Any] = Field(default_factory=dict)   # 如 sign / kernel_kind / weight_default
    source: Optional[SourceRef] = None
    freshness: Freshness = "fresh"


class Edge(BaseModel):
    src: str                        # 实体 id
    rel: EdgeType
    dst: str                        # 实体 id（或外部引用串）
    source: Optional[SourceRef] = None
    note: str = ""


class RewardTermView(BaseModel):
    """装配好的一项:durable 结构 + 经边现读的 live 权重值。给 agent 的一次性答案。"""
    term: str
    sign: Literal["reward", "penalty", "mixed", "unknown"] = "unknown"
    kernel_kind: str = ""                                   # exp_kernel/far_kernel/linear/cited
    weight_param: Optional[str] = None                      # 主权重参数名
    weight_params: List[str] = Field(default_factory=list)  # 全部权重(如基权重 + 后期/尾部混合)
    weight_value: Optional[float] = None                    # LIVE,主权重现读值,永不入库
    weight_source: Literal["effective_config", "code_default", "unknown"] = "unknown"
    weight_default: Optional[float] = None                  # RewardConfig 字段默认(durable 结构默认)
    gates: List[str] = Field(default_factory=list)
    measured_as: str = ""                                   # 如 "reward.tracking_lin"
    trades_off_with: List[str] = Field(default_factory=list)
    computed_in: Optional[SourceRef] = None
    live_snippet: str = ""                                  # 答题时现读的源码窗口,不转写核数学
    freshness: Freshness = "fresh"
    note: str = ""


class KnowledgeAudit(BaseModel):
    """覆盖 / 漂移自审(镜像 scoreboard 覆盖自检):让盲区可见,不悄悄烂掉。"""
    fresh: List[str] = Field(default_factory=list)
    changed: List[str] = Field(default_factory=list)
    vanished: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)       # 代码里有、库里没有的奖励项
    dead_params: List[str] = Field(default_factory=list)   # RewardConfig 有 w_*、没被任何项引用
    unmeasured: List[str] = Field(default_factory=list)    # reward.<key> 未被遥测 emit
    notes: List[str] = Field(default_factory=list)


class GateThreshold(BaseModel):
    """一个阶段门控/课程阈值:代码默认(durable,带出处)+ 本次 run 现值(live)。"""
    name: str
    label: str = ""
    code_default: Optional[float] = None                    # 从 taili_amp_env_cfg.py 推导
    live_value: Optional[float] = None                      # 从 effective_config 现读
    live_source: Literal["effective_config", "code_default", "unknown"] = "unknown"
    source: Optional[SourceRef] = None
    freshness: Freshness = "fresh"
    note: str = ""


class CurriculumModelResult(BaseModel):
    """get_curriculum_model 的返回:阶段门控阈值(代码默认 + 现值)+ 只读边界。"""
    available: bool = True
    gates: List[GateThreshold] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    permission_boundary: str = (
        "只读派生视图:门控阈值结构/默认从 taili_amp_env_cfg.py AST 推导,当前值从 effective_config 现读。"
        "不写、不执行、不碰训练。"
    )


class RobotFact(BaseModel):
    """机器人本体的一条事实(基座高度/默认关节角/执行器/自由度),带来源与新鲜度。"""
    key: str
    label: str
    value: Any = None
    source: Optional[SourceRef] = None
    note: str = ""
    freshness: Freshness = "fresh"


class RobotModelResult(BaseModel):
    """get_robot_model 的返回:本体事实 + 只读边界。全部为稳定结构(无 live 值)。"""
    available: bool = True
    facts: List[RobotFact] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    permission_boundary: str = (
        "只读派生视图:机器人本体从资产 taili.py AST 推导(稳定结构,无 live 值);"
        "自由度精确值以 URDF 为准。不写、不执行、不碰训练。"
    )


class RewardModelResult(BaseModel):
    """get_reward_model 的返回:命中的项 + 覆盖自审 + 只读边界声明。"""
    available: bool = True
    terms: List[RewardTermView] = Field(default_factory=list)
    audit: KnowledgeAudit = Field(default_factory=KnowledgeAudit)
    permission_boundary: str = (
        "只读派生视图:结构从本地 allowlist 代码 AST 推导,当前权重值从 effective_config 现读;"
        "不写、不执行、不碰训练。"
    )
    notes: List[str] = Field(default_factory=list)
