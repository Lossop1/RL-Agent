"""分层知识设施（Stage 1：奖励模型域）。

设计原则（见计划文件 idempotent-spinning-cupcake）:
- 从代码 AST 推导,不手抄——从文件推的,不可能和文件矛盾。
- 对齐 = "永不悄悄错":每条 durable 记录挂来源代码段的 sha256,变了判过时、重推;漂移审计列 missing/dangling。
- 关系 = 实体 + 有类型的边(不是图数据库)。
- 只放稳定结构;当前权重等易变值永不入库,答题时从 effective_config 现读、经边拼上。

公共 API 在本模块底部从 store/reward_deriver 汇出。
"""
from __future__ import annotations

from .schema import (
    CurriculumModelResult,
    Edge,
    Entity,
    GateThreshold,
    KnowledgeAudit,
    RewardModelResult,
    RewardTermView,
    RobotFact,
    RobotModelResult,
    SourceRef,
)
from .code_facts import get_code_facts
from .store import (
    KnowledgeStore,
    assemble_reward_term,
    check_alignment,
    get_curriculum_model,
    get_reward_model,
    get_robot_model,
    reward_model_audit,
)

__all__ = [
    # schema
    "SourceRef",
    "Entity",
    "Edge",
    "RewardTermView",
    "KnowledgeAudit",
    "RewardModelResult",
    "RobotFact",
    "RobotModelResult",
    "GateThreshold",
    "CurriculumModelResult",
    # store / public API
    "KnowledgeStore",
    "get_code_facts",
    "get_reward_model",
    "assemble_reward_term",
    "reward_model_audit",
    "check_alignment",
    "get_robot_model",
    "get_curriculum_model",
]
