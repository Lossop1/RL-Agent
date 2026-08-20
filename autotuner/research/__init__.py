"""与具体机器人、训练框架和界面无关的强化学习研究闭环。

研究模块采用惰性导出，避免仅解析产品合同时加载 LLM、远程传输或实验后端。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "COORDINATOR_SCHEMA": ("coordinator", "COORDINATOR_SCHEMA"),
    "CoordinatedDeployment": ("coordinator", "CoordinatedDeployment"),
    "CoordinatedTrainingStart": ("coordinator", "CoordinatedTrainingStart"),
    "CoordinatedEvaluation": ("coordinator", "CoordinatedEvaluation"),
    "DispositionAssessment": ("coordinator", "DispositionAssessment"),
    "EvaluationSubmission": ("coordinator", "EvaluationSubmission"),
    "EvaluatorResult": ("coordinator", "EvaluatorResult"),
    "EvidenceIngestion": ("coordinator", "EvidenceIngestion"),
    "ResearchCoordinator": ("coordinator", "ResearchCoordinator"),
    "ResearchCoordinatorError": ("coordinator", "ResearchCoordinatorError"),
    "ResearchTrainingStarter": ("coordinator", "ResearchTrainingStarter"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), symbol)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
