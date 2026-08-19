"""
LLM 协调边界。

This package wraps an external LLM (DeepSeek / OpenAI / etc.) for two
narrowly-scoped jobs:

  1. task_intake：依据当前产品合同把自然语言任务编译为 TaskRequest；
     旧 TaskSpec 入口仅保留兼容。
  2. `verdict_narrate` — translate a structured verdict back into a
     human-readable Chinese summary.

Hard rules:

  - LLM NEVER sees raw cfg, raw log, raw rollout CSV.
  - LLM sees only summaries we explicitly chose to surface.
  - All LLM responses are validated against a JSON schema.
  - Validation failure → 1 retry → fallback to a deterministic default.
  - All LLM calls are logged to disk for replay.
  - temperature=0, response_format=json_schema, top_p=1.

LLM 可以提出候选配置和新机制，但不能直接执行。合同编译、静态校验、
远程变更、诊断验收和回滚保持确定性，并记录完整证据与谱系。
"""

from .task_intake import DynamicTaskIntakeResult, translate, translate_dynamic

__all__ = ["DynamicTaskIntakeResult", "translate", "translate_dynamic"]
