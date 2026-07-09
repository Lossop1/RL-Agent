"""
LLM frontend — USER-FACING layer ONLY.

This package wraps an external LLM (DeepSeek / OpenAI / etc.) for two
narrowly-scoped jobs:

  1. `task_intake` — translate user natural-language task descriptions
     into a typed TaskSpec the autotuner can act on.
  2. `verdict_narrate` — translate a structured verdict back into a
     human-readable Chinese summary.

Hard rules:

  - LLM NEVER sees raw cfg, raw log, raw rollout CSV.
  - LLM sees only summaries we explicitly chose to surface.
  - All LLM responses are validated against a JSON schema.
  - Validation failure → 1 retry → fallback to a deterministic default.
  - All LLM calls are logged to disk for replay.
  - temperature=0, response_format=json_schema, top_p=1.

Anything that smells like "LLM decides what cfg to use" is out of scope.
The orchestrator + cfg writer + diagnostic gate are deterministic; LLM
only translates between user and that core.
"""
