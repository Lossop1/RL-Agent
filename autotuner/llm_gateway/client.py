"""
Tiny OpenAI-SDK-compatible LLM client wrapper.

Reuses the same `config/llm.json` convention as the legacy
`autotuner/tuning/llm_openai_advisor.py`, but exposes a minimal
"call with schema, get validated dict back" interface for the
user-facing frontend layer.

Constraints enforced here (so individual entry points don't repeat):

  - temperature=0 (deterministic)
  - response_format=json_object (structured)
  - 1 retry on JSON parse failure
  - Hard failure → caller's responsibility to provide deterministic
    fallback; we don't pretend the LLM said something useful.
  - Every call written to local audit log for replay.
"""

from __future__ import annotations

import itertools
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_LLM_CONFIG_PATH = Path("config/llm.json")


def _state_root() -> Path:
    return Path(os.environ.get(
        "LOCOMOTION_CONSOLE_STATE_ROOT",
        os.path.join(tempfile.gettempdir(), "locomotion_console_state"),
    ))


def _audit_log_dir() -> Path:
    override = os.environ.get("LOCOMOTION_CONSOLE_LLM_AUDIT_DIR", "")
    return Path(override) if override else _state_root() / "llm_gateway_audit"


@dataclass(frozen=True)
class LLMResponse:
    parsed: Optional[Dict[str, Any]]
    raw_text: str
    model: str
    elapsed_s: float
    attempt: int                    # which retry attempt produced this
    error: Optional[str] = None


def _resolve_api_key(raw: str) -> str:
    if raw.startswith("sk-") or raw.startswith("key-"):
        return raw
    return os.environ.get(raw, "")


def _load_config(path: Path = DEFAULT_LLM_CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"config/llm.json not found; LLM gateway requires it. "
            f"See autotuner/tuning/llm_openai_advisor.py for format."
        )
    cfg = json.loads(path.read_text(encoding="utf-8"))
    override_path = _state_root() / "llm_profile.json"
    if override_path.exists():
        try:
            override = json.loads(override_path.read_text(encoding="utf-8"))
            for key in ("api_key_env_var", "base_url", "model"):
                if override.get(key):
                    cfg[key] = override[key]
        except Exception:
            pass
    return cfg


_audit_seq = itertools.count()


def _audit_log(entry: Dict[str, Any]) -> None:
    audit_dir = _audit_log_dir()
    audit_dir.mkdir(parents=True, exist_ok=True)
    # Second-granularity stamps + overwrite lost audit records when two LLM calls landed in the
    # same second (and concurrent calls raced). Make the name unique: sub-second + pid + a counter.
    now = time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    uniq = f"{int((now % 1) * 1_000_000):06d}_{os.getpid()}_{next(_audit_seq)}"
    f = audit_dir / f"llm_call_{stamp}_{uniq}.json"
    f.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")


def call_llm_with_schema(system_prompt: str,
                            user_prompt: str,
                            schema_name: str,
    max_retries: int = 1,
                            ) -> LLMResponse:
    """Call configured LLM, expect JSON object back, return parsed.

    Schema validation is the CALLER'S responsibility — we just deliver
    a parsed dict (or None on failure).  This keeps `client.py` agnostic
    about which schema is being enforced for any given call.
    """
    try:
        cfg = _load_config()
    except Exception as e:
        return LLMResponse(parsed=None, raw_text="", model="", elapsed_s=0.0,
                            attempt=0, error=f"config_load:{e!r}")

    api_key = _resolve_api_key(cfg.get("api_key_env_var", ""))
    if not api_key:
        return LLMResponse(parsed=None, raw_text="", model=cfg.get("model", ""),
                            elapsed_s=0.0, attempt=0,
                            error="no_api_key_resolved")

    try:
        from openai import OpenAI
    except ImportError:
        return LLMResponse(parsed=None, raw_text="", model="", elapsed_s=0.0,
                            attempt=0, error="openai_sdk_not_installed")

    timeout_s = float(os.environ.get("LOCOMOTION_CONSOLE_LLM_TIMEOUT_S", cfg.get("timeout_s", 35.0)))
    client = OpenAI(api_key=api_key, base_url=cfg.get("base_url"), timeout=timeout_s)
    model = cfg.get("model", "deepseek-chat")
    temperature = float(cfg.get("temperature", 0.0))

    t0 = time.time()
    last_error: Optional[str] = None
    raw_text = ""
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                top_p=1.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw_text = completion.choices[0].message.content or ""
            parsed = json.loads(raw_text)
            elapsed = time.time() - t0
            _audit_log({
                "schema_name": schema_name,
                "model": model,
                "attempt": attempt,
                "elapsed_s": elapsed,
                "system": system_prompt,
                "user": user_prompt,
                "raw": raw_text,
                "parsed": parsed,
            })
            return LLMResponse(parsed=parsed, raw_text=raw_text, model=model,
                                elapsed_s=elapsed, attempt=attempt)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                continue

    elapsed = time.time() - t0
    _audit_log({
        "schema_name": schema_name,
        "model": model,
        "attempt": max_retries,
        "elapsed_s": elapsed,
        "system": system_prompt,
        "user": user_prompt,
        "raw": raw_text,
        "parsed": None,
        "error": last_error,
    })
    return LLMResponse(parsed=None, raw_text=raw_text, model=model,
                        elapsed_s=elapsed, attempt=max_retries,
                        error=last_error)
