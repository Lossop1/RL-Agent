"""call_llm_text 必须是自由文本（不带 response_format=json_object），这样最终答案
完整落进 content，而不是被挤进一个短 JSON 字段。它仍要捕获 reasoning_content、
支持 model 覆盖，并在没有 key 时干净失败。"""
import sys
import types

from autotuner.llm_gateway import client


class _Msg:
    def __init__(self, content, reasoning=None):
        self.content = content
        if reasoning is not None:
            self.reasoning_content = reasoning


class _Completion:
    def __init__(self, message):
        self.choices = [types.SimpleNamespace(message=message)]


def _install(monkeypatch, message, seen=None):
    """让 `from openai import OpenAI` 得到一个假 client，其 create() 返回 message，
    并把传给 create 的 kwargs 记进 seen（用来断言没传 response_format）。"""
    def _create(**kw):
        if seen is not None:
            seen.update(kw)
        return _Completion(message)

    class _FakeClient:
        def __init__(self, *a, **k):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=_create))

    fake = types.ModuleType("openai")
    fake.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.setattr(client, "_load_config",
                        lambda *a, **k: {"api_key_env_var": "sk-test", "base_url": "http://x",
                                         "model": "cfg-model", "temperature": 0.0, "timeout_s": 5.0})
    monkeypatch.setattr(client, "_audit_log", lambda entry: None)


def test_call_llm_text_is_free_form_no_json(monkeypatch):
    seen = {}
    _install(monkeypatch, _Msg("完整的一段答案，不是被榨干的一句话。", reasoning="thinking"), seen)
    resp = client.call_llm_text("sys", "user")
    assert "response_format" not in seen           # 关键：不强制 JSON
    assert resp.parsed is None
    assert resp.raw_text == "完整的一段答案，不是被榨干的一句话。"
    assert resp.reasoning == "thinking"


def test_call_llm_text_model_override_reaches_api(monkeypatch):
    seen = {}
    _install(monkeypatch, _Msg("ok"), seen)
    client.call_llm_text("sys", "user", model="fast-model")
    assert seen["model"] == "fast-model"


def test_call_llm_text_reasoning_defaults_empty(monkeypatch):
    _install(monkeypatch, _Msg("ok"))                # 消息对象没有 reasoning_content 属性
    resp = client.call_llm_text("sys", "user")
    assert resp.raw_text == "ok" and resp.reasoning == ""


def test_call_llm_text_no_api_key(monkeypatch):
    monkeypatch.setattr(client, "_load_config",
                        lambda *a, **k: {"api_key_env_var": "MISSING_ENV_VAR_XYZ", "model": "m"})
    monkeypatch.setattr(client, "_audit_log", lambda entry: None)
    resp = client.call_llm_text("s", "u")
    assert resp.parsed is None and resp.error == "no_api_key_resolved"


# ── thinking 开关（速度根因：便宜跳默认关思考）──────────────────────────────
def test_think_false_disables_thinking_via_extra_body(monkeypatch):
    seen = {}
    _install(monkeypatch, _Msg("完整答案"), seen)
    client.call_llm_text("s", "u", think=False)
    assert seen.get("extra_body") == {"thinking": {"type": "disabled"}}


def test_think_none_sends_no_thinking_param(monkeypatch):
    seen = {}
    _install(monkeypatch, _Msg("答案"), seen)
    client.call_llm_text("s", "u")               # think 默认 None → 不传，走模型默认
    assert "extra_body" not in seen


def test_schema_call_think_false_disables_thinking(monkeypatch):
    seen = {}
    _install(monkeypatch, _Msg('{"reply":"ok"}'), seen)
    client.call_llm_with_schema("s", "u", "schema", think=False)
    assert seen.get("extra_body") == {"thinking": {"type": "disabled"}}


def test_thinking_param_rejected_falls_back_without_it(monkeypatch):
    # 若 API 不认 thinking 参数：第一次带 extra_body 报错 → 去掉重试并成功，绝不因此让调用挂。
    calls = {"n": 0, "second_kwargs": None}

    def _create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            assert "extra_body" in kw            # 第一次确实带了 thinking 参数
            raise RuntimeError("unknown param: thinking")
        calls["second_kwargs"] = kw
        return _Completion(_Msg("去掉参数后恢复成功的答案"))

    class _FakeClient:
        def __init__(self, *a, **k):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=_create))

    fake = types.ModuleType("openai")
    fake.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.setattr(client, "_load_config",
                        lambda *a, **k: {"api_key_env_var": "sk-test", "base_url": "http://x",
                                         "model": "m", "temperature": 0.0, "timeout_s": 5.0})
    monkeypatch.setattr(client, "_audit_log", lambda entry: None)

    resp = client.call_llm_text("s", "u", think=False)    # max_retries=1 → 有第二次
    assert resp.raw_text == "去掉参数后恢复成功的答案"
    assert calls["n"] == 2
    assert "extra_body" not in calls["second_kwargs"]      # 第二次不再带 thinking 参数
