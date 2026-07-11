"""The gateway must stop discarding the model's thinking.

DeepSeek returns `reasoning_content` alongside `content` — even under
response_format=json_object, where `content` is squeezed to a sentence and the
substance lands in `reasoning_content`. This locks that `call_llm_with_schema`
captures it onto `LLMResponse.reasoning`, and defaults to "" when a provider
doesn't return it (so nothing downstream breaks on a plain model)."""
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


def _install_fake_openai(monkeypatch, message):
    """Make `from openai import OpenAI` yield a client whose create() returns `message`."""
    class _FakeClient:
        def __init__(self, *a, **k):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kw: _Completion(message))
            )

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    # sk- key resolves without env; skip disk writes.
    monkeypatch.setattr(client, "_load_config",
                        lambda *a, **k: {"api_key_env_var": "sk-test", "base_url": "http://x",
                                         "model": "cfg-model", "temperature": 0.0, "timeout_s": 5.0})
    monkeypatch.setattr(client, "_audit_log", lambda entry: None)


def test_reasoning_content_is_captured(monkeypatch):
    _install_fake_openai(monkeypatch, _Msg('{"reply": "hi"}', reasoning="deep thoughts here"))
    resp = client.call_llm_with_schema("sys", "user", "some_schema")
    assert resp.parsed == {"reply": "hi"}
    assert resp.reasoning == "deep thoughts here"


def test_reasoning_defaults_empty_when_provider_omits_it(monkeypatch):
    # A plain model whose message object has no reasoning_content attribute at all.
    _install_fake_openai(monkeypatch, _Msg('{"reply": "hi"}'))
    resp = client.call_llm_with_schema("sys", "user", "some_schema")
    assert resp.parsed == {"reply": "hi"}
    assert resp.reasoning == ""


def test_model_override_reaches_the_api(monkeypatch):
    # The loop routes cheap hops to a fast model via the `model=` override; confirm it wins over config.
    seen = {}

    class _FakeClient:
        def __init__(self, *a, **k):
            def _create(**kw):
                seen["model"] = kw.get("model")
                return _Completion(_Msg('{"reply": "ok"}', reasoning="r"))
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(client, "_load_config",
                        lambda *a, **k: {"api_key_env_var": "sk-test", "base_url": "http://x",
                                         "model": "cfg-model", "temperature": 0.0, "timeout_s": 5.0})
    monkeypatch.setattr(client, "_audit_log", lambda entry: None)

    resp = client.call_llm_with_schema("sys", "user", "some_schema", model="fast-model")
    assert seen["model"] == "fast-model"
    assert resp.reasoning == "r"
