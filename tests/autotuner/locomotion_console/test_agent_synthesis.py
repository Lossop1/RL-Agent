"""最终答案要被综合成完整文本（不再是被榨干的一句话），且综合绝不能丢答案：
调用失败或返回空时，回退到循环给出的草答。"""
import asyncio
import json

from autotuner.llm_gateway.client import LLMResponse
from autotuner.locomotion_console import agent


def _resp(text, reasoning="r"):
    return LLMResponse(parsed=None, raw_text=text, model="m", elapsed_s=0.1, attempt=0, reasoning=reasoning)


def test_synthesis_returns_full_answer(monkeypatch):
    monkeypatch.setattr("autotuner.llm_gateway.client.call_llm_text",
                        lambda *a, **k: _resp("完整答案：progress_gate=0.55，低于门槛 0.68，所以卡在推进。"))
    out = agent._synthesize_final_answer(
        "为什么卡住？", [{"tool": "snapshot", "result": {"progress_gate": 0.55}}], [], "卡住了")
    assert out["reply"] != "卡住了" and "0.55" in out["reply"]
    assert out["reasoning"] == "r"


def test_synthesis_falls_back_to_draft_on_empty(monkeypatch):
    monkeypatch.setattr("autotuner.llm_gateway.client.call_llm_text",
                        lambda *a, **k: _resp(""))
    out = agent._synthesize_final_answer("q", [], [], "草答")
    assert out["reply"] == "草答"


def test_synthesis_falls_back_to_draft_on_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("api down")
    monkeypatch.setattr("autotuner.llm_gateway.client.call_llm_text", _boom)
    out = agent._synthesize_final_answer("q", [], [], "草答")
    assert out["reply"] == "草答"


def test_evidence_text_includes_question_and_tool_results():
    ev = agent._synthesis_evidence("为什么卡住？", [{"tool": "snapshot", "result": {"progress_gate": 0.55}}], [])
    assert "为什么卡住？" in ev and "progress_gate" in ev and "snapshot" in ev


def test_short_draft_needs_synthesis():
    assert agent._needs_final_synthesis("卡在推进门槛。") is True


def test_complete_draft_skips_synthesis():
    draft = (
        "当前训练停在第一阶段，遥测中的 progress_gate 为 0.55，而配置门槛为 0.68。"
        "诊断报告没有提供足部接触序列，因此目前只能确认推进指标未过门，不能据此断言具体步态原因。"
        "下一步应先运行步态诊断，再决定是否修改奖励权重；现有证据不支持直接调参。"
    )
    assert agent._needs_final_synthesis(draft) is False


def test_request_telemetry_cache_reuses_one_snapshot():
    class Source:
        calls = 0

        async def training_telemetry(self):
            self.calls += 1
            return {"sample": self.calls}

    source = Source()
    agent._cache_training_telemetry_for_request(source)
    first = asyncio.run(source.training_telemetry())
    second = asyncio.run(source.training_telemetry())
    assert first == second == {"sample": 1}
    assert source.calls == 1


def test_tool_cache_key_is_stable_for_argument_order():
    assert agent._tool_cache_key("get_config", {"grep": "phase", "limit": 2}) == agent._tool_cache_key(
        "get_config", {"limit": 2, "grep": "phase"}
    )


def test_static_formula_question_does_not_need_live_evidence():
    assert agent._question_needs_live_evidence("占空平衡是怎么算的？") is False
    hint = agent._intent_tool_hint("占空平衡是怎么算的？")
    assert hint["tool"] == "get_reward_model"


def test_current_reward_question_requires_live_evidence():
    assert agent._question_needs_live_evidence("当前占空平衡权重是多少？") is True
    assert agent._intent_tool_hint("当前占空平衡权重是多少？")["tool"] == "get_reward_model"


def test_robot_definition_routes_to_robot_model():
    hint = agent._intent_tool_hint("机器人有多少自由度？")
    assert hint["tool"] == "get_robot_model"


def test_static_agent_question_skips_live_preload(monkeypatch):
    from autotuner.locomotion_console.config import LocomotionConsoleSettings

    reply = "机器人本体定义应由 get_robot_model 提供。" * 8
    monkeypatch.setattr(
        "autotuner.llm_gateway.client.call_llm_with_schema",
        lambda *args, **kwargs: LLMResponse(
            parsed={"reply": reply}, raw_text="", model="fake", elapsed_s=0.0, attempt=0
        ),
    )
    out = agent.run_agent(
        "机器人有多少自由度？",
        LocomotionConsoleSettings(source="fake"),
        allow_slash=False,
        preload_operator_context=True,
    )
    tools = [item.get("tool") for item in out["transcript"]]
    assert "intent_router" in tools
    assert "live_state" not in tools
    assert "get_evidence_context" not in tools


def test_causal_runtime_field_requires_code_evidence():
    profile = agent._build_investigation_profile("为什么 phase_count 一直不增加？")
    assert profile["route"] == "investigation"
    assert profile["requires_code_evidence"] is True
    assert "phase_count" in profile["runtime_identifiers"]


def test_plain_live_status_does_not_force_code_evidence():
    profile = agent._build_investigation_profile("当前训练状态怎么样？")
    assert profile["requires_live_evidence"] is True
    assert profile["requires_code_evidence"] is False


def test_contradiction_escalates_without_domain_specific_keyword():
    profile = agent._build_investigation_profile("明明都满足了，但为什么还是没有变化？")
    assert profile["contradiction_detected"] is True
    assert profile["requires_code_evidence"] is True


def test_investigation_path_prefetches_code_before_answer(monkeypatch):
    from autotuner.locomotion_console.config import LocomotionConsoleSettings

    reply = "代码证据表明该状态只在控制条件成立时更新；当前证据需要按源码窗口逐项核对。" * 5
    monkeypatch.setattr(
        "autotuner.llm_gateway.client.call_llm_with_schema",
        lambda *args, **kwargs: LLMResponse(
            parsed={"reply": reply}, raw_text="", model="fake", elapsed_s=0.0, attempt=0
        ),
    )
    out = agent.run_agent(
        "为什么 phase_count 一直不增加？",
        LocomotionConsoleSettings(source="fake"),
        allow_slash=False,
    )
    code_rows = [item for item in out["transcript"] if item.get("tool") == "get_code_facts"]
    assert code_rows and code_rows[0]["result"]["complete"] is True
    assert any(item.get("tool") == "evidence_gate" for item in out["transcript"])


def test_missing_required_code_evidence_blocks_causal_claim(monkeypatch):
    from autotuner.locomotion_console.config import LocomotionConsoleSettings

    monkeypatch.setattr(agent, "_tool_get_code_facts", lambda *args, **kwargs: {"complete": False})
    monkeypatch.setattr(
        "autotuner.llm_gateway.client.call_llm_with_schema",
        lambda *args, **kwargs: LLMResponse(
            parsed={"reply": "原因已经确定。" * 20}, raw_text="", model="fake", elapsed_s=0.0, attempt=0
        ),
    )
    out = agent.run_agent(
        "明明状态满足了，但为什么 counter 还是没有变化？",
        LocomotionConsoleSettings(source="fake"),
        allow_slash=False,
    )
    assert out["reply"].startswith("我还不能可靠地给出这个结论")
    assert out["evidence_gaps"]


def test_claim_gate_rejects_unscoped_completeness_claim():
    transcript = [{"tool": "thresholds", "result": {"condition_set_complete": False}}]
    audit = agent._reply_claim_gate("所有条件都已满足。", transcript)
    assert audit["ok"] is False


def test_claim_gate_accepts_explicit_complete_evaluation():
    transcript = [{"tool": "gate_evaluator", "result": {"all_conditions_met": True}}]
    audit = agent._reply_claim_gate("所有条件都已满足。", transcript)
    assert audit["ok"] is True


def test_run_agent_asks_model_to_scope_unsupported_completeness_claim(monkeypatch):
    from autotuner.locomotion_console.config import LocomotionConsoleSettings

    replies = iter((
        "所有条件都已满足，但计数器仍未更新。" * 8,
        "现有证据只能确认界面展示的配置阈值已满足，不能证明运行时代码中的完整条件集已满足。"
        "源码窗口还需要逐项核对计数器的写入条件，因此不能把原因归结为等待或波动。" * 3,
    ))

    def fake_call(*args, **kwargs):
        return LLMResponse(
            parsed={"reply": next(replies)},
            raw_text="",
            model="fake",
            elapsed_s=0.0,
            attempt=0,
        )

    monkeypatch.setattr("autotuner.llm_gateway.client.call_llm_with_schema", fake_call)
    out = agent.run_agent(
        "明明界面条件都满足了，但为什么 phase_count 还是不增加？",
        LocomotionConsoleSettings(source="fake"),
        allow_slash=False,
    )

    claim_rows = [item for item in out["transcript"] if item.get("tool") == "claim_gate"]
    assert len(claim_rows) == 1
    assert claim_rows[0]["result"]["ok"] is False
    assert "只能确认界面展示" in out["reply"]
    assert "所有条件都已满足" not in out["reply"]


def test_render_keeps_ranked_runtime_assignment_inside_code_budget():
    from autotuner.locomotion_console.knowledge_model.code_facts import get_code_facts

    result = get_code_facts("为什么 phase_count 一直不增加？ phase_count")
    prompt = agent._render(
        "为什么 phase_count 一直不增加？",
        [{"tool": "get_code_facts", "args": {"query": "phase_count"}, "result": result}],
        [],
    )

    assert "_phase_count" in prompt
    assert any(marker in prompt for marker in ("_phase_count =", "_phase_count +="))


def test_run_agent_emits_public_progress_without_raw_evidence(monkeypatch):
    from autotuner.locomotion_console.config import LocomotionConsoleSettings

    reply = "根据已读取的定义，可以确认这是一个静态知识问题；当前回答不需要远程运行态。" * 5
    monkeypatch.setattr(
        "autotuner.llm_gateway.client.call_llm_with_schema",
        lambda *args, **kwargs: LLMResponse(
            parsed={"reply": reply}, raw_text="", model="fake", elapsed_s=0.0, attempt=0
        ),
    )
    events = []
    out = agent.run_agent(
        "机器人本体定义是什么？",
        LocomotionConsoleSettings(source="fake"),
        allow_slash=False,
        progress=events.append,
    )

    assert out["reply"] == reply
    assert [event["stage"] for event in events] == ["routing", "analysis", "complete"]
    assert all(set(event) <= {"stage", "label", "detail", "tool"} for event in events)
    assert all("result" not in event and "args" not in event for event in events)


def test_chat_stream_emits_progress_deltas_and_complete_response():
    from autotuner.locomotion_console.app import chat_stream
    from autotuner.locomotion_console.schemas import ChatRequest

    async def collect_events():
        response = await chat_stream(ChatRequest(message="/help"))
        payload = bytearray()
        async for chunk in response.body_iterator:
            payload.extend(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
        return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]

    events = asyncio.run(collect_events())
    event_types = [event["type"] for event in events]
    assert event_types[0] == "progress"
    assert "answer_start" in event_types
    assert "answer_delta" in event_types
    assert event_types[-1] == "complete"
    streamed = "".join(event["delta"] for event in events if event["type"] == "answer_delta")
    assert streamed == events[-1]["response"]["reply"]
