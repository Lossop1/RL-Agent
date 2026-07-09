"""Deterministic context routing for LLM turns.

The frontend may tell us where the operator is looking, but it must not decide
which evidence the LLM should trust.  This module turns a chat request into a
small, auditable context envelope: route, source budget, optional remote-machine
summary, and prompt text.
"""
from __future__ import annotations

from typing import Any

from .evidence_router import route_evidence, selected_source_ids
from .schemas import ContextEnvelopeInfo, ContextRequestInfo, RemoteMachineStatus


def build_context_envelope(
    *,
    message: str,
    ui_mode: str = "",
    legacy_context: str = "",
    context_request: ContextRequestInfo | None = None,
    remote_status: RemoteMachineStatus | None = None,
) -> ContextEnvelopeInfo:
    request = context_request or ContextRequestInfo()
    page = (request.page or ui_mode or "").strip()
    intent = (request.intent or message or "").strip()
    budget = max(1000, min(int(request.max_chars or 3500), 9000))
    focus_text = " ".join(str(item) for item in request.focus[:8])
    route_query = "\n".join(part for part in [intent, focus_text, legacy_context[:800]] if part)
    route = route_evidence(query=route_query or message, ui_mode=page or ui_mode)
    selected = selected_source_ids(route)
    required = [
        str(item.get("id"))
        for item in route.get("selected_sources", [])
        if isinstance(item, dict) and item.get("required", True)
    ]
    remote_summary = _remote_summary(remote_status) if remote_status is not None else {}

    lines = [
        "[ContextEnvelope]",
        f"page={page or 'unknown'}",
        f"intent={_one_line(intent or message, 220)}",
        f"selected_sources={', '.join(selected) or 'none'}",
        f"required_sources={', '.join(required) or 'none'}",
        "rules:",
    ]
    lines.extend(f"- {rule}" for rule in route.get("answer_rules", [])[:8])
    gaps = route.get("gap_checks", [])
    if gaps:
        lines.append("gap_checks:")
        for gap in gaps[:8]:
            if isinstance(gap, dict):
                lines.append(
                    f"- {gap.get('source')}: {gap.get('gap')} -> {gap.get('impact')}"
                )
    if remote_summary:
        lines.append(f"remote_summary={remote_summary}")
    if request.visible:
        lines.append(f"visible_ui={_compact_mapping(request.visible, 900)}")
    elif legacy_context:
        lines.append(f"legacy_visible_ui={_one_line(legacy_context, 700)}")
    prompt = "\n".join(lines)
    if len(prompt) > budget:
        prompt = prompt[: budget - 80].rstrip() + "\n[ContextEnvelope truncated by budget]"
    return ContextEnvelopeInfo(
        page=page,
        intent=intent,
        budget_chars=budget,
        injected_chars=len(prompt),
        selected_sources=selected,
        required_sources=required,
        route=route,
        remote_summary=remote_summary,
        prompt=prompt,
    )


def should_probe_remote_for_context(
    *, message: str, ui_mode: str = "", context_request: ContextRequestInfo | None = None
) -> bool:
    request = context_request or ContextRequestInfo()
    if not request.include_remote_status:
        return False
    page = request.page or ui_mode
    intent = request.intent or message
    focus_text = " ".join(str(item) for item in request.focus[:8])
    route = route_evidence(query="\n".join([intent, focus_text]), ui_mode=page)
    return "remote_status" in selected_source_ids(route)


def _remote_summary(status: RemoteMachineStatus) -> dict[str, Any]:
    if not status.available:
        return {
            "available": False,
            "source": status.source,
            "host": status.host,
            "error": status.error,
        }
    gpu = status.gpus[0] if status.gpus else None
    return {
        "available": True,
        "host": status.host,
        "load_avg": status.load_avg,
        "memory_available_mb": status.memory_available_mb,
        "gpu": None if gpu is None else {
            "name": gpu.name,
            "memory_used_mb": gpu.memory_used_mb,
            "memory_total_mb": gpu.memory_total_mb,
            "utilization_gpu_pct": gpu.utilization_gpu_pct,
            "temperature_c": gpu.temperature_c,
            "processes": len(gpu.processes),
        },
        "tmux_sessions": [item.name for item in status.tmux_sessions[:8]],
        "training_processes": len(status.training_processes),
        "disks": [
            {"mount": item.mount, "avail": item.avail, "use_pct": item.use_pct}
            for item in status.disks[:5]
        ],
    }


def _one_line(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _compact_mapping(value: dict[str, Any], limit: int) -> str:
    parts: list[str] = []
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)) or item is None:
            parts.append(f"{key}={item}")
        else:
            parts.append(f"{key}={type(item).__name__}")
    text = "; ".join(parts)
    return _one_line(text, limit)
