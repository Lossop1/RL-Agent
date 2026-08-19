"""Durable current state for an evolving RL research program.

The research ledger records immutable history.  This module complements it
with a small, recoverable *current* state used by supervisors and agents.  A
write is accepted only against the revision the caller read, preventing an
old chat turn or a concurrent monitor from silently replacing newer facts.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field


STATE_SCHEMA_VERSION = "rl-agent.research-state/v1"
EVENT_SCHEMA_VERSION = "rl-agent.research-state-event/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class GoalState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str
    acceptance: list[str] = Field(default_factory=list)
    priority: int = 0
    status: Literal["active", "satisfied", "retired"] = "active"
    contract_ref: str = ""


class FactState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: Literal["observed", "established", "contested", "rejected", "superseded"] = "observed"
    scope: str = ""
    updated_at: str = Field(default_factory=_utc_now)


class HypothesisState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    claim: str
    predicts: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: Literal["active", "validated", "rejected", "superseded"] = "active"
    last_tested_by: str = ""
    updated_at: str = Field(default_factory=_utc_now)


class ProtectedCapabilityState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    baseline_ref: str
    evaluator_ref: str
    tolerance: dict[str, Any] = Field(default_factory=dict)
    last_result: dict[str, Any] = Field(default_factory=dict)
    status: Literal["unknown", "passing", "regressed"] = "unknown"


class ObservationWindowState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    started_at: str = Field(default_factory=_utc_now)
    ends_at: str = ""
    run_ref: str = ""
    checkpoint_start_ref: str = ""
    required_evidence: list[str] = Field(default_factory=list)
    collected_evidence: list[str] = Field(default_factory=list)
    intervention_conditions: list[str] = Field(default_factory=list)
    status: Literal["planned", "observing", "complete", "intervened", "cancelled"] = "planned"


class ResearchCaseState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    problem_statement: str
    symptoms: list[str] = Field(default_factory=list)
    capability_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    hypothesis_refs: list[str] = Field(default_factory=list)
    candidate_refs: list[str] = Field(default_factory=list)
    experiment_refs: list[str] = Field(default_factory=list)
    status: Literal[
        "observed", "attributing", "evidence_needed", "proposed",
        "experimenting", "evaluating", "open", "resolved",
    ] = "observed"
    resolution: str = ""
    updated_at: str = Field(default_factory=_utc_now)


class ResearchState(BaseModel):
    """Small authoritative snapshot restored after process/context loss."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = STATE_SCHEMA_VERSION
    state_id: str = "default"
    revision: int = Field(default=0, ge=0)
    updated_at: str = Field(default_factory=_utc_now)
    program_ref: str = ""
    contract_ref: str = ""
    baseline_ref: str = ""
    goals: list[GoalState] = Field(default_factory=list)
    active_run_ref: str = ""
    active_checkpoint_ref: str = ""
    active_case_ref: str = ""
    cases: dict[str, ResearchCaseState] = Field(default_factory=dict)
    facts: dict[str, FactState] = Field(default_factory=dict)
    hypotheses: dict[str, HypothesisState] = Field(default_factory=dict)
    protected_capabilities: dict[str, ProtectedCapabilityState] = Field(default_factory=dict)
    observation_window: ObservationWindowState | None = None
    pending_interventions: list[str] = Field(default_factory=list)
    no_repeat_fingerprints: dict[str, str] = Field(default_factory=dict)
    unresolved_questions: list[str] = Field(default_factory=list)
    last_decision_ref: str = ""
    last_outcome_ref: str = ""


class StateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = EVENT_SCHEMA_VERSION
    event_id: str
    sequence: int = Field(ge=1)
    action: Literal["initialize", "update", "recover"]
    actor: str
    reason: str
    created_at: str = Field(default_factory=_utc_now)
    previous_event_hash: str = ""
    state_revision: int = Field(ge=0)
    state_hash: str
    state: dict[str, Any]
    event_hash: str


class ResearchStateError(RuntimeError):
    pass


class RevisionConflict(ResearchStateError):
    def __init__(self, expected: int, actual: int):
        self.expected = expected
        self.actual = actual
        super().__init__(f"research state revision conflict: expected {expected}, actual {actual}")


class StateIntegrityError(ResearchStateError):
    pass


class ResearchStateStore:
    """CAS state store backed by an atomic snapshot and hash-chained events."""

    _thread_lock = threading.RLock()

    def __init__(
        self,
        root: str | os.PathLike[str] = "output/research_state",
        *,
        lock_timeout_s: float = 5.0,
        stale_lock_s: float = 120.0,
    ) -> None:
        self.root = Path(root)
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        self.lock_path = self.root / ".write.lock"
        self.lock_timeout_s = lock_timeout_s
        self.stale_lock_s = stale_lock_s

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.lock_timeout_s
        fd: int | None = None
        while fd is None:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"pid={os.getpid()} created={time.time()}\n".encode("ascii"))
                os.fsync(fd)
            except FileExistsError:
                try:
                    age = time.time() - self.lock_path.stat().st_mtime
                    if age > self.stale_lock_s:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise ResearchStateError(f"timed out acquiring research state lock: {self.lock_path}")
                time.sleep(0.02)
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
            self.lock_path.unlink(missing_ok=True)

    @staticmethod
    def _event_hash_body(event: StateEvent | dict[str, Any]) -> dict[str, Any]:
        data = event.model_dump(mode="json") if isinstance(event, StateEvent) else dict(event)
        data.pop("event_hash", None)
        return data

    def _read_events(self) -> list[StateEvent]:
        if not self.events_path.is_file():
            return []
        events: list[StateEvent] = []
        previous = ""
        for expected_sequence, line in enumerate(self.events_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = StateEvent.model_validate_json(line)
            except Exception as exc:
                raise StateIntegrityError(f"invalid state event at line {expected_sequence}: {exc}") from exc
            if event.sequence != expected_sequence:
                raise StateIntegrityError(
                    f"state event sequence gap: expected {expected_sequence}, got {event.sequence}"
                )
            if event.previous_event_hash != previous:
                raise StateIntegrityError(f"state event chain break at sequence {event.sequence}")
            if event.state_hash != _sha256(event.state):
                raise StateIntegrityError(f"state payload hash mismatch at sequence {event.sequence}")
            if event.event_hash != _sha256(self._event_hash_body(event)):
                raise StateIntegrityError(f"state event hash mismatch at sequence {event.sequence}")
            state = ResearchState.model_validate(event.state)
            if state.revision != event.state_revision:
                raise StateIntegrityError(f"state revision mismatch at sequence {event.sequence}")
            events.append(event)
            previous = event.event_hash
        return events

    def _atomic_write_state(self, state: ResearchState) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f".{self.state_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        payload = json.dumps(state.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        try:
            with tmp.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.state_path)
        finally:
            tmp.unlink(missing_ok=True)

    def _append_event(self, state: ResearchState, *, action: Literal["initialize", "update", "recover"], actor: str, reason: str, events: list[StateEvent]) -> StateEvent:
        state_dict = state.model_dump(mode="json")
        body = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": f"state-event:{uuid.uuid4().hex}",
            "sequence": len(events) + 1,
            "action": action,
            "actor": actor,
            "reason": reason,
            "created_at": _utc_now(),
            "previous_event_hash": events[-1].event_hash if events else "",
            "state_revision": state.revision,
            "state_hash": _sha256(state_dict),
            "state": state_dict,
        }
        event = StateEvent(**body, event_hash=_sha256(body))
        self.root.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(event.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def _load_locked(self, *, recover: bool) -> ResearchState:
        events = self._read_events()
        latest = ResearchState.model_validate(events[-1].state) if events else None
        snapshot: ResearchState | None = None
        snapshot_error: Exception | None = None
        if self.state_path.is_file():
            try:
                snapshot = ResearchState.model_validate_json(self.state_path.read_text(encoding="utf-8"))
            except Exception as exc:
                snapshot_error = exc

        if latest is None:
            if snapshot_error:
                raise StateIntegrityError(f"state snapshot is invalid and no recovery event exists: {snapshot_error}")
            if snapshot is None:
                raise ResearchStateError("research state is not initialized")
            return snapshot

        if snapshot is not None and snapshot.revision == latest.revision and _sha256(snapshot) == _sha256(latest):
            return snapshot
        if not recover:
            detail = f"snapshot revision={getattr(snapshot, 'revision', None)}, event revision={latest.revision}"
            raise StateIntegrityError(f"research state snapshot is not synchronized with its event log: {detail}")
        self._atomic_write_state(latest)
        return latest

    def initialize(self, state: ResearchState, *, actor: str, reason: str = "initialize research state") -> ResearchState:
        if not actor.strip():
            raise ResearchStateError("actor is required")
        with self._thread_lock, self._write_lock():
            events = self._read_events()
            if events or self.state_path.exists():
                raise ResearchStateError("research state is already initialized")
            initial = state.model_copy(update={"revision": 0, "updated_at": _utc_now()})
            self._append_event(initial, action="initialize", actor=actor, reason=reason, events=[])
            self._atomic_write_state(initial)
            return initial

    def load(self, *, recover: bool = True) -> ResearchState:
        with self._thread_lock:
            if recover:
                with self._write_lock():
                    return self._load_locked(recover=True)
            return self._load_locked(recover=False)

    def compare_and_set(
        self,
        expected_revision: int,
        update: ResearchState | dict[str, Any] | Callable[[ResearchState], ResearchState],
        *,
        actor: str,
        reason: str,
    ) -> ResearchState:
        if not actor.strip() or not reason.strip():
            raise ResearchStateError("actor and reason are required")
        with self._thread_lock, self._write_lock():
            current = self._load_locked(recover=True)
            if current.revision != expected_revision:
                raise RevisionConflict(expected_revision, current.revision)
            if callable(update):
                candidate = update(current.model_copy(deep=True))
            elif isinstance(update, ResearchState):
                candidate = update
            else:
                candidate = current.model_copy(update=update, deep=True)
            if not isinstance(candidate, ResearchState):
                raise ResearchStateError("research state update callable must return ResearchState")
            if candidate.state_id != current.state_id:
                raise ResearchStateError("state_id is immutable")
            next_state = candidate.model_copy(
                update={"revision": current.revision + 1, "updated_at": _utc_now()},
                deep=True,
            )
            events = self._read_events()
            self._append_event(next_state, action="update", actor=actor, reason=reason, events=events)
            self._atomic_write_state(next_state)
            return next_state

    def history(self) -> list[StateEvent]:
        with self._thread_lock:
            return self._read_events()

    def summary(self) -> dict[str, Any]:
        state = self.load()
        events = self.history()
        return {
            "state_id": state.state_id,
            "revision": state.revision,
            "updated_at": state.updated_at,
            "program_ref": state.program_ref,
            "contract_ref": state.contract_ref,
            "active_run_ref": state.active_run_ref,
            "active_checkpoint_ref": state.active_checkpoint_ref,
            "active_case_ref": state.active_case_ref,
            "active_goals": [goal.id for goal in state.goals if goal.status == "active"],
            "case_count": len(state.cases),
            "event_count": len(events),
            "chain_verified": True,
            "last_event_hash": events[-1].event_hash if events else "",
        }
