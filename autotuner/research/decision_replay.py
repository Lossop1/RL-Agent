"""Blind historical decision replay for the RL research agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from .research_supervisor import evidence_policy


class ReplayExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    required_actions: frozenset[str] = frozenset()
    forbidden_actions: frozenset[str] = frozenset()
    required_evidence: frozenset[str] = frozenset()
    protected_capabilities: frozenset[str] = frozenset()
    allowed_bootstrap: frozenset[Literal["fresh", "resume", "either", "none"]] = frozenset({"none"})
    allowed_disposition: frozenset[Literal["continue", "experiment", "promote", "rollback", "block"]] = frozenset({"continue"})
    required_concepts: frozenset[str] = frozenset()


class DecisionReplayCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    split: Literal["train", "holdout"] = "holdout"
    visible: dict[str, Any]
    expectation: ReplayExpectation
    withheld_outcome: dict[str, Any]
    source_refs: tuple[str, ...] = ()


class ReplayDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: frozenset[str]
    evidence: frozenset[str] = frozenset()
    protected_capabilities: frozenset[str] = frozenset()
    bootstrap: Literal["fresh", "resume", "either", "none"] = "none"
    disposition: Literal["continue", "experiment", "promote", "rollback", "block"] = "continue"
    concepts: frozenset[str] = frozenset()
    rationale: str = ""


class ReplayPolicy(Protocol):
    def decide(self, visible: dict[str, Any]) -> ReplayDecision: ...


class ReplayCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    passed: bool
    score: float
    dimension_scores: dict[str, float]
    missing: dict[str, list[str]]
    decision: ReplayDecision
    outcome_after_decision: dict[str, Any]


class ReplayReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_count: int
    pass_count: int
    mean_score: float
    results: tuple[ReplayCaseResult, ...]


def load_replay_cases(path: str | Path, *, split: str = "") -> list[DecisionReplayCase]:
    cases: list[DecisionReplayCase] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = DecisionReplayCase.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"invalid replay case at line {line_no}: {exc}") from exc
        if not split or case.split == split:
            cases.append(case)
    ids = [item.id for item in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("decision replay contains duplicate case ids")
    return cases


def _coverage(required: frozenset[str], actual: frozenset[str]) -> tuple[float, list[str]]:
    if not required:
        return 1.0, []
    missing = sorted(required - actual)
    return (len(required) - len(missing)) / len(required), missing


class DecisionReplayRunner:
    def run_case(self, case: DecisionReplayCase, policy: ReplayPolicy) -> ReplayCaseResult:
        # Only visible facts cross this call boundary.  Hidden outcomes become
        # available solely after the decision has been frozen by Pydantic.
        decision = ReplayDecision.model_validate(policy.decide(json.loads(json.dumps(case.visible))))
        expected = case.expectation
        action_score, missing_actions = _coverage(expected.required_actions, decision.actions)
        evidence_score, missing_evidence = _coverage(expected.required_evidence, decision.evidence)
        protected_score, missing_protected = _coverage(expected.protected_capabilities, decision.protected_capabilities)
        concept_score, missing_concepts = _coverage(expected.required_concepts, decision.concepts)
        forbidden = sorted(expected.forbidden_actions & decision.actions)
        if forbidden:
            action_score = 0.0
        bootstrap_score = 1.0 if decision.bootstrap in expected.allowed_bootstrap or "either" in expected.allowed_bootstrap else 0.0
        disposition_score = 1.0 if decision.disposition in expected.allowed_disposition else 0.0
        dimensions = {
            "actions": action_score,
            "evidence": evidence_score,
            "protected_capabilities": protected_score,
            "bootstrap": bootstrap_score,
            "disposition": disposition_score,
            "reasoning_concepts": concept_score,
        }
        score = sum(dimensions.values()) / len(dimensions)
        return ReplayCaseResult(
            case_id=case.id,
            passed=score >= 0.85 and not forbidden,
            score=score,
            dimension_scores=dimensions,
            missing={
                "actions": missing_actions,
                "evidence": missing_evidence,
                "protected_capabilities": missing_protected,
                "concepts": missing_concepts,
                "forbidden_actions": forbidden,
            },
            decision=decision,
            outcome_after_decision=case.withheld_outcome,
        )

    def run(self, cases: list[DecisionReplayCase], policy: ReplayPolicy) -> ReplayReport:
        results = tuple(self.run_case(case, policy) for case in cases)
        return ReplayReport(
            case_count=len(results),
            pass_count=sum(item.passed for item in results),
            mean_score=sum(item.score for item in results) / len(results) if results else 0.0,
            results=results,
        )


class GlobalResearchDecisionPolicy:
    """Deterministic safety baseline derived from the global tuning method."""

    def decide(self, visible: dict[str, Any]) -> ReplayDecision:
        deployment_mismatch = bool(visible.get("deployment_mismatch"))
        log_quality = str(visible.get("log_quality", "unknown"))
        behavior_quality = str(visible.get("behavior_quality", "unknown"))
        high_risk = bool(visible.get("high_risk"))
        policy = evidence_policy(
            log_quality=log_quality if log_quality in {"bad", "good", "unknown"} else "unknown",
            behavior_quality=behavior_quality if behavior_quality in {"known", "unknown", "contradictory"} else "unknown",
            deployment_mismatch=deployment_mismatch,
            high_risk=high_risk,
        )
        actions = {policy.default_action}
        evidence = set(policy.required_kinds)
        protected = set(visible.get("protected_capabilities", []))
        concepts = {"global_contract", "preserve_effective_adjust_missing", "independent_evaluator"}
        bootstrap: Literal["fresh", "resume", "either", "none"] = "none"
        disposition: Literal["continue", "experiment", "promote", "rollback", "block"] = "continue"

        if deployment_mismatch:
            actions.add("establish_policy_parity")
            disposition = "block"
        if visible.get("logs_sufficient_to_reject"):
            actions.add("reject_progress_claim")
            actions.discard("continue_observation")
        if visible.get("transient_improvement") or visible.get("trend") in {"oscillating", "plateau"}:
            actions.add("observe_full_window")
            concepts.add("sustained_evidence_not_frontier")
        if visible.get("protected_regression"):
            actions.add("rollback_candidate")
            disposition = "rollback"
        if visible.get("historical_anchor"):
            actions.add("compare_history_diff")
            concepts.add("historical_resume_chain")
        if visible.get("known_mature_checkpoint"):
            bootstrap = "resume"
            actions.add("bounded_resume_experiment")
            disposition = "experiment"
        elif visible.get("fresh_reproducibility_question"):
            bootstrap = "either"
            actions.add("compare_fresh_and_resume_hypotheses")
            disposition = "experiment"
        if visible.get("mechanism_semantics_unknown"):
            actions.add("inspect_reward_drive_and_exploit_paths")
            concepts.add("positive_drive_before_guard")
        if visible.get("diagnostic_needed"):
            actions.add("run_targeted_diagnostic")
            evidence.add("physical_diag")
        if visible.get("stage_mismatch"):
            actions.add("move_training_to_matching_distribution")
            concepts.add("train_problem_in_its_distribution")
        if visible.get("frontier_only"):
            actions.add("ignore_frontier_as_capability_proof")
            concepts.add("sustained_evidence_not_frontier")
        return ReplayDecision(
            actions=frozenset(actions), evidence=frozenset(evidence),
            protected_capabilities=frozenset(protected), bootstrap=bootstrap,
            disposition=disposition, concepts=frozenset(concepts), rationale=policy.rationale,
        )
