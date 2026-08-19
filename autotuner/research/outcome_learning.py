"""Turn bounded experiment outcomes into durable research memory."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from autotuner.mechanisms.mechanism_synthesis import SynthesisCandidate
from .research_ledger import (
    EvidenceRecord,
    ExperimentOutcome,
    ExperimentPlan,
    KnowledgeClaim,
    ResearchLedgerStore,
)
from .research_state import FactState, ResearchState, ResearchStateStore
from .research_supervisor import ExperimentExecution


@dataclass(frozen=True)
class OutcomeLearningResult:
    outcome_ref: str
    evidence_ref: str
    knowledge_refs: tuple[str, ...]
    state_revision: int
    disposition: str
    idempotent: bool = False


def _prediction_results(candidate: SynthesisCandidate, execution: ExperimentExecution) -> list[dict[str, Any]]:
    evaluation = dict(execution.evaluation or {})
    explicit = evaluation.get("predictions")
    results: list[dict[str, Any]] = []
    if isinstance(explicit, list):
        for item in explicit:
            if isinstance(item, Mapping):
                results.append(dict(item))
    if results:
        return results
    status = "supported" if execution.disposition == "promote" else (
        "contradicted" if execution.disposition == "rollback" else "unknown"
    )
    return [
        {"prediction": prediction, "status": status, "source": "experiment disposition"}
        for prediction in candidate.predictions
    ]


def _confidence(current: float, status: str) -> float:
    if status == "supported":
        return min(1.0, current + 0.25 * (1.0 - current))
    if status == "contradicted":
        return max(0.0, current * 0.60)
    return current


class OutcomeLearner:
    def __init__(self, state_store: ResearchStateStore, ledger: ResearchLedgerStore):
        self.state_store = state_store
        self.ledger = ledger

    def learn(
        self,
        *,
        expected_state_revision: int,
        plan: ExperimentPlan,
        candidate: SynthesisCandidate,
        execution: ExperimentExecution,
        actor: str,
    ) -> OutcomeLearningResult:
        suffix = candidate.candidate_bundle_fingerprint[:16]
        outcome_id = f"outcome:{plan.id}:{suffix}"
        evidence_id = f"evidence:{plan.id}:{suffix}:evaluation"
        existing = self.ledger.latest("experiment_outcome", outcome_id)
        if existing is not None:
            state = self.state_store.load()
            return OutcomeLearningResult(
                outcome_ref=outcome_id,
                evidence_ref=evidence_id,
                knowledge_refs=tuple(existing.get("knowledge_updates", [])),
                state_revision=state.revision,
                disposition=str(existing.get("disposition", "inconclusive")),
                idempotent=True,
            )

        workspace = Path(execution.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        evaluation_path = workspace / "independent_evaluation.json"
        evaluation_payload = {
            "experiment_ref": plan.id,
            "candidate_ref": candidate.id,
            "candidate_bundle_fingerprint": candidate.candidate_bundle_fingerprint,
            "disposition": execution.disposition,
            "backend_state": execution.backend_state,
            "stop_reason": execution.stop_reason,
            "evaluation": dict(execution.evaluation or {}),
            "protected_results": dict(execution.protected_results or {}),
        }
        evaluation_path.write_text(json.dumps(evaluation_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        evidence = EvidenceRecord(
            id=evidence_id,
            kind="physical_diag",
            run_ref=plan.id,
            checkpoint_ref=str((execution.evaluation or {}).get("checkpoint_ref", "")),
            contract_ref=str((execution.evaluation or {}).get("contract_ref", "")),
            raw_artifact_ref=str(evaluation_path),
            extracted_facts=[
                {"name": "disposition", "value": execution.disposition},
                {"name": "candidate_bundle_fingerprint", "value": candidate.candidate_bundle_fingerprint},
                {"name": "protected_results", "value": dict(execution.protected_results or {})},
            ],
            observer="machine",
            freshness="current",
            source="outcome_learner",
        )
        self.ledger.append("evidence", evidence, actor=actor)

        prediction_results = _prediction_results(candidate, execution)
        evaluation = dict(execution.evaluation or {})
        knowledge: list[KnowledgeClaim] = []
        for index, prediction in enumerate(candidate.predictions):
            status = "tentative" if execution.disposition == "promote" else "rejected"
            if execution.disposition == "promote" and evaluation.get("replicated") is True:
                status = "validated"
            statement = (
                f"Under experiment {plan.id}, candidate {candidate.id} supported: {prediction}"
                if execution.disposition == "promote"
                else f"Under experiment {plan.id}, candidate {candidate.id} did not establish: {prediction}"
            )
            knowledge.append(KnowledgeClaim(
                id=f"claim:{plan.id}:{suffix}:{index}",
                statement=statement,
                status=status,
                scope="config_range",
                evidence_refs=[evidence_id],
                source="outcome_learner",
            ))
        for claim in knowledge:
            self.ledger.append("knowledge_claim", claim, actor=actor)

        outcome = ExperimentOutcome(
            id=outcome_id,
            experiment_ref=plan.id,
            actual_diff_ref=candidate.patch.id,
            exposure_reached=dict(evaluation.get("exposure_reached", {})),
            capability_delta=dict(evaluation.get("capability_delta", {})),
            protected_capability_results=dict(execution.protected_results or {}),
            prediction_results=prediction_results,
            disposition=execution.disposition,
            knowledge_updates=[item.id for item in knowledge],
            source="outcome_learner",
        )
        self.ledger.append("experiment_outcome", outcome, actor=actor)

        prediction_status = "supported" if execution.disposition == "promote" else (
            "contradicted" if execution.disposition == "rollback" else "unknown"
        )

        def update_state(state: ResearchState) -> ResearchState:
            for hypothesis_ref in candidate.patch.hypothesis_refs:
                hypothesis = state.hypotheses.get(hypothesis_ref)
                if hypothesis is None:
                    continue
                confidence = _confidence(hypothesis.confidence, prediction_status)
                hypothesis.confidence = confidence
                hypothesis.last_tested_by = plan.id
                if prediction_status == "supported":
                    hypothesis.supporting_evidence = list(dict.fromkeys(hypothesis.supporting_evidence + [evidence_id]))
                    if evaluation.get("replicated") is True:
                        hypothesis.status = "validated"
                elif prediction_status == "contradicted":
                    hypothesis.contradicting_evidence = list(dict.fromkeys(hypothesis.contradicting_evidence + [evidence_id]))
                    if confidence < 0.20:
                        hypothesis.status = "rejected"

            case = state.cases.get(state.active_case_ref or plan.problem_statement)
            if case is not None:
                case.experiment_refs = list(dict.fromkeys(case.experiment_refs + [plan.id]))
                case.candidate_refs = list(dict.fromkeys(case.candidate_refs + [candidate.id]))
                if execution.disposition == "promote" and evaluation.get("resolves_case") is True:
                    case.status = "resolved"
                    case.resolution = f"promoted {candidate.id}"
                else:
                    case.status = "open"
            state.no_repeat_fingerprints[candidate.patch.fingerprint()] = outcome_id
            state.no_repeat_fingerprints[candidate.candidate_bundle_fingerprint] = outcome_id
            state.pending_interventions = [
                item for item in state.pending_interventions
                if item not in {candidate.id, candidate.patch.id, plan.id}
            ]
            state.last_outcome_ref = outcome_id
            if execution.disposition == "promote":
                fact_id = f"fact:{plan.id}:{suffix}"
                state.facts[fact_id] = FactState(
                    id=fact_id,
                    statement=f"Independent evaluation promoted {candidate.id} under {plan.id}",
                    evidence_refs=[evidence_id], confidence=0.75,
                    status="established" if evaluation.get("replicated") else "observed",
                    scope="config_range",
                )
            return state

        updated = self.state_store.compare_and_set(
            expected_state_revision,
            update_state,
            actor=actor,
            reason=f"learn experiment outcome {outcome_id}",
        )
        return OutcomeLearningResult(
            outcome_ref=outcome_id,
            evidence_ref=evidence_id,
            knowledge_refs=tuple(item.id for item in knowledge),
            state_revision=updated.revision,
            disposition=execution.disposition,
        )
