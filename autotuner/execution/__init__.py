"""Lazy public facade for deterministic execution-layer primitives.

Keeping imports lazy is intentional: resolving a product contract only needs
the runtime identity module and must not load YAML patching or remote transport
helpers as an accidental side effect.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "AppliedChangeSet": ("changeset", "AppliedChangeSet"),
    "Change": ("changeset", "Change"),
    "ChangeSet": ("changeset", "ChangeSet"),
    "ChangeSetError": ("changeset", "ChangeSetError"),
    "ResumeCompatibility": ("compatibility", "ResumeCompatibility"),
    "ResumeProof": ("compatibility", "ResumeProof"),
    "check_resume_compatibility": ("compatibility", "check_resume_compatibility"),
    "evaluate_resume_proof": ("compatibility", "evaluate_resume_proof"),
    "PayloadFile": ("payload", "PayloadFile"),
    "PayloadManifest": ("payload", "PayloadManifest"),
    "build_payload_manifest": ("payload", "build_payload_manifest"),
    "load_payload_manifest": ("payload", "load_payload_manifest"),
    "verify_payload": ("payload", "verify_payload"),
    "verify_payload_archive": ("payload", "verify_payload_archive"),
    "DeploymentResult": ("deployment", "DeploymentResult"),
    "DeploymentSpec": ("deployment", "DeploymentSpec"),
    "RemoteArtifact": ("deployment", "RemoteArtifact"),
    "RemoteLayout": ("deployment", "RemoteLayout"),
    "RemoteTransport": ("deployment", "RemoteTransport"),
    "VersionedRemoteDeployer": ("deployment", "VersionedRemoteDeployer"),
    "RuntimeIdentity": ("runtime", "RuntimeIdentity"),
    "capture_runtime_identity": ("runtime", "capture_runtime_identity"),
    "resolve_runtime_identity": ("runtime", "resolve_runtime_identity"),
    "TrainingStartPlan": ("training", "TrainingStartPlan"),
    "TrainingStartResult": ("training", "TrainingStartResult"),
    "TRAINING_START_SCHEMA": ("training", "TRAINING_START_SCHEMA"),
    "VersionedRemoteTrainingStarter": ("training", "VersionedRemoteTrainingStarter"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), symbol)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
