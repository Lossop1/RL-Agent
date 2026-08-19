from __future__ import annotations

from types import SimpleNamespace

from autotuner.locomotion_console import ops_state


def test_declarations_keep_exact_environment_names_and_explicit_prefixes(monkeypatch) -> None:
    runtime = SimpleNamespace(
        product_id="sample",
        deployment={
            "runs_root": "/runs/sample",
            "monitor_log_files": ["train.log"],
            "training_launch": {
                "environment": {"SAMPLE_INTERVAL": "10"},
                "environment_prefixes": ["SAMPLE_"],
                "resume_state_environment": {"phase": "SAMPLE_PHASE"},
            },
        },
        runtime={
            "schema_version": "rl-agent.runtime-identity/v1",
            "declaration": {"training_process_pattern": "sample_train"},
        },
    )
    monkeypatch.setattr(ops_state, "resolve_product_runtime", lambda _product_id: runtime)
    source = SimpleNamespace(settings=SimpleNamespace(product_id="sample"))

    declarations = ops_state._declarations(source)

    assert declarations["environment_names"] == (
        "PYTHONPATH",
        "SAMPLE_INTERVAL",
        "SAMPLE_PHASE",
    )
    assert declarations["environment_prefixes"] == ("SAMPLE_",)


def test_remote_probe_environment_filter_has_exact_and_prefix_boundaries() -> None:
    script = ops_state._build_remote_probe_script(
        {
            "run_glob": "/runs/sample/*/",
            "process_pattern": "sample_train",
            "log_files": ("train.log",),
            "environment_names": ("PYTHONPATH", "SAMPLE_INTERVAL", "SAMPLE_PHASE"),
            "environment_prefixes": ("SAMPLE_DYNAMIC_",),
            "phase_environment": "SAMPLE_PHASE",
        }
    )

    assert "^(PYTHONPATH|SAMPLE_INTERVAL|SAMPLE_PHASE|SAMPLE_DYNAMIC_[A-Za-z0-9_]*)=" in script
    assert "TAILI_" not in script


def test_declarations_reject_unsafe_environment_prefix(monkeypatch) -> None:
    runtime = SimpleNamespace(
        product_id="sample",
        deployment={
            "runs_root": "/runs/sample",
            "training_launch": {"environment_prefixes": ["SAMPLE.*"]},
        },
        runtime={
            "schema_version": "rl-agent.runtime-identity/v1",
            "declaration": {"training_process_pattern": "sample_train"},
        },
    )
    monkeypatch.setattr(ops_state, "resolve_product_runtime", lambda _product_id: runtime)

    declarations = ops_state._declarations(SimpleNamespace(settings=SimpleNamespace(product_id="sample")))

    assert "error" in declarations
    assert "ending in '_'" in declarations["error"]
