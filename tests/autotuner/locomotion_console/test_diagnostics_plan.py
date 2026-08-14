import autotuner.locomotion_console.diagnostics as diagnostics_module
from types import SimpleNamespace

from autotuner.locomotion_console.diagnostics import (
    _CHECKPOINT_CATALOG_LIMIT,
    _PRESET_BY_ID,
    DiagnosticsController,
    _normalize_plan,
)
from autotuner.locomotion_console.schemas import DiagnosticCheckpoint, DiagnosticRunRequest


def test_dr_plan_preserves_population_profile_and_factor_selectors():
    request = DiagnosticRunRequest(
        preset="dr",
        plan={
            "dr_cases": [
                {
                    "level": 2,
                    "label": "DR2 contact",
                    "population": "forced",
                    "stress_profile": "single",
                    "stress_factor": "contact",
                }
            ]
        },
    )

    case = request.plan.dr_cases[0]
    assert case.population == "forced"
    assert case.stress_profile == "single"
    assert case.stress_factor == "contact"

    plan = _normalize_plan(_PRESET_BY_ID["dr"], request.plan)
    assert plan["dr_cases"] == [
        {
            "level": 2,
            "label": "DR2 contact",
            "population": "forced",
            "stress_profile": "single",
            "stress_factor": "contact",
        }
    ]


def test_stairs_direction_up_normalizes_to_stairs_up():
    plan = _normalize_plan(
        _PRESET_BY_ID["terrain"],
        {
            "terrains": [
                {"type": "stairs", "level": 5, "params": {"direction": "up", "step_height": 0.18}},
            ],
        },
    )
    assert plan["terrains"][0]["type"] == "stairs_up"
    assert plan["terrains"][0]["params"]["direction"] == "up"


def test_terrain_default_uses_explicit_stairs_up():
    plan = _normalize_plan(_PRESET_BY_ID["terrain"])
    assert any(item["type"] == "stairs_up" and item["params"].get("direction") == "up" for item in plan["terrains"])


def test_direction_change_respects_explicit_transition_settle():
    plan = _normalize_plan(
        _PRESET_BY_ID["directions"],
        {
            "commands": [
                {"id": "fwd", "vx": 0.5, "duration_s": 2.0},
                {"id": "back", "vx": -0.4, "duration_s": 2.0, "settle_s": 0.2},
                {"id": "yaw", "wz": 0.6, "duration_s": 2.0, "settle_s": 0.0},
            ]
        },
    )

    assert plan["commands"][1]["settle_s"] == 0.20
    assert plan["commands"][2]["settle_s"] == 0.0


def test_checkpoint_catalog_retains_extended_history_per_framework(monkeypatch):
    frameworks = [
        type("Framework", (), {"id": "current"})(),
        type("Framework", (), {"id": "reference"})(),
    ]
    monkeypatch.setattr(diagnostics_module, "list_framework_profiles", lambda: frameworks)
    controller = object.__new__(DiagnosticsController)
    observed_limits = []

    def fake_recent(_remote, framework, limit):
        observed_limits.append(limit)
        return [
            DiagnosticCheckpoint(
                path=f"/{framework.id}/run/checkpoints/agent_{iteration}.pt",
                name=f"agent_{iteration}.pt",
                run_name="run",
                iteration=iteration,
                kind="iteration",
                framework_id=framework.id,
            )
            for iteration in range(70)
        ]

    controller._recent_checkpoints_for_framework = fake_recent
    checkpoints = controller._recent_checkpoints(object())

    assert observed_limits == [_CHECKPOINT_CATALOG_LIMIT, _CHECKPOINT_CATALOG_LIMIT]
    assert len(checkpoints) == _CHECKPOINT_CATALOG_LIMIT
    assert checkpoints[0].is_default is True


def test_payload_shell_prefers_checkpoint_run_metadata():
    controller = object.__new__(DiagnosticsController)
    controller.settings = SimpleNamespace(diagnostic_tool_root="/root/gpufree-data/tools/diag")

    shell = controller._payload_root_shell(
        "/root/gpufree-data/taili_runs/example/checkpoints/agent_20000.pt"
    )

    assert "RUN_DIR=" in shell
    assert '"$RUN_DIR/run.json"' in shell
    assert "payload_root" in shell
    assert "payload selected from checkpoint run metadata" in shell
    assert 'if [ -z "$PAYLOAD" ]' in shell


def test_payload_shell_fallback_includes_recovered_payloads():
    controller = object.__new__(DiagnosticsController)
    controller.settings = SimpleNamespace(diagnostic_tool_root="/root/gpufree-data/tools/diag")

    shell = controller._payload_root_shell()

    assert "/root/gpufree-data/training_payloads/taili_blind_runtime_*" in shell
    assert "/root/gpufree-data/training_payloads/taili_recovered_*" in shell


def test_active_run_latest_checkpoint_is_the_default(monkeypatch):
    framework = type("Framework", (), {"id": "current"})()
    monkeypatch.setattr(diagnostics_module, "list_framework_profiles", lambda: [framework])
    controller = object.__new__(DiagnosticsController)
    controller.source = SimpleNamespace(
        _newest_run_with_checkpoint=lambda _remote: "/runs/current_run"
    )

    def fake_recent(_remote, _framework, _limit):
        return [
            DiagnosticCheckpoint(
                path="/runs/old_run/checkpoints/agent_128000.pt",
                name="agent_128000.pt",
                run_name="old_run",
                iteration=128000,
                kind="iteration",
                framework_id="current",
            ),
            DiagnosticCheckpoint(
                path="/runs/current_run/checkpoints/agent_46000.pt",
                name="agent_46000.pt",
                run_name="current_run",
                iteration=46000,
                kind="iteration",
                framework_id="current",
            ),
        ]

    controller._recent_checkpoints_for_framework = fake_recent

    checkpoints = controller._recent_checkpoints(object())

    assert checkpoints[0].run_name == "current_run"
    assert checkpoints[0].name == "agent_46000.pt"
    assert checkpoints[0].is_default is True


def test_active_run_history_is_not_evicted_by_other_run_latest_checkpoints(monkeypatch):
    framework = type("Framework", (), {"id": "current"})()
    monkeypatch.setattr(diagnostics_module, "list_framework_profiles", lambda: [framework])
    controller = object.__new__(DiagnosticsController)
    controller.source = SimpleNamespace(
        _newest_run_with_checkpoint=lambda _remote: "/runs/current_run"
    )

    def checkpoint(run_name, iteration):
        return DiagnosticCheckpoint(
            path=f"/runs/{run_name}/checkpoints/agent_{iteration}.pt",
            name=f"agent_{iteration}.pt",
            run_name=run_name,
            iteration=iteration,
            kind="iteration",
            framework_id="current",
        )

    def fake_recent(_remote, _framework, _limit):
        other_runs = [checkpoint(f"old_run_{index:03d}", 100000 + index) for index in range(105)]
        return other_runs + [
            checkpoint("current_run", 2000),
            checkpoint("current_run", 8000),
            checkpoint("current_run", 12000),
        ]

    controller._recent_checkpoints_for_framework = fake_recent

    checkpoints = controller._recent_checkpoints(object())

    assert [item.iteration for item in checkpoints[:3]] == [12000, 8000, 2000]
    assert all(item.run_name == "current_run" for item in checkpoints[:3])
    assert checkpoints[0].is_default is True
