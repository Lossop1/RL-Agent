from autotuner.locomotion_console.diagnostics import _PRESET_BY_ID, _normalize_plan


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
