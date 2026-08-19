from __future__ import annotations

from dataclasses import replace

import pytest

from autotuner.adapter.derive import derive
from autotuner.adapter.pipeline import build_config_set, plan_adaptation
from autotuner.adapter.reward_scale import scale_rewards


def test_derive_uses_declared_roles_without_dof_assumption(tmp_path) -> None:
    urdf = tmp_path / "single_joint.urdf"
    urdf.write_text(
        """
<robot name="minimal">
  <link name="base"><inertial><mass value="2"/><inertia iyy="0.2"/></inertial></link>
  <link name="arm"><inertial><mass value="1"/><inertia iyy="0.1"/></inertial></link>
  <joint name="arm_drive" type="revolute">
    <parent link="base"/><child link="arm"/>
    <limit lower="-1" upper="1" effort="20" velocity="5"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )

    result = derive(
        str(urdf),
        {
            "joint_roles": {"arm": r"arm_drive$"},
            "kp_per_effort": 2.0,
            "damping_inertia_floor": 0.01,
        },
    )

    assert result["n_actuated_joints"] == 1
    assert result["mass_kg"] == 3.0
    assert result["effort_limit"] == {"arm": 20.0}
    assert result["Kp_per_joint"] == {"arm": 40}


def test_reward_scaling_has_no_implicit_product_anchor() -> None:
    anchor = {
        "leg_length": 0.5,
        "mass_kg": 10.0,
        "thresholds": {
            "stand_height": 0.4,
            "base_clearance": 0.05,
            "clr_rough_bonus_max": 0.03,
            "air_time_min": 0.1,
            "torque_limit_frac": 0.8,
            "cmd_fwd_range": [0.2, 0.4],
            "cmd_back_range": [0.1, 0.3],
            "cmd_lat_range": [0.1, 0.2],
            "cmd_yaw_range": [0.2, 0.6],
            "dr_mass_range_1": [-1, 1],
            "dr_mass_range_2": [-2, 2],
            "dr_mass_range_3": [-3, 3],
        },
    }

    scaled = scale_rewards(1.0, 20.0, anchor)

    assert scaled["stand_height"] == 0.8
    assert scaled["cmd_fwd_range"] == (0.4, 0.8)
    assert scaled["dr_mass_range_3"] == (-6.0, 6.0)


def test_product_config_set_binds_contract_and_rejects_drift(tmp_path) -> None:
    config_set = build_config_set("taili")
    assert config_set.product_id == "taili"
    assert config_set.product_contract_digest
    assert config_set.framework_composition == "taili_amp_proven"

    stale = replace(config_set, product_contract_digest="stale")
    with pytest.raises(ValueError, match="product contract changed"):
        plan_adaptation(stale, str(tmp_path), "test")


def test_taili_adaptation_preview_roundtrips_without_legacy_deploy(tmp_path) -> None:
    bundle = plan_adaptation(build_config_set("taili"), str(tmp_path), "test")

    assert bundle.roundtrip_ok
    assert bundle.adapted["provenance"]["product_id"] == "taili"
    assert bundle.adapted["dims"]["n_actuated_joints"] == 12
    assert not bundle.legacy_deploy_enabled
