# Torch is optional in the test environment; importorskip must run first.
# ruff: noqa: E402
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

torch = pytest.importorskip("torch")

from products.taili.blind_locomotion import parametric_ref
from products.taili.core import taili_amp_reference, taili_export, taili_geometry


def test_urdf_foot_spheres_match_the_runtime_geometry_constant():
    root = Path(__file__).resolve().parents[4]
    urdf = root / "locomotion-console-ui" / "public" / "robot" / "taili_dog_description" / "urdf" / "robot.urdf"
    tree = ET.parse(urdf)
    radii = []
    for link in tree.getroot().findall("link"):
        if not str(link.attrib.get("name", "")).endswith("_foot"):
            continue
        sphere = link.find("./collision/geometry/sphere")
        assert sphere is not None
        radii.append(float(sphere.attrib["radius"]))

    assert len(radii) == 4
    assert radii == pytest.approx([taili_geometry.FOOT_RADIUS] * 4)


def test_default_height_places_the_foot_sphere_on_the_ground():
    assert taili_amp_reference.H0 == pytest.approx(
        taili_geometry.DEFAULT_FOOT_CENTER_HEIGHT,
        abs=1e-7,
    )
    assert taili_amp_reference.BASE_HEIGHT_REF == pytest.approx(
        taili_geometry.NOMINAL_BASE_HEIGHT
    )
    assert parametric_ref.BASE_Z == pytest.approx(taili_geometry.NOMINAL_BASE_HEIGHT)
    assert taili_export.NOMINAL_BASE_H == pytest.approx(taili_geometry.NOMINAL_BASE_HEIGHT)

    command = torch.zeros((1, 3))
    _, _, base_height, _, feet = taili_amp_reference.flat_reference(command, torch.zeros(1))
    foot_center_world_z = base_height[0, 0] + feet[0, :, 2]
    assert torch.allclose(
        foot_center_world_z,
        torch.full((4,), taili_geometry.FOOT_RADIUS),
        atol=1e-6,
    )


def test_sole_clearance_removes_the_foot_collision_radius_exactly():
    ground = torch.tensor([0.0, 0.15, -0.08])
    expected = torch.tensor([0.0, 0.06, 0.12])
    foot_center = ground + taili_geometry.FOOT_RADIUS + expected

    measured = taili_geometry.sole_clearance(foot_center, ground)

    assert torch.allclose(measured, expected, atol=1e-7)


def test_flat_stand_reset_removes_moving_clip_velocity_and_uses_nominal_height():
    root = torch.randn((3, 13))
    joint_pos = torch.randn((3, 12))
    joint_vel = torch.randn((3, 12))
    commands = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    origins = torch.tensor([[1.0, 2.0, 0.1], [3.0, 4.0, 0.2], [5.0, 6.0, 0.3]])
    default_joint_pos = torch.zeros((3, 12))
    flat_mask = torch.tensor([True, True, False])
    moving_root = root[1].clone()
    terrain_root = root[2].clone()

    stand_mask = taili_amp_reference.apply_flat_stand_reset(
        root,
        joint_pos,
        joint_vel,
        commands,
        origins,
        default_joint_pos,
        flat_mask,
        sole_clearance=0.003,
    )

    assert stand_mask.tolist() == [True, False, False]
    assert root[0, :2].tolist() == pytest.approx(origins[0, :2].tolist())
    assert root[0, 2].item() == pytest.approx(taili_geometry.NOMINAL_BASE_HEIGHT + 0.103)
    assert root[0, 3:7].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert torch.count_nonzero(root[0, 7:13]) == 0
    assert torch.count_nonzero(joint_pos[0]) == 0
    assert torch.count_nonzero(joint_vel[0]) == 0
    assert torch.equal(root[1], moving_root)
    assert torch.equal(root[2], terrain_root)


def test_moving_reset_neutralizes_reference_velocity_credit_for_every_environment():
    root = torch.randn((3, 13))
    joint_vel = torch.randn((3, 12))
    pose_before = root[:, :7].clone()

    taili_amp_reference.neutralize_reset_velocities(root, joint_vel)

    assert torch.equal(root[:, :7], pose_before)
    assert torch.count_nonzero(root[:, 7:13]) == 0
    assert torch.count_nonzero(joint_vel) == 0


def test_terminal_failure_keeps_the_failed_command_for_the_next_attempt():
    sampled = torch.tensor([
        [0.0, 0.4, 0.0],
        [0.0, 0.0, 0.6],
        [0.5, 0.0, 0.0],
    ])
    previous = torch.tensor([
        [0.5, 0.0, 0.0],
        [-0.4, 0.0, 0.0],
        [0.0, -0.4, 0.0],
    ])
    failed = torch.tensor([True, False, True])

    restored = taili_amp_reference.preserve_failed_command_targets(
        sampled, previous, failed
    )

    assert torch.equal(restored[failed], previous[failed])
    assert torch.equal(restored[~failed], sampled[~failed])
