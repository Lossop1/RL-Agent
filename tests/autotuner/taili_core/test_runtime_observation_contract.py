import pytest
import torch

from autotuner.taili_core import taili_amp_reference, taili_models, taili_obs, taili_symmetry


def test_body57_layout_and_actor_input_dimension():
    n = 3
    body = taili_obs.assemble_body57(
        torch.zeros(n, 3),
        torch.zeros(n, 3),
        torch.zeros(n, 3),
        torch.zeros(n, 3),
        torch.zeros(n),
        torch.zeros(n, 12),
        torch.zeros(n, 12),
        torch.zeros(n, 12),
        torch.zeros(n, 8),
    )
    assert body.shape == (n, 57)
    assert taili_models.BODY_DIM == 57
    actor = taili_models.EquivariantActor(hidden=(32,))
    assert actor.mean(body, torch.zeros(n, 32)).shape == (n, 12)


def test_body57_mirror_is_an_involution_and_preserves_command_age():
    body = torch.randn(5, 57)
    body[:, 12] = torch.linspace(0.0, 1.0, 5)
    mirrored = taili_symmetry.mirror_actor_body57(body)
    restored = taili_symmetry.mirror_actor_body57(mirrored)
    assert torch.allclose(restored, body)
    assert torch.allclose(mirrored[:, 12], body[:, 12])


def test_previous_command_uses_the_same_physical_mirror_as_current_command():
    body = torch.zeros(1, 57)
    body[:, 6:9] = torch.tensor([[0.4, 0.2, 0.3]])
    body[:, 9:12] = torch.tensor([[-0.3, -0.1, -0.2]])
    mirrored = taili_symmetry.mirror_actor_body57(body)
    assert torch.allclose(mirrored[:, 6:9], torch.tensor([[0.4, -0.2, -0.3]]))
    assert torch.allclose(mirrored[:, 9:12], torch.tensor([[-0.3, 0.1, 0.2]]))


def test_foot_mask_maps_to_role_major_joint_order():
    mask = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    joint_mask = taili_obs.foot_mask_to_joint12(mask)
    assert torch.equal(joint_mask, torch.tensor([[
        1.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 1.0,
    ]]))


def test_amp_frame51_uses_only_motion_and_deployable_command_conditioning():
    motion = torch.randn(3, 43)
    commands = torch.tensor([
        [0.5, 0.0, 0.0],
        [0.0, -0.3, 0.0],
        [0.0, 0.0, 0.6],
    ])

    frame = taili_amp_reference.conditioned_frame51(motion, commands)

    assert frame.shape == (3, 51)
    assert torch.equal(frame[:, :43], motion)
    assert torch.equal(frame[:, 43:46], commands)
    assert torch.equal(frame[:, 46:], taili_amp_reference.mode_onehot(commands))


def test_amp_frame51_rejects_non_motion43_inputs():
    with pytest.raises(ValueError, match="motion width must be 43"):
        taili_amp_reference.conditioned_frame51(torch.zeros(2, 46), torch.zeros(2, 3))
