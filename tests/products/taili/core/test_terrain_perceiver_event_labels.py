import torch

from products.taili.core import taili_models, taili_symmetry
from products.taili.core.taili_terrain_labels import assemble_risk8


def test_terrain_perceiver_risk_head_matches_post_contact_contract():
    perceiver = taili_models.TerrainPerceiver()
    history = torch.zeros(2, taili_models.HISTORY_LEN, taili_models.TICK_DIM)
    geom, risk = perceiver.aux(perceiver.encode(history))

    assert geom.shape == (2, 9)
    assert taili_models.RISK_DIM == 8
    assert risk.shape == (2, 8)


def test_event_labels_are_masked_until_physical_event_is_visible():
    label, mask = assemble_risk8(
        impact_n=torch.tensor([0.1, 0.2, 0.3]),
        support_instability=torch.tensor([0.4, 0.5, 0.6]),
        event_direction=torch.tensor([1, 1, -1]),
        lead_foot4=torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]),
        event_visible=torch.tensor([False, True, True]),
    )

    assert label.shape == mask.shape == (3, 8)
    assert torch.equal(mask[:, :2], torch.ones(3, 2))
    assert torch.equal(mask[0, 2:], torch.zeros(6))
    assert torch.allclose(label[1], torch.tensor([0.2, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(label[2], torch.tensor([0.3, 0.6, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]))


def test_event_risk_mirror_swaps_only_left_and_right_lead_foot():
    value = torch.tensor([[0.1, 0.2, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    mirrored = taili_symmetry.mirror_risk_label8(value)

    assert torch.allclose(
        mirrored,
        torch.tensor([[0.1, 0.2, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
    )
    assert torch.equal(taili_symmetry.mirror_risk_label8(mirrored), value)
