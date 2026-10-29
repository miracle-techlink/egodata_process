import pytest
import torch

from src.encoding.encoder import HOIEncoder


def relation_features(coordinate_mode, object_position):
    encoder = HOIEncoder({
        "d_model": 16, "n_heads": 2, "n_layers": 1, "dim_ff": 32,
        "n_max": 2, "max_frames": 4, "dropout": 0.0,
        "coordinate_mode": coordinate_mode,
    }).eval()
    hands = torch.zeros(1, 3, 2, 21, 3)
    hands[:, :, 1, :, 0] = 0.5
    objects = torch.zeros(1, 3, 2, 13)
    objects[:, :, :, :3] = object_position
    observed = []
    handle = encoder.rel_mlp.register_forward_pre_hook(
        lambda module, inputs: observed.append(inputs[0].detach().clone())
    )
    with torch.no_grad():
        encoder(hands, objects, torch.ones(1, 3, 2, dtype=torch.bool))
    handle.remove()
    return observed[0]


def test_mixed_coordinates_exclude_cross_space_geometry():
    first = relation_features("mixed", 1.0)
    translated = relation_features("mixed", 100.0)
    torch.testing.assert_close(first, translated)
    assert first[:, :, :3].abs().sum() > 0


def test_camera_coordinates_enable_cross_entity_geometry():
    first = relation_features("camera", 1.0)
    translated = relation_features("camera", 100.0)
    assert not torch.allclose(first[:, :, :3], translated[:, :, :3])


def test_legacy_inputs_default_to_safe_mixed_coordinates():
    assert HOIEncoder({"n_max": 1}).coordinate_mode == "mixed"


def test_unknown_coordinate_mode_rejected():
    with pytest.raises(ValueError, match="coordinate_mode"):
        HOIEncoder({"coordinate_mode": "world"})
