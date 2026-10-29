import pytest
import torch

from src.mamba.graph_v5 import (
    DirectedGraphV5Head, GraphV5Encoder, GraphV5Mamba,
    decode_graph, masked_graph_bce, sparse_graph_bce,
)


def batch(device="cpu", feature_dim=12):
    torch.manual_seed(112)
    features = torch.randn(2, 16, 6, feature_dim, device=device)
    valid = torch.ones(2, 16, 6, dtype=torch.bool, device=device)
    valid[0, :4, 4] = False
    valid[1, :, 5] = False
    features[~valid] = float("nan")
    return {"entity_features": features, "entity_valid": valid,
            "entity_kind": torch.tensor([[0, 1, 2, 3, 3, 3]] * 2, device=device)}


def permute_batch(original, permutation):
    return {"entity_features": original["entity_features"][:, :, permutation],
            "entity_valid": original["entity_valid"][:, :, permutation],
            "entity_kind": original["entity_kind"][:, permutation]}


def test_encoder_and_directed_head_preserve_object_permutation():
    data = batch()
    encoder = GraphV5Encoder(12, d_model=24, n_heads=4).eval()
    head = DirectedGraphV5Head(24, 5).eval()
    permutation = torch.tensor([0, 1, 2, 5, 3, 4])
    original = encoder(data["entity_features"], data["entity_valid"], data["entity_kind"])
    changed = permute_batch(data, permutation)
    reordered = encoder(changed["entity_features"], changed["entity_valid"], changed["entity_kind"])
    torch.testing.assert_close(reordered, original[:, :, permutation], rtol=1e-5, atol=1e-6)
    logits, mask = head(original[:, -1], data["entity_valid"].any(1))
    perm_logits, perm_mask = head(reordered[:, -1], changed["entity_valid"].any(1))
    torch.testing.assert_close(perm_logits, logits[:, permutation][:, :, permutation], rtol=1e-5, atol=1e-6)
    assert torch.equal(perm_mask, mask[:, permutation][:, :, permutation])
    assert (logits - logits.transpose(1, 2)).abs().max() > 1e-3


def test_missing_tokens_cannot_change_valid_entities_or_poison_gradients():
    data = batch()
    data["entity_valid"][0] = False
    data["entity_features"][0] = float("nan")
    encoder = GraphV5Encoder(12, d_model=24, n_heads=4).eval()
    features = data["entity_features"].clone().requires_grad_()
    result = encoder(features, data["entity_valid"], data["entity_kind"])
    altered = features.detach().clone()
    altered[~data["entity_valid"]] = 1e30
    other = encoder(altered, data["entity_valid"], data["entity_kind"])
    torch.testing.assert_close(result, other)
    assert torch.isfinite(result).all() and not result[0].any()
    result.square().sum().backward()
    assert torch.isfinite(features.grad).all()
    assert not features.grad[~data["entity_valid"]].any()
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)


def test_observed_nan_is_rejected_instead_of_silently_fabricated():
    data = batch()
    data["entity_features"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Observed entity"):
        GraphV5Encoder(12)(data["entity_features"], data["entity_valid"], data["entity_kind"])


def test_masked_loss_unknown_values_do_not_become_negatives_or_gradients():
    logits = torch.randn(1, 4, 4, 3, requires_grad=True)
    values = torch.full_like(logits, float("nan"))
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, 0, 3, 0] = mask[0, 3, 1, 2] = True
    values[0, 0, 3, 0], values[0, 3, 1, 2] = 1, 0
    output = masked_graph_bce(logits, values, mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], values[mask])
    torch.testing.assert_close(output["loss"], expected)
    assert output["valid_count"] == 2 and output["positive_count"] == output["negative_count"] == 1
    output["loss"].backward()
    assert not logits.grad[~mask].any() and torch.isfinite(logits.grad).all()
    changed = values.clone()
    changed[~mask] = 99999
    torch.testing.assert_close(masked_graph_bce(logits, changed, mask)["loss"], expected)


def test_sparse_and_dense_supervision_are_identical_and_direction_matters():
    logits = torch.arange(48, dtype=torch.float32).reshape(1, 4, 4, 3) / 11
    indices = torch.tensor([[0, 0, 3, 0], [0, 3, 1, 2]])
    values = torch.tensor([1., 0.])
    mask = torch.zeros_like(logits, dtype=torch.bool)
    targets = torch.zeros_like(logits)
    at = tuple(indices.unbind(-1))
    mask[at], targets[at] = True, values
    dense = masked_graph_bce(logits, targets, mask)
    sparse = sparse_graph_bce(logits, indices, values)
    torch.testing.assert_close(dense["loss"], sparse["loss"])
    reverse = indices[:, [0, 2, 1, 3]]
    assert sparse_graph_bce(logits, reverse, values)["loss"] != sparse["loss"]
    with pytest.raises(ValueError, match="Duplicate"):
        sparse_graph_bce(logits, indices[[0, 0]], values)


def test_empty_loss_is_a_skip_not_a_successful_training_batch():
    logits = torch.zeros(1, 3, 3, 2, requires_grad=True)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    with pytest.raises(ValueError, match="No explicit graph supervision"):
        masked_graph_bce(logits, torch.zeros_like(logits), mask)
    output = masked_graph_bce(logits, torch.zeros_like(logits), mask, require_supervision=False)
    assert output["skipped"] and output["valid_count"] == 0


def test_labels_on_unseen_entities_cannot_silently_enter_loss():
    logits = torch.zeros(1, 4, 4, 2)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, 0, 3, 0] = True
    predicted = torch.ones(1, 4, 4, dtype=torch.bool)
    predicted[:, 3] = False
    predicted[:, :, 3] = False
    with pytest.raises(ValueError, match="never observed"):
        masked_graph_bce(logits, torch.ones_like(logits), mask, predicted)


def test_chinese_decode_preserves_directed_entities_and_padding_mask():
    logits = torch.full((1, 5, 5, 2), -10.)
    logits[0, 1, 3, 0], logits[0, 3, 4, 1], logits[0, 4, 3, 1] = 5, 6, -8
    prediction_mask = torch.ones(1, 5, 5, dtype=torch.bool)
    prediction_mask[:, 2] = prediction_mask[:, :, 2] = False
    logits[0, 2, 4, 0] = 100
    events = [{"layer": "relation", "code": "grasp_control", "name_zh": "抓持控制"},
              {"layer": "relation", "code": "supported_by", "name_zh": "受到支撑"}]
    metadata = [{"entity_ids": ["left_hand", "right_hand", "both_hands", "scanner", "table"],
                 "entity_names_zh": ["左手", "右手", "双手", "扫码枪", "桌面"]}]
    result = decode_graph(logits, events, metadata, prediction_mask=prediction_mask)[0]
    assert len(result) == 2
    assert result[0]["source_name_zh"] == "扫码枪" and result[0]["target_name_zh"] == "桌面"
    assert result[0]["type_name_zh"] == "受到支撑"
    assert result[1]["source_name_zh"] == "右手" and result[1]["target_name_zh"] == "扫码枪"


@pytest.fixture
def real_cuda_model():
    if not torch.cuda.is_available():
        pytest.skip("Real Mamba verification requires CUDA")
    pytest.importorskip("mamba_ssm")
    torch.manual_seed(113)
    return GraphV5Mamba(12, event_type_count=5, d_model=32, n_layer=1,
                        d_state=8, n_heads=4, dropout=0.).cuda().eval()


def test_real_mamba_graph_is_permutation_equivariant(real_cuda_model):
    model, data = real_cuda_model, batch("cuda")
    permutation = torch.tensor([0, 1, 2, 5, 3, 4], device="cuda")
    with torch.no_grad():
        first = model(data)
        second = model(permute_batch(data, permutation))
    torch.testing.assert_close(second["graph_logits"], first["graph_logits"][:, permutation][:, :, permutation], rtol=1e-4, atol=1e-5)
    assert first["graph_logits"].shape == (2, 6, 6, 5)
    assert not first["graph_prediction_mask"][1, 5].any()
    assert not first["entity_states"][1, 5].any()


def test_real_mamba_prefix_and_historical_encode_ignore_future_and_targets(real_cuda_model):
    model, data = real_cuda_model, batch("cuda")
    with torch.no_grad():
        original = model.encode_history(data)
        altered = {k: v.clone() for k, v in data.items()}
        altered["entity_features"][:, 8:] = 100
        altered["entity_valid"][:, 8:] = True
        future_changed = model.encode_history(altered)
        torch.testing.assert_close(original[:, :8], future_changed[:, :8], rtol=1e-4, atol=1e-5)
        poisoned = {**data, "graph_values": torch.full((2, 6, 6, 5), float("nan"), device="cuda"),
                    "graph_label_valid": torch.ones((2, 6, 6, 5), device="cuda", dtype=torch.bool),
                    "target_entity_features": torch.randn(2, 6, 12, device="cuda") * 1e9,
                    "metadata": [{"future_label": "not model input"}] * 2}
        torch.testing.assert_close(original, model.encode_history(poisoned), rtol=0, atol=0)


@pytest.mark.parametrize("amp", [False, True])
def test_real_mamba_backward_has_nonzero_finite_gradients_without_optimizer(real_cuda_model, amp):
    model, data = real_cuda_model, batch("cuda")
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        output = model(data)
    values = torch.zeros_like(output["graph_logits"])
    mask = torch.zeros_like(values, dtype=torch.bool)
    for b, source, target, kind, value in [(0, 1, 3, 0, 1), (0, 3, 4, 1, 0), (1, 0, 4, 3, 1)]:
        mask[b, source, target, kind], values[b, source, target, kind] = True, value
    stats = masked_graph_bce(output["graph_logits"], values, mask, output["graph_prediction_mask"])
    assert stats["valid_count"] == 3 and stats["loss"].item() > 0
    stats["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum().item() for g in gradients) > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert any(type(m).__module__.startswith("mamba_ssm") for m in model.modules())
