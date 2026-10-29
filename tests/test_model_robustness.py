import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.encoding.decoder import HOIDecoder
from src.encoding.encoder import HOIEncoder
from src.mamba.train import (
    DEFAULT_CONFIG,
    IntentModel,
    LSTMBackbone,
    build_model,
    compute_losses,
    mask_object_logits,
    run_epoch,
    save_checkpoint,
)


def small_config(object_count=3):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["model"].update(backbone="lstm", d_model=8, n_layer=2, dropout=0.0)
    config["model"]["encoder"].update(n_heads=2, n_layers=1, dim_ff=16, max_frames=8)
    config["model"]["decoder"]["n_max"] = object_count
    config["train"].update(log_interval=0, grad_clip=0.0)
    return config


def small_batch(object_count=3):
    torch.manual_seed(902)
    slot_mask = torch.ones(2, object_count, dtype=torch.bool)
    if object_count:
        slot_mask[:, -1] = False
    return {
        "hand_seq": torch.randn(2, 5, 2, 21, 3),
        "obj_seq": torch.randn(2, 5, object_count, 13),
        "obj_mask": slot_mask,
        "subject": torch.tensor([0, 1]),
        "object": torch.tensor([0 if object_count > 1 else object_count, object_count]),
        "interaction_type": torch.tensor([2, 4]),
        "label_valid": torch.ones(2, dtype=torch.bool),
    }


def assert_finite_parameter_gradients(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("coordinate_mode", ["mixed", "camera"])
@pytest.mark.parametrize("padding_value", [float("nan"), float("inf"), 1e30, 7.0])
def test_encoder_padding_is_inert_before_nonlinear_operations(coordinate_mode, padding_value):
    torch.manual_seed(17)
    encoder = HOIEncoder({
        "d_model": 8, "n_heads": 2, "n_layers": 1, "dim_ff": 16,
        "n_max": 3, "dropout": 0.0, "coordinate_mode": coordinate_mode,
    }).eval()
    batch = small_batch()
    frame_mask = batch["obj_mask"][:, None].expand(-1, 5, -1).clone()
    frame_mask[:, 1, 0] = False
    clean_objects = batch["obj_seq"].masked_fill(~frame_mask[..., None], 0.0).requires_grad_()
    dirty_objects = clean_objects.detach().masked_fill(~frame_mask[..., None], padding_value)
    reference = encoder(batch["hand_seq"], clean_objects, frame_mask)
    reference.square().sum().backward()
    reference_gradients = {name: parameter.grad.clone() for name, parameter in encoder.named_parameters()}
    encoder.zero_grad(set_to_none=True)
    dirty_objects.requires_grad_()
    result = encoder(batch["hand_seq"], dirty_objects, frame_mask)
    torch.testing.assert_close(result, reference)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result[:, :, 3:][~frame_mask]) == 0
    result.square().sum().backward()
    assert_finite_parameter_gradients(encoder)
    assert torch.isfinite(dirty_objects.grad).all()
    assert torch.count_nonzero(dirty_objects.grad[~frame_mask]) == 0
    torch.testing.assert_close(dirty_objects.grad, clean_objects.grad)
    for name, parameter in encoder.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_gradients[name])


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_unobserved_hand_has_no_residual_features_or_gradients(invalid_value):
    torch.manual_seed(18)
    encoder = HOIEncoder({
        "d_model": 8, "n_heads": 2, "n_layers": 1, "dim_ff": 16,
        "n_max": 3, "dropout": 0.0,
    }).eval()
    batch = small_batch()
    missing = batch["hand_seq"].clone()
    missing[:, 1, 0] = float("nan")
    malformed = batch["hand_seq"].clone()
    malformed[:, 1, 0] = 1e30
    malformed[:, 1, 0, 0, 0] = invalid_value
    malformed.requires_grad_()
    frame_mask = batch["obj_mask"][:, None].expand(-1, 5, -1)
    reference = encoder(missing, batch["obj_seq"], frame_mask)
    result = encoder(malformed, batch["obj_seq"], frame_mask)
    torch.testing.assert_close(result, reference)
    result.square().sum().backward()
    assert_finite_parameter_gradients(encoder)
    assert torch.isfinite(malformed.grad).all()
    assert torch.count_nonzero(malformed.grad[:, 1, 0]) == 0


@pytest.mark.parametrize("time_axis", [False, True])
@pytest.mark.parametrize("padding_value", [float("nan"), float("inf"), 1e30])
def test_decoder_masks_padding_before_pooling(time_axis, padding_value):
    torch.manual_seed(19)
    decoder = HOIDecoder({"d_model": 8, "n_max": 3, "dropout": 0.0}).eval()
    tokens = torch.randn(2, 5, 6, 8) if time_axis else torch.randn(2, 6, 8)
    object_mask = torch.tensor([[True, False, False], [False, False, False]])
    entity_mask = torch.cat([torch.ones(2, 3, dtype=torch.bool), object_mask], dim=1)
    token_mask = entity_mask[:, None, :, None] if time_axis else entity_mask[..., None]
    clean_tokens = tokens.masked_fill(~token_mask, 0.0)
    dirty_tokens = clean_tokens.masked_fill(~token_mask, padding_value).requires_grad_()
    decoder_mask = object_mask[:, None].expand(-1, 5, -1) if time_axis else object_mask
    reference = decoder(clean_tokens, decoder_mask)
    result = decoder(dirty_tokens, decoder_mask)
    for head in reference:
        torch.testing.assert_close(result[head], reference[head])
    (result["subject"].square().sum() + result["interaction_type"].square().sum()).backward()
    assert_finite_parameter_gradients(decoder)
    assert torch.isfinite(dirty_tokens.grad).all()
    assert torch.count_nonzero(dirty_tokens.grad.masked_select(~token_mask)) == 0


@pytest.mark.parametrize("object_count", [0, 3])
def test_fully_missing_observations_train_without_selecting_padding(object_count):
    config = small_config(object_count)
    model = build_model(config).eval()
    batch = small_batch(object_count)
    batch["hand_seq"].fill_(float("nan"))
    batch["obj_seq"].fill_(float("nan"))
    batch["obj_mask"].fill_(False)
    batch["object"].fill_(object_count)
    with torch.no_grad():
        model.decoder.object_head.bias[-1] = -20000.0
    logits = model(batch)
    masked = mask_object_logits(logits["object"], batch["obj_mask"])
    assert torch.equal(masked.argmax(-1), batch["object"])
    loss, parts = compute_losses(logits, batch, config["train"]["loss_weights"])
    assert parts["object"] == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert_finite_parameter_gradients(model)
    metrics = run_epoch(model, [batch], torch.device("cpu"), config, False)
    assert metrics["acc_object"] == 1.0
    assert metrics["loss_object"] == 0.0


@pytest.mark.parametrize("all_ignored", [False, True])
def test_ignored_nonfinite_logits_have_zero_gradients(all_ignored):
    class_counts = {"subject": 3, "object": 4, "interaction_type": 7}
    logits = {}
    batch = {
        "obj_mask": torch.tensor([[True, False, False]] * 3),
        "label_valid": torch.ones(3, dtype=torch.bool),
    }
    for head, class_count in class_counts.items():
        head_logits = torch.linspace(-1, 1, 3 * class_count).reshape(3, class_count)
        head_logits[1] = float("nan")
        head_logits[2] = float("inf")
        if all_ignored:
            head_logits[0] = float("nan")
        logits[head] = head_logits.requires_grad_()
        batch[head] = torch.tensor([-1 if all_ignored else 0, -1, -1])
    weights = DEFAULT_CONFIG["train"]["loss_weights"]
    loss, parts = compute_losses(logits, batch, weights)
    assert torch.isfinite(loss)
    if all_ignored:
        assert loss.item() == 0.0
        assert all(value == 0.0 for value in parts.values())
    loss.backward()
    for head in class_counts:
        assert torch.isfinite(logits[head].grad).all()
        assert torch.count_nonzero(logits[head].grad[batch[head] == -1]) == 0
        if not all_ignored:
            assert torch.count_nonzero(logits[head].grad[0]) > 0


class SuppliedLogits(nn.Module):
    def forward(self, batch):
        return batch["logits"]


def test_partially_ignored_nonfinite_rows_do_not_pollute_epoch_metrics():
    batch = {
        "obj_mask": torch.zeros(3, 3, dtype=torch.bool),
        "logits": {},
        "label_valid": torch.ones(3, dtype=torch.bool),
    }
    for head, class_count in {"subject": 3, "object": 4, "interaction_type": 7}.items():
        target = 3 if head == "object" else 1
        batch[head] = torch.tensor([target, -1, -1])
        logits = torch.zeros(3, class_count)
        logits[0, target] = 2.0
        logits[1] = float("nan")
        logits[2] = float("inf")
        batch["logits"][head] = logits
    metrics = run_epoch(SuppliedLogits(), [batch], torch.device("cpu"), small_config(), False)
    for head, class_count in {"subject": 3, "object": 4, "interaction_type": 7}.items():
        assert metrics[f"acc_{head}"] == 1.0
        assert metrics[f"f1_macro_{head}"] == pytest.approx(1 / class_count)
        expected = 0.0 if head == "object" else functional.cross_entropy(
            batch["logits"][head][:1], batch[head][:1]
        ).item()
        assert metrics[f"loss_{head}"] == pytest.approx(expected)
    assert metrics["acc_joint"] == 1.0


def test_full_model_batch_outputs_and_input_gradients_are_independent():
    torch.manual_seed(21)
    model = build_model(small_config()).eval()
    batch = small_batch()
    batch["hand_seq"].requires_grad_()
    batch["obj_seq"].requires_grad_()
    combined = model(batch)
    for batch_index in range(2):
        single = {name: value[batch_index:batch_index + 1] for name, value in batch.items()}
        separate = model(single)
        for head in combined:
            torch.testing.assert_close(combined[head][batch_index:batch_index + 1], separate[head])
    combined["subject"][0].square().sum().backward()
    assert torch.count_nonzero(batch["hand_seq"].grad[1]) == 0
    assert torch.count_nonzero(batch["obj_seq"].grad[1]) == 0
    assert torch.count_nonzero(batch["hand_seq"].grad[0]) > 0


def test_full_model_padding_does_not_change_loss_or_parameter_gradients():
    torch.manual_seed(29)
    config = small_config()
    model = build_model(config).eval()
    reference_batch = small_batch()
    reference_batch["obj_seq"][:, :, -1] = 0.0
    reference_loss, reference_parts = compute_losses(
        model(reference_batch), reference_batch, config["train"]["loss_weights"]
    )
    reference_loss.backward()
    reference_gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    dirty_batch = {name: value.clone() for name, value in reference_batch.items()}
    dirty_batch["obj_seq"][:, :, -1] = float("nan")
    dirty_batch["obj_seq"][:, 2, -1] = 1e30
    dirty_batch["obj_seq"].requires_grad_()
    actual_loss, actual_parts = compute_losses(
        model(dirty_batch), dirty_batch, config["train"]["loss_weights"]
    )
    torch.testing.assert_close(actual_loss, reference_loss)
    assert actual_parts == pytest.approx(reference_parts)
    actual_loss.backward()
    assert_finite_parameter_gradients(model)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_gradients[name])
    assert torch.count_nonzero(dirty_batch["obj_seq"].grad[:, :, -1]) == 0


class IdentityTokenEncoder(nn.Module):
    def forward(self, hand_seq, obj_seq, obj_mask=None):
        return hand_seq


class IdentityTokenDecoder(nn.Module):
    def forward(self, tokens, obj_mask=None):
        return {"tokens": tokens}


def test_real_lstm_keeps_entity_trajectories_separate_and_causal():
    torch.manual_seed(22)
    model = IntentModel(IdentityTokenEncoder(), LSTMBackbone(8, 2, 0.0), IdentityTokenDecoder()).eval()
    tokens = torch.randn(2, 5, 4, 8).requires_grad_()
    batch = {"hand_seq": tokens, "obj_seq": torch.zeros(2, 5, 1, 13),
             "obj_mask": torch.ones(2, 1, dtype=torch.bool)}
    result = model(batch)["tokens"]
    for entity_index in range(4):
        separate = model.backbone(tokens[:, :, entity_index])
        torch.testing.assert_close(result[:, :, entity_index], separate)
    result[0, :3, 1].square().sum().backward()
    assert torch.count_nonzero(tokens.grad[1]) == 0
    assert torch.count_nonzero(tokens.grad[0, :, [0, 2, 3]]) == 0
    assert torch.count_nonzero(tokens.grad[0, 3:, 1]) == 0
    assert torch.count_nonzero(tokens.grad[0, :3, 1]) > 0


def test_encoder_and_lstm_prefix_does_not_depend_on_future_frames():
    torch.manual_seed(23)
    model = build_model(small_config()).eval()
    batch = small_batch()
    captured = []
    handle = model.decoder.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    )
    model(batch)
    altered = {name: value.clone() for name, value in batch.items()}
    altered["hand_seq"][:, 3:] = torch.randn_like(altered["hand_seq"][:, 3:]) * 50
    altered["obj_seq"][:, 3:] = torch.randn_like(altered["obj_seq"][:, 3:]) * 50
    model(altered)
    prefix = {name: value[:, :3] if name in {"hand_seq", "obj_seq"} else value
              for name, value in batch.items()}
    model(prefix)
    handle.remove()
    torch.testing.assert_close(captured[0][:, :3], captured[1][:, :3])
    torch.testing.assert_close(captured[0][:, :3], captured[2])
    assert not torch.allclose(captured[0][:, 3:], captured[1][:, 3:])


def test_checkpoint_reload_preserves_logits_and_next_optimizer_update(tmp_path):
    torch.manual_seed(24)
    config = small_config()
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = small_batch()
    run_epoch(model, [batch], torch.device("cpu"), config, False, optimizer)
    model.eval()
    before = model(batch)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, model, optimizer, 3, config, 0.7)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["contract_version"] == config["contract_version"]
    assert checkpoint["epoch"] == 3
    assert checkpoint["best_metric"] == 0.7
    restored = build_model(checkpoint["config"])
    restored.load_state_dict(checkpoint["model"])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored.eval()
    after = restored(batch)
    for head in before:
        torch.testing.assert_close(before[head], after[head], rtol=0, atol=0)
    run_epoch(model, [batch], torch.device("cpu"), config, False, optimizer)
    run_epoch(restored, [batch], torch.device("cpu"), config, False, restored_optimizer)
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, restored.state_dict()[name], rtol=0, atol=0)
