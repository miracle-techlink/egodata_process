"""CPU trainer regressions; run with pytest or python tests/test_training.py."""
import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mamba.train import (
    DEFAULT_CONFIG, IntentModel, SamplesDataset, build_model, collate_fn,
    compute_losses, run_epoch,
)


HEAD_CLASSES = {"subject": 3, "object": 4, "interaction_type": 7}


class TokenEncoder(nn.Module):
    def forward(self, hand_seq, obj_seq, obj_mask=None, gaze_seq=None):
        self.obj_mask = obj_mask
        self.gaze_seq = gaze_seq
        return hand_seq


class CumulativeBackbone(nn.Module):
    def forward(self, tokens):
        self.tokens = tokens
        return tokens.cumsum(dim=1)


class TokenDecoder(nn.Module):
    def forward(self, tokens, obj_mask=None):
        self.obj_mask = obj_mask
        return {"tokens": tokens}


class BatchLogitsModel(nn.Module):
    def forward(self, batch):
        return {head: batch[f"logits_{head}"] for head in HEAD_CLASSES}


class TrainableLogitsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.ParameterDict({
            head: nn.Parameter(torch.ones(class_count))
            for head, class_count in HEAD_CLASSES.items()
        })

    def forward(self, batch):
        batch_size = batch["subject"].shape[0]
        return {
            head: logits.unsqueeze(0).expand(batch_size, -1)
            for head, logits in self.logits.items()
        }


def make_logits_batch(targets, predictions):
    batch_size = len(targets["subject"])
    batch = {head: torch.tensor(labels, dtype=torch.long)
             for head, labels in targets.items()}
    batch["obj_mask"] = torch.ones(batch_size, HEAD_CLASSES["object"] - 1,
                                   dtype=torch.bool)
    batch["label_valid"] = torch.ones(batch_size, dtype=torch.bool)
    for head, class_count in HEAD_CLASSES.items():
        predicted = torch.tensor(predictions[head], dtype=torch.long)
        batch[f"logits_{head}"] = F.one_hot(predicted, class_count).float() * 8.0 - 4.0
    return batch


def slice_batch(batch, start, stop):
    return {name: value[start:stop] for name, value in batch.items()}


def assert_close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7), (actual, expected)


def assert_zero_metrics(metrics):
    for name, value in metrics.items():
        assert math.isfinite(value), (name, value)
        if name != "seconds":
            assert value == 0.0, (name, value)


def test_temporal_backbone_preserves_batch_time_entity_identities():
    batch_size, window_size, entity_count, feature_dim = 2, 3, 4, 2
    tokens = (1000 * torch.arange(batch_size)[:, None, None, None]
              + 100 * torch.arange(window_size)[None, :, None, None]
              + 10 * torch.arange(entity_count)[None, None, :, None]
              + torch.arange(feature_dim)[None, None, None, :] + 1).float()
    noncontiguous = tokens.transpose(1, 2).contiguous().transpose(1, 2)
    assert not noncontiguous.is_contiguous()
    for input_tokens in (tokens, noncontiguous):
        input_tokens = input_tokens.detach().requires_grad_(True)
        encoder, backbone, decoder = TokenEncoder(), CumulativeBackbone(), TokenDecoder()
        model = IntentModel(encoder, backbone, decoder)
        batch = {
            "hand_seq": input_tokens,
            "obj_seq": torch.zeros(batch_size, window_size, 1, 13),
            "obj_mask": torch.tensor([[True], [False]]),
            "gaze_seq": torch.zeros(batch_size, window_size, 3),
        }
        actual = model(batch)["tokens"]
        torch.testing.assert_close(actual, input_tokens.cumsum(dim=1))
        for batch_index in range(batch_size):
            for entity_index in range(entity_count):
                torch.testing.assert_close(
                    backbone.tokens[batch_index * entity_count + entity_index],
                    input_tokens[batch_index, :, entity_index],
                )
        torch.testing.assert_close(
            encoder.obj_mask, batch["obj_mask"][:, None, :].expand(-1, window_size, -1)
        )
        assert encoder.gaze_seq is batch["gaze_seq"]
        assert decoder.obj_mask is batch["obj_mask"]
        actual.sum().backward()
        expected_gradient = torch.arange(window_size, 0, -1).view(1, window_size, 1, 1)
        torch.testing.assert_close(input_tokens.grad, expected_gradient.expand_as(tokens).float())


def test_pooled_tokens_keep_the_original_temporal_path():
    tokens = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    backbone = CumulativeBackbone()
    model = IntentModel(TokenEncoder(), backbone, TokenDecoder())
    batch = {"hand_seq": tokens, "obj_seq": torch.zeros(2, 3, 1, 13),
             "obj_mask": torch.ones(2, 1, dtype=torch.bool)}
    torch.testing.assert_close(model(batch)["tokens"], tokens.cumsum(dim=1))
    assert backbone.tokens is tokens


def test_epoch_macro_f1_joint_accuracy_and_object_mask():
    batch = make_logits_batch(
        {"subject": [0, 0, 1, 1, 2, 2], "object": [0, 1, 1, 3, 3, 0],
         "interaction_type": [0, 1, 1, 2, 2, 0]},
        {"subject": [0, 1, 1, 2, 2, 2], "object": [0, 1, 0, 3, 1, 0],
         "interaction_type": [0, 1, 2, 2, 0, 0]},
    )
    batch["obj_mask"][:, 2] = False
    batch["logits_object"][:, 2] = 1000.0
    metrics = run_epoch(BatchLogitsModel(), [slice_batch(batch, 0, 2), slice_batch(batch, 2, 6)],
                        torch.device("cpu"), DEFAULT_CONFIG, False)
    for head in HEAD_CLASSES:
        assert_close(metrics[f"acc_{head}"], 2 / 3)
    assert_close(metrics["acc_macro"], 2 / 3)
    assert_close(metrics["acc_joint"], 1 / 3)
    assert_close(metrics["f1_macro_subject"], (2 / 3 + 1 / 2 + 4 / 5) / 3)
    assert_close(metrics["f1_macro_object"], (4 / 5 + 1 / 2 + 0 + 2 / 3) / 4)
    assert_close(metrics["f1_macro_interaction_type"], (4 / 5 + 2 / 3 + 1 / 2) / 7)
    unsplit = run_epoch(BatchLogitsModel(), [batch], torch.device("cpu"), DEFAULT_CONFIG, False)
    for name in metrics:
        if name != "seconds":
            assert_close(metrics[name], unsplit[name])


def test_ignored_targets_use_per_head_and_complete_joint_denominators():
    batch = make_logits_batch(
        {"subject": [0, -1, 1, 2, -1], "object": [0, 1, -1, 3, -1],
         "interaction_type": [0, 1, 1, -1, -1]},
        {"subject": [0, 1, 2, 2, 0], "object": [0, 1, 1, 3, 2],
         "interaction_type": [0, 2, 1, 0, 3]},
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["train"]["loss_weights"] = {"subject": 2.0, "object": 0.5, "interaction_type": 1.5}
    loader = [slice_batch(batch, index, index + 1) for index in range(5)]
    metrics = run_epoch(BatchLogitsModel(), loader, torch.device("cpu"), cfg, False)
    assert_close(metrics["acc_subject"], 2 / 3)
    assert_close(metrics["acc_object"], 1)
    assert_close(metrics["acc_interaction_type"], 2 / 3)
    assert_close(metrics["acc_macro"], 7 / 9)
    assert_close(metrics["acc_joint"], 1)
    assert_close(metrics["f1_macro_subject"], 5 / 9)
    assert_close(metrics["f1_macro_object"], 3 / 4)
    assert_close(metrics["f1_macro_interaction_type"], 5 / 21)
    expected_loss = 0.0
    for head in HEAD_CLASSES:
        valid = batch[head] != -1
        head_loss = F.cross_entropy(batch[f"logits_{head}"][valid], batch[head][valid]).item()
        assert_close(metrics[f"loss_{head}"], head_loss)
        expected_loss += cfg["train"]["loss_weights"][head] * head_loss
    assert_close(metrics["loss"], expected_loss)
    unsplit = run_epoch(BatchLogitsModel(), [batch], torch.device("cpu"), cfg, False)
    for name in metrics:
        if name != "seconds":
            assert_close(metrics[name], unsplit[name])


def test_joint_accuracy_is_zero_without_complete_targets():
    batch = make_logits_batch(
        {"subject": [0, -1, -1], "object": [-1, 3, -1], "interaction_type": [-1, -1, 0]},
        {"subject": [0, 1, 2], "object": [0, 3, 1], "interaction_type": [2, 3, 0]},
    )
    metrics = run_epoch(BatchLogitsModel(), [batch], torch.device("cpu"), DEFAULT_CONFIG, False)
    assert metrics["acc_joint"] == 0.0
    assert metrics["acc_macro"] == 1.0
    for head, class_count in HEAD_CLASSES.items():
        assert_close(metrics[f"f1_macro_{head}"], 1 / class_count)


def test_entirely_ignored_head_has_zero_accuracy_f1_and_loss():
    batch = make_logits_batch(
        {"subject": [0, 0], "object": [-1, -1], "interaction_type": [0, 0]},
        {"subject": [0, 0], "object": [3, 3], "interaction_type": [0, 0]},
    )
    metrics = run_epoch(BatchLogitsModel(), [batch], torch.device("cpu"), DEFAULT_CONFIG, False)
    assert metrics["loss_object"] == 0.0
    assert metrics["acc_object"] == 0.0
    assert metrics["f1_macro_object"] == 0.0
    assert metrics["acc_joint"] == 0.0
    assert_close(metrics["acc_macro"], 2 / 3)
    assert all(math.isfinite(value) for value in metrics.values())


def test_empty_loader_and_empty_batch_return_zero_metrics():
    empty_batch = make_logits_batch({head: [] for head in HEAD_CLASSES},
                                    {head: [] for head in HEAD_CLASSES})
    for loader in ([], [empty_batch]):
        metrics = run_epoch(BatchLogitsModel(), loader, torch.device("cpu"), DEFAULT_CONFIG, False)
        assert_zero_metrics(metrics)
        assert {f"f1_macro_{head}" for head in HEAD_CLASSES}.issubset(metrics)
        assert "acc_joint" in metrics


def test_all_ignored_batches_return_zero_metrics_without_optimizer_updates():
    batch = make_logits_batch({head: [-1, -1] for head in HEAD_CLASSES},
                              {head: [0, 0] for head in HEAD_CLASSES})
    model = TrainableLogitsModel()
    original = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    metrics = run_epoch(model, [batch], torch.device("cpu"), DEFAULT_CONFIG, False, optimizer)
    assert_zero_metrics(metrics)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])
    assert not optimizer.state


def test_all_ignored_or_empty_losses_are_zero_with_finite_gradients():
    for batch_size in (0, 2):
        logits = {head: torch.zeros(batch_size, class_count, requires_grad=True)
                  for head, class_count in HEAD_CLASSES.items()}
        batch = {head: torch.full((batch_size,), -1, dtype=torch.long) for head in HEAD_CLASSES}
        batch["obj_mask"] = torch.zeros(batch_size, 3, dtype=torch.bool)
        batch["label_valid"] = torch.ones(batch_size, dtype=torch.bool)
        loss, parts = compute_losses(logits, batch, DEFAULT_CONFIG["train"]["loss_weights"])
        assert loss.item() == 0.0
        assert all(value == 0.0 for value in parts.values())
        loss.backward()
        for values in logits.values():
            assert values.grad is not None
            assert torch.isfinite(values.grad).all()
            assert torch.count_nonzero(values.grad) == 0


def test_label_valid_false_excludes_otherwise_valid_targets():
    batch = make_logits_batch(
        {"subject": [0, 1], "object": [0, 1], "interaction_type": [1, 2]},
        {"subject": [0, 1], "object": [0, 1], "interaction_type": [1, 2]},
    )
    batch["label_valid"] = torch.tensor([True, False])
    metrics = run_epoch(BatchLogitsModel(), [batch], torch.device("cpu"), DEFAULT_CONFIG, False)
    assert metrics["acc_subject"] == 1.0
    assert metrics["acc_joint"] == 1.0
    model = TrainableLogitsModel()
    original = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    invalid_only = slice_batch(batch, 1, 2)
    metrics = run_epoch(model, [invalid_only], torch.device("cpu"), DEFAULT_CONFIG, False, optimizer)
    assert_zero_metrics(metrics)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])
    assert not optimizer.state


def test_missing_label_valid_is_rejected_by_loss_and_epoch():
    batch = make_logits_batch(
        {"subject": [0], "object": [0], "interaction_type": [1]},
        {"subject": [0], "object": [0], "interaction_type": [1]},
    )
    del batch["label_valid"]
    logits = {head: batch[f"logits_{head}"] for head in HEAD_CLASSES}
    with pytest.raises(ValueError, match="缺少 label_valid"):
        compute_losses(logits, batch, DEFAULT_CONFIG["train"]["loss_weights"])
    with pytest.raises(ValueError, match="缺少 label_valid"):
        run_epoch(BatchLogitsModel(), [batch], torch.device("cpu"), DEFAULT_CONFIG, False)


def test_missing_label_valid_legacy_opt_in_warns_and_runs(capsys):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["data"]["allow_legacy_missing_label_valid"] = True
    batch = make_logits_batch(
        {"subject": [0], "object": [0], "interaction_type": [1]},
        {"subject": [0], "object": [0], "interaction_type": [1]},
    )
    del batch["label_valid"]
    metrics = run_epoch(BatchLogitsModel(), [batch], torch.device("cpu"), cfg, False)
    assert metrics["acc_joint"] == 1.0
    assert metrics["warning_legacy_missing_label_valid"] == 1.0
    warning = capsys.readouterr().out
    assert "LEGACY/SYNTHETIC SMOKE ONLY" in warning
    assert "不是正式数据许可" in warning

    logits = {head: batch[f"logits_{head}"] for head in HEAD_CLASSES}
    loss, _ = compute_losses(
        logits, batch, cfg["train"]["loss_weights"],
        allow_legacy_missing_label_valid=True,
    )
    assert math.isfinite(loss.item())


def test_samples_dataset_rejects_missing_label_valid_unless_legacy_opt_in(tmp_path, capsys):
    (tmp_path / "train").mkdir()
    arrays = {
        "hand_seq": np.zeros((1, 2, 21, 3), np.float32),
        "obj_seq": np.zeros((1, 2, 13), np.float32),
        "obj_mask": np.ones(2, np.float32),
        "label_subject": np.int64(2), "label_object_slot": np.int64(-1),
        "label_interaction": np.int64(0),
    }
    np.savez(tmp_path / "train" / "invalid.npz", **arrays, label_valid=np.int64(0))
    np.savez(tmp_path / "train" / "legacy.npz", **arrays)

    dataset = SamplesDataset(tmp_path, "train", n_max=2)
    invalid = dataset[dataset.ids.index("invalid")]
    assert not invalid["label_valid"].item()
    with pytest.raises(ValueError, match="缺少 label_valid"):
        dataset[dataset.ids.index("legacy")]

    legacy_dataset = SamplesDataset(
        tmp_path, "train", n_max=2, allow_legacy_missing_label_valid=True,
    )
    items = {sid: legacy_dataset[index] for index, sid in enumerate(legacy_dataset.ids)}
    assert not items["invalid"]["label_valid"].item()
    assert items["legacy"]["label_valid"].item()
    batch = collate_fn([items["invalid"], items["legacy"]])
    assert batch["label_valid"].tolist() == [False, True]
    warning = capsys.readouterr().out
    assert "LEGACY/SYNTHETIC SMOKE ONLY" in warning
    assert "不是正式数据许可" in warning


def test_cpu_lstm_training_step_updates_parameters_with_finite_metrics():
    torch.manual_seed(7)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["model"].update(backbone="lstm", d_model=8, n_layer=1, dropout=0.0)
    cfg["model"]["decoder"]["n_max"] = 2
    model = build_model(cfg)
    batch = {
        "hand_seq": torch.randn(2, 3, 2, 21, 3),
        "obj_seq": torch.randn(2, 3, 2, 13),
        "obj_mask": torch.tensor([[True, False], [False, False]]),
        "subject": torch.tensor([0, 1]),
        "object": torch.tensor([0, 2]),
        "interaction_type": torch.tensor([1, 2]),
        "label_valid": torch.tensor([True, True]),
    }
    original = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = run_epoch(model, [batch], torch.device("cpu"), cfg, False, optimizer)
    assert model.training
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(previous, parameter)
               for previous, parameter in zip(original, model.parameters()))
    validation = run_epoch(model, [batch], torch.device("cpu"), cfg, False)
    assert not model.training
    assert all(math.isfinite(value) for value in validation.values())


if __name__ == "__main__":
    functions = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for function in functions:
        function()
        print(f"PASS {function.__name__}")
    print(f"\n{len(functions)} tests passed")
