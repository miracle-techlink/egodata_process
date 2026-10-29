import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from src.mamba import train_v5


def make_batch(batch_size=2, window=16, entities=6, feature_dim=12, types=3):
    torch.manual_seed(5)
    features = torch.randn(batch_size, window, entities, feature_dim)
    valid = torch.ones(batch_size, window, entities, dtype=torch.bool)
    valid[:, :4, entities - 1] = False
    features[~valid] = 0.0
    kind = torch.tensor([[0, 1, 2, 3, 3, 3]] * batch_size)
    logits_shape = (batch_size, entities, entities, types)
    values = torch.zeros(logits_shape)
    label_valid = torch.zeros(logits_shape, dtype=torch.bool)
    cells = [(0, 0, 3, 0, 1), (0, 1, 4, 1, 0), (1, 2, 5, 2, 1)]
    if batch_size >= 4:
        cells += [(2, 0, 3, 0, 1), (2, 1, 4, 1, 0), (3, 2, 5, 2, 1)]
    for b, s, t, k, v in cells:
        label_valid[b, s, t, k] = True
        values[b, s, t, k] = v
    return {"entity_features": features, "entity_valid": valid, "entity_kind": kind,
            "graph_values": values, "graph_label_valid": label_valid}


def test_lstm_model_forward_backward_and_unknown_zero_gradient():
    model = train_v5.GraphV5LSTM(12, event_type_count=3, d_model=24, n_layer=2, n_heads=4)
    data = make_batch()
    out = model(data)
    assert out["graph_logits"].shape == (2, 6, 6, 3)
    from src.mamba.graph_v5 import masked_graph_bce
    stats = masked_graph_bce(out["graph_logits"], data["graph_values"],
                             data["graph_label_valid"], out["graph_prediction_mask"])
    stats["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_state_dict_keys_match_graph_v5_mamba():
    try:
        pytest.importorskip("mamba_ssm")
    except RuntimeError as exc:
        pytest.skip(f"mamba_ssm runtime unavailable: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("state-dict parity check runs where mamba_ssm is importable")
    mamba = train_v5.build_v5_model({"model": {**train_v5.DEFAULT_CONFIG["model"], "backbone": "mamba"}})
    lstm = train_v5.build_v5_model({"model": {**train_v5.DEFAULT_CONFIG["model"], "backbone": "lstm"}})
    transformer = train_v5.build_v5_model({"model": {**train_v5.DEFAULT_CONFIG["model"], "backbone": "transformer"}})
    assert {k for k in mamba.state_dict() if not k.startswith("backbone.")} == {k for k in lstm.state_dict() if not k.startswith("backbone.")}
    assert {k for k in mamba.state_dict() if not k.startswith("backbone.")} == {k for k in transformer.state_dict() if not k.startswith("backbone.")}


def test_transformer_backbone_is_causal_and_shapes_match():
    torch.manual_seed(7)
    backbone = train_v5.TransformerBackbone(d_model=24, n_layer=2, n_heads=4)
    x = torch.randn(3, 16, 24)
    out = backbone(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    altered = x.clone()
    altered[:, 8:] = 1e3  # scramble future frames
    with torch.no_grad():
        prefix_original = backbone(x)[:, :8]
        prefix_altered = backbone(altered)[:, :8]
    torch.testing.assert_close(prefix_original, prefix_altered, rtol=1e-5, atol=1e-6)


def test_transformer_model_forward_backward():
    model = train_v5.GraphV5Transformer(12, event_type_count=3, d_model=24, n_layer=2, n_heads=4)
    data = make_batch()
    out = model(data)
    assert out["graph_logits"].shape == (2, 6, 6, 3)
    from src.mamba.graph_v5 import masked_graph_bce
    stats = masked_graph_bce(out["graph_logits"], data["graph_values"],
                             data["graph_label_valid"], out["graph_prediction_mask"])
    stats["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_build_v5_model_rejects_mamba_without_cuda(tmp_path):
    cfg = {"model": {**train_v5.DEFAULT_CONFIG["model"], "backbone": "mamba"}, "device": "cpu",
           "data": {"root": str(tmp_path)}, "train": train_v5.DEFAULT_CONFIG["train"]}
    args = train_v5.parse_args(["--config", "x", "--device", "cpu"])
    with mock.patch("src.mamba.train_v5.load_config_v5", return_value=cfg):
        # mamba on CPU must abort before any optimizer step
        with pytest.raises(SystemExit, match="CUDA"):
            train_v5.main(args)


def test_checkpoint_roundtrip_and_data_binding(tmp_path):
    model = train_v5.GraphV5LSTM(12, event_type_count=3, d_model=24, n_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    binding = {"training_index_sha256": "a", "training_release_sha256": "b"}
    path = tmp_path / "last.pt"
    train_v5.save_checkpoint_v5(path, model, optimizer, 3, {"k": 1}, 0.5, binding)
    loaded = train_v5.load_checkpoint_v5(path, model, optimizer, binding)
    assert loaded["epoch"] == 3 and loaded["best_metric"] == 0.5
    with pytest.raises(SystemExit, match="数据绑定"):
        train_v5.load_checkpoint_v5(path, model, optimizer, {"training_index_sha256": "changed"})


def test_config_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  backbone: transformer_xl\n")
    with pytest.raises(SystemExit, match="mamba\\|lstm\\|transformer"):
        train_v5.load_config_v5(str(bad))
    bad_metric = tmp_path / "metric.yaml"
    bad_metric.write_text("train:\n  selection_metric: val_auc\n")
    with pytest.raises(SystemExit, match="selection_metric"):
        train_v5.load_config_v5(str(bad_metric))
    good = tmp_path / "good.yaml"
    good.write_text("train:\n  epochs: 3\n")
    cfg = train_v5.load_config_v5(str(good))
    assert cfg["train"]["epochs"] == 3 and cfg["model"]["d_model"] == 128


def test_cross_check_seen_types():
    release = {"positive_types_absent_from_train": ["operation:hold", "relation:stabilizes"]}
    seen = {"operation:pick": 5, "relation:contact": 9}
    check = train_v5.cross_check_seen_types(seen, release)
    assert check["seen_types"] == ["operation:pick", "relation:contact"]
    with pytest.raises(SystemExit):
        train_v5.cross_check_seen_types({"operation:hold": 1}, release)


def test_run_epoch_reports_loss_and_metrics():
    model = train_v5.GraphV5LSTM(12, event_type_count=3, d_model=24, n_heads=4)
    data = make_batch(batch_size=4)
    from torch.utils.data import DataLoader, Dataset

    class Tiny(Dataset):
        def __init__(self):
            self.event_types = [{"layer": "operation", "code": "a"},
                                {"layer": "operation", "code": "b"},
                                {"layer": "relation", "code": "c"}]
        def __len__(self):
            return 4
        def __getitem__(self, i):
            item = {}
            for k, v in data.items():
                value = v[i]
                item[k] = value
            return item

    loader = DataLoader(Tiny(), batch_size=2)
    cfg = train_v5.deep_merge(train_v5.DEFAULT_CONFIG, {"data": {"root": "x"}})
    metrics = train_v5.run_epoch_v5(model, loader, torch.device("cpu"), cfg, False)
    assert metrics["loss"] > 0 and metrics["windows"] == 4
    assert metrics["counts"]["known"] == 6  # 3 known cells per sample
    assert metrics["windows_per_second"] > 0
