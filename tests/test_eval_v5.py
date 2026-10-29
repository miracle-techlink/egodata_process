import torch

from src.mamba.eval_v5 import (
    GraphMetricAccumulator,
    all_negative_baseline,
    all_positive_baseline,
    prf,
    seen_unseen_split,
)


EVENTS = [
    {"layer": "operation", "code": "pick", "name_zh": "拿取"},
    {"layer": "relation", "code": "contact", "name_zh": "接触"},
]


def test_prf_and_baselines():
    assert prf(2, 1, 1) == {"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3, "tp": 2, "fp": 1, "fn": 1}
    assert all_positive_baseline({"known": 4, "positive": 1})["recall"] == 1.0
    assert all_negative_baseline({"known": 4})["f1"] == 0.0


def test_accumulator_ignores_unknown_and_reports_seen_unseen():
    # Four observed pair/type cells: one TP, one FN, one FP, one TN.
    logits = torch.full((1, 2, 2, 2), -10.0)
    values = torch.zeros_like(logits)
    valid = torch.zeros_like(logits, dtype=torch.bool)
    logits[0, 0, 1, 0] = 10.0  # TP
    values[0, 0, 1, 0] = 1.0
    valid[0, 0, 1, 0] = True
    logits[0, 1, 0, 0] = -10.0  # FN
    values[0, 1, 0, 0] = 1.0
    valid[0, 1, 0, 0] = True
    logits[0, 0, 0, 1] = 10.0  # FP
    valid[0, 0, 0, 1] = True
    valid[0, 1, 1, 1] = True  # TN
    logits[0, 1, 1, 1] = -10.0
    logits[0, 0, 0, 0] = 1000.0  # unknown: must not count as FP

    acc = GraphMetricAccumulator(EVENTS)
    acc.update(logits, values, valid, torch.ones(1, 2, 2, dtype=torch.bool))
    report = acc.summarize({"operation:pick"})
    assert report["counts"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "known": 4, "positive": 2}
    assert report["micro"]["f1"] == 0.5
    assert report["window_positive_hit_rate"] == 1.0
    assert report["per_type"]["operation:pick"]["seen"]
    assert report["per_type"]["relation:contact"]["seen"] is False
    buckets = seen_unseen_split(report["per_type"], {"operation:pick"})
    assert "operation:pick" in buckets["seen"] and "relation:contact" in buckets["unseen"]


def test_prediction_mask_excludes_unobserved_entities():
    logits = torch.zeros(1, 3, 3, 1)
    values = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    prediction_mask = torch.ones(1, 3, 3, dtype=torch.bool)
    prediction_mask[:, 2] = False
    prediction_mask[:, :, 2] = False
    acc = GraphMetricAccumulator([{"layer": "relation", "code": "x"}])
    acc.update(logits, values, valid, prediction_mask)
    assert acc.summarize()["counts"]["known"] == 4
