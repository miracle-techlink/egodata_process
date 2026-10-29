import json
from pathlib import Path
import runpy

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = runpy.run_path(str(ROOT / "tests/check_contracts.py"))
FIXTURES = runpy.run_path(str(ROOT / "tests/make_fixtures.py"))


def sample_with_horizon():
    payload = FIXTURES["make_sample_npz"](np.random.default_rng(0))
    payload.update(
        target_frame=np.int64(61), horizon_frames=np.int64(30),
        horizon_seconds=np.float32(1), split_group=np.array("episode-a"),
        fps=np.float32(30),
    )
    return payload


def test_valid_prediction_metadata():
    report = CHECKER["Report"]()
    CHECKER["_check_prediction_metadata"](sample_with_horizon(), "sample", report)
    assert report.n_errors == 0


@pytest.mark.parametrize("field,value", [
    ("target_frame", np.int64(31)), ("horizon_frames", np.int64(0)),
    ("horizon_seconds", np.float32(2)), ("fps", np.float32(np.nan)),
    ("split_group", np.array("")),
])
def test_invalid_prediction_metadata(field, value):
    payload = sample_with_horizon()
    payload[field] = value
    report = CHECKER["Report"]()
    CHECKER["_check_prediction_metadata"](payload, "sample", report)
    assert report.n_errors > 0


def test_partial_prediction_metadata_rejected():
    payload = sample_with_horizon()
    del payload["horizon_frames"]
    report = CHECKER["Report"]()
    CHECKER["_check_prediction_metadata"](payload, "sample", report)
    assert report.n_errors > 0


def test_cross_camera_episode_leakage_rejected(tmp_path):
    first = sample_with_horizon()
    second = sample_with_horizon()
    second.update(clip_id=np.array("camera2_001"), source_video=np.array("camera2"))
    FIXTURES["write_samples_case"](
        tmp_path, [("first", first, "demo01")], [("second", second, "camera2")]
    )
    root = tmp_path / "samples"
    index_path = root / "dataset_index.json"
    index = json.loads(index_path.read_text())
    for entries in index.values():
        for entry in entries:
            entry["split_group"] = "episode-a"
    index_path.write_text(json.dumps(index))
    report = CHECKER["Report"]()
    CHECKER["check_samples"](root, report)
    assert any(key == "split_group" and "multiple splits" in message
               for level, path, key, message in report.entries)
