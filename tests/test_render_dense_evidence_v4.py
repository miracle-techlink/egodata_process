import hashlib
import json

import cv2
import numpy as np
import pytest

from src.annotation.render_dense_evidence_v4 import frame_batches, render_dense, validate_request


def test_dense_batches_cover_every_frame_exactly_once():
    batches = frame_batches(17, 44, 12)
    assert [f for page in batches for f in page] == list(range(17, 44))
    assert [len(p) for p in batches] == [12, 12, 3]


@pytest.mark.parametrize("start,end,crop", [(0, 513, [0, 0, 32, 24]),
    (1, 1, [0, 0, 32, 24]), (0, 10, [-1, 0, 32, 24]), (0, 10, [0, 0, 33, 24])])
def test_invalid_native_requests_are_rejected(start, end, crop):
    with pytest.raises(ValueError):
        validate_request(start, end, 1000, crop, [32, 24])


def test_real_video_decode_retains_frame_pixels_timestamps_and_unreviewed_status(tmp_path):
    source = tmp_path / "source"
    sequence = source / "sequences" / "pilot"
    sequence.mkdir(parents=True)
    video = source / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 30, (32, 24))
    assert writer.isOpened()
    frames = []
    for i in range(8):
        frame = np.zeros((24, 32, 3), np.uint8)
        frame[:, :, 0] = i * 20
        frame[:, :, 1] = np.arange(32)[None]
        frames.append(frame)
        writer.write(frame)
    writer.release()
    metadata = {"sequence_id": "pilot", "frame_count": 8, "image_size": [32, 24],
                "video_path": str(video), "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest()}
    (sequence / "metadata.json").write_text(json.dumps(metadata))
    timestamps = np.array([0, .03, .07, .10, .15, .18, .22, .25])
    np.savez(sequence / "hands.npz", timestamps=timestamps)
    result = render_dense(source, tmp_path / "output", "pilot", 2, 7, [4, 3, 28, 20], tile_width=128, frames_per_page=3)
    manifest = json.loads(open(result["manifest_path"]).read())
    assert manifest["rendered_frame_count"] == 5
    assert [r["frame_index"] for r in manifest["frames"]] == list(range(2, 7))
    assert not manifest["visual_reviewed"] and not manifest["annotation_created"]
    for record in manifest["frames"]:
        i = record["frame_index"]
        crop = cv2.imread(str(tmp_path / "output" / record["crop_path"]))
        assert np.array_equal(crop, frames[i][3:20, 4:28])
        assert record["timestamp_s"] == timestamps[i]
        expected_hash = hashlib.sha256(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
        assert record["decoded_native_rgb_sha256"] == expected_hash
    with pytest.raises(FileExistsError):
        render_dense(source, tmp_path / "output", "pilot", 2, 7, [4, 3, 28, 20], tile_width=128, frames_per_page=3)
