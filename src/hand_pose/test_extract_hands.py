"""Pure-numpy tests for extract_hands (runnable with plain python or pytest).

    python src/hand_pose/test_extract_hands.py
    pytest src/hand_pose/test_extract_hands.py

MediaPipe/cv2 are not needed here; detect_video is monkeypatched in the
end-to-end test. Real model inference is verified on the server (ash_hands).
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_hands as eh  # noqa: E402


def _lm(seed):
    rng = np.random.default_rng(seed)
    return rng.random((21, 3), dtype=np.float32)


def test_build_basic_and_nan_rule():
    lm_l, lm_r = _lm(0), _lm(1)
    frames = [
        [("Left", 0.9, lm_l), ("Right", 0.8, lm_r)],
        None,
        [("Right", 0.7, lm_r)],
    ]
    left, right, conf, dropped = eh.build_hand_arrays(frames)
    assert left.shape == (3, 21, 3) and left.dtype == np.float32
    assert right.shape == (3, 21, 3) and conf.shape == (3, 2)
    assert dropped == 0
    np.testing.assert_allclose(left[0], lm_l, atol=1e-6)
    np.testing.assert_allclose(right[0], lm_r, atol=1e-6)
    assert np.isnan(left[1]).all() and np.isnan(right[1]).all()
    assert np.isnan(left[2]).all() and not np.isnan(right[2]).any()
    assert conf[0, 0] == np.float32(0.9) and conf[0, 1] == np.float32(0.8)
    assert np.isnan(conf[1]).all()
    assert np.isnan(conf[2, 0]) and conf[2, 1] == np.float32(0.7)


def test_swap_handedness():
    lm = _lm(2)
    left, right, conf, _ = eh.build_hand_arrays(
        [[("Left", 0.9, lm)]], swap_handedness=True
    )
    assert np.isnan(left[0]).all()
    np.testing.assert_allclose(right[0], lm, atol=1e-6)
    assert np.isnan(conf[0, 0]) and conf[0, 1] == np.float32(0.9)


def test_low_score_becomes_nan():
    lm = _lm(3)
    left, right, conf, dropped = eh.build_hand_arrays(
        [[("Left", 0.3, lm)]], min_handedness_score=0.5
    )
    assert np.isnan(left[0]).all() and np.isnan(conf[0]).all()
    assert dropped == 0  # low score is filtered, not "dropped" garbage


def test_duplicate_label_keeps_higher_score():
    lm_lo, lm_hi = _lm(4), _lm(5)
    left, _, conf, dropped = eh.build_hand_arrays(
        [[("Left", 0.6, lm_lo), ("Left", 0.9, lm_hi)]]
    )
    np.testing.assert_allclose(left[0], lm_hi, atol=1e-6)
    assert conf[0, 0] == np.float32(0.9) and dropped == 1


def test_bad_label_and_bad_shape_dropped():
    _, _, _, dropped = eh.build_hand_arrays(
        [[("Middle", 0.9, _lm(6)), ("Left", 0.9, np.zeros((20, 3), np.float32))]]
    )
    assert dropped == 2


def test_detection_stats():
    frames = [[("Left", 0.9, _lm(7))], None, [("Left", 0.9, _lm(8)), ("Right", 0.9, _lm(9))]]
    left, right, _, _ = eh.build_hand_arrays(frames)
    s = eh.detection_stats(left, right)
    assert s["frames"] == 3
    assert s["left_detected_frames"] == 2 and s["right_detected_frames"] == 1
    assert abs(s["left_detection_rate"] - 2 / 3) < 1e-6
    assert abs(s["both_detected_rate"] - 1 / 3) < 1e-6


def test_save_validate_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "demo01_007.npz"
        frames = [[("Left", 0.9, _lm(10))], None]
        left, right, conf, _ = eh.build_hand_arrays(frames)
        eh.save_npz(p, left, right, conf)
        assert eh.validate_npz(p, expect_frames=2) == []
        with np.load(p) as z:
            assert set(z.files) == {"left_hand", "right_hand", "conf"}
            assert z["left_hand"].dtype == np.float32
            np.testing.assert_allclose(z["conf"][:, 0], [0.9, np.nan])


def test_validate_catches_violations():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        left = np.full((2, 21, 3), np.nan, np.float32)
        right = np.full((2, 21, 3), np.nan, np.float32)
        conf = np.full((2, 2), np.nan, np.float32)

        p = d / "partial_nan.npz"  # half-NaN frame is illegal
        bad = left.copy()
        bad[0, :10] = 0.0
        eh.save_npz(p, bad, right, conf)
        assert any("partially NaN" in m for m in eh.validate_npz(p))

        p = d / "wrong_dtype.npz"
        np.savez_compressed(p, left_hand=left.astype(np.float64),
                            right_hand=right, conf=conf)
        assert any("dtype" in m for m in eh.validate_npz(p))

        p = d / "missing_key.npz"
        np.savez_compressed(p, left_hand=left, right_hand=right)
        assert any("keys" in m for m in eh.validate_npz(p))

        p = d / "conf_mismatch.npz"  # conf says detected but landmarks are NaN
        bad_conf = conf.copy()
        bad_conf[0, 0] = 0.9
        eh.save_npz(p, left, right, bad_conf)
        assert any("NaN pattern" in m for m in eh.validate_npz(p))

        p = d / "t_mismatch.npz"
        assert any("expect_frames" in m for m in eh.validate_npz(d / "partial_nan.npz", expect_frames=5))


def test_write_json_sanitizes():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        eh.write_json(p, {"fps": float("nan"), "n": np.float32(1.5), "ok": 1})
        meta = json.loads(p.read_text())
        assert meta["fps"] is None and meta["n"] == 1.5 and meta["ok"] == 1


def test_collect_clips():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "b_002.mp4").write_bytes(b"")
        (d / "a_001.mp4").write_bytes(b"")
        (d / "notes.txt").write_text("x")
        clips = eh.collect_clips(d)
        assert [c.name for c in clips] == ["a_001.mp4", "b_002.mp4"]
        assert eh.collect_clips(d / "a_001.mp4") == [d / "a_001.mp4"]


def test_process_clip_end_to_end_monkeypatched():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        clip = d / "demo01_007.mp4"
        clip.write_bytes(b"not a real video")  # detect_video is patched out
        lm_l, lm_r = _lm(11), _lm(12)
        fake_frames = [
            [("Left", 0.95, lm_l), ("Right", 0.85, lm_r)],
            None,
            [("Right", 0.75, lm_r)],
        ]
        orig = eh.detect_video
        eh.detect_video = lambda path, params: (fake_frames, 30.0, 640, 480)
        try:
            rec = eh.process_clip(clip, d, dict(eh.DEFAULT_PARAMS))
        finally:
            eh.detect_video = orig
        assert rec["status"] == "ok" and rec["frames"] == 3
        assert rec["left_detected_frames"] == 1 and rec["right_detected_frames"] == 2

        npz_path = d / "demo01_007.npz"
        assert eh.validate_npz(npz_path, expect_frames=3) == []
        with np.load(npz_path) as z:
            np.testing.assert_allclose(z["left_hand"][0], lm_l, atol=1e-6)
            assert np.isnan(z["left_hand"][2]).all()
            np.testing.assert_allclose(z["right_hand"][2], lm_r, atol=1e-6)

        meta = json.loads((d / "demo01_007.meta.json").read_text())
        assert meta["clip_id"] == "demo01_007" and meta["fps"] == 30.0
        assert meta["frames"] == 3 and meta["width"] == 640
        assert meta["handedness"]["swap_applied"] is False
        assert meta["stats"]["right_detection_rate"] == 2 / 3
        assert "generated_at" in meta and "environment" in meta

        rec2 = eh.process_clip(clip, d, dict(eh.DEFAULT_PARAMS))  # resume: skip
        assert rec2["status"] == "skipped"


def test_worker_catches_errors():
    with tempfile.TemporaryDirectory() as d:
        orig = eh.process_clip
        def boom(*a, **k):
            raise RuntimeError("boom")
        eh.process_clip = boom
        try:
            rec = eh._worker(("/x/bad_001.mp4", d, dict(eh.DEFAULT_PARAMS), False))
        finally:
            eh.process_clip = orig
        assert rec["status"] == "error" and "boom" in rec["error"]
        assert rec["clip_id"] == "bad_001"


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
