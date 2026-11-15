import json
from pathlib import Path

import numpy as np
import pytest

from src.pose6d import extract_pose6d as ep

W, H, T, FPS = 320, 240, 30, 30
K4 = [500.0, 500.0, 160.0, 120.0]


def box_a(f):  # area 50*40 = 2000, moves +1 px/frame
    return [20 + f, 20, 70 + f, 60]


BOX_B = [200, 150, 260, 200]  # area 60*50 = 3000, static; missing frames 5..9


def make_clip(tmp_path: Path) -> Path:
    import cv2
    frames = []
    for f in range(T):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        x1, y1, x2, y2 = box_a(f)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), -1)
        cv2.rectangle(img, (BOX_B[0], BOX_B[1]), (BOX_B[2], BOX_B[3]), (0, 0, 255), -1)
        frames.append(img)
    for codec, ext in [("mp4v", ".mp4"), ("MJPG", ".avi")]:
        p = tmp_path / f"demo01_007{ext}"
        wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*codec), FPS, (W, H))
        if not wr.isOpened():
            wr.release()
            continue
        for fr in frames:
            wr.write(fr)
        wr.release()
        cap = cv2.VideoCapture(str(p))
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        cap.release()
        if n == T:
            return p
    pytest.skip("no working video codec in this environment")


def make_bbox_file(tmp_path: Path) -> Path:
    data = {}
    for f in range(T):
        dets = [{"bbox": box_a(f), "label": "obj_a"}]
        if not 5 <= f <= 9:
            dets.append({"bbox": BOX_B, "label": "obj_b"})
        data[str(f)] = dets
    p = tmp_path / "boxes.json"
    p.write_text(json.dumps(data))
    return p


def run_cli(clip, boxes, out_dir, extra=()):
    return ep.main([
        "--clip", str(clip), "--out-dir", str(out_dir),
        "--bbox-from-file", str(boxes),
        "--intrinsics", *[str(v) for v in K4],
        *extra,
    ])


def check_contract_npz(npz_path, n, t):
    z = np.load(npz_path)
    assert set(z.files) == {"object_ids", "poses", "bboxes"}
    ids, poses, bboxes = z["object_ids"], z["poses"], z["bboxes"]
    assert ids.shape == (n,) and ids.dtype.kind == "U"
    assert poses.shape == (n, t, 4, 4) and poses.dtype == np.float32
    assert bboxes.shape == (n, t, 4) and bboxes.dtype == np.int32
    return ids, poses, bboxes


def test_end_to_end_mock(tmp_path, capsys):
    clip = make_clip(tmp_path)
    boxes = make_bbox_file(tmp_path)
    out_dir = tmp_path / "pose6d"
    summary = run_cli(clip, boxes, out_dir,
                      extra=["--pose-estimator", "mock", "--mock-z", "2.5"])
    capsys.readouterr()

    assert summary["clip_id"] == "demo01_007"
    assert Path(summary["npz"]).exists() and Path(summary["meta"]).exists()
    assert summary["num_objects"] == 2 and summary["num_frames"] == T

    ids, poses, bboxes = check_contract_npz(summary["npz"], 2, T)
    # larger first-frame area (B: 3000) sorts before A (2000)
    assert ids.tolist() == ["obj_000", "obj_001"]
    obj_b, obj_a = 0, 1

    # undetected frames 5..9 for B: bbox -1, pose NaN
    assert (bboxes[obj_b, 5:10] == -1).all()
    assert np.isnan(poses[obj_b, 5:10]).all()
    # B reappears with the same track -> same object_id row
    assert not np.isnan(poses[obj_b, 10]).any()
    assert (bboxes[obj_b, 10] == np.array(BOX_B)).all()

    # mock pose: identity rotation, back-projected bbox center, z = mock_z
    fx, fy, cx, cy = K4
    for f in (0, 7, 29):
        x1, y1, x2, y2 = box_a(f)
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        expected_t = np.array([(bx - cx) * 2.5 / fx, (by - cy) * 2.5 / fy, 2.5])
        assert np.allclose(poses[obj_a, f, :3, :3], np.eye(3), atol=1e-6)
        assert np.allclose(poses[obj_a, f, :3, 3], expected_t, atol=1e-5)
    assert (bboxes[obj_a, 0] == np.array(box_a(0))).all()

    meta = json.loads(Path(summary["meta"]).read_text())
    assert meta["depth_source"] == "none"
    assert meta["detector"] == "file" and meta["pose_estimator"] == "mock"
    assert meta["fps"] == pytest.approx(FPS)
    assert meta["num_frames"] == T and meta["num_objects"] == 2
    assert meta["object_ids"] == ["obj_000", "obj_001"]
    assert meta["intrinsics"]["source"] == "cli"
    assert meta["intrinsics"]["fx"] == 500.0
    assert meta["bbox_undetected_value"] == -1
    assert meta["quality_flags"] == ["mock_pose"]
    assert any("not a real 6D estimate" in warning for warning in meta["warnings"])
    assert "generated_at" in meta and "environment" in meta


def test_pose_none_chain_mode(tmp_path, capsys):
    clip = make_clip(tmp_path)
    boxes = make_bbox_file(tmp_path)
    out_dir = tmp_path / "pose6d_none"
    summary = run_cli(clip, boxes, out_dir)  # default --pose-estimator none
    capsys.readouterr()
    ids, poses, bboxes = check_contract_npz(summary["npz"], 2, T)
    assert np.isnan(poses).all()
    assert not (bboxes[0, 0] == -1).all()  # bboxes still produced
    meta = json.loads(Path(summary["meta"]).read_text())
    assert meta["pose_estimator"] == "none"


def test_mock_uses_file_depth(tmp_path, capsys):
    clip = make_clip(tmp_path)
    boxes = make_bbox_file(tmp_path)
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    for f in range(T):
        np.save(depth_dir / f"{f:06d}.npy", np.full((H, W), 3.0, dtype=np.float32))
    out_dir = tmp_path / "pose6d_depth"
    summary = run_cli(clip, boxes, out_dir, extra=[
        "--pose-estimator", "mock", "--depth-source", "file",
        "--depth-dir", str(depth_dir), "--mock-z", "2.5"])
    capsys.readouterr()
    ids, poses, _ = check_contract_npz(summary["npz"], 2, T)
    assert np.allclose(poses[1, 0, 2, 3], 3.0, atol=1e-5)  # file depth wins over mock-z
    meta = json.loads(Path(summary["meta"]).read_text())
    assert meta["depth_source"] == "file"


def test_heuristic_intrinsics_flagged(tmp_path, capsys):
    clip = make_clip(tmp_path)
    boxes = make_bbox_file(tmp_path)
    out_dir = tmp_path / "pose6d_heur"
    summary = ep.main(["--clip", str(clip), "--out-dir", str(out_dir),
                       "--bbox-from-file", str(boxes)])
    capsys.readouterr()
    meta = json.loads(Path(summary["meta"]).read_text())
    assert meta["intrinsics"]["source"] == "heuristic"
    assert "heuristic_intrinsics" in meta["quality_flags"]
    assert meta["intrinsics"]["fx"] == float(max(W, H))


def test_write_npz_shape_validation(tmp_path):
    with pytest.raises(ValueError):
        ep.write_pose6d_npz(str(tmp_path / "x.npz"), ["obj_000"],
                            np.zeros((2, 3, 4, 4)), np.zeros((1, 3, 4)))
    with pytest.raises(ValueError):
        ep.write_pose6d_npz(str(tmp_path / "x.npz"), ["obj_000"],
                            np.zeros((1, 3, 4, 4)), np.zeros((1, 4, 4)))
    # N = 0 is valid
    p = tmp_path / "empty.npz"
    ep.write_pose6d_npz(str(p), [], np.zeros((0, 5, 4, 4)), np.zeros((0, 5, 4)))
    z = np.load(p)
    assert z["object_ids"].shape == (0,)
    assert z["poses"].shape == (0, 5, 4, 4)


def test_import_without_gpu_deps():
    # module already imported at top of this file without torch/FoundationPose
    assert ep.NullDepthProvider().depth(None, 0) is None
    assert ep.NullPoseBackend().requires_depth is False
    with pytest.raises(RuntimeError, match="foundationpose-repo"):
        ep.FoundationPoseBackend(repo=None)
    with pytest.raises(RuntimeError, match="depth-anything-repo"):
        ep.DepthAnythingProvider(repo=None, ckpt="x.pth")


@pytest.mark.parametrize("failure", ["create", "estimate", "depth"])
@pytest.mark.parametrize("strict", [True, False])
def test_runtime_failure_obeys_strict_mode(tmp_path, monkeypatch, failure, strict):
    class BrokenTracker:
        def estimate(self, frame_bgr, depth, intrinsics, bbox, is_new):
            raise RuntimeError("pose estimation failed")

    class BrokenBackend:
        name = "foundationpose"
        requires_depth = failure == "depth"

        def create(self, track_id):
            if failure == "create":
                raise RuntimeError("tracker creation failed")
            return BrokenTracker()

    monkeypatch.setattr(ep, "build_pose_backend", lambda *args: BrokenBackend())
    clip = make_clip(tmp_path)
    boxes = make_bbox_file(tmp_path)
    output = tmp_path / "failed_pose"
    arguments = ["--pose-estimator", "foundationpose"]
    if strict:
        arguments.append("--strict")
        with pytest.raises(RuntimeError, match="creation failed|estimation failed|required depth is missing"):
            run_cli(clip, boxes, output, arguments)
        assert not list(output.glob("*.npz"))
        assert not list(output.glob("*.meta.json"))
    else:
        result = run_cli(clip, boxes, output, arguments)
        with np.load(result["npz"], allow_pickle=False) as archive:
            assert np.isnan(archive["poses"]).all()
        metadata = json.loads(Path(result["meta"]).read_text())
        expected_flag = {"create": "pose_backend_create_failed", "estimate": "pose_failures", "depth": "no_depth_for_estimator"}[failure]
        assert expected_flag in metadata["quality_flags"]


def test_clip_iterator_releases_capture_on_early_close():
    class FakeCapture:
        released = False

        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    reader = ep.ClipReader.__new__(ep.ClipReader)
    reader.cap = FakeCapture()
    frames = reader.frames()
    next(frames)
    frames.close()
    assert reader.cap.released
