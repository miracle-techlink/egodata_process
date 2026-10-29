"""Small synthetic CPU fixtures test runner code, NOT FoundationPose success/accuracy."""

import csv
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

from src.pose6d import run_rgbd_sequence as runner


def synthetic_pose(index=0):
    angle = index * 0.1
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    pose[:3, 3] = [0.01 * index, 0, 1 + 0.02 * index]
    return pose


class FakeEstimator:
    """Injection-only deterministic code fixture; no learned model is involved."""

    def __init__(self, fail_at=None):
        self.calls = []
        self.pose_last = None
        self.fail_at = fail_at

    def predict(self, method, inputs):
        index = len(self.calls)
        self.calls.append((method, inputs))
        if index == self.fail_at:
            raise RuntimeError("simulated CUDA failure")
        self.pose_last = synthetic_pose(index)
        return self.pose_last

    def register(self, **inputs):
        assert set(inputs) == {"rgb", "depth", "K", "ob_mask", "iteration"}
        return self.predict("register", inputs)

    def track_one(self, **inputs):
        assert set(inputs) == {"rgb", "depth", "K", "iteration"}
        return self.predict("track_one", inputs)


@pytest.fixture
def scene(tmp_path):
    directory = tmp_path / "benchmark_demo"
    for kind in ("rgb", "depth", "masks"):
        (directory / kind).mkdir(parents=True)
    mask = np.zeros((32, 40), dtype=np.uint8)
    mask[10:23, 12:17] = 255
    mask[18:23, 12:28] = 255
    for index in range(3):
        stem = f"{index:04d}"
        color = np.full((32, 40, 3), [11 + index, 42, 91], dtype=np.uint8)
        assert cv2.imwrite(str(directory / "rgb" / f"{stem}.png"), color)
        assert cv2.imwrite(str(directory / "depth" / f"{stem}.png"), np.full((32, 40), 1000, np.uint16))
        assert cv2.imwrite(str(directory / "masks" / f"{stem}.png"), mask)
    np.savetxt(directory / "cam_K.txt", [[50, 0, 20], [0, 50, 16], [0, 0, 1]])
    mesh_file = tmp_path / "mesh.ply"
    trimesh.creation.box(extents=[0.16, 0.10, 0.08]).export(mesh_file)
    args = runner.build_parser().parse_args([
        "--foundationpose-repo", str(tmp_path / "no_real_model"), "--scene-dir", str(directory),
        "--mesh-file", str(mesh_file), "--out-dir", str(tmp_path / "output"), "--limit-frames", "3"])
    return args, directory, mask


def read_report(args):
    return json.loads((Path(args.out_dir) / "run_report.json").read_text())


def add_references(args, directory):
    reference_dir = directory / "annotated_poses"
    reference_dir.mkdir()
    for index in range(3):
        np.savetxt(reference_dir / f"{index:04d}.txt", synthetic_pose(index))
    args.reference_dir = str(reference_dir)
    args.reference_frame_confirmed = True
    args.translation_threshold_m = 0.001
    args.rotation_threshold_deg = 0.1
    return reference_dir


def test_defaults_and_lazy_import_without_cuda():
    code = "from src.pose6d import run_rgbd_sequence; import sys; assert not ({'torch','estimater','nvdiffrast'} & sys.modules.keys())"
    subprocess.run([sys.executable, "-B", "-c", code], check=True)
    args = runner.build_parser().parse_args([
        "--foundationpose-repo", "/repo", "--scene-dir", "/scene", "--mesh-file", "/mesh", "--out-dir", "/new"])
    assert (args.depth_scale, args.est_refine_iter, args.track_refine_iter) == (0.001, 5, 2)
    assert args.limit_frames is None
    assert args.reference_dir is None
    subprocess.run([sys.executable, "-B", str(Path(runner.__file__)), "--help"], check=True, capture_output=True)


def test_three_frame_exports_are_only_cpu_code_evidence(scene):
    args, directory, mask = scene
    estimator = FakeEstimator()
    report = runner.run_sequence(args, estimator=estimator)
    output = Path(args.out_dir)
    assert report["status"] == "test_only_success"
    assert report["real_model"] is False
    assert report["dataset_role"] == "benchmark_demo" and report["is_egodemo"] is False
    assert report["inference_success_rate"] == 1
    assert report["accuracy"]["evaluated"] is False
    assert all(report["accuracy"][name] is None for name in ("translation_accuracy", "rotation_accuracy", "joint_accuracy"))
    assert [method for method, _ in estimator.calls] == ["register", "track_one", "track_one"]
    assert [inputs["iteration"] for _, inputs in estimator.calls] == [5, 2, 2]
    inputs = estimator.calls[0][1]
    np.testing.assert_array_equal(inputs["ob_mask"], mask > 0)
    assert inputs["ob_mask"].dtype == np.bool_
    assert not inputs["ob_mask"][12, 25]
    np.testing.assert_array_equal(inputs["rgb"][0, 0], [91, 42, 11])
    assert inputs["depth"].dtype == np.float32
    np.testing.assert_allclose(inputs["depth"], 1)
    with np.load(output / "poses.npz", allow_pickle=False) as saved:
        assert saved["poses"].shape == (1, 3, 4, 4)
        assert saved["poses"].dtype == np.float32
        assert saved["bboxes"].shape == (1, 3, 4) and saved["bboxes"].dtype == np.int32
        assert saved["object_ids"].tolist() == ["obj_000"]
        np.testing.assert_array_equal(saved["bboxes"][0], [[12, 10, 27, 22]] * 3)
        np.testing.assert_allclose(saved["poses"][0], [synthetic_pose(index) for index in range(3)])
    metadata = json.loads((output / "poses.meta.json").read_text())
    assert metadata["pose_estimator"] == "injected_cpu_test"
    assert metadata["depth_scale"] == 0.001 and metadata["fps"] is None
    assert metadata["reference_files"] == []
    assert "injected_test_only" in metadata["quality_flags"]
    assert metadata["intrinsics"]["source"]["sha256"] == hashlib.sha256((directory / "cam_K.txt").read_bytes()).hexdigest()
    for frame in metadata["frames"]:
        for kind in ("rgb", "depth", "masks"):
            record = frame[kind]
            assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    for record in (metadata["mesh"], metadata["runner"], metadata["npz_writer"]):
        assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    assert report["video"]["decoded_frames"] == 3
    assert len(list((output / "overlays").glob("*.png"))) == 3
    overlay = cv2.imread(str(output / "overlays" / "0000.png"))
    assert overlay.shape == (32, 40, 3)
    assert np.any(overlay != cv2.imread(str(directory / "rgb" / "0000.png")))
    with (output / "trajectory.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["stem"] for row in rows] == ["0000", "0001", "0002"]
    assert all(row["translation_error_m"] == "" and row["rotation_error_deg"] == "" for row in rows)
    assert read_report(args) == report


def test_references_read_and_hashed_only_after_every_prediction(scene, monkeypatch):
    args, directory, _ = scene
    reference_dir = add_references(args, directory)
    estimator, reads = FakeEstimator(), []
    original_load, original_hash = runner.np.loadtxt, runner.file_record

    def guarded_load(path, *positional, **keywords):
        if Path(path).parent == reference_dir:
            assert len(estimator.calls) == 3
            reads.append(Path(path).stem)
        return original_load(path, *positional, **keywords)

    def guarded_hash(path):
        if Path(path).parent == reference_dir:
            assert len(estimator.calls) == 3
        return original_hash(path)

    monkeypatch.setattr(runner.np, "loadtxt", guarded_load)
    monkeypatch.setattr(runner, "file_record", guarded_hash)
    report = runner.run_sequence(args, estimator=estimator)
    assert reads == ["0000", "0001", "0002"]
    assert report["accuracy"]["joint_accuracy"] == 1
    assert report["accuracy"]["alignment"] == "none"
    assert report["status"] == "test_only_success"
    assert len(report["accuracy"]["reference_files"]) == 3
    for record in report["accuracy"]["reference_files"]:
        assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()


def test_scene_gt_is_not_automatically_discovered(scene, monkeypatch):
    args, directory, _ = scene
    add_references(args, directory)
    args.reference_dir, args.reference_frame_confirmed = None, False
    args.translation_threshold_m = args.rotation_threshold_deg = None
    original = runner.np.loadtxt

    def reject_gt(path, *positional, **keywords):
        assert Path(path).parent.name != "annotated_poses"
        return original(path, *positional, **keywords)

    monkeypatch.setattr(runner.np, "loadtxt", reject_gt)
    report = runner.run_sequence(args, estimator=FakeEstimator())
    assert report["accuracy"]["joint_accuracy"] is None


def test_evaluation_does_not_align_constant_offsets(tmp_path):
    pose = synthetic_pose(0)
    reference = pose.copy()
    reference[0, 3] += 0.2
    reference[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    np.savetxt(tmp_path / "frame.txt", reference)
    result = runner.evaluate_reference(tmp_path, ["frame"], [pose], 0.1, 45)
    assert result["joint_accuracy"] == 0
    assert result["translation_error_m"] == pytest.approx([0.2])
    assert result["rotation_error_deg"] == pytest.approx([90])
    reference[:3, :3] = np.diag([-1, -1, 1])
    np.savetxt(tmp_path / "frame.txt", reference)
    assert runner.evaluate_reference(tmp_path, ["frame"], [pose], 1, 179)["rotation_accuracy"] == 0


@pytest.mark.parametrize("fault", ["missing", "nonrigid", "wrong_shape"])
def test_bad_reference_fails_after_inference_without_accuracy(scene, fault):
    args, directory, _ = scene
    reference_dir = add_references(args, directory)
    target = reference_dir / "0001.txt"
    if fault == "missing":
        target.rename(reference_dir / "different_stem.txt")
    else:
        np.savetxt(target, np.ones((4, 4)) if fault == "nonrigid" else np.eye(3))
    estimator = FakeEstimator()
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.run_sequence(args, estimator=estimator)
    report = read_report(args)
    assert len(estimator.calls) == 3 and report["inference_success_rate"] == 1
    assert report["status"] == "failed" and report["phase"] == "post_inference_evaluation"
    assert report["accuracy"]["joint_accuracy"] is None


@pytest.mark.parametrize("field,value", [
    ("limit_frames", 0), ("est_refine_iter", 0), ("track_refine_iter", -1),
    ("depth_scale", 0), ("depth_scale", float("nan")), ("preview_fps", float("inf")),
    ("translation_threshold_m", 0.1), ("reference_frame_confirmed", True)])
def test_invalid_configuration_is_explicit_failure(scene, field, value):
    args, _, _ = scene
    setattr(args, field, value)
    estimator = FakeEstimator()
    with pytest.raises(ValueError):
        runner.run_sequence(args, estimator=estimator)
    assert estimator.calls == []
    assert read_report(args)["status"] == "failed"


@pytest.mark.parametrize("field,value", [
    ("reference_frame_confirmed", False), ("translation_threshold_m", None),
    ("rotation_threshold_deg", None), ("translation_threshold_m", -1),
    ("rotation_threshold_deg", 181), ("rotation_threshold_deg", float("nan"))])
def test_reference_requires_confirmation_and_predeclared_thresholds(scene, field, value):
    args, directory, _ = scene
    add_references(args, directory)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        runner.run_sequence(args, estimator=FakeEstimator())
    assert read_report(args)["phase"] == "validation"


@pytest.mark.parametrize("fault", ["missing_depth", "missing_first_mask", "extra_mask", "bad_png",
                                       "depth_size", "mask_size", "rgb_size", "grayscale_rgb", "depth_rgb", "zero_depth", "empty_mask"])
def test_scene_rejects_mismatched_or_invalid_inputs(scene, fault):
    _, directory, _ = scene
    if fault in ("missing_depth", "missing_first_mask"):
        kind = "depth" if fault == "missing_depth" else "masks"
        (directory / kind / "0000.png").unlink()
    elif fault == "extra_mask":
        (directory / "masks" / "orphan.png").write_bytes((directory / "masks" / "0000.png").read_bytes())
    elif fault == "bad_png":
        (directory / "rgb" / "0001.png").write_bytes(b"not a PNG")
    else:
        kind, image = {
            "depth_size": ("depth", np.ones((31, 40), np.uint16)),
            "mask_size": ("masks", np.ones((31, 40), np.uint8)),
            "rgb_size": ("rgb", np.ones((30, 40, 3), np.uint8)),
            "grayscale_rgb": ("rgb", np.ones((32, 40), np.uint8)),
            "depth_rgb": ("depth", np.ones((32, 40, 3), np.uint8)),
            "zero_depth": ("depth", np.zeros((32, 40), np.uint16)),
            "empty_mask": ("masks", np.zeros((32, 40), np.uint8)),
        }[fault]
        assert cv2.imwrite(str(directory / kind / ("0001.png" if fault == "rgb_size" else "0000.png")), image)
    with pytest.raises(ValueError):
        runner.load_scene(directory)


@pytest.mark.parametrize("intrinsics", [np.eye(4), np.zeros((3, 3)), np.full((3, 3), np.nan),
                                           [[50, 0, 20], [0, -50, 16], [0, 0, 1]],
                                           [[50, 0, 20], [0, 50, 16], [0, 1, 1]]])
def test_invalid_intrinsics(scene, intrinsics):
    _, directory, _ = scene
    np.savetxt(directory / "cam_K.txt", intrinsics)
    with pytest.raises(ValueError, match="cam_K"):
        runner.load_scene(directory)


@pytest.mark.parametrize("fault", ["nan", "inf", "reflection", "scale", "last_row", "zero_z", "negative_z", "wrong_shape"])
def test_pose_validation(fault):
    pose = synthetic_pose()
    if fault == "wrong_shape":
        pose = pose[:3]
    else:
        row, column, value = {
            "nan": (0, 3, np.nan), "inf": (1, 3, np.inf), "reflection": (0, 0, -1),
            "scale": (1, 1, 1.1), "last_row": (3, 0, 1), "zero_z": (2, 3, 0), "negative_z": (2, 3, -1),
        }[fault]
        pose[row, column] = value
    with pytest.raises(ValueError):
        runner.validate_pose(pose)


def test_projection_and_missing_later_mask_bbox(scene):
    args, directory, _ = scene
    (directory / "masks" / "0001.png").unlink()
    cv2.imwrite(str(directory / "masks" / "0002.png"), np.zeros((32, 40), np.uint8))
    report = runner.run_sequence(args, estimator=FakeEstimator())
    assert report["status"] == "test_only_success"
    meta = json.loads((Path(args.out_dir) / "poses.meta.json").read_text())
    assert meta["bboxes_source_per_frame"] == ["real_mask"] + ["mesh_projection_missing_or_empty_mask"] * 2
    intrinsics = np.array([[50, 0, 20], [0, 50, 16], [0, 0, 1]])
    np.testing.assert_allclose(runner.project_points([[0, 0, 0], [0.1, 0, 0]], synthetic_pose(), intrinsics), [[20, 16], [25, 16]])
    with pytest.raises(ValueError, match="positive"):
        runner.project_points([[0, 0, -2]], synthetic_pose(), intrinsics)


def test_limit_and_custom_scale_and_iterations(scene):
    args, _, _ = scene
    args.limit_frames, args.depth_scale = 1, 0.002
    args.est_refine_iter, args.track_refine_iter = 7, 3
    estimator = FakeEstimator()
    runner.run_sequence(args, estimator=estimator)
    assert len(estimator.calls) == 1 and estimator.calls[0][1]["iteration"] == 7
    np.testing.assert_allclose(estimator.calls[0][1]["depth"], 2)
    metadata = json.loads((Path(args.out_dir) / "poses.meta.json").read_text())
    assert metadata["depth_scale"] == 0.002 and metadata["num_frames"] == 1


def test_full_sequence_with_extra_depth_and_only_first_mask(scene):
    args, directory, _ = scene
    args.limit_frames = None
    (directory / "depth" / "orphan.png").write_bytes(b"unused depth must never be decoded")
    for stem in ("0001", "0002"):
        (directory / "masks" / f"{stem}.png").unlink()
    estimator = FakeEstimator()
    report = runner.run_sequence(args, estimator=estimator)
    assert len(estimator.calls) == report["inference_frames_completed"] == 3
    assert report["accuracy"]["joint_accuracy"] is None
    assert "ob_mask" in estimator.calls[0][1]
    assert all("ob_mask" not in inputs for _, inputs in estimator.calls[1:])
    _, records, _ = runner.load_scene(directory)
    assert all("rgb" not in record and "depth" not in record for record in records)


def test_existing_output_even_empty_is_never_overwritten(scene):
    args, _, _ = scene
    directory = Path(args.out_dir)
    directory.mkdir()
    with pytest.raises(FileExistsError):
        runner.run_sequence(args, estimator=FakeEstimator())
    assert list(directory.iterdir()) == []
    (directory / "keep.txt").write_text("untouched")
    with pytest.raises(FileExistsError):
        runner.run_sequence(args, estimator=FakeEstimator())
    assert (directory / "keep.txt").read_text() == "untouched"


def test_gpu_exception_propagates_no_fallback_or_reference_reads(scene, monkeypatch):
    args, directory, _ = scene
    add_references(args, directory)
    original = runner.np.loadtxt

    def reject_gt(path, *positional, **keywords):
        assert Path(path).parent.name != "annotated_poses"
        return original(path, *positional, **keywords)

    monkeypatch.setattr(runner.np, "loadtxt", reject_gt)
    with pytest.raises(RuntimeError, match="frame 1.*simulated CUDA failure"):
        runner.run_sequence(args, estimator=FakeEstimator(fail_at=1))
    report = read_report(args)
    assert report["status"] == "failed" and report["phase"] == "inference"
    assert report["inference_success_rate"] == pytest.approx(1 / 3)
    assert report["accuracy"]["joint_accuracy"] is None
    assert not (Path(args.out_dir) / "poses.npz").exists()


def test_backend_initialization_error_is_not_mock_fallback(scene, monkeypatch):
    args, _, _ = scene

    def failing_factory(*arguments):
        raise RuntimeError("missing real weights")

    monkeypatch.setattr(runner, "create_estimator", failing_factory)
    with pytest.raises(RuntimeError, match="missing real weights"):
        runner.run_sequence(args)
    report = read_report(args)
    assert report["phase"] == "backend_initialization" and report["status"] == "failed"
    assert report["inference_frames_completed"] == 0


def test_factory_seed_order_and_provenance_with_stub_modules(tmp_path, monkeypatch):
    """Only constructor plumbing is exercised; no real torch/CUDA/checkpoint load."""
    repo = tmp_path / "fake_repository"
    repo.mkdir()
    for name in ("estimater.py", "Utils.py"):
        (repo / name).write_text("synthetic CPU fixture")
    for name in ("scorer", "refiner"):
        (repo / name).mkdir()
        (repo / name / "model_best.pth").write_bytes(b"not real model weights")
        (repo / name / "config.yml").write_text("synthetic: true")
    events, constructor_inputs = [], {}

    class StubPredictor:
        def __init__(self, name):
            events.append(name)
            self.cfg = {"ckpt_dir": str(repo / name / "model_best.pth")}

    def construct(**kwargs):
        events.append("foundationpose")
        constructor_inputs.update(kwargs)
        return object()

    torch_stub = types.ModuleType("torch")
    torch_stub.__version__ = "CPU_TEST_STUB"
    torch_stub.version = types.SimpleNamespace(cuda="NOT_REAL_CUDA")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda: "CPU_TEST_STUB", synchronize=lambda: None)
    rasterizer = types.ModuleType("nvdiffrast.torch")
    rasterizer.RasterizeCudaContext = lambda: events.append("cuda_context") or "CPU_TEST_CONTEXT"
    rasterizer_parent = types.ModuleType("nvdiffrast")
    rasterizer_parent.torch = rasterizer
    module = types.ModuleType("estimater")
    module.__file__ = str(repo / "estimater.py")
    module.set_seed = lambda seed: events.append(("seed", seed))
    module.ScorePredictor = lambda: StubPredictor("scorer")
    module.PoseRefinePredictor = lambda: StubPredictor("refiner")
    module.FoundationPose = construct
    for name, stub in (("torch", torch_stub), ("nvdiffrast", rasterizer_parent), ("nvdiffrast.torch", rasterizer), ("estimater", module)):
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(runner.subprocess, "run", lambda *arguments, **keywords: types.SimpleNamespace(stdout="fixture_only"))
    _, backend, synchronize = runner.create_estimator(repo, trimesh.creation.box(), tmp_path / "debug")
    assert events == [("seed", 0), "scorer", "refiner", "cuda_context", "foundationpose"]
    assert backend["seed"] == 0 and backend["device"] == "CPU_TEST_STUB"
    assert constructor_inputs["glctx"] == "CPU_TEST_CONTEXT" and constructor_inputs["debug"] == 0
    assert "gt_pose" not in constructor_inputs
    assert synchronize is torch_stub.cuda.synchronize
    for predictor in backend["weights"].values():
        assert predictor["checkpoint"]["sha256"] == hashlib.sha256(b"not real model weights").hexdigest()


def test_progress_every_fifty_frames_and_at_completion(scene, capsys):
    args, directory, _ = scene
    args.limit_frames = None
    for index in range(3, 51):
        for kind in ("rgb", "depth"):
            (directory / kind / f"{index:04d}.png").write_bytes((directory / kind / "0000.png").read_bytes())
    report = runner.run_sequence(args, estimator=FakeEstimator())
    messages = capsys.readouterr().err.splitlines()
    assert len(messages) == 2
    assert "50/51 frames; elapsed=" in messages[0]
    assert "51/51 frames; elapsed=" in messages[1]
    assert 0 < report["inference_elapsed_seconds"] <= report["total_elapsed_seconds"]
    assert report["video"]["decoded_frames"] == 51


def test_upstream_placeholder_registration_is_rejected(scene):
    _, directory, _ = scene
    intrinsics, frames, _ = runner.load_scene(directory)
    estimator = FakeEstimator()
    estimator.register = lambda **inputs: synthetic_pose()
    with pytest.raises(RuntimeError, match="refusing upstream placeholder"):
        list(runner.infer_sequence(frames, intrinsics, estimator))


def test_cuda_synchronization_errors_are_not_silenced(scene):
    _, directory, _ = scene
    intrinsics, frames, _ = runner.load_scene(directory)

    def fail_sync():
        raise RuntimeError("asynchronous CUDA error")

    with pytest.raises(RuntimeError, match="asynchronous CUDA error"):
        list(runner.infer_sequence(frames, intrinsics, FakeEstimator(), synchronize=fail_sync))


def test_original_returned_pose_not_internal_centered_tracking_state(scene):
    _, directory, _ = scene
    intrinsics, frames, _ = runner.load_scene(directory)

    class CenteredStateEstimator(FakeEstimator):
        def predict(self, method, inputs):
            returned_pose = super().predict(method, inputs).copy()
            self.pose_last[0, 3] += 3
            return returned_pose

    predictions = list(runner.infer_sequence(frames, intrinsics, CenteredStateEstimator()))
    np.testing.assert_array_equal(predictions, [synthetic_pose(index) for index in range(3)])


def test_mutable_predictor_results_are_snapshots(scene):
    _, directory, _ = scene
    intrinsics, frames, _ = runner.load_scene(directory)

    class SharedBufferEstimator(FakeEstimator):
        def __init__(self):
            super().__init__()
            self.shared = np.empty((4, 4), np.float32)

        def predict(self, method, inputs):
            self.shared[:] = super().predict(method, inputs)
            return self.shared

    predictions = list(runner.infer_sequence(frames, intrinsics, SharedBufferEstimator()))
    np.testing.assert_array_equal(predictions, [synthetic_pose(index) for index in range(3)])


def test_source_mutation_after_preflight_is_detected(scene):
    _, directory, _ = scene
    intrinsics, frames, _ = runner.load_scene(directory)
    (directory / "depth" / "0001.png").write_bytes((directory / "rgb" / "0000.png").read_bytes())
    estimator = FakeEstimator()
    with pytest.raises(RuntimeError, match="frame 1.*source changed"):
        list(runner.infer_sequence(frames, intrinsics, estimator))
    assert len(estimator.calls) == 1


def test_odd_image_dimensions_preserved_in_png_padded_only_in_mp4(scene):
    args, directory, _ = scene
    for kind in ("rgb", "depth", "masks"):
        for path in (directory / kind).glob("*.png"):
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            assert cv2.imwrite(str(path), cv2.copyMakeBorder(image[:31], 0, 0, 0, 1, cv2.BORDER_REPLICATE))
    report = runner.run_sequence(args, estimator=FakeEstimator())
    assert report["video"]["size"] == [42, 32] and report["video"]["decoded_frames"] == 3
    assert cv2.imread(str(Path(args.out_dir) / "overlays" / "0000.png")).shape[:2] == (31, 41)


def test_alpha_channel_does_not_turn_background_into_initial_mask(scene):
    _, directory, mask = scene
    alpha_mask = np.dstack((mask, mask, mask, np.full_like(mask, 255)))
    assert cv2.imwrite(str(directory / "masks" / "0000.png"), alpha_mask)
    intrinsics, frames, _ = runner.load_scene(directory)
    estimator = FakeEstimator()
    list(runner.infer_sequence(frames, intrinsics, estimator))
    np.testing.assert_array_equal(estimator.calls[0][1]["ob_mask"], mask > 0)


@pytest.mark.parametrize("invalid_depth", [float("nan"), float("inf"), -1])
def test_nonfinite_and_negative_depth_rejected(scene, monkeypatch, invalid_depth):
    _, directory, _ = scene
    depth_bytes = (directory / "depth" / "0000.png").read_bytes()
    original = cv2.imdecode

    def inject_invalid_depth(contents, flags):
        if contents.tobytes() == depth_bytes:
            return np.full((32, 40), invalid_depth, dtype=np.float32)
        return original(contents, flags)

    monkeypatch.setattr(cv2, "imdecode", inject_invalid_depth)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        runner.load_scene(directory)


@pytest.mark.parametrize("fault", ["encoder", "png", "truncated_mp4"])
def test_export_failures_are_explicit_not_success(scene, monkeypatch, fault):
    args, _, _ = scene
    if fault == "png":
        monkeypatch.setattr(cv2, "imwrite", lambda *arguments: False)
    elif fault == "encoder":
        class ClosedWriter:
            def isOpened(self):
                return False

            def release(self):
                pass

        monkeypatch.setattr(cv2, "VideoWriter", lambda *arguments: ClosedWriter())
    else:
        class TruncatedCapture:
            def isOpened(self):
                return True

            def read(self):
                return False, None

            def release(self):
                pass

        monkeypatch.setattr(cv2, "VideoCapture", lambda *arguments: TruncatedCapture())
    with pytest.raises(RuntimeError):
        runner.run_sequence(args, estimator=FakeEstimator())
    report = read_report(args)
    assert report["status"] == "failed" and report["phase"] == "export"
    assert report["inference_success_rate"] == 1
    assert not (Path(args.out_dir) / "poses.npz").exists()
