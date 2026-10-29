"""CPU regressions for forecast frame math and acquisition-isolated samples."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import build_samples


EPISODE_A = "abcdef01-2345-4678-9abc-0123456789ab"
EPISODE_B = "12345678-abcd-4123-8abc-0123456789ab"
EPISODE_C = "87654321-abcd-4123-8abc-0123456789ab"


@pytest.fixture
def sample_inputs():
    frame_count = 14
    frame_numbers = np.arange(frame_count)
    left = np.broadcast_to(frame_numbers[:, None, None], (frame_count, 21, 3)).copy()
    clip = {
        "clip_id": "legacy_camera_001",
        "fps": 30.0,
        "left": left,
        "right": -left,
        "object_ids": ["obj_000", "obj_001"],
    }
    feats = np.broadcast_to(frame_numbers[:, None, None], (frame_count, 2, 13)).copy()
    valid = np.ones((frame_count, 2), np.float32)
    labels = (frame_numbers % 3, frame_numbers % 2, frame_numbers % 7)
    return clip, feats, valid, labels


def write_clip(root, clip_id, frame_count=12, fps=10.0, hand_meta=None, pose_meta=None):
    hand_dir = root / "hand_pose"
    hand_dir.mkdir(parents=True, exist_ok=True)
    hand = np.zeros((frame_count, 21, 3), np.float32)
    np.savez(hand_dir / f"{clip_id}.npz", left_hand=hand, right_hand=hand)
    metadata = {"fps": fps, "width": 640, "height": 480}
    metadata.update(hand_meta or {})
    (hand_dir / f"{clip_id}.meta.json").write_text(json.dumps(metadata))
    if pose_meta is not None:
        pose_dir = root / "pose6d"
        pose_dir.mkdir(parents=True, exist_ok=True)
        np.savez(pose_dir / f"{clip_id}.npz", object_ids=np.array([], dtype=str),
                 poses=np.empty((0, frame_count, 4, 4), np.float32),
                 bboxes=np.empty((0, frame_count, 4), np.int32))
        (pose_dir / f"{clip_id}.meta.json").write_text(json.dumps(pose_meta))


def read_index(root):
    return json.loads((root / "samples" / "dataset_index.json").read_text())


@pytest.mark.parametrize("seconds,fps,expected", [
    (None, 30.0, 1), (None, 2.5, 1), (0.5, 30.0, 15), (0.5, 25.0, 13),
    (0.5, 29.97, 15), (0.5, 2.5, 2), (1.0, 24.0, 24), (2.0, 30.0, 60),
    (0.0001, 30.0, 1), (np.nextafter(0.0, 1.0), 0.1, 1),
])
def test_horizon_rounding(seconds, fps, expected):
    assert build_samples.resolve_horizon_frames(seconds, fps) == expected


@pytest.mark.parametrize("seconds", [0, -0.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_horizon_seconds(seconds):
    with pytest.raises(ValueError, match="horizon-seconds must be finite and positive"):
        build_samples.resolve_horizon_frames(seconds, 30.0)


def test_horizon_product_overflow():
    with pytest.raises(ValueError, match=r"horizon-seconds \* fps must be finite"):
        build_samples.resolve_horizon_frames(1e308, 30.0)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_fps_not_silently_replaced(fps):
    with pytest.raises(ValueError, match="fps must be finite and positive"):
        build_samples.resolve_fps({"fps": fps}, {"fps": 30.0}, 25.0)
    with pytest.raises(ValueError, match="fps must be finite and positive"):
        build_samples.resolve_horizon_frames(None, fps)


def test_fps_metadata_precedence_and_fallback():
    assert build_samples.resolve_fps({"fps": 25}, {"fps": 30}, 60) == 25
    assert build_samples.resolve_fps({}, {"fps": 30}, 60) == 30
    assert build_samples.resolve_fps({}, {}, 60) == 60


def test_old_make_sample_call_preserves_next_frame(sample_inputs):
    sample = build_samples.make_sample(*sample_inputs, 2, 4, 3)
    assert sample["target_frame"] == 6
    assert sample["horizon_frames"] == 1
    assert sample["horizon_seconds"] == pytest.approx(1 / 30)
    assert sample["source_video"] == "legacy_camera"
    assert sample["split_group"] == "legacy_camera"
    assert sample["label_subject"] == sample_inputs[3][0][6]
    assert sample["label_interaction"] == sample_inputs[3][2][6]
    assert sample["hand_seq"].shape == (4, 2, 21, 3)
    assert sample["obj_seq"].shape == (4, 3, 13)


@pytest.mark.parametrize("horizon_frames", [1, 2, 5, np.int64(8)])
def test_target_labels_and_observation_boundary(sample_inputs, horizon_frames):
    sample = build_samples.make_sample(*sample_inputs, 2, 4, 3, horizon_frames)
    target = 5 + horizon_frames
    assert sample["target_frame"] == target
    assert sample["label_subject"] == sample_inputs[3][0][target]
    assert sample["label_object_slot"] == sample_inputs[3][1][target]
    assert sample["label_interaction"] == sample_inputs[3][2][target]
    assert sample["label_object_id"] == sample_inputs[0]["object_ids"][target % 2]
    assert sample["horizon_seconds"] == pytest.approx(horizon_frames / 30)
    np.testing.assert_array_equal(sample["hand_seq"][:, 0, 0, 0], np.arange(2, 6))
    np.testing.assert_array_equal(sample["obj_seq"][:, 0, 0], np.arange(2, 6))
    assert np.max(sample["hand_seq"][:, 0, 0, 0]) < target


@pytest.mark.parametrize("horizon_frames", [0, -1, 1.5, float("nan"), float("inf"), True])
def test_invalid_frame_horizon(sample_inputs, horizon_frames):
    with pytest.raises(ValueError, match="horizon_frames must be a positive integer"):
        build_samples.make_sample(*sample_inputs, 0, 4, 3, horizon_frames)


def test_target_exactly_past_clip_is_rejected(sample_inputs):
    with pytest.raises(ValueError, match="target_frame 14 is outside clip"):
        build_samples.make_sample(*sample_inputs, 2, 4, 3, 9)


def test_legacy_observation_only_arrays_remain_supported(sample_inputs):
    clip, feats, valid, labels = sample_inputs
    clip["left"] = clip["left"][:4]
    clip["right"] = clip["right"][:4]
    sample = build_samples.make_sample(clip, feats[:4], valid[:4], labels, 0, 4, 3)
    assert sample["target_frame"] == 4
    assert sample["label_interaction"] == labels[2][4]


def test_truncated_observation_window_is_rejected(sample_inputs):
    clip, feats, valid, labels = sample_inputs
    with pytest.raises(ValueError, match="observation window ends at 4, beyond 3 frames"):
        build_samples.make_sample(clip, feats[:3], valid, labels, 0, 4, 3)


def test_make_sample_without_precomputed_group_uses_uuid(sample_inputs):
    clip, feats, valid, labels = sample_inputs
    clip["clip_id"] = f"standard_{EPISODE_B}_head_left_000"
    clip["source_episode_uuid"] = EPISODE_A
    sample = build_samples.make_sample(clip, feats, valid, labels, 0, 4, 3)
    assert sample["split_group"] == EPISODE_A
    assert sample["source_video"] == f"standard_{EPISODE_B}_head_left"


@pytest.mark.parametrize("start,window", [(-1, 4), (0, 0)])
def test_invalid_observation_window(sample_inputs, start, window):
    with pytest.raises(ValueError, match="start must be nonnegative and window must be positive"):
        build_samples.make_sample(*sample_inputs, start, window, 3)


@pytest.mark.parametrize("clip_id,metadata,expected", [
    (f"raw_{EPISODE_A}_head_left_000", {}, EPISODE_A),
    (f"raw_{EPISODE_A.upper()}_hand_right_000", {}, EPISODE_A),
    (f"standard_{EPISODE_A}_head_left_002", {}, EPISODE_A),
    ("short_alias_000", {"source_video": f"raw/{EPISODE_A}/head_left.mp4"}, EPISODE_A),
    ("legacy_camera_001", {}, "legacy_camera"),
    ("legacy", {}, "legacy"),
    (f"partial_{EPISODE_A[:-1]}_000", {}, f"partial_{EPISODE_A[:-1]}"),
    (f"extended_{EPISODE_A}a_000", {}, f"extended_{EPISODE_A}a"),
])
def test_uuid_inference_and_legacy_fallback(clip_id, metadata, expected):
    assert build_samples.resolve_split_group(clip_id, metadata) == expected


def test_explicit_episode_metadata_overrides_format_uuid():
    group = build_samples.resolve_split_group(
        f"standard_{EPISODE_B}_000",
        {"source_episode_uuid": EPISODE_A.upper(), "source_video": EPISODE_C},
        {"source_episode_uuid": EPISODE_A},
    )
    assert group == EPISODE_A


def test_pose_metadata_episode_is_used():
    assert build_samples.resolve_split_group(
        "alias_000", {}, {"source_episode_uuid": EPISODE_A},
    ) == EPISODE_A


def test_conflicting_upstream_episode_metadata_rejected():
    with pytest.raises(ValueError, match="conflicting source_episode_uuid metadata"):
        build_samples.resolve_split_group(
            "alias_000", {"source_episode_uuid": EPISODE_A},
            {"source_episode_uuid": EPISODE_B},
        )


@pytest.mark.parametrize("episode_id", [123, [], "   "])
def test_invalid_episode_metadata_rejected(episode_id):
    with pytest.raises(ValueError, match="source_episode_uuid must be a nonempty string"):
        build_samples.resolve_split_group("alias_000", {"source_episode_uuid": episode_id})


def test_ambiguous_inferred_uuid_rejected():
    with pytest.raises(ValueError, match="conflicting UUIDs in clip/source_video"):
        build_samples.resolve_split_group(
            f"raw_{EPISODE_A}_000", {"source_video": f"raw/{EPISODE_B}/head.mp4"},
        )


def test_load_clip_preserves_source_video_with_authoritative_uuid(tmp_path):
    clip_id = f"standard_{EPISODE_B}_head_left_000"
    write_clip(tmp_path, clip_id, hand_meta={"source_episode_uuid": EPISODE_A},
               pose_meta={"source_episode_uuid": EPISODE_A.upper()})
    clip = build_samples.load_clip(tmp_path / "hand_pose", tmp_path / "pose6d", clip_id)
    assert clip["source_video"] == f"standard_{EPISODE_B}_head_left"
    assert clip["split_group"] == EPISODE_A


@pytest.fixture
def video_groups():
    return {
        "raw_head_left": EPISODE_A,
        "raw_head_right": EPISODE_A,
        "standard_head_left": EPISODE_A,
        "body_head_left": EPISODE_A,
        "other_video": EPISODE_B,
        "last_video": EPISODE_C,
    }


def test_random_splits_are_deterministic_and_group_isolated(video_groups):
    splits = build_samples.split_videos(video_groups, 0.1, 0.1, 13, video_groups=video_groups)
    assert splits == build_samples.split_videos(
        reversed(list(video_groups)), 0.1, 0.1, 13, video_groups=video_groups,
    )
    seen_groups = {}
    seen_videos = set()
    for split, videos in splits.items():
        assert videos
        for video in videos:
            assert video not in seen_videos
            seen_videos.add(video)
            assert seen_groups.setdefault(video_groups[video], split) == split
    assert seen_videos == set(video_groups)


def test_explicit_video_lists_expand_to_entire_groups(video_groups):
    splits = build_samples.split_videos(
        video_groups, 0.1, 0.1, 0, ["raw_head_left"], ["other_video"], video_groups,
    )
    assert splits == {
        "train": ["last_video"],
        "val": ["body_head_left", "raw_head_left", "raw_head_right", "standard_head_left"],
        "test": ["other_video"],
    }


@pytest.mark.parametrize("test_video", ["raw_head_left", "standard_head_left"])
def test_explicit_group_conflicts_rejected(video_groups, test_video):
    with pytest.raises(ValueError, match="split_group cannot be in both val and test"):
        build_samples.split_videos(
            video_groups, 0.1, 0.1, 0, ["raw_head_left"], [test_video], video_groups,
        )


def test_unknown_explicit_video_rejected(video_groups):
    with pytest.raises(ValueError, match="unknown videos"):
        build_samples.split_videos(
            video_groups, 0.1, 0.1, 0, ["typo"], video_groups=video_groups,
        )


def test_incomplete_group_mapping_rejected():
    with pytest.raises(ValueError, match="missing split_group for videos"):
        build_samples.split_videos(["missing"], 0.1, 0.1, 0, video_groups={})


def test_explicit_empty_lists_do_not_trigger_random_split(video_groups):
    splits = build_samples.split_videos(video_groups, 0.1, 0.1, 0, [], [], video_groups)
    assert splits == {"train": sorted(video_groups), "val": [], "test": []}


def test_one_episode_with_many_cameras_stays_in_train(video_groups):
    videos = [video for video, group in video_groups.items() if group == EPISODE_A]
    splits = build_samples.split_videos(videos, 0.1, 0.1, 0, video_groups=video_groups)
    assert splits == {"train": sorted(videos), "val": [], "test": []}


def test_old_split_videos_call_remains_compatible():
    splits = build_samples.split_videos(["a", "b", "c"], 0.1, 0.1, 0, ["b"], ["c"])
    assert splits == {"train": ["a"], "val": ["b"], "test": ["c"]}


@pytest.mark.parametrize("frame_count,horizon,expected_starts,expected_targets", [
    (4, None, [], []), (5, None, [0], [4]), (6, 0.3, [], []),
    (7, 0.3, [0], [6]), (12, 0.3, [0, 2, 4], [6, 8, 10]),
    (13, 0.3, [0, 2, 4, 6], [6, 8, 10, 12]),
])
def test_cli_window_end_boundaries(tmp_path, frame_count, horizon, expected_starts, expected_targets):
    write_clip(tmp_path, "legacy_000", frame_count=frame_count)
    args = ["--data-root", str(tmp_path), "--window", "4", "--stride", "2"]
    if horizon is not None:
        args.extend(["--horizon-seconds", str(horizon)])
    build_samples.main(args)
    entries = read_index(tmp_path)["train"]
    assert [entry["start_frame"] for entry in entries] == expected_starts
    assert [entry["target_frame"] for entry in entries] == expected_targets
    assert len(list((tmp_path / "samples").rglob("*.npz"))) == len(entries)


def test_cli_per_clip_fps_and_npz_index_agree(tmp_path):
    write_clip(tmp_path, "fast_000", frame_count=24, fps=29.97)
    write_clip(tmp_path, "slow_000", frame_count=24, fps=25.0)
    build_samples.main(["--data-root", str(tmp_path), "--window", "4", "--stride", "3",
                        "--horizon-seconds", "0.5"])
    entries = read_index(tmp_path)["train"]
    assert entries
    for entry in entries:
        fps = 29.97 if entry["clip_id"] == "fast_000" else 25.0
        expected_frames = 15 if fps == 29.97 else 13
        assert entry["horizon_frames"] == expected_frames
        assert entry["target_frame"] == entry["start_frame"] + 3 + expected_frames
        assert entry["horizon_seconds"] == pytest.approx(expected_frames / fps)
        assert entry["horizon_seconds"] >= 0.5
        with np.load(tmp_path / "samples" / "train" / f"{entry['sample_id']}.npz") as sample:
            for field in ("target_frame", "horizon_frames", "horizon_seconds", "split_group",
                          "source_video", "start_frame"):
                assert sample[field].item() == entry[field]
            assert sample["hand_seq"].shape[0] == 4


def test_cli_cross_camera_cross_format_group_isolation(tmp_path):
    clip_ids = [f"raw_{EPISODE_A}_head_left_000", f"raw_{EPISODE_A}_hand_right_001",
                f"standard_{EPISODE_B}_head_left_000", "legacy_000", "heldout_000"]
    for clip_id in clip_ids:
        metadata = {"source_episode_uuid": EPISODE_A} if clip_id.startswith("standard") else {}
        write_clip(tmp_path, clip_id, hand_meta=metadata)
    build_samples.main([
        "--data-root", str(tmp_path), "--window", "4", "--stride", "3",
        "--horizon-seconds", "0.3", "--val-videos", f"raw_{EPISODE_A}_head_left",
        "--test-videos", "heldout",
    ])
    index = read_index(tmp_path)
    assert {entry["clip_id"] for entry in index["val"]} == set(clip_ids[:3])
    assert {entry["split_group"] for entry in index["val"]} == {EPISODE_A}
    assert {entry["source_video"] for entry in index["val"]} == {
        build_samples.parse_source_video(clip_id) for clip_id in clip_ids[:3]
    }
    assert {entry["split_group"] for entry in index["train"]} == {"legacy"}
    assert {entry["split_group"] for entry in index["test"]} == {"heldout"}
    for split, entries in index.items():
        for entry in entries:
            with np.load(tmp_path / "samples" / split / f"{entry['sample_id']}.npz") as sample:
                assert sample["split_group"].item() == entry["split_group"]
                assert sample["source_video"].item() == entry["source_video"]


def test_cli_conflicting_groups_for_same_video_rejected_before_writing(tmp_path):
    write_clip(tmp_path, "same_video_000", hand_meta={"source_episode_uuid": EPISODE_A})
    write_clip(tmp_path, "same_video_001", hand_meta={"source_episode_uuid": EPISODE_B})
    with pytest.raises(ValueError, match="conflicting split_group across clips"):
        build_samples.main(["--data-root", str(tmp_path), "--window", "4"])
    assert not (tmp_path / "samples").exists()


def test_cli_conflicting_explicit_group_lists_fail_before_writing(tmp_path):
    write_clip(tmp_path, f"raw_{EPISODE_A}_left_000")
    write_clip(tmp_path, f"raw_{EPISODE_A}_right_000")
    with pytest.raises(ValueError, match="split_group cannot be in both val and test"):
        build_samples.main([
            "--data-root", str(tmp_path), "--window", "4",
            "--val-videos", f"raw_{EPISODE_A}_left", "--test-videos", f"raw_{EPISODE_A}_right",
        ])
    assert not (tmp_path / "samples").exists()


@pytest.mark.parametrize("horizon", ["0", "-1", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_horizons_before_reading(tmp_path, capsys, horizon):
    with pytest.raises(SystemExit) as raised:
        build_samples.main(["--data-root", str(tmp_path), f"--horizon-seconds={horizon}"])
    assert raised.value.code == 2
    assert "horizon-seconds must be finite and positive" in capsys.readouterr().err
    assert not (tmp_path / "samples").exists()
