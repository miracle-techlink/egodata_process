import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.dataset.audit_training_v5 import (
    EvidenceAuditor, audit_review, audit_review_labels, audit_seeds, audit_transitions,
)


VOCAB = {"operation": {"codes": ["hold", "press"]}, "relation": {"codes": ["contact"]},
         "cooperation": {"codes": ["bimanual_joint"]},
         "transition": {"codes": ["contact_onset", "release", "grasp_onset"]}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value))
    return sha(path)


def review_fixture():
    state = dict(layer="relation", type="contact", subject="right_hand", object_id="visual_tool",
                 secondary_object_id=None, start_frame=0, end_frame=2, value=0, evidence_frames=[0, 1],
                 sampling_scope="continuous_native_frames_reviewed", annotation_source="assistant_visual_review",
                 silver_label_valid=True, human_label_valid=False)
    event = dict(layer="transition", type="contact_onset", subject="right_hand", object_id="visual_tool",
                 secondary_object_id=None, boundary_frame_range=[2, 5], evidence_frames=list(range(6)),
                 frame_resolved=False, censored=False, occurrence_confirmed=True, before_after_observed=True,
                 before_evidence_frame=1, after_evidence_frame=4, reason_zh="1未接触，4已接触，保留中间范围。",
                 event_family="physical_interaction", sampling_scope="continuous_native_frames_reviewed",
                 annotation_source="assistant_visual_review", silver_label_valid=True, human_label_valid=False)
    return dict(schema_version="assistant_annotation_v4_dense_review", sequence_id="uid", frame_count=6,
                annotation_source="assistant_visual_review", reviewer_agent="other-agent", reviewed_at="test",
                human_label_valid=False, training_allowed=False, reviewed_ranges=[[0, 6]], unreviewed_ranges=[],
                entities=[dict(entity_id="visual_tool")], state_segments=[state], transition_events=[event],
                unresolved_ranges=[])


def test_bounded_occurrence_is_distinct_from_precise_and_old_censoring():
    r = review_fixture()
    got = audit_review_labels(r, set(range(6)), 6, VOCAB)
    assert got["counts"]["physical_interaction_confirmed_occurrence_ranges"] == 1
    assert got["counts"].get("physical_interaction_frame_resolved", 0) == 0
    r["transition_events"][0].update(censored=True, occurrence_confirmed=False)
    got = audit_transitions(r, set(range(6)), 6, VOCAB)
    assert got["physical_interaction_censored_ranges"] == 1
    assert not got.get("physical_interaction_confirmed_occurrence_ranges")


@pytest.mark.parametrize("change", [
    {"after_evidence_frame": 5},  # Exclusive hi=5 must include the first known after frame.
    {"before_evidence_frame": 2},
    {"occurrence_confirmed": False},
    {"censored": True},
    {"frame_resolved": True},
    {"before_after_observed": False},
    {"value": 0},
])
def test_invalid_event_promotion_or_half_open_end_is_rejected(change):
    r = review_fixture()
    r["transition_events"][0].update(change)
    with pytest.raises(ValueError):
        audit_transitions(r, set(range(6)), 6, VOCAB)


def test_bounded_transition_cannot_bridge_unviewed_middle():
    r = review_fixture()
    r["reviewed_ranges"] = [[0, 2], [4, 6]]
    with pytest.raises(ValueError, match="unreviewed"):
        audit_transitions(r, {0, 1, 4, 5}, 6, VOCAB)


def test_unknown_is_not_negative_and_deformable_does_not_get_rigid_pose():
    r = review_fixture()
    r["unresolved_ranges"] = [dict(r["state_segments"][0], value=None, silver_label_valid=False)]
    with pytest.raises(ValueError, match="unknown overlaps"):
        audit_review_labels(r, set(range(6)), 6, VOCAB)
    r = review_fixture()
    r["entities"][0].update(material_kind="deformable", whole_object_6d_valid=True)
    with pytest.raises(ValueError, match="deformable"):
        audit_review_labels(r, set(range(6)), 6, VOCAB)


@pytest.fixture
def pixel_fixture(tmp_path):
    cv2 = pytest.importorskip("cv2")
    Image = pytest.importorskip("PIL.Image")
    video = tmp_path / "native.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 30, (32, 24))
    assert writer.isOpened(), "Lossless FFV1 test encoder unavailable"
    for f in range(6):
        bgr = np.zeros((24, 32, 3), np.uint8)
        bgr[:, :, f % 3] = 30 + f * 20
        bgr[5:15, 7 + f:17 + f] = [20, 50, 210]
        writer.write(bgr)
    writer.release()
    hands = tmp_path / "hands.npz"
    np.savez(hands, hand_valid=np.ones((6, 2), bool))
    metadata = tmp_path / "metadata.json"
    write(metadata, dict(video_path=str(video), video_sha256=sha(video), image_size=[32, 24]))
    times = np.arange(6) / 30
    frames, pages = [], []
    cap = cv2.VideoCapture(str(video))
    for f in range(6):
        ok, bgr = cap.read(); assert ok
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop = tmp_path / f"crop_{f}.png"
        page = tmp_path / f"page_{f}.png"
        Image.fromarray(rgb).save(crop); Image.fromarray(rgb).save(page)
        frames.append(dict(frame_index=f, timestamp_s=float(times[f]), decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                           crop_path=crop.name, crop_sha256=sha(crop), crop_xyxy_native=[0, 0, 32, 24],
                           page_index=f, page_cell_index=0))
        pages.append(dict(path=page.name, sha256=sha(page), frame_indices=[f], cells=[dict(frame_index=f)]))
    cap.release()
    manifest_path = tmp_path / "manifest.json"
    manifest = dict(schema_version="assistant_annotation_v4_dense_evidence", sequence_id="uid", native_frame_range=[0, 6],
                    native_image_size=[32, 24], frames=frames, pages=pages,
                    source_bindings=dict(metadata_path=str(metadata), metadata_sha256=sha(metadata), hands_path=str(hands),
                                         hands_sha256=sha(hands), video_path=str(video), video_sha256=sha(video)))
    manifest_sha = write(manifest_path, manifest)
    ref = dict(path=manifest_path.name, sha256=manifest_sha, reviewed_page_paths=[p["path"] for p in pages],
               reviewed_page_sha256=[p["sha256"] for p in pages])
    r = review_fixture()
    annotation = tmp_path / "annotation.json"
    write(annotation, {"sequence_id": "uid"})
    r["source_bindings"] = dict(v3_annotation_path=str(annotation), v3_annotation_sha256=sha(annotation),
                                 v2_dataset_index_sha256="base", v3_overlay_index_sha256="overlay")
    r["evidence_manifests"] = [ref]
    r["reviewed_gallery_page_count"] = 6
    review_path = tmp_path / "review.json"
    write(review_path, r)
    seed = dict(entity_id="visual_tool", frame_index=2, bbox_xyxy_native=[8, 5, 22, 18], name_zh="工具",
                evidence_manifest_path=manifest_path.name, evidence_manifest_sha256=manifest_sha,
                evidence_crop_path=frames[2]["crop_path"], evidence_crop_sha256=frames[2]["crop_sha256"],
                annotation_source="assistant_visual_review", human_label_valid=False, reason_zh="实际看当前帧")
    seed_path = tmp_path / "seeds.json"
    write(seed_path, dict(schema_version="assistant_annotation_v5_visual_seeds", sequence_id="uid", native_image_size=[32, 24],
                         seeds=[seed], human_label_valid=False))
    return dict(root=tmp_path, review=r, review_path=review_path, ref=ref, timestamps=times,
                seed_path=seed_path, manifest=manifest, manifest_path=manifest_path)


def test_actual_native_rgb_and_crops_pass_then_tamper_is_rejected(pixel_fixture):
    f = pixel_fixture
    result = audit_review(f["review_path"], f["root"], timestamps=f["timestamps"], vocab=VOCAB, expected_uid="uid",
                          base_index_sha256="base", v3_overlay_sha256="overlay")
    assert result["evidence"][0]["native_pixels_verified"]
    assert result["counts"]["reviewed_frames"] == 6
    got = audit_seeds(f["seed_path"], f["root"], timestamps=f["timestamps"], review=f["review"])
    assert got["seed_count"] == 1
    crop = f["root"] / "crop_2.png"
    crop.write_bytes(crop.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="crop SHA"):
        audit_seeds(f["seed_path"], f["root"], timestamps=f["timestamps"], review=f["review"])


@pytest.mark.parametrize("name", ["metadata.json", "page_2.png", "native.avi", "manifest.json", "annotation.json"])
def test_changed_frozen_source_or_artifact_is_rejected(pixel_fixture, name):
    f = pixel_fixture
    path = f["root"] / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        audit_review(f["review_path"], f["root"], timestamps=f["timestamps"], vocab=VOCAB, expected_uid="uid")


@pytest.mark.parametrize("change", [
    {"activation_frame": 0},
    {"evidence_crop_path": "crop_3.png"},
    {"entity_id": "visual_not_reviewed"},
    {"bbox_xyxy_native": [-1, 3, 10, 20]},
    {"bbox_xyxy_native": [1, 3, float("nan"), 20]},
])
def test_seed_cannot_use_future_crop_hidden_identity_or_bad_box(pixel_fixture, change):
    f = pixel_fixture
    doc = json.loads(f["seed_path"].read_text())
    doc["seeds"][0].update(change)
    write(f["seed_path"], doc)
    with pytest.raises(ValueError):
        audit_seeds(f["seed_path"], f["root"], timestamps=f["timestamps"], review=f["review"])


def test_wrong_decoded_native_pixels_are_rejected_even_with_updated_crop_sha(pixel_fixture):
    f = pixel_fixture
    manifest = copy.deepcopy(f["manifest"])
    manifest["frames"][2]["decoded_native_rgb_sha256"] = "0" * 64
    updated = write(f["manifest_path"], manifest)
    ref = dict(f["ref"], sha256=updated)
    with pytest.raises(ValueError, match="native decoded"):
        EvidenceAuditor().verify(f["root"], ref, f["timestamps"], "uid")


def test_cached_evidence_is_invalidated_by_artifact_mutation(pixel_fixture):
    f = pixel_fixture
    auditor = EvidenceAuditor()
    auditor.verify(f["root"], f["ref"], f["timestamps"], "uid")
    crop = f["root"] / "crop_2.png"
    crop.write_bytes(crop.read_bytes() + b"changed after first verification")
    with pytest.raises(ValueError, match="crop SHA"):
        auditor.verify(f["root"], f["ref"], f["timestamps"], "uid")


def test_changed_parent_review_is_rejected(pixel_fixture):
    f = pixel_fixture
    parent = f["root"] / "parent.json"
    write(parent, f["review"])
    review = copy.deepcopy(f["review"])
    review["source_bindings"].update(parent_review_path=str(parent), parent_review_sha256=sha(parent))
    write(f["review_path"], review)
    parent.write_bytes(parent.read_bytes() + b" ")
    with pytest.raises(ValueError, match="parent review changed"):
        audit_review(f["review_path"], f["root"], timestamps=f["timestamps"], vocab=VOCAB, expected_uid="uid")


def test_full_frame_grid_evidence_is_bound_to_its_actual_native_frame(pixel_fixture):
    import cv2
    from PIL import Image, ImageDraw
    f = pixel_fixture
    cap = cv2.VideoCapture(str(f["root"] / "native.avi"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
    ok, bgr = cap.read(); cap.release(); assert ok
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((16, 12))
    draw = ImageDraw.Draw(image)
    draw.line((0, 0, 0, 12), fill=(100, 180, 160), width=1)
    draw.text((2, 4), "0", fill="white", stroke_width=1, stroke_fill="black")
    draw.line((0, 0, 16, 0), fill=(100, 180, 160), width=1)
    draw.text((2, 2), "0", fill="white", stroke_width=1, stroke_fill="black")
    draw.text((650, 705), "FRAME 2 | native 32x24", fill="white", stroke_width=1, stroke_fill="black")
    preview = f["root"] / "preview.jpg"
    image.save(preview, quality=94)
    seed = dict(frame_index=2, evidence_frame_path=preview.name, evidence_frame_sha256=sha(preview))
    auditor = EvidenceAuditor()
    result = auditor.verify_seed_full_frame(f["root"], seed, f["manifest"], f["manifest"]["frames"][2])
    assert result["native_pixels_verified"]
    with pytest.raises(ValueError, match="current native frame"):
        auditor.verify_seed_full_frame(f["root"], seed, f["manifest"], f["manifest"]["frames"][3])
    preview.write_bytes(preview.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="preview SHA"):
        auditor.verify_seed_full_frame(f["root"], seed, f["manifest"], f["manifest"]["frames"][2])


@pytest.fixture
def graph_fixture(tmp_path):
    from src.dataset.export_training_v5 import export
    base, root = tmp_path / 'base', tmp_path / 'v5'
    base.mkdir(); root.mkdir()
    uid = 'uid'; length = 80
    times = np.arange(length, dtype=np.float64) / 30
    source = dict(timestamps=times, hands_world=np.ones((length, 2, 21, 3), np.float32),
        hand_valid=np.ones((length, 2), bool), camera_to_world=np.tile(np.eye(4), (length, 1, 1)))
    np.savez(base / 'sequence.npz', **source)
    vocab = {layer: dict(codes=codes, names_zh=codes) for layer, codes in
             [('operation', ['hold']), ('relation', ['contact']), ('transition', ['release']), ('cooperation', ['bimanual_joint'])]}
    write(base / 'vocab.json', vocab)
    write(base / 'split.json', {'assignments': {uid: 'test'}})
    base_row = dict(sequence_id=uid, split='test', sequence_path='sequence.npz', frame_count=length,
                    sha256={'sequence': sha(base / 'sequence.npz')})
    write(base / 'dataset_index.json', dict(sequences=[base_row], split_manifest_path='split.json',
        split_manifest_sha256=sha(base / 'split.json'), vocab_path='vocab.json', vocab_sha256=sha(base / 'vocab.json'),
        source_selection_sha256='frozen-source'))
    folder = root / 'visual_observations' / uid; folder.mkdir(parents=True)
    review_dir = root / 'dense_reviews'; review_dir.mkdir()
    review = dict(sequence_id=uid, reviewed_ranges=[[0, length]], entities=[{'entity_id': 'visual_tool', 'name_zh': '工具'}],
        state_segments=[dict(id='contact', layer='relation', type='contact', subject='left_hand',
            object_id='visual_tool', secondary_object_id=None, start_frame=0, end_frame=length,
            evidence_frames=list(range(length)), annotation_source='assistant_visual_review', silver_label_valid=True, value=0)],
        transition_events=[])
    write(review_dir / (uid + '.json'), review)
    seeds = [dict(frame_index=f, entity_id='visual_tool', bbox_xyxy_native=[5, 5, 20, 20]) for f in [0, 30, 60]]
    write(folder / 'seeds.json', dict(seeds=seeds))
    last = (np.arange(length) // 30 * 30).astype(np.int32)[:, None]
    code = np.where(last == np.arange(length)[:, None], 1, 2).astype(np.uint8)
    obs = dict(timestamps=times, frame_index=np.arange(length), max_frame_accessed=np.arange(length),
        object_ids=np.array(['visual_tool']), object_valid=np.ones((length, 1), bool),
        boxes_xyxy_native=np.tile(np.array([5, 5, 20, 20], np.float32), (length, 1, 1)),
        rgb_stats=np.full((length, 1, 6), .3, np.float32), source_code=code, last_seed_frame=last,
        tracking_inliers=np.full((length, 1), 8), hand_uv_native=np.full((length, 2, 21, 2), 10, np.float32),
        hand_projection_valid=np.ones((length, 2), bool))
    np.savez(folder / 'observations.npz', **obs)
    native_meta = root / 'native_metadata.json'
    write(native_meta, dict(video_sha256='audited-video', K=[[10, 0, 0], [0, 10, 0], [0, 0, 1]]))
    write(folder / 'metadata.json', dict(observations_sha256=sha(folder / 'observations.npz'),
        source_bindings=dict(seeds_sha256=sha(folder / 'seeds.json'), base_sequence_sha256=sha(base / 'sequence.npz'),
                             source_metadata_sha256=sha(native_meta), video_sha256='audited-video'),
        tracking_config=dict(max_seed_age=30, minimum_points=6), native_image_size=[32, 24],
        observation_mode='causal_replay_with_annotated_initialization', whole_object_6d_certified=False))
    gallery = root / 'gallery.jpg'; gallery.write_bytes(b'gallery already pixel-reviewed in separate stage')
    frames = [dict(frame_index=f, timestamp_s=float(times[f]), objects=[dict(entity_id='visual_tool',
        bbox_xyxy_native=[5, 5, 20, 20], source_code=int(code[f, 0]), last_seed_frame=int(last[f, 0]))]) for f in range(length)]
    manifest = dict(sequence_id=uid, source_bindings={'observations_sha256': sha(folder / 'observations.npz'),
        'source_metadata_path': str(native_meta), 'source_metadata_sha256': sha(native_meta), 'source_video_sha256': 'audited-video',
        'tracking_metadata_sha256': sha(folder / 'metadata.json'), 'seeds_sha256': sha(folder / 'seeds.json')},
        frames=frames, pages=[dict(path=gallery.name, sha256=sha(gallery), frame_indices=list(range(length)))])
    write(root / 'tracking_manifest.json', manifest)
    write(folder / 'tracking_review.json', dict(sequence_id=uid, review_status='assistant_reviewed',
        observations_sha256=sha(folder / 'observations.npz'), evidence_manifest_path='tracking_manifest.json',
        evidence_manifest_sha256=sha(root / 'tracking_manifest.json'), reviewed_page_paths=[gallery.name],
        accepted_ranges=[dict(entity_id='visual_tool', start_frame=0, end_frame=length, reason_zh='fixture证据')],
        rejected_ranges=[], human_label_valid=False))
    # This fixture isolates index auditing. Native review auditing has its own
    # real-video tests above; this does not certify the synthetic gallery.
    write(root / 'review_seed_audit.json', dict(failed=0, native_pixels_verified=True,
        base_index_sha256=sha(base / 'dataset_index.json'), split_manifest_sha256=sha(base / 'split.json'),
        records=[dict(sequence_id=uid, sha256=sha(review_dir / (uid + '.json')),
                      seeds={'sha256': sha(folder / 'seeds.json')})]))
    export(base, root)
    return root


def test_all_graph_windows_are_actually_loaded_and_independently_checked(graph_fixture):
    from src.dataset.audit_training_v5 import audit_training_index
    result = audit_training_index(graph_fixture)
    assert result['failed'] == 0 and result['actual_dataset_windows_loaded'] > 0
    assert result['counts']['test']['known_negative_cells'] > 0
    assert result['counts']['test']['known_positive_cells'] == 0
    assert result['counts']['train'] == {}
    assert result['type_coverage']['test']['known_type_count'] == 1
    assert result['type_coverage']['test']['positive_type_count'] == 0
    assert result['type_coverage']['test']['negative_type_count'] == 1


@pytest.mark.parametrize('name', ['dense_reviews/uid.json', 'graph_windows/test.jsonl',
    'visual_observations/uid/observations.npz', 'visual_observations/uid/metadata.json',
    'sequences/uid/hands.npz'])
def test_actual_graph_audit_rejects_changed_frozen_artifact(graph_fixture, name):
    from src.dataset.audit_training_v5 import audit_training_index
    p = graph_fixture / name
    p.write_bytes(p.read_bytes() + b' ')
    with pytest.raises(ValueError, match='SHA changed'):
        audit_training_index(graph_fixture)


def test_independent_graph_preserves_direction_and_unknown_partial_horizon():
    from src.dataset.audit_training_v5 import independent_graph
    state = dict(layer='relation', type='contact', subject='object', object_id='a', secondary_object_id='b',
        start_frame=0, end_frame=10, evidence_frames=list(range(10)), value=0,
        annotation_source='assistant_visual_review', silver_label_valid=True)
    event = review_fixture()['transition_events'][0]
    event.update(object_id='a', type='release', subject='right_hand')
    review = {'state_segments': [state], 'transition_events': [event]}
    types = [{'layer': 'relation', 'code': 'contact'}, {'layer': 'transition', 'code': 'release'}]
    ids = ['right_hand', 'a', 'b']; observed = np.ones(3, bool)
    values, mask = independent_graph(review, ids, types, 0, 4, observed)
    assert mask[1, 2, 0] and not mask[2, 1, 0] and values[1, 2, 0] == 0
    assert mask[0, 1, 1]
    assert not independent_graph(review, ids, types, 2, 4, observed)[1][0, 1, 1]
