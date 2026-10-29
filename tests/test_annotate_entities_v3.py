import json

import numpy as np
import pytest

from src.annotation.annotate_entities_v3 import (
    apply_reviews, geometry_status, mention_candidates, runs, text_candidates, verify_review,
)
from src.dataset.complete_egodemo import digest, save


def test_observation_gaps_remain_explicit():
    assert runs(np.array([1, 1, 0, 1, 0, 0], bool)) == [
        {"start_frame": 0, "end_frame": 2}, {"start_frame": 3, "end_frame": 4}]
    assert runs(np.zeros(5, bool)) == []


def test_role_and_material_rules_cannot_create_visual_silver():
    roles, materials = text_candidates({"label": "plastic bag", "name_zh": "透明塑料袋"})
    assert roles[0]["role"] == "container"
    assert any(row["type"] == "polymer_film" for row in materials)
    assert all(row["review_status"] == "estimated" and not row["silver_label_valid"] for row in roles + materials)


def test_same_name_remains_multiple_identity_candidates():
    entities = [{"id": "obj_001", "label": "blue box"}, {"id": "obj_002", "label": "blue box"}]
    assert mention_candidates("box", entities) == ["obj_001", "obj_002"]
    assert geometry_status({"rigidity": "deformable"}, 0, 100, True)[0] == "surface_only"
    assert geometry_status({"rigidity": "rigid"}, 20, 100, False)[0] == "blocked_calibration"


def test_visual_review_requires_actual_bound_image_and_scope(tmp_path):
    image = tmp_path / "page.jpg"
    image.write_bytes(b"test evidence bytes")
    review = {"schema_version": "assistant_annotation_v3_review", "sequence_id": "sample",
              "reviewer_agent": "test-agent", "reviewed_at": "2026-10-23T00:00:00Z",
              "source_bindings": {"proposal_sha256": "abc"},
              "interval_reviews": [{"source_interval_id": "source_0000", "review_status": "assistant_reviewed",
                  "annotation_source": "assistant_visual_review", "evidence_scope": "sampled_native_frames_only",
                  "evidence": [{"gallery_path": str(image), "gallery_sha256": digest(image), "frames": [2, 5]}],
                  "entity_assessments": [{"object_id": "obj_001", "status": "uncertain"}]}]}
    path = tmp_path / "review.json"
    save(path, review)
    doc = {"sequence_id": "sample", "source_bindings": {"proposal_sha256": "abc"},
           "entities": [{"entity_id": "obj_001"}],
           "interval_annotations": [{"source_interval_id": "source_0000", "review_status": "estimated"}],
           "coverage": {}}
    apply_reviews(doc, [path], tmp_path)
    assert doc["coverage"]["source_intervals_visually_reviewed"] == 1
    assert not doc["entities"][0]["visual_assessments"][0]["silver_label_valid"]
    assert not doc["entities"][0]["visual_assessments"][0]["human_label_valid"]
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image hash"):
        verify_review(review, "abc", tmp_path)
