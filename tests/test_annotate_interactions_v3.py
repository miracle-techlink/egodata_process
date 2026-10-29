import hashlib
import json
import tempfile
from pathlib import Path
import unittest

from src.annotation.annotate_interactions_v3 import (
    interval_baseline, geometry_candidates, source_transition_candidates,
    apply_visual_reviews, uncovered_ranges,
    source_operation_quality,
)


def interval(description="Place the cup on the table", operation="place"):
    return interval_baseline({"source_interval_id": "source_0000", "start_frame": 10, "end_frame": 30,
                              "description_original": description, "description_zh": "将杯子放在桌上",
                              "operations": [{"type": operation}],
                              "object_mentions": [{"source_phrase": "cup"}, {"source_phrase": "table"}]})


class InteractionV3Tests(unittest.TestCase):
    def test_noun_false_matches_preserved_but_inactive_and_cap_added(self):
        text = "Put the cap on the marker and place it in the pen holder"
        a = text.index("holder")
        rows = source_operation_quality({"description_original": text,
            "operations": [{"type": "hold", "text_evidence": [{"text": "holder", "span": [a, a+6]}]}]})
        self.assertEqual(rows[0]["source_text_quality"], "rejected_false_match")
        self.assertFalse(rows[0]["active_weak_candidate"])
        self.assertEqual(rows[1]["type"], "cap")
        self.assertEqual(rows[0]["original_text_evidence"][0]["text"], "holder")

    def test_place_does_not_certify_support_or_release(self):
        row = interval()
        transitions = source_transition_candidates([row])
        self.assertEqual([x["type"] for x in transitions], ["operation_onset"])
        self.assertFalse(row["operations"][0]["silver_label_valid"])
        self.assertEqual(row["physical_layer_observability"]["support"]["status"], "unresolved")
        self.assertIsNone(row["entity_bindings"][0]["entity_id"])

    def test_plug_nouns_and_box_opening_are_not_actions(self):
        for text, code, token in [("Put the blue plug into the paper box", "plug", "plug into"),
                                  ("Put multiple white plugs into the paper box", "plug", "plugs into"),
                                  ("Seal the box opening with tape", "open", "opening"),
                                  ("Arrange the wet wipes", "wipe", "wipes")]:
            a = text.index(token)
            result = source_operation_quality({"description_original": text, "operations": [
                {"type": code, "text_evidence": [{"text": token, "span": [a, a + len(token)]}]}]})
            self.assertFalse(result[0]["active_weak_candidate"], text)
        result = source_operation_quality({"description_original": "Plug the lamp into the power strip",
            "operations": [{"type": "plug", "text_evidence": [{"text": "Plug the lamp into", "span": [0, 18]}]}]})
        self.assertTrue(result[0]["active_weak_candidate"])

    def test_explicit_release_still_has_uncertain_source_range(self):
        row = interval("Release the cup", "place")
        transitions = source_transition_candidates([row])
        self.assertEqual(transitions[-1]["type"], "release")
        self.assertEqual(transitions[-1]["boundary_range"], [10, 30])
        self.assertTrue(transitions[-1]["censored"])
        self.assertFalse(transitions[-1]["silver_label_valid"])

    def test_geometry_one_frame_onset_not_promoted_to_exact_boundary(self):
        graph = geometry_candidates({"interactions": [{"layer": "transition", "type": "grasp_onset",
                                    "subject": "left_hand", "object_id": "obj_001",
                                    "start_frame": 14, "end_frame": 15}]}, [interval()], 40)
        record = graph["transitions"][0]
        self.assertEqual(record["raw_geometry_candidate_range"], [14, 15])
        self.assertEqual(record["boundary_range"], [10, 30])
        self.assertFalse(record["physical_relation_verified"])

    def test_visual_support_is_sampled_only_not_whole_source_interval(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "view.jpg"
            path.write_bytes(b"test-image-provenance-only")
            evidence = {"gallery_path": str(path), "gallery_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "frames": [12, 20, 28]}
            manifest_path = Path(folder) / "evidence" / "fixture" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            native_path = manifest_path.parent / "frames" / "000020.jpg"
            native_path.parent.mkdir()
            native_path.write_bytes(b"native-image-provenance-fixture")
            native_sha = hashlib.sha256(native_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps({"pages": [{"path": str(path), "sha256": evidence["gallery_sha256"],
                                                            "interval_ids": ["source_0000"]}],
                                                 "frames": [{"path": str(native_path), "sha256": native_sha, "frame_index": 20}],
                                                 "intervals": [{"source_interval_id": "source_0000", "frames": [12, 20, 28]}]}))
            checked = {"source_interval_id": "source_0000", "review_status": "assistant_reviewed",
                       "annotation_source": "assistant_visual_review", "evidence": [evidence],
                       "handedness_assessment": {"status": "uncertain"},
                       "boundary_assessment": {"censored": True},
                       "operation_assessments": [{"type": "place", "status": "supported", "silver_label_valid": True,
                                                  "evidence_frames": [12, 20, 28],
                                                  "temporal_coverage": "sampled_evidence_interval_presence"}],
                       "relations": [{"type": "supported_by", "source": "visual_cup", "target": "visual_table",
                                      "secondary_object_id": "visual_table", "evidence_frames": [28],
                                      "silver_label_valid": True, "status": "supported"}]}
            review = {"schema_version": "assistant_annotation_v3_review", "sequence_id": "fixture",
                      "source_bindings": {"proposal_sha256": "sha",
                                          "evidence_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
                      "reviewer_agent": "fixture", "reviewed_at": "2026-10-23T00:00:00Z", "interval_reviews": [checked]}
            annotation = {"source_bindings": {"proposal_sha256": "sha"}, "interval_annotations": [interval()],
                          "relations": [], "transitions": [], "cooperations": [], "review_records": []}
            out = apply_visual_reviews(annotation, [review], folder)
            self.assertEqual(out["relations"][0]["temporal_coverage"], "sampled_points")
            self.assertEqual(out["relations"][0]["evidence_frames"], [28])
            self.assertEqual(out["relations"][0]["secondary_object_id"], "visual_table")
            self.assertFalse(out["interval_annotations"][0]["operations"][0]["silver_label_valid"])
            self.assertTrue(out["interval_annotations"][0]["silver_label_valid"])
            self.assertEqual(out["interval_annotations"][0]["visual_operation_assessments"][0]["temporal_coverage"],
                             "sampled_evidence_interval_presence")
            self.assertTrue(out["interval_annotations"][0]["boundary_assessment"]["censored"])
            self.assertEqual(out["interval_annotations"][0]["handedness_assessment"]["status"], "uncertain")
            checked["evidence"].append({"gallery_path": str(native_path), "gallery_sha256": native_sha, "frames": [20]})
            apply_visual_reviews(annotation, [review], folder)
            checked["evidence"][-1]["frames"] = [12]
            with self.assertRaisesRegex(ValueError, "exactly"):
                apply_visual_reviews(annotation, [review], folder)
            checked["evidence"].pop()
            checked["operation_assessments"][0]["status"] = "uncertain"
            with self.assertRaisesRegex(ValueError, "supported"):
                apply_visual_reviews(annotation, [review], folder)
            checked["operation_assessments"][0]["status"] = "supported"
            checked["relations"][0]["evidence_frames"] = [25]
            with self.assertRaisesRegex(ValueError, "subset"):
                apply_visual_reviews(annotation, [review], folder)

    def test_uncovered_frames_remain_unknown(self):
        self.assertEqual(uncovered_ranges([interval()], 40), [[0, 10], [30, 40]])

    def test_tool_pair_is_only_text_role_hypothesis(self):
        row = interval_baseline({"source_interval_id": "s", "start_frame": 0, "end_frame": 10,
                                 "description_original": "Cut paper using scissors", "operations": [{"type": "cut"}],
                                 "object_mentions": [{"source_phrase": "scissors"}, {"source_phrase": "paper"}]})
        edge = row["intent_graph_candidates"][0]
        self.assertIsNone(edge["source"])
        self.assertIsNone(edge["target"])
        self.assertFalse(edge["physical_relation_asserted"])
        self.assertFalse(edge["silver_label_valid"])


if __name__ == "__main__":
    unittest.main()
