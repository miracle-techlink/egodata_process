import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/dataset/evaluate_scale.py"
SPEC = importlib.util.spec_from_file_location("evaluate_scale", MODULE)
scale = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scale)


def record(sequence_id="seq_main", acquisition_id="acq_1", **updates):
    result = {
        "sequence_id": sequence_id,
        "dataset": "HO-Cap",
        "acquisition_id": acquisition_id,
        "subject_id": "subject_1",
        "scene_id": "scene_1",
        "object_ids": ["G02_1"],
        "action_labels": ["hocap_task_1"],
        "download_status": "downloaded",
        "pool": None,
        "split": None,
    }
    result.update(updates)
    return result


def event(event_id="event_1", sequence_id="seq_main", **updates):
    result = {
        "event_id": event_id,
        "sequence_id": sequence_id,
        "subject": "left",
        "object_id": "G02_1",
        "object_hint": "object",
        "interaction_type": "grasp",
        "start_frame": 0,
        "end_frame": 2,
        "evidence_frames": [0],
        "label_source": "human",
        "review_status": "pending",
        "geometry_status": "not_checked",
        "source_event_ids": [],
        "notes": "",
    }
    result.update(updates)
    return result


def none_interval(interval_id="none_1", sequence_id="seq_main", **updates):
    result = {
        "interval_id": interval_id,
        "sequence_id": sequence_id,
        "start_frame": 2,
        "end_frame": 4,
        "evidence_frames": [2],
        "label_source": "human",
        "review_status": "pending",
        "notes": "",
    }
    result.update(updates)
    return result


def accepted(item):
    item.update(review_status="accepted", reviewer="reviewer", reviewed_at="2026-10-22T00:00:00Z")
    return item


def projected_sample(sample_id="w1", sequence_id="seq_main", split_group="acq_1", **updates):
    result = {
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "split_group": split_group,
        "target_frame": 1,
        "label_valid": True,
        "label_event_ids": ["event_1"],
        "label_subject": 0,
        "label_object_slot": 0,
        "label_object_id": "G02_1",
        "label_interaction": 2,
        "label_interaction_name": "grasp",
        "label_source": "human",
        "annotation_mode": "offline_noncausal",
    }
    result.update(updates)
    return result


class ScaleEvaluationTests(unittest.TestCase):
    def inputs(self, root, records, entries=None, ready=None, training_allowed=False, target=120):
        records = list(records)
        ready = set(ready or [])
        if entries is None:
            entries = []
            for item in records:
                split = "candidate" if item.get("pool") == "auxiliary" else "train"
                entries.append({
                    "sequence_id": item["sequence_id"],
                    "split": split,
                    "acquisition_group_id": item["acquisition_id"],
                    "joint_pose_ready": item["sequence_id"] in ready,
                })
        manifest = root / "candidate.json"
        raw = (json.dumps({"schema_version": "human_pose_dataset_v1", "records": records}) + "\n").encode()
        manifest.write_bytes(raw)
        freeze = root / "freeze"
        freeze.mkdir()
        (freeze / "candidate_split.json").write_text(json.dumps({"records": entries}))
        (freeze / "audit_report.json").write_text(json.dumps({
            "training_allowed": training_allowed,
            "source_ledger": {"sha256": hashlib.sha256(raw).hexdigest()},
            "records": [{"downloaded": True, "joint_pose_ready": item["sequence_id"] in ready}
                        for item in records],
        }))
        (freeze / "freeze.json").write_text(json.dumps({"training_allowed": training_allowed}))
        entry_by_sequence = {entry["sequence_id"]: entry for entry in entries}
        joint_groups = {
            entry_by_sequence[item["sequence_id"]]["acquisition_group_id"]
            for item in records if item.get("pool") != "auxiliary"
        }
        qualified = {}
        for item in records:
            if item["sequence_id"] in ready:
                entry = entry_by_sequence[item["sequence_id"]]
                qualified[entry["acquisition_group_id"]] = entry["split"]
        if training_allowed:
            (freeze / "training_manifest.json").write_text(json.dumps({
                "status": "accepted",
                "training_allowed": True,
                "source_ledger_sha256": hashlib.sha256(raw).hexdigest(),
                "splits": {
                    split: [entry for entry in entries
                            if entry["split"] == split and entry["sequence_id"] in ready]
                    for split in scale.SPLITS
                },
            }))
        progress = root / "progress.json"
        progress.write_text(json.dumps({
            "observed_at": "2026-10-22T17:21:44+08:00",
            "target_qualified_independent_acquisitions": target,
            "downloaded_joint_source_candidates": len(joint_groups),
            "accepted_qualified_joint_acquisitions": len(qualified),
            "qualified_gap_to_120": target - len(qualified),
            "current_qualified_split": {
                split: sum(group_split == split for group_split in qualified.values()) for split in scale.SPLITS
            },
            "training_allowed": training_allowed,
        }))
        return progress, manifest, freeze

    def evaluate(self, root, records, **kwargs):
        entries = kwargs.pop("entries", None)
        ready = kwargs.pop("ready", None)
        training_allowed = kwargs.pop("training_allowed", False)
        target = kwargs.pop("target", 120)
        progress, manifest, freeze = self.inputs(root, records, entries, ready, training_allowed, target)
        return scale.evaluate_scale(progress, manifest, freeze, **kwargs)

    def write_complete_annotations(self, root, records, accepted_sequence="seq_main"):
        annotations = root / "annotations"
        for item in records:
            sequence_dir = annotations / item["sequence_id"]
            sequence_dir.mkdir(parents=True)
            accepted_event = accepted(event(sequence_id=item["sequence_id"])) if item["sequence_id"] == accepted_sequence else None
            (sequence_dir / "events.jsonl").write_text(
                json.dumps(accepted_event) + "\n" if accepted_event else ""
            )
            (sequence_dir / "none_intervals.jsonl").write_text("")
        return annotations

    def test_independent_acquisitions_are_not_views_or_windows(self):
        root = self.enter_context(self.temporary_directory())
        records = [
            record(),
            record("seq_low", camera_id="low"),
            record("seq_aux", "aux_1", dataset="EgoDemo", pool="auxiliary", subject_id=None),
        ]
        sample_index = root / "dataset_index.json"
        sample_index.write_text(json.dumps({
            "train": [{"sample_id": f"window_{index}", "split_group": "acq_1",
                       "target_frame": index + 32, "label_valid": False} for index in range(7)],
            "val": [], "test": [],
        }))
        report = self.evaluate(root, records, sample_index_path=sample_index)
        joint = report["layers"]["downloaded_joint_candidates"]
        self.assertEqual(joint["records"], 2)
        self.assertEqual(joint["independent_acquisitions"], 1)
        self.assertEqual(joint["duplicate_view_or_record_excess"], 1)
        self.assertEqual(report["layers"]["downloaded_auxiliary"]["independent_acquisitions"], 1)
        self.assertEqual(report["windows"]["generated"], 7)
        self.assertEqual(report["windows"]["formal_usable"], 0)
        self.assertNotEqual(report["windows"]["generated"], joint["independent_acquisitions"])
        self.assertEqual(report["target_frames"]["invalid"], 7)

    def test_pending_events_and_labels_are_not_supervision(self):
        root = self.enter_context(self.temporary_directory())
        annotations = root / "annotations" / "seq_main"
        annotations.mkdir(parents=True)
        (annotations / "events.jsonl").write_text(
            json.dumps(event()) + "\n" + json.dumps(event("event_2", interaction_type="reach", review_status="rejected")) + "\n"
        )
        (annotations / "none_intervals.jsonl").write_text(json.dumps(none_interval()) + "\n")
        sample_index = root / "dataset_index.json"
        sample_index.write_text(json.dumps({
            "train": [
                {"sample_id": "w1", "split_group": "acq_1", "target_frame": 1,
                 "label_valid": True, "label_event_ids": ["event_1"]},
                {"sample_id": "w2", "split_group": "acq_1", "target_frame": 2, "label_valid": False},
            ], "val": [], "test": [],
        }))
        report = self.evaluate(root, [record()], annotations_path=annotations.parent,
                               sample_index_path=sample_index)
        self.assertEqual(report["annotations"]["interaction_events"]["by_status"]["pending"], 1)
        self.assertEqual(report["annotations"]["interaction_events"]["by_status"]["rejected"], 1)
        self.assertEqual(report["annotations"]["interaction_events"]["by_status"]["accepted"], 0)
        self.assertEqual(report["annotations"]["none_intervals"]["by_status"]["pending"], 1)
        self.assertEqual(report["annotations"]["none_intervals"]["accepted_coverage_frames_union"], 0)
        self.assertEqual(report["target_frames"]["valid"], 1)
        self.assertEqual(report["target_frames"]["invalid"], 1)
        self.assertEqual(report["target_frames"]["usable_valid"], 0)
        self.assertEqual(report["current_supervision"]["usable_independent_acquisitions"], 0)
        self.assertFalse(report["training_allowed"])

    def test_unknown_dimensions_are_not_inferred_from_ids_or_task_text(self):
        root = self.enter_context(self.temporary_directory())
        item = record(subject_id=None, scene_id=None, object_ids=["G02_1", "G02_2"], action_labels=[],
                      task_name="do not map this to an action")
        report = self.evaluate(root, [item])
        dimensions = report["layers"]["downloaded_joint_candidates"]["dimensions"]
        self.assertEqual(dimensions["object_instances"]["count"], 2)
        for name in ("subjects", "scenes", "object_semantic_categories", "actions"):
            self.assertEqual(dimensions[name]["count"], 0, name)
            self.assertEqual(dimensions[name]["status"], "unknown", name)
            self.assertEqual(dimensions[name]["unknown_acquisitions"], 1, name)

    def test_two_learning_curve_budgets_have_distinct_total_and_train_units(self):
        config = json.loads((ROOT / "configs/scale_benchmark_v1.json").read_text())
        designs = {item["design_id"]: item for item in scale.benchmark_design_summary(config)}
        fixed = designs["fixed_total_120_train_30_60_84"]
        expanded = designs["train_30_60_90_120_plus_holdout"]
        self.assertEqual(fixed["train_acquisition_budgets"], [30, 60, 84])
        self.assertEqual(fixed["train_pool"], 84)
        self.assertEqual(fixed["minimum_total_independent_acquisitions"], 120)
        self.assertTrue(fixed["holdout_included_in_fixed_total"])
        self.assertEqual(expanded["train_acquisition_budgets"], [30, 60, 90, 120])
        self.assertEqual(expanded["train_pool"], 120)
        self.assertEqual(expanded["additional_independent_holdout"], 36)
        self.assertEqual(expanded["minimum_total_independent_acquisitions"], 156)
        self.assertFalse(expanded["holdout_included_in_fixed_total"])

    def test_cluster_bootstrap_resamples_whole_acquisitions_not_windows(self):
        rows = [
            {"split_group": "acq_a", "window_id": "a1", "value": 0},
            {"split_group": "acq_a", "window_id": "a2", "value": 0},
            {"split_group": "acq_b", "window_id": "b1", "value": 1},
            {"split_group": "acq_b", "window_id": "b2", "value": 1},
        ]
        result = scale.cluster_bootstrap(rows, resamples=200, seed=7, include_replicates=True)
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result["row_count"], 4)
        self.assertEqual(result["observed"], 0.5)
        self.assertEqual(result["resampling_unit"], "independent_acquisition_cluster_not_windows")
        self.assertTrue(set(result["replicates"]).issubset({0.0, 0.5, 1.0}))
        self.assertIn(0.5, result["replicates"])

    def test_accepted_actual_event_and_explicit_none_are_separate_and_usable(self):
        root = self.enter_context(self.temporary_directory())
        annotations = root / "annotations" / "seq_main"
        annotations.mkdir(parents=True)
        (annotations / "events.jsonl").write_text(json.dumps(accepted(event())) + "\n")
        (annotations / "none_intervals.jsonl").write_text(json.dumps(accepted(none_interval())) + "\n")
        sample_index = root / "dataset_index.json"
        sample_index.write_text(json.dumps({
            "train": [
                projected_sample(),
                projected_sample("w2", target_frame=2, label_event_ids=["none_1"], label_subject=2,
                                 label_object_slot=-1, label_object_id="", label_interaction=0,
                                 label_interaction_name="none"),
            ], "val": [], "test": [],
        }))
        report = self.evaluate(root, [record()], ready={"seq_main"}, annotations_path=annotations.parent,
                               sample_index_path=sample_index)
        self.assertEqual(report["annotations"]["joint_accepted_interaction_events"], 1)
        self.assertEqual(report["annotations"]["joint_accepted_none_intervals"], 1)
        self.assertEqual(report["annotations"]["none_intervals"]["accepted_coverage_frames_union"], 2)
        self.assertEqual(
            {(item["id"], item["record_type"], item["start_frame"], item["end_frame"])
             for item in report["annotations"]["accepted_records"]},
            {("event_1", "event", 0, 2), ("none_1", "none_interval", 2, 4)},
        )
        self.assertNotIn("none", report["annotations"]["interaction_events"]["by_type"])
        self.assertEqual(report["target_frames"]["usable_valid"], 2)
        self.assertEqual(report["current_supervision"]["usable_independent_acquisitions"], 1)
        self.assertEqual(report["windows"]["formal_usable"], 2)

    def test_formal_index_integer_label_valid_and_invalid_types_stay_unknown(self):
        root = self.enter_context(self.temporary_directory())
        annotations = root / "annotations" / "seq_main"
        annotations.mkdir(parents=True)
        (annotations / "events.jsonl").write_text(json.dumps(accepted(event())) + "\n")
        (annotations / "none_intervals.jsonl").write_text("")
        valid_formal = {
            "sample_id": "seq_main_000000",
            "sequence_id": "seq_main",
            "clip_id": "seq_main",
            "source_video": "seq_main.mp4",
            "split": "train",
            "split_group": "acq_1",
            "acquisition_group_id": "acq_1",
            "subject_group_id": "subject_1",
            "start_index": 0,
            "target_index": 1,
            "observation_end_index": 0,
            "start_frame": 0,
            "target_frame": 1,
            "observation_end_frame": 0,
            "horizon_frames": 1,
            "fps": 30.0,
            "label_valid": 1,
            "label_subject": 0,
            "label_object_slot": 0,
            "label_object_id": "G02_1",
            "label_interaction": 2,
            "label_interaction_name": "grasp",
            "label_event_ids": ["event_1"],
            "label_source": "human",
            "annotation_mode": "offline_noncausal",
            "reviewer": "reviewer",
            "reviewed_at": "2026-10-22T00:00:00Z",
            "invalid_reason": "",
            "training_allowed": True,
        }
        invalid_formal = {
            **valid_formal,
            "sample_id": "seq_main_000001",
            "target_index": 2,
            "target_frame": 2,
            "observation_end_index": 1,
            "observation_end_frame": 1,
            "label_valid": 0,
            "label_subject": -1,
            "label_object_slot": -1,
            "label_object_id": "",
            "label_interaction": -1,
            "label_interaction_name": "",
            "label_event_ids": [],
            "label_source": "",
            "reviewer": "",
            "reviewed_at": "",
            "invalid_reason": "pending_annotation",
            "training_allowed": False,
        }
        rejected_values = [2, -1, 1.0, 0.0, "1", "0"]
        rejected = [
            {**valid_formal, "sample_id": f"invalid_type_{index}", "label_valid": value}
            for index, value in enumerate(rejected_values)
        ]
        sample_index = root / "dataset_index.json"
        sample_index.write_text(json.dumps({"train": [valid_formal, invalid_formal, *rejected],
                                            "val": [], "test": []}))
        serialized = json.loads(sample_index.read_text())
        self.assertIs(type(serialized["train"][0]["label_valid"]), int)
        self.assertIs(type(serialized["train"][1]["label_valid"]), int)

        report = self.evaluate(root, [record()], ready={"seq_main"}, annotations_path=annotations.parent,
                               sample_index_path=sample_index)

        self.assertEqual(report["target_frames"]["valid"], 1)
        self.assertEqual(report["target_frames"]["invalid"], 1)
        self.assertEqual(report["target_frames"]["unknown_validity"], len(rejected_values))
        self.assertIsNone(report["target_frames"]["usable_valid"])
        self.assertEqual(report["target_frames"]["verified_usable_valid"], 1)
        self.assertEqual(report["target_frames"]["valid_but_unverified"], 0)
        self.assertIsNone(report["current_supervision"]["usable_independent_acquisitions"])
        self.assertEqual(report["current_supervision"]["verified_usable_independent_acquisitions"], 1)
        self.assertEqual(report["windows"]["generated"], 2 + len(rejected_values))
        self.assertIsNone(report["windows"]["formal_usable"])
        self.assertEqual(report["windows"]["verified_formal_usable"], 1)
        self.assertEqual(report["target"]["qualified_actual"], 1)

    def test_qualified_count_comes_from_audit_not_window_count(self):
        root = self.enter_context(self.temporary_directory())
        report = self.evaluate(root, [record()], ready={"seq_main"})
        self.assertEqual(report["layers"]["qualified_joint"]["independent_acquisitions"], 1)
        self.assertEqual(report["layers"]["qualified_joint"]["by_split"]["train"]["independent_acquisitions"], 1)
        self.assertEqual(report["target"]["qualified_gap"], 119)
        self.assertEqual(report["annotations"]["coverage_status"], "not_provided")
        self.assertIsNone(report["target_frames"]["valid"])
        self.assertIsNone(report["target_frames"]["invalid"])
        self.assertIsNone(report["current_supervision"]["usable_target_frames"])
        self.assertFalse(report["training_manifest_present"])

    def test_cli_emits_machine_json_and_chinese_markdown(self):
        root = self.enter_context(self.temporary_directory())
        progress, manifest, freeze = self.inputs(root, [record()])
        json_output = root / "scale.json"
        markdown_output = root / "scale.md"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([
            sys.executable, str(MODULE), "--progress", str(progress),
            "--manifest", str(manifest), "--freeze", str(freeze), "--json-output", str(json_output),
            "--markdown-output", str(markdown_output),
        ], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(json_output.read_text())
        self.assertEqual(payload["schema_version"], "dataset_scale_evaluation_v1")
        self.assertEqual(payload["layers"]["qualified_joint"]["independent_acquisitions"], 0)
        self.assertIn("120 是项目内部门槛", markdown_output.read_text())
        self.assertIn("cluster bootstrap", markdown_output.read_text())

    def test_custom_group_id_resolves_and_record_split_conflict_disqualifies(self):
        root = self.enter_context(self.temporary_directory())
        item = record()
        entries = [{"sequence_id": "seq_main", "split": "train",
                    "acquisition_group_id": "frozen_custom_group", "joint_pose_ready": True}]
        annotations = self.write_complete_annotations(root, [item])
        sample_index = root / "dataset_index.json"
        sample_index.write_text(json.dumps({
            "train": [projected_sample(split_group="frozen_custom_group")], "val": [], "test": [],
        }))
        report = self.evaluate(root, [item], entries=entries, ready={"seq_main"},
                               annotations_path=annotations, sample_index_path=sample_index)
        self.assertEqual(report["target_frames"]["usable_valid"], 1)
        self.assertEqual(report["issues"], [])

        conflict_root = self.enter_context(self.temporary_directory())
        conflict = record(split="test")
        report = self.evaluate(conflict_root, [conflict], ready={"seq_main"})
        self.assertEqual(report["layers"]["qualified_joint"]["independent_acquisitions"], 0)
        self.assertIn("acquisition_record_split_conflict", {issue["code"] for issue in report["issues"]})

    def test_sample_split_must_match_frozen_group_split(self):
        root = self.enter_context(self.temporary_directory())
        annotations = self.write_complete_annotations(root, [record()])
        sample_index = root / "dataset_index.json"
        sample_index.write_text(json.dumps({
            "train": [], "val": [], "test": [projected_sample()],
        }))
        report = self.evaluate(root, [record()], ready={"seq_main"},
                               annotations_path=annotations, sample_index_path=sample_index)
        self.assertEqual(report["target_frames"]["usable_valid"], 0)
        self.assertEqual(report["target_frames"]["rejected_usable"], 1)
        self.assertIn("sample_split_mismatch", {issue["code"] for issue in report["issues"]})
        self.assertFalse(report["training_allowed"])

    def test_target_outside_accepted_boundary_is_rejected(self):
        root = self.enter_context(self.temporary_directory())
        annotations = self.write_complete_annotations(root, [record()])
        sample_index = root / "dataset_index.json"
        sample_index.write_text(json.dumps({
            "train": [projected_sample(target_frame=2)], "val": [], "test": [],
        }))
        report = self.evaluate(root, [record()], ready={"seq_main"},
                               annotations_path=annotations, sample_index_path=sample_index)
        self.assertEqual(report["target_frames"]["usable_valid"], 0)
        self.assertEqual(report["target_frames"]["rejected_usable"], 1)
        self.assertEqual(report["target_frames"]["projection_diagnostics"]["by_reason"],
                         {"unlabelled_or_rejected": 1})
        self.assertIn("sample_projection_mismatch", {issue["code"] for issue in report["issues"]})

    def test_overlapping_accepted_events_are_not_usable(self):
        root = self.enter_context(self.temporary_directory())
        annotations = root / "annotations" / "seq_main"
        annotations.mkdir(parents=True)
        (annotations / "events.jsonl").write_text(
            json.dumps(accepted(event())) + "\n"
            + json.dumps(accepted(event("event_2", subject="right", interaction_type="hold"))) + "\n"
        )
        (annotations / "none_intervals.jsonl").write_text("")
        sample_index = root / "dataset_index.json"
        sample_index.write_text(json.dumps({"train": [projected_sample()], "val": [], "test": []}))
        report = self.evaluate(root, [record()], ready={"seq_main"},
                               annotations_path=annotations.parent, sample_index_path=sample_index)
        self.assertEqual(report["target_frames"]["usable_valid"], 0)
        self.assertEqual(report["target_frames"]["projection_diagnostics"]["by_reason"],
                         {"overlapping_accepted_labels": 1})
        self.assertIn("sample_projection_mismatch", {issue["code"] for issue in report["issues"]})

    def test_unqualified_group_is_not_formally_usable(self):
        root = self.enter_context(self.temporary_directory())
        annotations = self.write_complete_annotations(root, [record()])
        sample_index = root / "dataset_index.json"
        sample_index.write_text(json.dumps({"train": [projected_sample()], "val": [], "test": []}))
        report = self.evaluate(root, [record()], annotations_path=annotations,
                               sample_index_path=sample_index)
        self.assertEqual(report["layers"]["qualified_joint"]["independent_acquisitions"], 0)
        self.assertEqual(report["target_frames"]["usable_valid"], 0)
        self.assertEqual(report["target_frames"]["projection_diagnostics"]["by_reason"],
                         {"group_not_qualified": 1})
        self.assertIn("sample_projection_mismatch", {issue["code"] for issue in report["issues"]})

    def test_missing_projection_fields_are_linked_but_not_fully_verified(self):
        root = self.enter_context(self.temporary_directory())
        annotations = self.write_complete_annotations(root, [record()])
        sample_index = root / "dataset_index.json"
        sample_index.write_text(json.dumps({
            "train": [{"sample_id": "w1", "sequence_id": "seq_main", "split_group": "acq_1",
                       "target_frame": 1, "label_valid": True, "label_event_ids": ["event_1"]}],
            "val": [], "test": [],
        }))
        report = self.evaluate(root, [record()], ready={"seq_main"},
                               annotations_path=annotations, sample_index_path=sample_index)
        self.assertIsNone(report["target_frames"]["usable_valid"])
        self.assertEqual(report["target_frames"]["verified_usable_valid"], 0)
        self.assertEqual(report["target_frames"]["linked_but_not_fully_verified"], 1)
        self.assertEqual(report["target_frames"]["projection_diagnostics"]["by_reason"],
                         {"missing_projection_fields": 1})
        self.assertIn("sample_projection_incomplete", {issue["code"] for issue in report["issues"]})

    def test_missing_inputs_keep_unknown_and_complete_zero_is_provable(self):
        missing_root = self.enter_context(self.temporary_directory())
        report = self.evaluate(missing_root, [record()], ready={"seq_main"})
        self.assertEqual(report["annotations"]["coverage_status"], "not_provided")
        self.assertEqual(report["target_frames"]["inventory_status"], "not_provided")
        self.assertIsNone(report["target_frames"]["valid"])
        self.assertIsNone(report["target_frames"]["invalid"])
        self.assertIsNone(report["target_frames"]["usable_valid"])

        zero_root = self.enter_context(self.temporary_directory())
        annotations = self.write_complete_annotations(zero_root, [record()], accepted_sequence="absent")
        report = self.evaluate(zero_root, [record()], ready={"seq_main"}, annotations_path=annotations)
        self.assertEqual(report["annotations"]["coverage_status"], "provided_complete_zero")
        self.assertEqual(report["target_frames"]["usable_valid"], 0)
        self.assertIsNone(report["target_frames"]["valid"])
        self.assertIsNone(report["target_frames"]["invalid"])

        incomplete_root = self.enter_context(self.temporary_directory())
        sequence_dir = incomplete_root / "annotations" / "seq_main"
        sequence_dir.mkdir(parents=True)
        (sequence_dir / "events.jsonl").write_text("")
        report = self.evaluate(incomplete_root, [record()], ready={"seq_main"},
                               annotations_path=sequence_dir.parent)
        self.assertEqual(report["annotations"]["coverage_status"], "incomplete")
        self.assertIsNone(report["target_frames"]["usable_valid"])

        empty_index_root = self.enter_context(self.temporary_directory())
        sample_index = empty_index_root / "dataset_index.json"
        sample_index.write_text(json.dumps({"train": [], "val": [], "test": []}))
        report = self.evaluate(empty_index_root, [record()], sample_index_path=sample_index)
        self.assertEqual(report["target_frames"]["inventory_status"], "provided_complete_zero")
        self.assertEqual(report["target_frames"]["valid"], 0)
        self.assertEqual(report["target_frames"]["invalid"], 0)
        self.assertEqual(report["target_frames"]["usable_valid"], 0)

    def test_markdown_uses_nonzero_accepted_counts(self):
        root = self.enter_context(self.temporary_directory())
        annotations = self.write_complete_annotations(root, [record()])
        sample_index = root / "dataset_index.json"
        sample_index.write_text(json.dumps({"train": [projected_sample()], "val": [], "test": []}))
        report = self.evaluate(root, [record()], ready={"seq_main"},
                               annotations_path=annotations, sample_index_path=sample_index)
        markdown = scale.render_markdown(report)
        self.assertIn("accepted 交互事件=1", markdown)
        self.assertIn("当前已验证 1 个正式可用目标帧", markdown)
        self.assertNotIn("当前 accepted 监督为 0", markdown)
        self.assertNotIn("当前没有 accepted 事件", markdown)

    def test_complete_120_formal_gate_and_independent_blockers(self):
        root = self.enter_context(self.temporary_directory())
        records = [record(f"seq_{index}", f"acq_{index}") for index in range(120)]
        entries = []
        for index, item in enumerate(records):
            split = "train" if index < 84 else "val" if index < 102 else "test"
            entries.append({"sequence_id": item["sequence_id"], "split": split,
                            "acquisition_group_id": item["acquisition_id"], "joint_pose_ready": True})
        ready = {item["sequence_id"] for item in records}
        progress, manifest, freeze = self.inputs(root, records, entries, ready, True)
        annotations = self.write_complete_annotations(root, records, "seq_0")
        sample_index = root / "dataset_index.json"
        sample_index.write_text(json.dumps({
            "train": [projected_sample(sequence_id="seq_0", split_group="acq_0")], "val": [], "test": [],
        }))
        report = scale.evaluate_scale(progress, manifest, freeze, annotations, sample_index)
        self.assertTrue(report["training_allowed"], report["formal_training_gate"])
        self.assertEqual(report["issues"], [])
        self.assertTrue(report["formal_training_gate"]["checks"]["qualified_total"]["pass"])

        progress_payload = json.loads(progress.read_text())
        progress_payload["target_qualified_independent_acquisitions"] = 121
        progress_payload["qualified_gap_to_120"] = 1
        progress.write_text(json.dumps(progress_payload))
        report = scale.evaluate_scale(progress, manifest, freeze, annotations, sample_index)
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["formal_training_gate"]["checks"]["qualified_total"]["pass"])

        progress_payload["target_qualified_independent_acquisitions"] = 120
        progress_payload["qualified_gap_to_120"] = 0
        progress.write_text(json.dumps(progress_payload))
        audit_path = freeze / "audit_report.json"
        audit_payload = json.loads(audit_path.read_text())
        audit_payload["source_ledger"]["sha256"] = "0" * 64
        audit_path.write_text(json.dumps(audit_payload))
        report = scale.evaluate_scale(progress, manifest, freeze, annotations, sample_index)
        self.assertFalse(report["training_allowed"])
        self.assertIn("manifest_freeze_source_sha_mismatch", {issue["code"] for issue in report["issues"]})

        audit_payload["source_ledger"]["sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        audit_path.write_text(json.dumps(audit_payload))
        (freeze / "training_manifest.json").unlink()
        report = scale.evaluate_scale(progress, manifest, freeze, annotations, sample_index)
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["formal_training_gate"]["checks"]["training_manifest_accepted"]["pass"])

    def test_formal_gate_requires_preregistered_split_minimums(self):
        root = self.enter_context(self.temporary_directory())
        records = [record(f"seq_{index}", f"acq_{index}") for index in range(120)]
        entries = []
        for index, item in enumerate(records):
            split = "train" if index < 60 else "val"
            entries.append({"sequence_id": item["sequence_id"], "split": split,
                            "acquisition_group_id": item["acquisition_id"], "joint_pose_ready": True})
        ready = {item["sequence_id"] for item in records}
        progress, manifest, freeze = self.inputs(root, records, entries, ready, True)
        annotations = self.write_complete_annotations(root, records, "seq_0")
        sample_index = root / "dataset_index.json"
        sample_index.write_text(json.dumps({
            "train": [projected_sample(sequence_id="seq_0", split_group="acq_0")], "val": [], "test": [],
        }))
        report = scale.evaluate_scale(progress, manifest, freeze, annotations, sample_index)
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["formal_training_gate"]["checks"]["qualified_splits"]["test"]["pass"])
        self.assertEqual(report["issues"], [])

    def enter_context(self, context):
        result = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return Path(result)

    def temporary_directory(self):
        import tempfile
        return tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main(verbosity=2)
