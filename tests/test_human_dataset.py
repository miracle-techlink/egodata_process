"""Standard-library fixtures only: python -S tests/test_human_dataset.py."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/dataset/human_dataset.py"
SPEC = importlib.util.spec_from_file_location("human_dataset", MODULE)
dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dataset)


class HumanDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ledger = self.root / "candidate.json"
        self.fixture_config = {"fixture_mode": True, "minimums": {key: 1 for key in dataset.MINIMUMS}}

    def asset(self, number, role, content=None, suffix=".bin", folder=None):
        folder = folder or ("gt" if role in ("hand_metric_gt", "object_pose_gt") else
                            "observations" if role in ("hand_observations", "object_observations") else "source")
        path = self.root / "assets" / str(number) / folder / (role + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = content if content is not None else f"test payload for {number} / {role}\n".encode()
        path.write_bytes(payload)
        return {"role": role, "path": str(path.relative_to(self.root)),
                "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}

    def record(self, number=0, subject=None, **updates):
        assets = [self.asset(number, role) for role in dataset.REQUIRED_ROLES]
        assets.append(self.asset(number, "provenance", json.dumps({"fixture_review": number}).encode(), ".json"))
        assets.append(self.asset(number, "frame_ids", json.dumps(list(range(301))).encode(), ".json"))
        record = {"sequence_id": f"seq_{number}", "dataset": "UnitFixtures", "acquisition_id": f"acq_{number}",
                  "subject_id": subject or f"person_{number}", "scene_id": f"scene_{number % 3}",
                  "object_ids": [f"object_{number % 8}"], "action_labels": [f"action_{number % 4}"],
                  "camera_id": "same_physical_camera", "view": "third_person", "frame_count": 301,
                  "fps": 30.0, "duration_seconds": 10.0, "duration_definition": "frame_span",
                  "timestamp_source": "official_fps", "human_verified": True, "assets": assets,
                  "capabilities": {key: True for key in dataset.CAPABILITIES},
                  "quality": {key: "pass" for key in dataset.QUALITY_KEYS}, "split": None,
                  "evidence": {key: {"asset_role": "provenance", "description": f"Fixture review: {key}"}
                               for key in dataset.EVIDENCE_KEYS},
                  "continuity": {"continuous": True, "concatenated": False, "repeated_frames": False,
                                 "interpolated": False, "fps_modified": False},
                  "coordinates": {"unit": "m", "hand_frame": "camera", "object_frame": "camera",
                                  "handedness": "right", "aligned": True}, "camera_motion": "fixed",
                  "gt_independent": True, "gt_usage": "supervision_or_evaluation_only"}
        record["capabilities"]["camera_motion_compensated"] = False
        record.update(updates)
        return record

    def write(self, records):
        self.ledger.write_text(json.dumps({"schema_version": dataset.SCHEMA_VERSION, "records": records}), encoding="utf-8")
        return self.ledger

    def audit(self, records, config=None):
        return dataset.audit_manifest(self.write(records), self.fixture_config if config is None else config)

    def freeze(self, records, name="frozen", config=None):
        return dataset.freeze_manifest(self.write(records), self.root / name,
                                       self.fixture_config if config is None else config)

    def codes(self, report, index=0):
        return {error["code"] for error in report["records"][index]["errors"]}

    def role(self, record, role):
        return next(asset for asset in record["assets"] if asset["role"] == role)

    def replace_role(self, record, asset):
        record["assets"] = [existing for existing in record["assets"] if existing["role"] != asset["role"]] + [asset]

    def assignments(self, report):
        return {entry["sequence_id"]: entry["split"] for entry in report["candidate_split"]["records"]}

    def test_default_config_exposes_contract_and_production_floors(self):
        config = json.loads(dataset.DEFAULT_CONFIG.read_text())
        self.assertEqual(config["minimums"], dataset.MINIMUMS)
        self.assertEqual(config["required_roles"], list(dataset.REQUIRED_ROLES))
        self.assertEqual(config["required_evidence"], list(dataset.EVIDENCE_KEYS))
        self.assertFalse(config["fixture_mode"])

    def test_pending_annotations_do_not_collapse_candidate_splits_to_train(self):
        records = [self.record(number, subject=f"person_{number // 4}") for number in range(24)]
        for record in records:
            record["quality"]["coordinates"] = "pending"
        report = self.audit(records)
        self.assertEqual([report["inventory_splits"][split]["independent_acquisitions"]
                          for split in dataset.SPLITS], [16, 4, 4])
        self.assertFalse(report["training_allowed"])
        self.assertEqual(report["statistics"]["joint_pose_ready"]["records"], 0)

    def test_quality_and_new_estimates_do_not_choose_holdout_subjects(self):
        records = [self.record(number, subject=f"person_{number // 4}") for number in range(24)]
        original = self.assignments(self.audit(records))
        for number, record in enumerate(records):
            record["quality"]["coordinates"] = "pending"
            record["assets"].append(self.asset(number, "object_observations", f"new estimate {number}".encode(),
                                               folder="new_observations"))
        self.assertEqual(self.assignments(self.audit(records)), original)

    def test_valid_candidate_is_auditable_but_not_frozen_or_large_enough(self):
        report = self.audit([self.record()])
        self.assertEqual(report["records"][0]["errors"], [])
        self.assertEqual(report["statistics"]["asset_verified_long"]["records"], 1)
        self.assertEqual(report["statistics"]["joint_pose_ready"]["records"], 1)
        self.assertFalse(report["training_allowed"])
        self.assertIsNone(report["accuracy"]["joint"])

    def test_missing_observations_does_not_erase_downloaded_long_assets(self):
        record = self.record()
        excluded = {"hand_observations", "object_observations", "hand_valid", "object_valid", "object_pose_gt"}
        record["assets"] = [asset for asset in record["assets"] if asset["role"] not in excluded]
        report = self.audit([record])
        for layer in ("downloaded", "asset_verified_long", "qualified_long_sequences", "auxiliary"):
            self.assertEqual(report["statistics"][layer]["records"], 1, layer)
        self.assertEqual(report["statistics"]["joint_pose_ready"]["records"], 0)
        self.assertEqual(report["records"][0]["pool"], "auxiliary")
        self.assertEqual(report["missing_roles"]["object_pose_gt"], 1)

    def test_capability_booleans_never_replace_files(self):
        record = self.record(assets=[])
        report = self.audit([record])
        self.assertEqual(set(report["records"][0]["missing_roles"]), set(dataset.REQUIRED_ROLES))
        self.assertEqual(report["statistics"]["downloaded"]["records"], 0)
        self.assertFalse(report["records"][0]["joint_pose_ready"])

    def test_world_hand_reference_is_auxiliary_not_camera_metric_gt(self):
        record = self.record(pool="auxiliary")
        record["assets"] = [asset for asset in record["assets"] if asset["role"] not in
                            ("hand_metric_gt", "object_pose_gt", "hand_observations", "object_observations")]
        record["assets"].append(self.asset(0, "hand_world_reference"))
        record["capabilities"]["hand_metric_gt"] = False
        record["coordinates"] = None
        report = self.audit([record])
        for layer in ("asset_verified_long", "auxiliary", "declared_auxiliary", "verified_auxiliary"):
            self.assertEqual(report["statistics"][layer]["records"], 1, layer)
        self.assertEqual(report["statistics"]["joint_pose_ready"]["records"], 0)
        self.assertEqual(report["missing_roles"]["hand_metric_gt"], 1)
        record["human_verified"] = False
        report = self.audit([record])
        self.assertEqual(report["statistics"]["declared_auxiliary"]["records"], 1)
        self.assertEqual(report["statistics"]["verified_auxiliary"]["records"], 0)

    def test_rounding_duration_cannot_extend_subten_second_frame_span(self):
        record = self.record(frame_count=300)
        self.replace_role(record, self.asset(0, "frame_ids", json.dumps(list(range(300))).encode(), ".json"))
        self.assertIn("sequence_too_short", self.codes(self.audit([record])))
        record["duration_definition"] = "frame_count_over_fps"
        self.assertTrue(self.audit([record])["records"][0]["asset_verified_long"])

    def test_every_capability_quality_and_evidence_is_required(self):
        original = self.record()
        for section, keys, code in (("capabilities", dataset.CAPABILITIES, "capability_missing"),
                                     ("quality", dataset.QUALITY_KEYS, "quality_not_pass"),
                                     ("evidence", dataset.EVIDENCE_KEYS, "evidence_missing")):
            for key in keys:
                with self.subTest(section=section, key=key):
                    record = copy.deepcopy(original)
                    del record[section][key]
                    report = self.audit([record])
                    self.assertIn(code, self.codes(report))
                    self.assertFalse(report["records"][0]["joint_pose_ready"])

    def test_missing_subject_scene_action_or_coordinate_reference(self):
        original = self.record()
        for field in ("subject_id", "scene_id", "action_labels", "coordinates", "timestamp_source"):
            with self.subTest(field=field):
                record = copy.deepcopy(original)
                record[field] = None
                report = self.audit([record])
                self.assertFalse(report["records"][0]["joint_pose_ready"])
        record = copy.deepcopy(original)
        record["coordinates"]["unit"] = "mm"
        self.assertIn("coordinates_unverified", self.codes(self.audit([record])))

    def test_malformed_records_do_not_crash_whole_inventory(self):
        records = [None, 17, [], "not an object", {}, {"sequence_id": "partial", "assets": [None, {}, "bad"]}]
        base = self.record()
        for field in ("fps", "frame_count", "timestamp_source", "split", "assets", "capabilities", "quality", "subject_id"):
            invalid = copy.deepcopy(base)
            invalid["sequence_id"] = f"malformed_{field}"
            invalid[field] = {"bad": []}
            records.append(invalid)
        records.append(base)
        report = self.audit(records)
        self.assertEqual(len(report["records"]), len(records))
        self.assertTrue(report["records"][-1]["joint_pose_ready"])
        self.assertGreater(report["record_error_count"], 0)

    def test_lfs_empty_zero_url_html_and_placeholder_are_not_payloads(self):
        payloads = [b"", bytes(100), b"pending", b"placeholder file", b"https://example.invalid/data",
                    b"<html>Access denied</html>",
                    b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 200\n"]
        for number, payload in enumerate(payloads):
            with self.subTest(payload=payload[:30]):
                record = self.record(number)
                self.replace_role(record, self.asset(number, "rgb", payload))
                report = self.audit([record])
                self.assertIn("rgb", report["records"][0]["missing_roles"])
                self.assertFalse(self.role(report["records"][0], "rgb")["verified"])

    def test_sha_size_and_actual_size_are_verified(self):
        record = self.record()
        self.role(record, "rgb")["sha256"] = "0" * 64
        self.role(record, "depth")["size_bytes"] = 999
        del self.role(record, "mesh")["size_bytes"]
        report = self.audit([record])
        self.assertIn("asset_sha256_mismatch", self.codes(report))
        self.assertIn("asset_size_mismatch", self.codes(report))
        self.assertGreater(self.role(report["records"][0], "mesh")["size_bytes"], 0)
        self.assertTrue(self.role(report["records"][0], "mesh")["verified"])

    def test_relative_paths_use_ledger_directory_even_from_other_cwd(self):
        self.write([self.record()])
        result = subprocess.run([sys.executable, "-S", str(MODULE), "audit", str(self.ledger)],
                                cwd="/", capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads(result.stdout)["statistics"]["joint_pose_ready"]["records"], 1)

    def test_timebase_values_fps_duration_and_definition(self):
        original = self.record()
        for field, value in (("fps", 0), ("fps", float("inf")), ("fps", float("nan")), ("fps", True),
                              ("fps", 1e-320), ("duration_seconds", -1), ("frame_count", 1),
                              ("duration_seconds", 30), ("fps", 60), ("duration_definition", None)):
            with self.subTest(field=field, value=value):
                record = copy.deepcopy(original)
                record[field] = value
                self.assertFalse(self.audit([record])["records"][0]["asset_verified_long"])
        for definition, duration in (("frame_span", 10.0), ("frame_count_over_fps", 301 / 30)):
            record = copy.deepcopy(original)
            record.update(duration_definition=definition, duration_seconds=duration)
            self.assertTrue(self.audit([record])["records"][0]["joint_pose_ready"])

    def test_short_sequence_cannot_be_rescued_by_nominal_default_fps(self):
        record = self.record(frame_count=70, duration_seconds=70 / 30, duration_definition="frame_count_over_fps")
        report = self.audit([record])
        self.assertIn("sequence_too_short", self.codes(report))
        record["timestamp_source"] = "default_30fps"
        self.assertIn("timestamp_source_invalid", self.codes(self.audit([record])))

    def test_official_fps_requires_actual_continuous_frame_ids(self):
        record = self.record()
        record["assets"] = [asset for asset in record["assets"] if asset["role"] != "frame_ids"]
        self.assertIn("frame_ids_missing", self.codes(self.audit([record])))
        for values in ([0, 1], list(range(300)) + [299], list(range(300)) + [301]):
            self.replace_role(record, self.asset(0, "frame_ids", json.dumps(values).encode(), ".json"))
            self.assertIn("frame_ids_invalid", self.codes(self.audit([record])))

    def test_source_timestamp_tables_check_count_order_span_and_gaps(self):
        record = self.record(timestamp_source="source_timestamps")
        values = [index / 30 for index in range(301)]
        self.replace_role(record, self.asset(0, "timestamps", json.dumps(values).encode(), ".json"))
        self.assertTrue(self.audit([record])["records"][0]["joint_pose_ready"])
        broken = [values[:-1], [0.0] * 301, list(reversed(values)), [value * 2 for value in values]]
        gap = list(values)
        gap[150] = gap[149]
        broken.append(gap)
        for invalid in broken:
            self.replace_role(record, self.asset(0, "timestamps", json.dumps(invalid).encode(), ".json"))
            self.assertIn("timestamps_invalid", self.codes(self.audit([record])))

    def test_csv_time_and_frame_ids_supported(self):
        record = self.record(timestamp_source="video_pts")
        timestamps = "timestamp_seconds\n" + "\n".join(str(index / 30) for index in range(301))
        self.replace_role(record, self.asset(0, "timestamps", timestamps.encode(), ".csv"))
        self.assertTrue(self.audit([record])["records"][0]["joint_pose_ready"])
        record["timestamp_source"] = "official_fps"
        frames = "frame_id\n" + "\n".join(str(index + 100) for index in range(301))
        self.replace_role(record, self.asset(0, "frame_ids", frames.encode(), ".csv"))
        self.assertTrue(self.audit([record])["records"][0]["joint_pose_ready"])

    def test_stitching_interpolation_and_repeated_frames_never_make_long_data(self):
        original = self.record()
        for key in ("concatenated", "interpolated", "repeated_frames", "fps_modified"):
            record = copy.deepcopy(original)
            record["continuity"][key] = True
            self.assertIn("continuity_unverified", self.codes(self.audit([record])))

    def test_fixed_camera_or_compensation_requires_real_compensation_asset(self):
        record = self.record(camera_motion="moving")
        self.assertIn("camera_motion_unverified", self.codes(self.audit([record])))
        record["camera_motion"] = "compensated"
        record["capabilities"]["camera_motion_compensated"] = True
        self.assertIn("camera_compensation_missing", self.codes(self.audit([record])))
        record["assets"].append(self.asset(0, "camera_motion_compensated"))
        self.assertTrue(self.audit([record])["records"][0]["joint_pose_ready"])

    def test_gt_is_separate_independent_and_supervision_not_inference(self):
        original = self.record()
        for usage in ("evaluation_only", "supervision_or_evaluation_only"):
            record = copy.deepcopy(original)
            record["gt_usage"] = usage
            self.assertTrue(self.audit([record])["records"][0]["joint_pose_ready"])
        for field in ("gt_used_for_initialization", "future_gt_used", "gt_used_for_perception"):
            record = copy.deepcopy(original)
            record[field] = True
            self.assertIn("gt_leakage", self.codes(self.audit([record])))
        record = copy.deepcopy(original)
        reference = copy.deepcopy(self.role(record, "hand_metric_gt"))
        reference["role"] = "hand_observations"
        self.replace_role(record, reference)
        codes = self.codes(self.audit([record]))
        self.assertIn("gt_payload_reused", codes)
        self.assertIn("gt_directory_not_separate", codes)

    def test_copied_rgb_files_merge_acquisitions_even_with_different_paths_and_shared_flag(self):
        records = [self.record(0), self.record(1)]
        rgb = self.role(records[0], "rgb")
        payload = (self.root / rgb["path"]).read_bytes()
        copied = self.asset(1, "rgb", payload)
        copied["identity_scope"] = "shared"
        self.replace_role(records[1], copied)
        report = self.audit(records)
        entries = report["candidate_split"]["records"]
        self.assertEqual(entries[0]["group_id"], entries[1]["group_id"])
        self.assertEqual(entries[0]["acquisition_group_id"], entries[1]["acquisition_group_id"])
        self.assertEqual(report["statistics"]["joint_pose_ready"]["independent_acquisitions"], 1)

    def test_views_fragments_and_transitive_aliases_cannot_cross_declared_splits(self):
        records = [self.record(0, split="train"), self.record(1, split="val"), self.record(2, split="test")]
        records[1].update(acquisition_id=records[0]["acquisition_id"], camera_id="another_camera")
        records[1]["source_aliases"] = ["common_original"]
        records[2]["source_content_id"] = "common_original"
        report = self.audit(records)
        self.assertIn("declared_split_leakage", {item["code"] for item in report["errors"]})
        self.assertEqual(len({entry["group_id"] for entry in report["candidate_split"]["records"]}), 1)
        self.assertNotIn("test", set(self.assignments(report).values()))

    def test_same_camera_does_not_join_unrelated_acquisitions(self):
        report = self.audit([self.record(0), self.record(1)])
        self.assertEqual(len(report["candidate_split"]["groups"]), 2)

    def test_shared_mesh_calibration_and_global_gt_archive_do_not_join_sources(self):
        records = [self.record(0), self.record(1)]
        for role in ("mesh", "intrinsics", "hand_metric_gt", "object_pose_gt"):
            shared = copy.deepcopy(self.role(records[0], role))
            shared["identity_scope"] = "shared"
            self.replace_role(records[1], shared)
        report = self.audit(records)
        self.assertEqual(len(report["candidate_split"]["groups"]), 2)
        self.assertEqual(report["statistics"]["joint_pose_ready"]["independent_acquisitions"], 2)

    def test_per_sequence_raw_or_gt_content_does_join_sources(self):
        for role in ("hand_metric_gt", "object_pose_gt", "raw_sequence"):
            records = [self.record(0), self.record(1)]
            shared = self.asset(0, role)
            shared["identity_scope"] = "per_sequence"
            self.replace_role(records[0], copy.deepcopy(shared))
            self.replace_role(records[1], copy.deepcopy(shared))
            report = self.audit(records)
            self.assertEqual(len(report["candidate_split"]["groups"]), 1, role)

    def test_subjects_are_whole_and_deterministic_under_ledger_reordering(self):
        records = [self.record(number, subject=f"person_{number // 2}") for number in range(40)]
        first = self.audit(records)
        random.Random(99).shuffle(records)
        second = self.audit(records)
        self.assertEqual(self.assignments(first), self.assignments(second))
        assignments = self.assignments(first)
        for number in range(0, 40, 2):
            self.assertEqual(assignments[f"seq_{number}"], assignments[f"seq_{number + 1}"])
        self.assertEqual(first["actual_ratios"], dataset.RATIOS)

    def test_aliases_of_one_subject_do_not_inflate_subject_minimum(self):
        records = [self.record(0), self.record(1)]
        for record in records:
            record["subject_aliases"] = ["global_person_1"]
        report = self.audit(records)
        self.assertEqual(report["statistics"]["release_eligible"]["subjects"], 1)

    def test_unknown_subject_quarantines_related_views_not_counted(self):
        records = [self.record(0), self.record(1, subject_id=None)]
        records[1]["acquisition_id"] = records[0]["acquisition_id"]
        report = self.audit(records)
        self.assertEqual(set(self.assignments(report).values()), {"candidate"})
        self.assertEqual(report["statistics"]["release_eligible"]["subjects"], 0)
        self.assertEqual(report["statistics"]["candidate"]["subjects"], 1)

    def test_subject_without_evidence_is_not_counted_or_split(self):
        record = self.record()
        del record["evidence"]["subject_id"]
        report = self.audit([record])
        self.assertEqual(self.assignments(report)["seq_0"], "candidate")
        self.assertEqual(report["statistics"]["candidate"]["subjects"], 0)

    def test_previously_evaluated_and_old_test_taint_whole_subject_against_new_test(self):
        records = [self.record(number, subject=f"person_{number // 2}") for number in range(40)]
        records[0]["previously_evaluated"] = True
        records[2]["split"] = "test"
        records[3]["split"] = "test"
        records[4]["previous_split"] = "test"
        report = self.audit(records)
        assignment = self.assignments(report)
        for number in range(6):
            self.assertNotEqual(assignment[f"seq_{number}"], "test")
        self.assertGreater(report["splits"]["test"]["independent_acquisitions"], 0)

    def test_all_previously_evaluated_cannot_create_unseen_test(self):
        report = self.audit([self.record(number, previously_evaluated=True) for number in range(20)])
        self.assertEqual(report["splits"]["test"]["independent_acquisitions"], 0)
        self.assertFalse(report["readiness_passed"])

    def test_small_dataset_freezes_candidate_report_but_no_accepted_training_manifest(self):
        report = self.freeze([self.record()], config={})
        root = self.root / "frozen"
        self.assertTrue((root / "candidate_split.json").is_file())
        self.assertTrue((root / "audit_report.json").is_file())
        self.assertFalse((root / "training_manifest.json").exists())
        self.assertFalse(report["training_allowed"])
        self.assertTrue(report["integrity_verified"])

    def test_fixture_threshold_pass_cannot_authorize_real_training(self):
        report = self.freeze([self.record(number) for number in range(20)])
        self.assertTrue(report["readiness_passed"])
        self.assertFalse(report["training_allowed"])
        self.assertFalse((self.root / "frozen/training_manifest.json").exists())

    def test_lower_production_thresholds_or_contract_weakening_are_blocked(self):
        records = [self.record(number) for number in range(20)]
        config = {"minimums": {key: 1 for key in dataset.MINIMUMS}}
        report = self.freeze(records, config=config)
        self.assertIn("production_minimum_lowered", {error["code"] for error in report["errors"]})
        self.assertFalse(report["training_allowed"])
        report = self.audit(records, {"required_roles": ["rgb"]})
        self.assertIn("config_contract_changed", {error["code"] for error in report["errors"]})

    def test_production_floor_and_group_ratio_pass_accepts_only_frozen_release(self):
        records = [self.record(number, subject=f"person_{number // 6}") for number in range(120)]
        before = self.audit(records, {})
        self.assertTrue(before["readiness_passed"])
        self.assertFalse(before["training_allowed"])
        report = self.freeze(records, config={})
        self.assertTrue(report["training_allowed"], report["errors"])
        self.assertEqual(report["actual_ratios"], dataset.RATIOS)
        accepted = json.loads((self.root / "frozen/training_manifest.json").read_text())
        self.assertEqual([len(accepted["splits"][split]) for split in dataset.SPLITS], [84, 18, 18])
        self.assertTrue(accepted["split_before_windows"])
        self.assertEqual(accepted["normalization_fit_split"], "train")
        self.assertIsNone(report["accuracy"]["joint"])

    def test_split_ratios_are_reported_not_forced_by_breaking_subject_groups(self):
        records = [self.record(number, subject=f"person_{number // 5}") for number in range(30)]
        report = self.audit(records)
        self.assertNotEqual(report["actual_ratios"], dataset.RATIOS)
        assignment = self.assignments(report)
        for person in range(6):
            self.assertEqual(len({assignment[f"seq_{person * 5 + offset}"] for offset in range(5)}), 1)

    def test_unseen_actions_and_objects_reported_by_split(self):
        report = self.audit([self.record(number, object_ids=[f"obj{number}"], action_labels=[f"act{number}"])
                             for number in range(20)])
        for split in ("val", "test"):
            self.assertEqual(report["unseen_categories"][split]["objects"], report["splits"][split]["object_ids"])
            self.assertEqual(report["unseen_categories"][split]["actions"], report["splits"][split]["action_labels"])

    def test_existing_freeze_destination_even_empty_or_symlink_is_refused(self):
        self.write([self.record()])
        for name, symlink in (("existing", False), ("dangling", True)):
            target = self.root / name
            if symlink:
                target.symlink_to(self.root / "missing")
            else:
                target.mkdir()
            with self.assertRaises(FileExistsError):
                dataset.freeze_manifest(self.ledger, target, self.fixture_config)

    def test_asset_tampering_invalidates_frozen_gate_and_integrity(self):
        record = self.record()
        self.freeze([record])
        path = self.root / self.role(record, "rgb")["path"]
        path.write_bytes(b"tampered payload")
        report = dataset.verify_freeze(self.root / "frozen")
        self.assertFalse(report["integrity_verified"])
        self.assertFalse(report["training_allowed"])
        self.assertIn("asset_sha256_mismatch", self.codes(report))

    def test_frozen_manifest_config_and_report_tampering_detected(self):
        records = [self.record()]
        for number, filename in enumerate(("candidate_split.json", "source_ledger.json", "config.json", "audit_report.json")):
            with self.subTest(filename=filename):
                destination = self.root / f"frozen_{number}"
                self.freeze(records, name=destination.name)
                path = destination / filename
                path.write_bytes(path.read_bytes() + b" ")
                report = dataset.verify_freeze(destination)
                self.assertFalse(report["integrity_verified"])
                self.assertFalse(report["training_allowed"])

    def test_original_ledger_and_original_configuration_hashes_checked(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.fixture_config))
        self.freeze([self.record()], config=config_path)
        config_path.write_text(json.dumps({**self.fixture_config, "seed": 1}))
        self.assertFalse(dataset.verify_freeze(self.root / "frozen")["integrity_verified"])
        self.freeze([self.record(1)], name="other")
        self.ledger.write_text(self.ledger.read_text() + " ")
        self.assertFalse(dataset.verify_freeze(self.root / "other")["integrity_verified"])

    def test_unexpected_accepted_manifest_and_trusted_digest_mismatch_rejected(self):
        report = self.freeze([self.record()])
        root = self.root / "frozen"
        self.assertTrue(dataset.verify_freeze(root, report["freeze_sha256"])["integrity_verified"])
        self.assertFalse(dataset.verify_freeze(root, "0" * 64)["integrity_verified"])
        (root / "training_manifest.json").write_text('{"training_allowed":true}')
        self.assertFalse(dataset.verify_freeze(root)["integrity_verified"])

    def test_resealing_split_hash_does_not_bypass_recomputation(self):
        self.freeze([self.record(number) for number in range(20)])
        root = self.root / "frozen"
        path = root / "candidate_split.json"
        plan = json.loads(path.read_text())
        plan["records"][0]["split"] = "candidate"
        path.write_text(json.dumps(plan))
        seal = json.loads((root / "freeze.json").read_text())
        seal["files"]["candidate_split.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (root / "freeze.json").write_text(json.dumps(seal))
        report = dataset.verify_freeze(root)
        self.assertFalse(report["integrity_verified"])
        self.assertIn("split_changed", {error["code"] for error in report["integrity_errors"]})

    def test_official_benchmark_thresholds_are_not_dataset_accuracy(self):
        config = {**self.fixture_config, "official_benchmark": {"hand_mm": 20, "object_add": 0.1, "accuracy": 1.0}}
        record = self.record()
        record["official_standard"] = {"accuracy": 1.0}
        report = self.audit([record], config)
        self.assertEqual(set(report["accuracy"].values()), {None})
        self.assertEqual(report["threshold_source"], "project_release_gate_not_official_benchmark")
        self.assertFalse(report["training_allowed"])

    def test_cli_audit_and_freeze_exit_codes_and_no_overwrite(self):
        self.write([self.record()])
        report_path = self.root / "report.json"
        command = [sys.executable, "-S", str(MODULE), "audit", str(self.ledger), "--output", str(report_path)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        destination = self.root / "cli_frozen"
        command = [sys.executable, "-S", str(MODULE), "freeze", str(self.ledger), "--output", str(destination)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        result = subprocess.run([sys.executable, "-S", str(MODULE), "audit", str(destination)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertTrue(json.loads(result.stdout)["integrity_verified"])

    def test_invalid_top_level_and_duplicate_ids_are_not_silently_accepted(self):
        self.ledger.write_text('{"records":{}}')
        with self.assertRaises(dataset.DatasetError):
            dataset.audit_manifest(self.ledger)
        records = [self.record(0), self.record(1, sequence_id="seq_0")]
        report = self.audit(records)
        self.assertIn("duplicate_sequence_id", {error["code"] for error in report["errors"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
