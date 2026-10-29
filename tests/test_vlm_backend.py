from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from src.annotation import schema
from src.annotation import vlm_annotate as vlm
from src.annotation.qwen_backend import GenerationResult, QwenBackend


ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.prompt = None
        self.evidence = None
        self.calls = 0

    @property
    def model_metadata(self):
        return {"model_id": "fake-test-double", "loaded": False}

    def generate(self, prompt, evidence):
        self.calls += 1
        self.prompt = prompt
        self.evidence = evidence
        return GenerationResult(
            raw_text=self.raw_text,
            elapsed_seconds=0.125,
            input_tokens=100,
            output_tokens=20,
            peak_gpu_memory_bytes=None,
        )


class AnnotationTestCase(unittest.TestCase):
    def setUp(self):
        self.timeline = vlm.VideoTimeline(
            pts_seconds=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
            duration_seconds=2.0,
            stream={"codec_name": "fake", "avg_frame_rate": "0/0"},
            format_duration_seconds=2.0,
            duration_source="test_pts",
        )
        self.indices = [0, 2, 4, 7]
        self.evidence = [
            {"frame_index": index, "pts_seconds": self.timeline.pts_seconds[index]}
            for index in self.indices
        ]

    def event(self, **changes):
        item = {
            "subject": "left",
            "object_hint": "small red plastic box on the table",
            "interaction_type": "grasp",
            "start_sec": 0.25,
            "end_sec": 1.25,
            "evidence_times_sec": [0.5, 1.0],
            "confidence": 0.7,
            "notes": "rough boundary; fingers partly occluded",
            "source_event_ids": [],
        }
        item.update(changes)
        return item

    def parse(self, payload, source_events=None):
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        return vlm.parse_model_response(
            raw, "sequence-a", "run-a", self.timeline, self.evidence, source_events
        )

    def extract_fake(self, video, timeline, indices, evidence_dir, max_pixels):
        records = []
        for index in indices:
            data = b"fake-jpeg-" + str(index).encode("ascii")
            path = evidence_dir / f"frame_{index:08d}.jpg"
            path.write_bytes(data)
            records.append({
                "frame_index": index,
                "pts_seconds": timeline.pts_seconds[index],
                "path": f"evidence/{path.name}",
                "image_path": str(path.resolve()),
                "sha256": hashlib.sha256(data).hexdigest(),
                "sha256_basis": "encoded_jpeg",
                "original_width": 8,
                "original_height": 8,
                "input_width": 8,
                "input_height": 8,
            })
        return records

    def run_fake(self, root, raw_text, source_events=None):
        video = root / "input.mp4"
        video.write_bytes(b"fake video bytes")
        backend = FakeBackend(raw_text)
        output = root / "new-run"
        with mock.patch.object(vlm, "probe_video", return_value=self.timeline), \
                mock.patch.object(vlm, "extract_evidence_frames", side_effect=self.extract_fake):
            manifest = vlm.run_annotation(
                video=video,
                sequence_id="sequence-a",
                output_dir=output,
                num_frames=4,
                max_new_tokens=128,
                seed=7,
                source_events=source_events,
                backend=backend,
                ffprobe_path="fake-ffprobe",
            )
        return output, backend, manifest


class ImportAndBackendTests(AnnotationTestCase):
    def test_import_and_cli_help_do_not_load_heavy_dependencies(self):
        code = (
            "import sys; "
            "import src.annotation.schema; "
            "import src.annotation.qwen_backend; "
            "import src.annotation.vlm_annotate; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert 'cv2' not in sys.modules"
        )
        for command in ([sys.executable, "-c", code],
                        [sys.executable, "-m", "src.annotation.vlm_annotate", "--help"]):
            with self.subTest(command=command):
                completed = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True, check=False
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def write_model_directory(self, model_dir):
        contents = {
            "config.json": json.dumps({"_commit_hash": "disk-revision"}).encode("utf-8"),
            "model.safetensors": b"fake weights",
            "tokenizer.json": b"fake tokenizer",
        }
        for name, data in contents.items():
            (model_dir / name).write_bytes(data)
        manifest = {
            "model": "trusted/example-qwen",
            "model_revision": "manifest-revision",
            "files": [
                {
                    "Path": name,
                    "Size": len(data),
                    "Sha256": hashlib.sha256(data).hexdigest(),
                }
                for name, data in sorted(contents.items())
            ],
        }
        (model_dir / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def fake_model_modules(self):
        class FakeModel:
            def __init__(self):
                self.config = SimpleNamespace(_commit_hash="loaded-model-revision")
                self.dtype = "torch.bfloat16"
                self.hf_device_map = {"": "cpu"}
                self.device = SimpleNamespace(type="cpu", index=None)

            def eval(self):
                return self

            def parameters(self):
                return iter([SimpleNamespace(device=self.device)])

        fake_torch = ModuleType("torch")
        fake_torch.manual_seed = mock.Mock()
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: False,
            manual_seed_all=mock.Mock(),
        )
        fake_transformers = ModuleType("transformers")
        processor = object()
        fake_transformers.AutoProcessor = mock.Mock()
        fake_transformers.AutoProcessor.from_pretrained = mock.Mock(return_value=processor)
        model = FakeModel()
        fake_transformers.Qwen2_5_VLForConditionalGeneration = mock.Mock()
        fake_transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained = mock.Mock(
            return_value=model
        )
        return fake_torch, fake_transformers, processor

    def test_real_loader_verifies_source_and_uses_local_qwen_class_sdpa(self):
        fake_torch, fake_transformers, processor = self.fake_model_modules()
        loader = fake_transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            self.write_model_directory(model_dir)
            with mock.patch.dict(sys.modules, {
                "torch": fake_torch,
                "transformers": fake_transformers,
            }):
                backend = QwenBackend(str(model_dir), max_new_tokens=32, seed=5)
                backend.load()
                metadata = backend.model_metadata
        self.assertIs(backend._processor, processor)
        self.assertEqual(loader.call_args.kwargs["attn_implementation"], "sdpa")
        self.assertEqual(loader.call_args.kwargs["torch_dtype"], "auto")
        self.assertEqual(loader.call_args.kwargs["device_map"], "auto")
        self.assertTrue(loader.call_args.kwargs["local_files_only"])
        self.assertTrue(
            fake_transformers.AutoProcessor.from_pretrained.call_args.kwargs["local_files_only"]
        )
        self.assertNotIn("flash_attention_2", loader.call_args.kwargs.values())
        self.assertEqual(metadata["revision"], "manifest-revision")
        self.assertEqual(metadata["model_id"], "trusted/example-qwen")
        self.assertEqual(metadata["verification_status"], "verified")
        self.assertEqual(metadata["verified_file_count"], 3)
        self.assertEqual(len(metadata["weight_files"]), 1)
        weight = metadata["weight_files"][0]
        expected_sha256 = hashlib.sha256(b"fake weights").hexdigest()
        self.assertEqual(weight["size_bytes"], len(b"fake weights"))
        self.assertEqual(weight["expected_size_bytes"], len(b"fake weights"))
        self.assertEqual(weight["expected_sha256"], expected_sha256)
        self.assertEqual(weight["actual_sha256"], expected_sha256)
        self.assertEqual(weight["verification_status"], "verified")

    def test_source_identity_and_revision_are_required_before_loading(self):
        cases = {
            "missing_manifest": None,
            "missing_model": "model",
            "missing_revision": "model_revision",
        }
        for name, field in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                model_dir = Path(temporary)
                self.write_model_directory(model_dir)
                manifest_path = model_dir / "source_manifest.json"
                if field is None:
                    manifest_path.unlink()
                else:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest.pop(field)
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                fake_torch, fake_transformers, _ = self.fake_model_modules()
                with mock.patch.dict(sys.modules, {
                    "torch": fake_torch,
                    "transformers": fake_transformers,
                }), self.assertRaises(vlm.BackendError):
                    QwenBackend(str(model_dir)).load()
                fake_transformers.AutoProcessor.from_pretrained.assert_not_called()
                fake_transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained.assert_not_called()

    def test_missing_and_tampered_source_files_fail_before_loading(self):
        for name in ("missing_weight", "size_mismatch", "sha_mismatch", "tampered_non_weight"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                model_dir = Path(temporary)
                self.write_model_directory(model_dir)
                if name == "missing_weight":
                    (model_dir / "model.safetensors").unlink()
                elif name == "size_mismatch":
                    (model_dir / "model.safetensors").write_bytes(b"fake weights!")
                elif name == "sha_mismatch":
                    (model_dir / "model.safetensors").write_bytes(b"FAKE weights")
                else:
                    config_path = model_dir / "config.json"
                    config_path.write_bytes(config_path.read_bytes().replace(b"disk", b"DISK"))
                fake_torch, fake_transformers, _ = self.fake_model_modules()
                with mock.patch.dict(sys.modules, {
                    "torch": fake_torch,
                    "transformers": fake_transformers,
                }), self.assertRaises(vlm.BackendError):
                    QwenBackend(str(model_dir)).load()
                fake_transformers.AutoProcessor.from_pretrained.assert_not_called()
                fake_transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained.assert_not_called()


class TimelineAndValidationTests(AnnotationTestCase):
    def test_ffprobe_pts_define_timeline_without_default_fps(self):
        payload = {
            "streams": [{"codec_name": "h264", "avg_frame_rate": "0/0", "duration": "2.1"}],
            "format": {"duration": "9.0"},
            "frames": [
                {"best_effort_timestamp_time": "0.4", "pkt_duration_time": "0.4"},
                {"best_effort_timestamp_time": "0.0", "pkt_duration_time": "0.4"},
                {"best_effort_timestamp_time": "1.7", "pkt_duration_time": "0.4"},
                {"best_effort_timestamp_time": "0.9", "pkt_duration_time": "0.8"},
            ],
        }
        completed = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(vlm.subprocess, "run", return_value=completed) as run:
            timeline = vlm.probe_video(Path(temporary) / "video.mp4", "ffprobe-test")
        self.assertEqual(timeline.pts_seconds, (0.0, 0.4, 0.9, 1.7))
        self.assertEqual(timeline.duration_seconds, 2.1)
        command = run.call_args.args[0]
        self.assertIn("-show_frames", command)
        self.assertIn("best_effort_timestamp_time", " ".join(command))
        self.assertNotIn("30", command)

    def test_sampling_is_bounded_unique_and_rejects_short_video(self):
        self.assertEqual(vlm.sample_frame_indices(8, 4), [0, 2, 4, 7])
        self.assertEqual(vlm.sample_frame_indices(4, 4), [0, 1, 2, 3])
        for count in (0, 3, 33, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                vlm.sample_frame_indices(100, count)
        with self.assertRaisesRegex(vlm.AnnotationError, "too short"):
            vlm.sample_frame_indices(3, 4)

    def test_seconds_convert_to_half_open_native_frame_bounds(self):
        response, events, none_intervals = self.parse({
            "events": [self.event()],
            "none_intervals": [],
        })
        self.assertEqual(response["status"], "valid", response["errors"])
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["start_frame"], events[0]["end_frame"]), (1, 5))
        self.assertEqual(events[0]["evidence_frames"], [2, 4])
        self.assertEqual(none_intervals, [])
        schema.validate_event(events[0], frame_count=8)

        final_frame_event = self.event(
            start_sec=1.75, end_sec=2.0, evidence_times_sec=[1.75]
        )
        response, events, _ = self.parse({"events": [final_frame_event], "none_intervals": []})
        self.assertEqual(response["status"], "valid", response["errors"])
        self.assertEqual((events[0]["start_frame"], events[0]["end_frame"]), (7, 8))

    def test_invalid_json_categories_bounds_and_evidence_are_rejected(self):
        cases = {
            "malformed": "```json\n{}\n```",
            "duplicate_keys": '{"events": [], "events": [], "none_intervals": []}',
            "unsupported_category": {"events": [self.event(interaction_type="none")], "none_intervals": []},
            "unknown_subject": {"events": [self.event(subject="both")], "none_intervals": []},
            "out_of_range": {"events": [self.event(end_sec=2.1)], "none_intervals": []},
            "empty_interval": {"events": [self.event(start_sec=1.0, end_sec=1.0)], "none_intervals": []},
            "missing_evidence": {"events": [self.event(evidence_times_sec=[])], "none_intervals": []},
            "unsampled_evidence": {"events": [self.event(evidence_times_sec=[0.36])], "none_intervals": []},
            "evidence_outside_event": {
                "events": [self.event(start_sec=0.75, evidence_times_sec=[0.5])],
                "none_intervals": [],
            },
            "accepted_from_model": {
                "events": [self.event(review_status="accepted")],
                "none_intervals": [],
            },
            "stable_id_from_model": {
                "events": [self.event(object_id="obj_000")],
                "none_intervals": [],
            },
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                raw = payload if isinstance(payload, str) else json.dumps(payload)
                response, events, none_intervals = self.parse(raw)
                self.assertEqual(response["status"], "invalid")
                self.assertTrue(response["errors"])
                self.assertEqual(response["raw_text"], raw)
                self.assertEqual(events, [])
                self.assertEqual(none_intervals, [])

    def test_evidence_time_snapping_is_bounded_and_audited(self):
        raw = json.dumps({
            "events": [self.event(evidence_times_sec=[0.51, 0.99])],
            "none_intervals": [],
        })
        response, events, _ = self.parse(raw)
        self.assertEqual(response["status"], "valid", response["errors"])
        self.assertEqual(events[0]["evidence_frames"], [2, 4])
        self.assertEqual(events[0]["review_status"], "pending")
        self.assertEqual(response["raw_text"], raw)
        corrections = response["evidence_time_corrections"]
        self.assertEqual(len(corrections), 2)
        self.assertEqual(corrections[0]["requested_time_seconds"], 0.51)
        self.assertEqual(corrections[0]["snapped_pts_seconds"], 0.5)
        self.assertAlmostEqual(corrections[0]["absolute_error_seconds"], 0.01)

        response, events, _ = self.parse({
            "events": [self.event(evidence_times_sec=[0.49, 0.51])],
            "none_intervals": [],
        })
        self.assertEqual(response["status"], "invalid")
        self.assertIn("unique native frames", response["errors"][0])
        self.assertEqual(events, [])
        self.assertEqual(response["evidence_time_corrections"], [])

    def test_boundary_rounding_snaps_to_sampled_pts_and_is_audited(self):
        # Reproduces the 20261022 run03 failure: the model rounded 18.066667 to
        # 18.1, which pushed start_frame past the evidence frame and rejected
        # the candidate. Rounding within the snap tolerance now snaps back.
        rounded = self.event(start_sec=0.51, end_sec=0.99, evidence_times_sec=[0.5])
        response, events, _ = self.parse({"events": [rounded], "none_intervals": []})
        self.assertEqual(response["status"], "valid", response["errors"])
        self.assertEqual((events[0]["start_frame"], events[0]["end_frame"]), (2, 4))
        self.assertEqual(events[0]["evidence_frames"], [2])
        corrections = response["boundary_time_corrections"]
        self.assertEqual(
            [item["field"] for item in corrections], ["start_sec", "end_sec"]
        )
        self.assertEqual(corrections[0]["requested_time_seconds"], 0.51)
        self.assertEqual(corrections[0]["snapped_pts_seconds"], 0.5)
        self.assertEqual(corrections[0]["snapped_frame_index"], 2)
        self.assertEqual(corrections[1]["requested_time_seconds"], 0.99)
        self.assertEqual(corrections[1]["snapped_pts_seconds"], 1.0)
        self.assertEqual(corrections[1]["snapped_frame_index"], 4)

        exact = self.event(start_sec=0.25, end_sec=1.25, evidence_times_sec=[0.5, 1.0])
        response, events, _ = self.parse({"events": [exact], "none_intervals": []})
        self.assertEqual(response["status"], "valid", response["errors"])
        self.assertEqual((events[0]["start_frame"], events[0]["end_frame"]), (1, 5))
        self.assertEqual(response["boundary_time_corrections"], [])

    def test_boundary_between_samples_is_not_snapped(self):
        between = self.event(start_sec=0.35, end_sec=1.35, evidence_times_sec=[0.5, 1.0])
        response, events, _ = self.parse({"events": [between], "none_intervals": []})
        self.assertEqual(response["status"], "valid", response["errors"])
        self.assertEqual((events[0]["start_frame"], events[0]["end_frame"]), (2, 6))
        self.assertEqual(response["boundary_time_corrections"], [])

        # A boundary rounding that would collapse the interval is still rejected.
        collapsed = self.event(start_sec=0.51, end_sec=0.53, evidence_times_sec=[0.0])
        response, events, _ = self.parse({"events": [collapsed], "none_intervals": []})
        self.assertEqual(response["status"], "invalid")
        self.assertTrue(response["errors"])

    def test_prompt_requires_merged_continuous_events(self):
        prompt = vlm.build_prompt("sequence-a", self.timeline, self.evidence, None)
        self.assertIn("Report each continuous interaction phase as ONE event", prompt)
        self.assertIn("Never emit one event per evidence image", prompt)
        self.assertIn("end_sec is exclusive", prompt)
        self.assertIn("under 20 words", prompt)
        self.assertIn('"evidence_times_sec": [0.0, 3.6, 7.2, 10.833333]', prompt)

    def test_end_sec_cited_as_evidence_extends_half_frame_and_is_audited(self):
        # Reproduces the 20261023 first rerun: the model listed both boundary
        # PTS values as evidence, so the exclusive end frame fell outside.
        cited = self.event(start_sec=0.0, end_sec=1.0, evidence_times_sec=[0.0, 1.0])
        response, events, _ = self.parse({"events": [cited], "none_intervals": []})
        self.assertEqual(response["status"], "valid", response["errors"])
        # min native gap is 0.25 s, so the end moves half a frame to 1.125.
        self.assertEqual((events[0]["start_frame"], events[0]["end_frame"]), (0, 5))
        self.assertEqual(events[0]["evidence_frames"], [0, 4])
        corrections = response["boundary_time_corrections"]
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["field"], "end_sec")
        self.assertEqual(corrections[0]["requested_time_seconds"], 1.0)
        self.assertAlmostEqual(corrections[0]["snapped_pts_seconds"], 1.125)
        self.assertIn("extended past the exclusive boundary", corrections[0]["reason"])

        # Evidence strictly outside the interval is still rejected outright.
        outside = self.event(start_sec=0.0, end_sec=1.0, evidence_times_sec=[0.5, 1.75])
        response, events, _ = self.parse({"events": [outside], "none_intervals": []})
        self.assertEqual(response["status"], "invalid")
        self.assertTrue(response["errors"])

    def test_partial_response_keeps_only_structurally_valid_pending_candidates(self):
        valid = self.event()
        invalid = self.event(interaction_type="touch")
        response, events, none_intervals = self.parse({
            "events": [valid, invalid], "none_intervals": [],
        })
        self.assertEqual(response["status"], "partial")
        self.assertEqual(response["candidate_counts"]["events"], 2)
        self.assertEqual(response["rejected_candidate_count"], 1)
        self.assertEqual(len(response["errors"]), 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["review_status"], "pending")
        self.assertEqual(events[0]["interaction_type"], "grasp")
        self.assertEqual(none_intervals, [])

    def test_overlapping_events_and_explicit_none_remain_pending(self):
        second = self.event(
            subject="right", interaction_type="hold", start_sec=0.5, end_sec=1.75
        )
        none = {
            "start_sec": 1.5,
            "end_sec": 2.0,
            "evidence_times_sec": [1.75],
            "confidence": 0.4,
            "notes": "explicit candidate only; requires review",
            "source_event_ids": [],
        }
        response, events, none_intervals = self.parse({
            "events": [self.event(), second], "none_intervals": [none]
        })
        self.assertEqual(response["status"], "valid", response["errors"])
        self.assertEqual(len(events), 2)
        self.assertLess(events[0]["start_frame"], events[1]["end_frame"])
        self.assertLess(events[1]["start_frame"], events[0]["end_frame"])
        for record in events + none_intervals:
            self.assertEqual(record["review_status"], "pending")
            self.assertEqual(record["geometry_status"], "not_checked")
            self.assertEqual(record["annotation_mode"], "offline_noncausal")
            self.assertFalse(record["allowed_as_model_input"])
        for event in events:
            self.assertIsNone(event["object_id"])
            self.assertEqual(event["label_source"], "vlm")

    def test_source_text_is_not_automatically_mapped_to_labels(self):
        source = {"subtask_annotations": [{
            "subtask_id": "source-1", "description": "grasp the box",
            "interaction_type": "grasp", "subject": "left",
            "start_time": 0.0, "end_time": 2.0,
        }]}
        response, events, none_intervals = self.parse(
            {"events": [], "none_intervals": []}, source_events=source
        )
        self.assertEqual(response["status"], "valid")
        self.assertEqual(events, [])
        self.assertEqual(none_intervals, [])
        prompt = vlm.build_prompt("sequence-a", self.timeline, self.evidence, source)
        self.assertIn("Do not copy or map source descriptions", prompt)
        self.assertIn("untrusted coarse time context", prompt)


class FakeTensor:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.shape = (1, len(self.tokens))

    def __getitem__(self, key):
        rows, columns = key
        assert rows == slice(None)
        return FakeTensor(self.tokens[columns])


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeRoundProcessor:
    header = "<|im_start|>assistant\n"

    def __init__(self):
        self.texts = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<|im_start|>user\nprompt<|im_end|>\n" + self.header

    def __call__(self, text, images=None, videos=None, padding=True, return_tensors="pt"):
        self.texts.append(text[0])
        return FakeInputs({"input_ids": FakeTensor(text[0])})

    def batch_decode(self, trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ["".join(trimmed.tokens)]


class FakeRoundModel:
    def __init__(self, rounds):
        self.rounds = [list(tokens) for tokens in rounds]
        self.max_new_tokens_calls = []

    def generate(self, input_ids=None, max_new_tokens=None, **kwargs):
        self.max_new_tokens_calls.append(max_new_tokens)
        return FakeTensor(input_ids.tokens + self.rounds.pop(0))


class TokenCapContinuationTests(AnnotationTestCase):
    def make_backend(self, model, processor, max_new_tokens=8, max_continuation_rounds=2):
        backend = QwenBackend(
            max_new_tokens=max_new_tokens, max_continuation_rounds=max_continuation_rounds
        )
        backend._torch = SimpleNamespace(
            inference_mode=lambda: contextlib.nullcontext(),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        backend._model = model
        backend._processor = processor
        backend._input_device = SimpleNamespace(type="cpu", index=None)
        return backend

    def generate(self, backend, root):
        image = root / "frame.jpg"
        image.write_bytes(b"fake jpeg")
        fake_qwen_vl_utils = ModuleType("qwen_vl_utils")
        fake_qwen_vl_utils.process_vision_info = mock.Mock(return_value=(None, None))
        with mock.patch.dict(sys.modules, {"qwen_vl_utils": fake_qwen_vl_utils}):
            return backend.generate(
                "describe the interaction",
                [{"frame_index": 0, "pts_seconds": 0.0, "image_path": str(image)}],
            )

    def test_token_cap_triggers_one_continuation_until_natural_stop(self):
        model = FakeRoundModel(["ABCDEFGH", "IJKL}"])
        processor = FakeRoundProcessor()
        backend = self.make_backend(model, processor)
        with tempfile.TemporaryDirectory() as temporary:
            result = self.generate(backend, Path(temporary))
        self.assertEqual(result.raw_text, "ABCDEFGHIJKL}")
        self.assertEqual(result.output_tokens, 13)
        self.assertEqual(result.continuation_rounds, 1)
        self.assertEqual(result.token_cap_rounds, 1)
        self.assertFalse(result.hit_length_limit)
        self.assertEqual(len(processor.texts), 2)
        self.assertTrue(processor.texts[0].endswith(FakeRoundProcessor.header))
        self.assertEqual(processor.texts[1], processor.texts[0] + "ABCDEFGH")
        self.assertEqual(model.max_new_tokens_calls, [8, 8])

    def test_complete_looking_cap_output_does_not_continue(self):
        model = FakeRoundModel(["ABCDEFG}"])
        backend = self.make_backend(model, FakeRoundProcessor())
        with tempfile.TemporaryDirectory() as temporary:
            result = self.generate(backend, Path(temporary))
        self.assertEqual(result.raw_text, "ABCDEFG}")
        self.assertEqual(result.continuation_rounds, 0)
        self.assertEqual(result.token_cap_rounds, 1)
        self.assertFalse(result.hit_length_limit)
        self.assertEqual(model.max_new_tokens_calls, [8])

    def test_exhausted_rounds_report_possible_truncation(self):
        model = FakeRoundModel(["AAAAAAAA", "BBBBBBBB"])
        backend = self.make_backend(model, FakeRoundProcessor(), max_continuation_rounds=1)
        with tempfile.TemporaryDirectory() as temporary:
            result = self.generate(backend, Path(temporary))
        self.assertEqual(result.raw_text, "AAAAAAAABBBBBBBB")
        self.assertEqual(result.continuation_rounds, 1)
        self.assertEqual(result.token_cap_rounds, 2)
        self.assertTrue(result.hit_length_limit)

    def test_natural_stop_never_continues(self):
        model = FakeRoundModel(["{}"])
        backend = self.make_backend(model, FakeRoundProcessor())
        with tempfile.TemporaryDirectory() as temporary:
            result = self.generate(backend, Path(temporary))
        self.assertEqual(result.raw_text, "{}")
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(result.continuation_rounds, 0)
        self.assertEqual(result.token_cap_rounds, 0)
        self.assertFalse(result.hit_length_limit)
        self.assertEqual(model.max_new_tokens_calls, [8])


class OrchestrationTests(AnnotationTestCase):
    def test_fake_inference_orchestration_writes_complete_noncausal_provenance(self):
        raw = json.dumps({"events": [self.event()], "none_intervals": []})
        with tempfile.TemporaryDirectory() as temporary:
            output, backend, manifest = self.run_fake(Path(temporary), raw)
            events = schema.read_jsonl(output / "events.jsonl", frame_count=8)
            response = json.loads((output / "responses.jsonl").read_text().strip())
            self.assertEqual(backend.calls, 1)
            self.assertEqual(len(backend.evidence), 4)
            self.assertEqual(manifest["response_status"], "valid")
            self.assertEqual(manifest["annotation_mode"], "offline_noncausal")
            self.assertFalse(manifest["allowed_as_model_input"])
            self.assertFalse(events[0]["allowed_as_model_input"])
            self.assertEqual(events[0]["review_status"], "pending")
            self.assertIsNone(events[0]["object_id"])
            self.assertEqual(events[0]["geometry_status"], "not_checked")
            self.assertEqual(response["raw_text"], raw)
            self.assertEqual(response["status"], "valid")
            self.assertEqual(response["generation"]["output_tokens"], 20)
            self.assertEqual(len(manifest["input_frames"]), 4)
            self.assertTrue(all(frame["sha256"] for frame in manifest["input_frames"]))
            self.assertEqual(manifest["model"]["model_id"], "fake-test-double")
            self.assertEqual(manifest["parameters"]["seed"], 7)
            self.assertEqual(len(manifest["code"]), 3)
            self.assertTrue((output / "evidence" / "frame_00000000.jpg").is_file())
            self.assertTrue((output / "none_intervals.jsonl").is_file())

    def test_manifest_records_continuation_and_boundary_correction_metadata(self):
        raw = json.dumps({
            "events": [self.event(start_sec=0.51, end_sec=0.99, evidence_times_sec=[0.5])],
            "none_intervals": [],
        })
        with tempfile.TemporaryDirectory() as temporary:
            output, _, manifest = self.run_fake(Path(temporary), raw)
            response = json.loads((output / "responses.jsonl").read_text().strip())
        self.assertEqual(manifest["response_status"], "valid")
        self.assertEqual(manifest["parameters"]["max_new_tokens"], 128)
        self.assertEqual(manifest["parameters"]["max_continuation_rounds"], 2)
        self.assertEqual(manifest["outputs"]["boundary_time_correction_count"], 2)
        self.assertEqual(manifest["outputs"]["evidence_time_correction_count"], 0)
        self.assertEqual(response["generation"]["continuation_rounds"], 0)
        self.assertEqual(response["generation"]["token_cap_rounds"], 0)
        self.assertFalse(response["generation"]["hit_length_limit"])

    def test_invalid_model_text_is_preserved_and_not_converted(self):
        raw = '{"events": [{"interaction_type": "touch"}'
        with tempfile.TemporaryDirectory() as temporary:
            output, _, manifest = self.run_fake(Path(temporary), raw)
            response = json.loads((output / "responses.jsonl").read_text().strip())
            self.assertEqual(response["status"], "invalid")
            self.assertEqual(response["raw_text"], raw)
            self.assertTrue(response["errors"])
            self.assertEqual(manifest["response_status"], "invalid")
            self.assertEqual((output / "events.jsonl").read_text(), "")
            self.assertEqual((output / "none_intervals.jsonl").read_text(), "")

    def test_nonfinite_json_number_is_invalid_and_raw_response_is_saved(self):
        raw = json.dumps({"events": [self.event()], "none_intervals": []}).replace(
            '"confidence": 0.7', '"confidence": 1e999'
        )
        with tempfile.TemporaryDirectory() as temporary:
            output, _, manifest = self.run_fake(Path(temporary), raw)
            response = json.loads((output / "responses.jsonl").read_text().strip())
            self.assertEqual(manifest["response_status"], "invalid")
            self.assertEqual(response["status"], "invalid")
            self.assertEqual(response["raw_text"], raw)
            self.assertIn("finite", response["errors"][0])
            self.assertEqual((output / "events.jsonl").read_text(), "")

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("unchanged")
            backend = FakeBackend("{}")
            with self.assertRaises(FileExistsError):
                vlm.run_annotation(
                    video=root / "missing.mp4",
                    sequence_id="sequence-a",
                    output_dir=output,
                    backend=backend,
                    ffprobe_path="fake-ffprobe",
                )
            self.assertEqual(sentinel.read_text(), "unchanged")
            self.assertEqual(backend.calls, 0)


if __name__ == "__main__":
    unittest.main()
