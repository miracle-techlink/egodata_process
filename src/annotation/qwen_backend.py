from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Mapping, Optional, Sequence

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MODEL_PATH = "/root/gpufree-data/models/Qwen2.5-VL-7B-Instruct"
DEFAULT_MAX_PIXELS = 768 * 768
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_CONTINUATION_ROUNDS = 2
# The Qwen2.5 chat template ends a fully rendered generation prompt with this
# exact header; continuation relies on it to feed the model its own tokens back.
ASSISTANT_GENERATION_HEADER = "<|im_start|>assistant\n"


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    raw_text: str
    elapsed_seconds: float
    input_tokens: int
    output_tokens: int
    peak_gpu_memory_bytes: Optional[int]
    continuation_rounds: int = 0
    token_cap_rounds: int = 0
    hit_length_limit: bool = False


def package_versions() -> dict:
    versions = {}
    for distribution in (
        "torch",
        "transformers",
        "qwen-vl-utils",
        "accelerate",
        "opencv-python-headless",
        "decord",
    ):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def environment_metadata() -> dict:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def _read_source_manifest(model_path: Path) -> dict:
    manifest_path = model_path / "source_manifest.json"
    try:
        source = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackendError(f"trusted model source manifest is required: {manifest_path}: {exc}") from exc
    if not isinstance(source, dict):
        raise BackendError(f"model source manifest must be a JSON object: {manifest_path}")
    for field in ("model", "model_revision"):
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            raise BackendError(f"model source manifest requires a nonempty {field}")
    if not isinstance(source.get("files"), list) or not source["files"]:
        raise BackendError("model source manifest requires a nonempty files list")
    return source


def _size_and_sha256(path: Path) -> tuple:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(8 * 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise BackendError(f"cannot read model source file {path}: {exc}") from exc
    return size, digest.hexdigest()


def _verify_model_source(model_path: Path) -> dict:
    source = _read_source_manifest(model_path)
    seen = set()
    verified_files = []
    weight_files = []
    for item in source["files"]:
        if not isinstance(item, dict):
            raise BackendError("model source manifest file entries must be JSON objects")
        relative = item.get("Path")
        expected_size = item.get("Size")
        expected_sha256 = item.get("Sha256")
        if not isinstance(relative, str) or not relative:
            raise BackendError("model source manifest file entry requires a nonempty Path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
            raise BackendError(f"invalid model source file path: {relative}")
        relative = relative_path.as_posix()
        if relative in seen:
            raise BackendError(f"duplicate model source file in manifest: {relative}")
        seen.add(relative)
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise BackendError(f"invalid expected size for model source file: {relative}")
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or not all(character in "0123456789abcdefABCDEF" for character in expected_sha256)):
            raise BackendError(f"invalid expected SHA-256 for model source file: {relative}")
        try:
            path = (model_path / relative_path).resolve()
            path.relative_to(model_path)
        except (OSError, ValueError) as exc:
            raise BackendError(f"model source file escapes the model directory: {relative}") from exc
        if not path.is_file():
            raise BackendError(f"model source file is missing: {path}")
        actual_size, actual_sha256 = _size_and_sha256(path)
        if actual_size != expected_size:
            raise BackendError(
                f"model source file size mismatch for {relative}: "
                f"expected {expected_size}, got {actual_size}"
            )
        if actual_sha256 != expected_sha256.lower():
            raise BackendError(f"model source file SHA-256 mismatch for {relative}")
        record = {
            "path": str(path),
            "size_bytes": actual_size,
            "expected_size_bytes": expected_size,
            "expected_sha256": expected_sha256.lower(),
            "actual_sha256": actual_sha256,
            "verification_status": "verified",
        }
        verified_files.append(record)
        if relative_path.name.endswith(".safetensors") or (
                relative_path.name.startswith("pytorch_model") and relative_path.name.endswith(".bin")):
            weight_files.append(record)
    if not weight_files:
        raise BackendError("model source manifest contains no model weight files")
    weight_files.sort(key=lambda item: item["path"])
    return {
        "model_id": source["model"].strip(),
        "revision": source["model_revision"].strip(),
        "weight_files": weight_files,
        "verification_status": "verified",
        "verified_file_count": len(verified_files),
    }


class QwenBackend:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        seed: int = 0,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        max_continuation_rounds: int = DEFAULT_MAX_CONTINUATION_ROUNDS,
    ):
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels < 28 * 28:
            raise ValueError("max_pixels is too small")
        if (isinstance(max_continuation_rounds, bool) or not isinstance(max_continuation_rounds, int)
                or not 0 <= max_continuation_rounds <= 16):
            raise ValueError("max_continuation_rounds must be an integer in [0, 16]")
        self.model_path = Path(model_path).expanduser().resolve()
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.max_pixels = max_pixels
        self.max_continuation_rounds = max_continuation_rounds
        self._torch = None
        self._model = None
        self._processor = None
        self._input_device = None
        self._verification = None

    @property
    def model_metadata(self) -> dict:
        verification = self._verification or {}
        result = {
            "model_id": verification.get("model_id"),
            "model_path": str(self.model_path),
            "weight_directory": str(self.model_path),
            "revision": verification.get("revision"),
            "weight_files": [dict(item) for item in verification.get("weight_files", [])],
            "verification_status": verification.get("verification_status", "not_verified"),
            "verified_file_count": verification.get("verified_file_count", 0),
            "loaded": self._model is not None,
            "attention_implementation": "sdpa",
        }
        if self._model is not None:
            result["dtype"] = str(self._model.dtype).replace("torch.", "")
            device_map = getattr(self._model, "hf_device_map", None)
            if device_map is not None:
                result["device_map"] = {str(key): str(value) for key, value in device_map.items()}
            if self._input_device is not None:
                result["input_device"] = str(self._input_device)
                if self._input_device.type == "cuda":
                    index = self._input_device.index
                    result["gpu_name"] = self._torch.cuda.get_device_name(index)
                    result["gpu_capability"] = list(self._torch.cuda.get_device_capability(index))
                    result["cuda_version"] = self._torch.version.cuda
        return result

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise BackendError(f"local model directory not found: {self.model_path}")
        self._verification = _verify_model_source(self.model_path)
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise BackendError(
                "Qwen inference requires torch and transformers with Qwen2.5-VL support"
            ) from exc

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        try:
            processor = AutoProcessor.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(self.model_path),
                torch_dtype="auto",
                attn_implementation="sdpa",
                device_map="auto",
                local_files_only=True,
            )
        except Exception as exc:
            raise BackendError(f"failed to load local Qwen model: {exc}") from exc
        model.eval()
        self._torch = torch
        self._processor = processor
        self._model = model
        self._input_device = next(model.parameters()).device

    def generate(self, prompt: str, evidence_frames: Sequence[Mapping[str, object]]) -> GenerationResult:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not evidence_frames:
            raise ValueError("at least one evidence frame is required")
        self.load()
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise BackendError("Qwen inference requires qwen-vl-utils[decord]") from exc

        content = [{"type": "text", "text": prompt}]
        for position, frame in enumerate(evidence_frames, start=1):
            image_path = Path(str(frame["image_path"])).expanduser().resolve()
            if not image_path.is_file():
                raise BackendError(f"evidence image not found: {image_path}")
            content.append({
                "type": "text",
                "text": (
                    f"Evidence image {position}: native frame {frame['frame_index']}, "
                    f"PTS {float(frame['pts_seconds']):.9f} seconds."
                ),
            })
            content.append({
                "type": "image",
                "image": image_path.as_uri(),
                "max_pixels": self.max_pixels,
            })
        messages = [{"role": "user", "content": content}]
        base_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not base_text.endswith(ASSISTANT_GENERATION_HEADER):
            raise BackendError(
                "chat template does not end with the Qwen assistant generation header; "
                "token-cap continuation is unsafe and was refused"
            )
        image_inputs, video_inputs = process_vision_info(messages)

        torch = self._torch
        use_cuda = self._input_device.type == "cuda"
        if use_cuda:
            torch.cuda.synchronize(self._input_device)
            torch.cuda.reset_peak_memory_stats(self._input_device)
        pieces = []
        continuation_rounds = 0
        token_cap_rounds = 0
        hit_length_limit = False
        output_tokens = 0
        input_tokens = 0
        elapsed = 0.0
        started = time.perf_counter()
        try:
            for round_index in range(self.max_continuation_rounds + 1):
                # Continuation feeds the model its own previously generated tokens back
                # as the assistant turn; no synthetic text is ever injected.
                text = base_text + "".join(pieces)
                inputs = self._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self._input_device)
                input_tokens = int(inputs["input_ids"].shape[1])
                with torch.inference_mode():
                    generated = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                if use_cuda:
                    torch.cuda.synchronize(self._input_device)
                trimmed = generated[:, input_tokens:]
                new_text = self._processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                pieces.append(new_text)
                new_tokens = int(trimmed.shape[1])
                output_tokens += new_tokens
                elapsed = time.perf_counter() - started
                if new_tokens < self.max_new_tokens:
                    hit_length_limit = False
                    break
                token_cap_rounds += 1
                hit_length_limit = True
                if round_index >= self.max_continuation_rounds:
                    break
                if "".join(pieces).rstrip().endswith("}"):
                    # The output already looks complete; do not burn another round.
                    hit_length_limit = False
                    break
                continuation_rounds += 1
        finally:
            if use_cuda:
                torch.cuda.synchronize(self._input_device)
        peak = int(torch.cuda.max_memory_allocated(self._input_device)) if use_cuda else None
        return GenerationResult(
            raw_text="".join(pieces),
            elapsed_seconds=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            peak_gpu_memory_bytes=peak,
            continuation_rounds=continuation_rounds,
            token_cap_rounds=token_cap_rounds,
            hit_length_limit=hit_length_limit,
        )
