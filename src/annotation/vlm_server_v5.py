"""Shared batched Qwen2.5-VL inference server for the auto-review pipeline.

One copy of the model serves N concurrent workers: incoming requests are
queued, and a batcher thread groups them (up to MAX_BATCH or MAX_WAIT
seconds) into a single padded generate() call.  Stdlib only.

Usage:
  python -m src.annotation.vlm_server_v5 --model /path/to/Qwen2.5-VL-7B --port 8100
Client contract (auto_review_v5 --vlm-url http://127.0.0.1:8100):
  POST /ask {"image_path": ..., "question": ..., "max_tokens": ...} -> {"text": ...}
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BATCH = 6
MAX_WAIT = 0.3

MODEL = None
PROCESSOR = None
REQUESTS = queue.Queue()


def batcher_loop():
    import torch
    from PIL import Image
    while True:
        first = REQUESTS.get()
        batch = [first]
        deadline = time.monotonic() + MAX_WAIT
        while len(batch) < MAX_BATCH:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(REQUESTS.get(timeout=remaining))
            except queue.Empty:
                break
        texts, images, requests = [], [], []
        for item in batch:
            try:
                image = Image.open(item["image_path"]).convert("RGB")
                messages = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": item["question"]}]}]
                text = PROCESSOR.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
                texts.append(text); images.append(image); requests.append(item)
            except Exception as error:  # noqa: BLE001 - report per request
                item["reply"]["error"] = str(error)
                item["done"].set()
        if not requests:
            continue
        try:
            inputs = PROCESSOR(text=texts, images=images, return_tensors="pt", padding=True).to(MODEL.device)
            with torch.inference_mode():
                out = MODEL.generate(**inputs, max_new_tokens=max(
                    r.get("max_tokens", 512) for r in requests), do_sample=False)
            decoded = PROCESSOR.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                             skip_special_tokens=True)
            for item, text in zip(requests, decoded):
                item["reply"]["text"] = text.strip()
        except Exception as error:  # noqa: BLE001 - report per request
            for item in requests:
                item["reply"]["error"] = str(error)
        for item in batch:
            item["done"].set()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - stdlib naming
        if self.path != "/ask":
            self.send_response(404); self.end_headers(); return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        item = {"image_path": body["image_path"], "question": body["question"],
                "max_tokens": int(body.get("max_tokens", 512)),
                "reply": {}, "done": threading.Event()}
        REQUESTS.put(item)
        item["done"].wait(900)
        payload = json.dumps(item["reply"]).encode()
        self.send_response(200 if "text" in item["reply"] else 500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep stdout quiet
        return


def main():
    global MODEL, PROCESSOR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/root/gpufree-data/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device)
    PROCESSOR = AutoProcessor.from_pretrained(args.model)
    threading.Thread(target=batcher_loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[vlm-server] listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
