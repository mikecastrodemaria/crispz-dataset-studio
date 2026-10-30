"""A fake crispz family engine for the CPU tests: the Gradio call API
(/gradio_api/call/<name> then /gradio_api/call/<name>/<event_id>) with cli_caps,
cli_edit, cli_gen and cli_faces, no torch, no model.

Deterministic "identity": the embedding of an image is a 512-vector derived from its
mean colour, so a card rendered from the reference (same colour) scores high and a
prompt containing the word DRIFT (a different colour) scores low. A prompt containing
FAIL makes cli_edit answer {ok: false}. The cards are 64 px PNGs.

Run standalone (the launcher test starts it as an "app.py"):
    GRADIO_SERVER_PORT=7899 python fake_engine.py
"""

import os
import io
import re
import sys
import json
import time
import base64
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, PngImagePlugin

TOOL = "fake-klein"
VERSION = "0.0.1"
_EVENTS = {}
_LOCK = threading.Lock()
STATE = {"edits": 0, "faces": 0, "delay": 0.0}


def _mean_color(img):
    from PIL import ImageStat
    return tuple(ImageStat.Stat(img.convert("RGB")).mean)


def _embedding(img):
    r, g, b = _mean_color(img)
    base = [r / 255.0, g / 255.0, b / 255.0]
    vec = []
    for i in range(512):
        vec.append(base[i % 3] + 0.001 * ((i * 7919) % 13))
    return vec


def caps():
    return {"ok": True, "protocol": 1, "tool": TOOL, "version": VERSION, "ops": ["caps", "gen", "upscale", "edit", "inpaint"],
            "supports": {"loras": True, "refs": True, "max_refs": 4, "seed": True, "negative": False,
                         "arbitrary_size": True, "faces": True, "detail_faces": True, "detail_hands": False,
                         "edit": True, "edit_presets": [], "inpaint": True},
            "model_loaded": STATE["edits"] > 0, "models": [], "loras": [], "edit_loras": []}


def edit(spec_json):
    spec = json.loads(spec_json)
    STATE["edits"] += 1
    if STATE["delay"]:
        time.sleep(STATE["delay"])
    prompt = spec.get("prompt") or ""
    if "FAIL" in prompt:
        return {"ok": False, "protocol": 1, "tool": TOOL, "error": "fake refusal: FAIL in prompt", "exit_code": 1}
    inp = spec.get("input")
    if not inp or not os.path.isfile(inp):
        return {"ok": False, "protocol": 1, "tool": TOOL, "error": f"input not found on disk: {inp}", "exit_code": 2}
    ref = Image.open(inp).convert("RGB")
    color = tuple(int(c) for c in _mean_color(ref))
    if "DRIFT" in prompt:
        color = (255 - color[0], 255 - color[1], 255 - color[2])
    seed = int(spec.get("seed", -1))
    if seed < 0:
        seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
    # a small seed-bound jitter so two tries differ (and so a DRIFT re-roll stays bad)
    j = seed % 7
    color = tuple(max(0, min(255, c + j)) for c in color)
    # No size in the spec -> the input's own size, as the real engines do (qwen-edit
    # edits in place, klein renders at the reference's size).
    w, h = int(spec.get("width") or ref.size[0]), int(spec.get("height") or ref.size[1])
    w, h = max(8, min(w, 128)), max(8, min(h, 128))
    img = Image.new("RGB", (w, h), color)
    if "COPY" in prompt:
        img = ref.resize((w, h))         # the input handed back: the copy check's case
    out_dir = spec.get("out_dir") or os.path.join(os.path.dirname(inp), "fake_out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "last_spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f)          # what reached the engine, for the tests
    path = os.path.join(out_dir, f"czp_omni_{seed}_{STATE['edits']}.png")
    # The real engines expand {a|b|c} themselves (seed-bound) and write the line they
    # rendered in the PNG's "crispz" chunk; the fake picks the option at seed % n.
    used = re.sub(r"\{([^{}]*)\}", lambda m: m.group(1).split("|")[seed % len(m.group(1).split("|"))], prompt)
    info = PngImagePlugin.PngInfo()
    info.add_text("crispz", json.dumps({"mode": "omni", "prompt": used, "seed": seed}))
    img.save(path, "PNG", pnginfo=info)
    return {"ok": True, "protocol": 1, "tool": TOOL, "version": VERSION, "route": "remote",
            "images": [os.path.abspath(path)], "seed_used": seed, "loras": [], "refs_used": 1,
            "faces_refined": 0, "hands_refined": 0, "timings": {"total_s": 0.01},
            "warnings": ["'guidance' has no effect on klein"] if spec.get("guidance") is not None else []}


def faces(b64):
    STATE["faces"] += 1
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if _mean_color(img) == (0.0, 0.0, 0.0):
        return {"ok": True, "faces": []}          # a black image has "no face"
    w, h = img.size
    return {"ok": True, "faces": [{"box": [0, 0, w, h], "mouth": [w // 2, h * 2 // 3],
                                   "embedding": _embedding(img)}]}


ENDPOINTS = {"cli_caps": lambda data: caps(),
             "cli_edit": lambda data: edit(data[0]),
             "cli_gen": lambda data: edit(data[0]),
             "cli_faces": lambda data: faces(data[0])}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if len(parts) != 3 or parts[:2] != ["gradio_api", "call"] or parts[2] not in ENDPOINTS:
            return self._send(404, b'{"detail":"Not Found"}')
        n = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(n) or b"{}").get("data") or []
        eid = hashlib.md5(f"{time.time()}{parts[2]}{len(_EVENTS)}".encode()).hexdigest()
        with _LOCK:
            _EVENTS[eid] = (parts[2], data)
        self._send(200, json.dumps({"event_id": eid}).encode())

    def do_GET(self):
        parts = self.path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["gradio_api", "call"]:
            with _LOCK:
                item = _EVENTS.pop(parts[3], None)
            if not item:
                return self._send(404, b"no such event")
            name, data = item
            try:
                out = ENDPOINTS[name](data)
                body = f"event: complete\ndata: {json.dumps([json.dumps(out)])}\n\n"
            except Exception as e:
                body = f"event: error\ndata: {json.dumps([str(e)])}\n\n"
            return self._send(200, body.encode(), "text/event-stream")
        self._send(404, b"not found")


def serve(port, ready=None):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    if ready is not None:
        ready.set()
    return srv


def start_in_thread(port):
    srv = serve(port)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT") or 7899)
    print(f"{TOOL} listening on {port}", flush=True)
    serve(port).serve_forever()
