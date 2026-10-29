"""crispz-dataset-studio - a dataset project: state on disk, the run loop, the export.

A project is ONE folder under projects/<name>/:
    project.json      everything below (re-read / re-written at every operation)
    reference.png     the identity photo (only its main face matters)
    cards/NNN_tK.png  every try K of card NNN (kept: the user can go back)
    export/           the last ZIP and its unpacked pairs

project.json:
    {"name", "created", "engine", "reference", "ref_embedding", "ref_faces",
     "settings": {"promptset", "prompts": [...], "width", "height", "steps",
                  "threshold", "max_rerolls", "caption_source", "trigger_word",
                  "detail_faces", "export_size", "export_format"},
     "cards": [{"index", "prompt", "status", "file", "seed", "score", "detail",
                "tries": [{"file", "seed", "score", "detail"}], "caption",
                "caption_source_used", "warnings", "error"}]}

Card statuses: pending (never rendered), done (rendered, kept), rejected (the user
dropped it: a run renders it again), failed (the engine refused; the error is kept).

The run loop (Runner) is a thread: for every card to render, 1 + max_rerolls tries
at most, each try = cli_edit(reference, prompt) -> cli_faces(result) -> likeness; the
first try above the threshold wins, else the best one is kept and the card says so.
Then the caption. project.json is saved after every card, so a crash or a Cancel
loses at most the card in flight and the next run resumes where it stopped (only
pending / rejected / failed cards are rendered, unless force=True).

Cancel is checked between tries: the engine's own queue finishes the image in
flight (3 s on klein, 15 s on flux3), we do not interrupt it remotely.
"""

import os
import io
import re
import json
import time
import random
import shutil
import zipfile
import threading

import ds_likeness
import ds_captions

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(HERE, "projects")
PROMPTSETS_DIR = os.path.join(HERE, "promptsets")

DEFAULT_SETTINGS = {
    "promptset": "character-25", "prompts": [], "width": 1024, "height": 1024,
    "steps": None, "threshold": 35, "max_rerolls": 2, "caption_source": "prompt",
    "trigger_word": "", "detail_faces": None, "export_size": 1024,
    "export_format": "png", "guidance": None,
    # The engine's checkpoint for this project (spec.model, the protocol's per-call
    # switch: klein calls set_zimage_model). Empty = the engine's own setting. Added
    # on 2026-10-29 so a dataset can run on klein-4B while the user's app stays on 9B.
    "model": "",
}

_LOCK = threading.RLock()
_LOG_HOOK = None


def set_log_hook(fn):
    global _LOG_HOOK
    _LOG_HOOK = fn


def _log(msg):
    line = f"[project] {msg}"
    (_LOG_HOOK or (lambda s: print(s, flush=True)))(line)


# ----------------------------------------------------------------------------
# Prompt sets
# ----------------------------------------------------------------------------
def parse_promptset(text):
    """One prompt per non-empty, non-comment line."""
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def list_promptsets():
    out = []
    if os.path.isdir(PROMPTSETS_DIR):
        for f in sorted(os.listdir(PROMPTSETS_DIR)):
            if f.lower().endswith(".txt"):
                p = os.path.join(PROMPTSETS_DIR, f)
                with open(p, encoding="utf-8") as fh:
                    n = len(parse_promptset(fh.read()))
                out.append({"name": f[:-4], "count": n})
    return out


def load_promptset(name):
    p = os.path.join(PROMPTSETS_DIR, f"{name}.txt")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"prompt set not found: {name}")
    with open(p, encoding="utf-8") as fh:
        return parse_promptset(fh.read())


# ----------------------------------------------------------------------------
# Project files
# ----------------------------------------------------------------------------
def safe_name(name):
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "").strip()).strip("-.")
    if not s:
        raise ValueError("empty project name")
    return s[:64]


def project_dir(name):
    return os.path.join(PROJECTS_DIR, safe_name(name))


def list_projects():
    out = []
    if os.path.isdir(PROJECTS_DIR):
        for d in sorted(os.listdir(PROJECTS_DIR)):
            pj = os.path.join(PROJECTS_DIR, d, "project.json")
            if os.path.isfile(pj):
                try:
                    p = load(d)
                except Exception:
                    continue
                cards = p.get("cards") or []
                out.append({"name": d, "created": p.get("created"),
                            "engine": p.get("engine"),
                            "has_reference": bool(p.get("reference")),
                            "cards": len(cards),
                            "done": sum(1 for c in cards if c.get("status") == "done")})
    return out


def create(name, engine=None, promptset="character-25"):
    d = project_dir(name)
    if os.path.isdir(d):
        raise FileExistsError(f"project already exists: {os.path.basename(d)}")
    os.makedirs(os.path.join(d, "cards"))
    settings = dict(DEFAULT_SETTINGS)
    settings["promptset"] = promptset
    try:
        settings["prompts"] = load_promptset(promptset)
    except FileNotFoundError:
        settings["prompts"] = []
    p = {"name": os.path.basename(d), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
         "engine": engine or "", "reference": "", "ref_embedding": None,
         "ref_faces": 0, "settings": settings, "cards": []}
    _rebuild_cards(p)
    save(p)
    return p


def load(name):
    with open(os.path.join(project_dir(name), "project.json"), encoding="utf-8") as fh:
        return json.load(fh)


def save(p):
    d = project_dir(p["name"])
    tmp = os.path.join(d, "project.json.tmp")
    with _LOCK:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(p, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(d, "project.json"))


def _rebuild_cards(p):
    """Cards follow the prompt list: an existing card keeps its renders when its
    prompt text is unchanged, a changed line resets the card to pending (the render
    no longer matches its prompt), extra cards are dropped."""
    old = {c["index"]: c for c in p.get("cards") or []}
    cards = []
    for i, prompt in enumerate(p["settings"].get("prompts") or [], start=1):
        c = old.get(i)
        if c and c.get("prompt") == prompt:
            cards.append(c)
        else:
            cards.append({"index": i, "prompt": prompt, "status": "pending", "file": "",
                          "seed": None, "score": None, "detail": {}, "tries": [],
                          "caption": "", "caption_source_used": "", "warnings": [],
                          "error": ""})
    p["cards"] = cards


def update_settings(p, new):
    """Applies a settings dict from the UI. Changing the prompt set reloads its
    lines; editing 'prompts' directly wins over 'promptset'."""
    s = p["settings"]
    if "promptset" in new and new["promptset"] != s.get("promptset") \
            and "prompts" not in new:
        s["promptset"] = new["promptset"]
        s["prompts"] = load_promptset(new["promptset"])
    for k, v in new.items():
        if k == "prompts":
            s["prompts"] = parse_promptset("\n".join(v) if isinstance(v, list) else v)
        elif k in DEFAULT_SETTINGS:
            s[k] = v
    for k in ("width", "height", "export_size"):
        s[k] = max(64, min(4096, int(s.get(k) or DEFAULT_SETTINGS[k])))
    s["threshold"] = max(0, min(100, int(s.get("threshold") or 0)))
    s["max_rerolls"] = max(0, min(10, int(s.get("max_rerolls") or 0)))
    s["steps"] = int(s["steps"]) if s.get("steps") not in (None, "", 0, "0") else None
    s["guidance"] = float(s["guidance"]) if s.get("guidance") not in (None, "") else None
    s["model"] = str(s.get("model") or "").strip()
    if s.get("caption_source") not in ("prompt", "ollama", "both"):
        s["caption_source"] = "prompt"
    if s.get("export_format") not in ("png", "jpg"):
        s["export_format"] = "png"
    _rebuild_cards(p)
    save(p)
    return p


def set_reference(p, data, engine=None):
    """Stores the reference bytes as reference.png and, when the engine answers,
    its main face embedding. Every card goes back to pending: a new identity."""
    from PIL import Image
    d = project_dir(p["name"])
    img = Image.open(io.BytesIO(data)).convert("RGB")
    path = os.path.join(d, "reference.png")
    img.save(path, "PNG")
    p["reference"] = "reference.png"
    p["ref_embedding"], p["ref_faces"] = None, 0
    for c in p["cards"]:
        c.update({"status": "pending", "file": "", "seed": None, "score": None,
                  "detail": {}, "tries": [], "caption": "", "error": ""})
    warn = None
    if engine is not None:
        warn = refresh_reference_embedding(p, engine)
    save(p)
    return p, warn


def refresh_reference_embedding(p, engine):
    """cli_faces on the reference; returns a warning string or None."""
    path = os.path.join(project_dir(p["name"]), p.get("reference") or "")
    if not p.get("reference") or not os.path.isfile(path):
        return "no reference image"
    r = engine.faces(path)
    if not r.get("ok"):
        return f"reference faces: {r.get('error')}"
    faces = r.get("faces") or []
    main = ds_likeness.main_face(faces)
    p["ref_faces"] = len(faces)
    p["ref_embedding"] = main.get("embedding") if main else None
    if not main:
        return "no face found on the reference: likeness will be 0 for every card"
    if len(faces) > 1:
        return f"{len(faces)} faces on the reference: the largest one is the identity"
    return None


def cards_to_render(p, force=False, only=None):
    out = []
    for c in p["cards"]:
        if only is not None and c["index"] not in only:
            continue
        if force or c.get("status") in ("pending", "rejected", "failed"):
            out.append(c)
    return out


# ----------------------------------------------------------------------------
# The run loop
# ----------------------------------------------------------------------------
class Runner:
    """One run at a time per process (the engine serializes the GPU anyway)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.cancel_flag = False
        self.state = {"running": False, "project": None, "done": 0, "total": 0,
                      "current": None, "text": "", "ok": None, "error": "",
                      "started": None, "finished": None, "log": []}

    def _say(self, text):
        self.state["text"] = text
        self.state["log"] = (self.state["log"] + [text])[-200:]
        _log(text)

    def status(self):
        s = dict(self.state)
        s["log"] = s["log"][-30:]
        return s

    def cancel(self):
        if self.state["running"]:
            self.cancel_flag = True
            self._say("cancel requested: the card in flight finishes on the engine, "
                      "then the run stops")
            return True
        return False

    def start(self, name, engines, ollama_cfg, force=False, only=None, engine_name=None):
        with self.lock:
            if self.state["running"]:
                return {"ok": False, "error": "a run is already in progress"}
            p = load(name)
            if not p.get("reference"):
                return {"ok": False, "error": "set a reference photo first"}
            todo = cards_to_render(p, force=force, only=only)
            if not todo:
                return {"ok": True, "started": False,
                        "message": "nothing to render: every card is done "
                                   "(reject some, or use force)"}
            self.cancel_flag = False
            self.state.update({"running": True, "project": p["name"], "done": 0,
                               "total": len(todo), "current": None, "text": "",
                               "ok": None, "error": "", "started": time.time(),
                               "finished": None, "log": []})
            self.thread = threading.Thread(
                target=self._run, args=(p, engines, ollama_cfg, todo, engine_name),
                daemon=True, name="ds-run")
            self.thread.start()
            return {"ok": True, "started": True, "total": len(todo)}

    def wait(self, timeout=None):
        if self.thread:
            self.thread.join(timeout)
        return not self.state["running"]

    # -- the loop -------------------------------------------------------------
    def _run(self, p, engines, ollama_cfg, todo, engine_name):
        try:
            name = engine_name or p.get("engine") or None
            self._say(f"engine: checking {name or engines.default}")
            engine, r = engines.ensure(name)
            if not r.get("ok"):
                raise RuntimeError(r.get("error"))
            if r.get("started"):
                self._say(r.get("message") or "engine started")
            p["engine"] = engine.name
            caps = engine.caps() or {}
            sup = caps.get("supports") or {}
            if not sup.get("edit", True):
                raise RuntimeError(f"{caps.get('tool')} does not support the edit op")
            if not sup.get("faces", True):
                self._say("WARNING: this engine has no face service (insightface "
                          "missing?): likeness scores will all be 0")
            if not p.get("ref_embedding"):
                w = refresh_reference_embedding(p, engine)
                if w:
                    self._say(f"reference: {w}")
                save(p)
            d = project_dir(p["name"])
            ref = os.path.join(d, p["reference"])
            s = p["settings"]
            for n, card in enumerate(todo, start=1):
                if self.cancel_flag:
                    self._say(f"cancelled after {n - 1} card(s)")
                    break
                self.state["current"] = card["index"]
                self._render_card(engine, p, card, ref, d, s, ollama_cfg)
                save(p)
                self.state["done"] = n
            self.state["ok"] = not self.cancel_flag
        except Exception as e:
            self.state["ok"] = False
            self.state["error"] = f"{type(e).__name__}: {e}"
            self._say(f"run failed: {self.state['error']}")
        finally:
            self.state["running"] = False
            self.state["current"] = None
            self.state["finished"] = time.time()

    def _render_card(self, engine, p, card, ref, d, s, ollama_cfg):
        idx = card["index"]
        card.update({"status": "rendering", "error": "", "warnings": []})
        best = None
        n_tries = 1 + int(s.get("max_rerolls") or 0)
        for t in range(1, n_tries + 1):
            if self.cancel_flag and t > 1:
                break
            seed = random.randint(0, 2 ** 31 - 1)
            spec = {"prompt": card["prompt"], "input": ref, "refs": [ref],
                    "width": int(s["width"]), "height": int(s["height"]), "seed": seed,
                    "out_dir": os.path.join(d, "cards")}
            if (s.get("model") or "").strip():
                spec["model"] = s["model"].strip()
            if s.get("steps"):
                spec["steps"] = int(s["steps"])
            if s.get("guidance") is not None:
                spec["guidance"] = float(s["guidance"])
            if s.get("detail_faces") is not None:
                spec["detail_faces"] = bool(s["detail_faces"])
            self._say(f"card {idx:03d} try {t}/{n_tries}: rendering (seed {seed})")
            t0 = time.time()
            r = engine.edit(spec)
            if not r.get("ok"):
                card["status"] = "failed"
                card["error"] = r.get("error") or "engine error"
                self._say(f"card {idx:03d}: FAILED: {card['error']}")
                return
            for w in r.get("warnings") or []:
                if w not in card["warnings"]:
                    card["warnings"].append(w)
            src = (r.get("images") or [None])[0]
            if not src or not os.path.isfile(src):
                card["status"] = "failed"
                card["error"] = f"engine returned no image file ({src})"
                return
            dst_name = f"{idx:03d}_t{len(card['tries']) + 1}.png"
            dst = os.path.join(d, "cards", dst_name)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.move(src, dst)
                # the engine writes a sidecar .json / .txt next to some images; tidy it
                for ext in (".json", ".txt"):
                    side = os.path.splitext(src)[0] + ext
                    if os.path.isfile(side):
                        try:
                            os.remove(side)
                        except OSError:
                            pass
            fr = engine.faces(dst)
            if fr.get("ok"):
                score, detail = ds_likeness.score(p.get("ref_embedding"), fr.get("faces"))
            else:
                score, detail = 0, {"reason": f"faces: {fr.get('error')}"}
            tr = {"file": f"cards/{dst_name}", "seed": r.get("seed_used", seed),
                  "score": score, "detail": detail,
                  "time_s": round(time.time() - t0, 1)}
            card["tries"].append(tr)
            self._say(f"card {idx:03d} try {t}: likeness {score} "
                      f"({detail.get('reason') or 'cos ' + str(detail.get('cosine'))}), "
                      f"{tr['time_s']} s")
            if best is None or score > best["score"]:
                best = tr
            if score >= int(s.get("threshold") or 0):
                break
        if best is None:
            card["status"] = "failed"
            card["error"] = "cancelled before the first try"
            return
        card.update({"status": "done", "file": best["file"], "seed": best["seed"],
                     "score": best["score"], "detail": best["detail"]})
        if best["score"] < int(s.get("threshold") or 0):
            card["warnings"].append(f"best of {len(card['tries'])} tries is below the "
                                    f"threshold ({best['score']} < {s.get('threshold')}): "
                                    f"review it")
        cap = ds_captions.caption_for(os.path.join(d, best["file"]), card["prompt"],
                                      s, ollama_cfg)
        card["caption"] = cap["caption"]
        card["caption_source_used"] = cap["source_used"]
        if cap.get("warning"):
            card["warnings"].append(cap["warning"])
            self._say(f"card {idx:03d}: {cap['warning']}")


RUNNER = Runner()


# ----------------------------------------------------------------------------
# Card edits and export
# ----------------------------------------------------------------------------
def update_card(p, index, caption=None, status=None, use_try=None):
    card = next((c for c in p["cards"] if c["index"] == int(index)), None)
    if card is None:
        raise KeyError(f"no card {index}")
    if caption is not None:
        card["caption"] = str(caption)
    if status in ("rejected", "done"):
        if status == "done" and not card.get("file"):
            raise ValueError("card has no render to restore")
        card["status"] = status
    if use_try is not None:
        tr = card["tries"][int(use_try)]
        card.update({"file": tr["file"], "seed": tr["seed"], "score": tr["score"],
                     "detail": tr["detail"], "status": "done"})
    save(p)
    return card


def export(p, size=None, fmt=None):
    """ZIP of the done cards as NNN.<ext> + NNN.txt, the kohya / ai-toolkit /
    OneTrainer pair layout. Images are resized so the LONG side = size (never
    upscaled). Returns {zip, count, skipped}."""
    from PIL import Image
    d = project_dir(p["name"])
    s = p["settings"]
    size = int(size or s.get("export_size") or 1024)
    fmt = (fmt or s.get("export_format") or "png").lower()
    ext = "jpg" if fmt == "jpg" else "png"
    out_dir = os.path.join(d, "export")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    count, skipped = 0, []
    for c in p["cards"]:
        if c.get("status") != "done" or not c.get("file"):
            skipped.append(c["index"])
            continue
        src = os.path.join(d, c["file"])
        if not os.path.isfile(src):
            skipped.append(c["index"])
            continue
        img = Image.open(src).convert("RGB")
        w, h = img.size
        if max(w, h) > size:
            r = size / float(max(w, h))
            img = img.resize((max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS)
        stem = os.path.join(out_dir, f"{c['index']:03d}")
        if ext == "jpg":
            img.save(stem + ".jpg", "JPEG", quality=95)
        else:
            img.save(stem + ".png", "PNG")
        with open(stem + ".txt", "w", encoding="utf-8") as fh:
            fh.write((c.get("caption") or "").strip() + "\n")
        count += 1
    zpath = os.path.join(d, f"{p['name']}-dataset.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out_dir)):
            z.write(os.path.join(out_dir, f), f)
    return {"zip": zpath, "zip_rel": os.path.basename(zpath), "count": count,
            "skipped": skipped, "size": size, "format": ext}
