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
import ds_quality

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(HERE, "projects")
PROMPTSETS_DIR = os.path.join(HERE, "promptsets")

# Exports follow the family layout (crispz-klein cz_imageio: default_output_dir "out"
# next to the app, a YYYY-MM-DD subfolder per day). On top of it, one folder per
# export, the LOT: <out>/<date>/<project>_<YYYYMMDD-HHMMSS>/ holding the NNN.png +
# NNN.txt pairs, with <lot>.zip beside it. A lot is never overwritten: two exports of
# the same project the same second get _2, _3 (the family's _unique_path rule). The
# working files (reference, cards/, tries) stay in projects/<name>/ (2026-10-29).
DEFAULT_OUTPUT_DIR = "out"


def output_root(output_dir=None):
    """The absolute export root: a relative setting is taken from the app folder."""
    d = (output_dir or "").strip() or DEFAULT_OUTPUT_DIR
    d = os.path.expanduser(d)
    if not os.path.isabs(d):
        d = os.path.join(HERE, d)
    return os.path.normpath(d)


def _unique_dir(path):
    if not os.path.exists(path) and not os.path.exists(path + ".zip"):
        return path
    i = 2
    while os.path.exists(f"{path}_{i}") or os.path.exists(f"{path}_{i}.zip"):
        i += 1
    return f"{path}_{i}"


def lot_dir(name, output_dir=None, date_subfolders=True, now=None):
    """(lot folder, path relative to the output root) for a new export of `name`."""
    root = output_root(output_dir)
    t = now or time.localtime()
    parent = os.path.join(root, time.strftime("%Y-%m-%d", t)) if date_subfolders else root
    lot = _unique_dir(os.path.join(parent, f"{safe_name(name)}_{time.strftime('%Y%m%d-%H%M%S', t)}"))
    return lot, os.path.relpath(lot, root).replace(os.sep, "/")

DEFAULT_SETTINGS = {
    "promptset": "character-25", "prompts": [], "width": 1024, "height": 1024,
    "steps": None, "threshold": 35, "max_rerolls": 2, "caption_source": "prompt",
    "trigger_word": "", "detail_faces": None, "export_size": 1024,
    "export_format": "png", "guidance": None,
    # "fixed" sends width x height; "reference" sends no size at all, so the engine
    # keeps the input's own dimensions (qwen-edit edits in place; klein renders at the
    # reference's size). Added on 2026-10-30: a 896x1344 portrait reference forced into
    # 1024x1024 on qwen-edit came back recomposed and doubled.
    "size_mode": "fixed",
    # spec.fast of the edit op (protocol v1, klein and qwen-edit): "" = the engine's
    # own setting, "off", "auto" (the model's profile), "lightning-4", "lightning-8"
    # (the Lightning LoRA stacked on the edit model). Without it qwen-edit ran its
    # 8-step default WITHOUT the Lightning LoRA (2026-10-30: ghosted, tiled faces).
    "fast": "",
    # Visual checks beside the likeness (ds_quality, 2026-10-30): a try that is the
    # reference copied back (correlation >= 0.9) or, with Ollama, one with ghosting
    # or doubling is flagged and re-rolled like a low score.
    "copy_check": True, "artifact_check": False,
    # The engine's checkpoint for this project (spec.model, the protocol's per-call
    # switch: klein calls set_zimage_model). Empty = the engine's own setting. Added
    # on 2026-10-29 so a dataset can run on klein-4B while the user's app stays on 9B.
    "model": "",
    # Prefixed to EVERY line sent to the engine ("the same black panther humanoid with
    # a hair bun, "): the identity phrase. "same person" alone loses a non-human
    # character's traits on close-ups (25-shot run of 2026-10-29). Captions do not
    # carry it: the trigger word is the training-side identity.
    "subject": "",
}

# A line the face model cannot judge (rear view, closed eyes, face hidden): the card
# is rendered once, shown as "unscored", never re-rolled on its score. Explicit with
# a trailing #noscore, or guessed from the wording (announced per card).
NOSCORE_MARK = "#noscore"
NOSCORE_WORDS = ("rear view", "from behind", "walking away", "back to the camera",
                 "eyes closed", "face hidden", "facing away")


def split_line(prompt):
    """(text without the mark, scored: bool, reason or None)."""
    t = (prompt or "").strip()
    if t.lower().endswith(NOSCORE_MARK):
        return t[:-len(NOSCORE_MARK)].rstrip(" ,"), False, "marked #noscore"
    low = t.lower()
    for w in NOSCORE_WORDS:
        if w in low:
            return t, False, f"'{w}' in the line: the face model cannot judge it"
    return t, True, None


def resolved_text(image_path, text, settings):
    """The card's text as the engine really rendered it. The engines expand the
    {a|b|c} groups and __wildcards__ of a prompt themselves, bound to the seed, and
    the protocol result does not carry the expanded line: only the PNG does, in the
    "crispz" chunk (json, key "prompt", subject prefix included). Exported lot of
    2026-10-30: two captions still read "{dark suit|elegant blouse}", so the
    caption is built from that chunk when the line has a group or a wildcard. The
    raw text is kept when the chunk is missing (another engine, a copied file)."""
    if "{" not in (text or "") and "__" not in (text or ""):
        return text
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            meta = json.loads(im.info.get("crispz") or "{}")
    except Exception:
        return text
    used = " ".join((meta.get("prompt") or "").split())
    if not used:
        return text
    subj = " ".join(((settings or {}).get("subject") or "").split()).rstrip(" ,")
    if subj and used.startswith(subj + ", "):
        return used[len(subj) + 2:]
    return used


def engine_prompt(prompt, settings):
    """The line as the engine gets it: subject prefix + text without the mark."""
    text, _, _ = split_line(prompt)
    subj = " ".join(((settings or {}).get("subject") or "").split()).rstrip(" ,")
    return f"{subj}, {text}" if subj and text else (subj or text)

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
    """One prompt per non-empty, non-comment line (a trailing #noscore is kept: it is
    part of the line, see split_line)."""
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
        _, scored, _ = split_line(prompt)
        if c and c.get("prompt") == prompt:
            c["scored"] = scored
            cards.append(c)
        else:
            cards.append({"index": i, "prompt": prompt, "status": "pending", "file": "",
                          "seed": None, "score": None, "detail": {}, "tries": [],
                          "caption": "", "caption_source_used": "", "warnings": [],
                          "error": "", "scored": scored})
    p["cards"] = cards


def update_settings(p, new):
    """Applies a settings dict from the UI. Changing the prompt set reloads its
    lines; editing 'prompts' directly wins over 'promptset'."""
    s = p["settings"]
    # Picking a prompt set ALWAYS reloads its lines (even under the same name: the
    # lines may have been edited by hand since, which is exactly when the user picks
    # the set again to get them back, 2026-10-29).
    if "promptset" in new and "prompts" not in new:
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
    if s.get("size_mode") not in ("fixed", "reference"):
        s["size_mode"] = "fixed"
    s["fast"] = str(s.get("fast") or "").strip().lower()
    if s["fast"] not in ("", "off", "auto", "lightning-4", "lightning-8"):
        s["fast"] = ""
    s["copy_check"] = bool(s.get("copy_check", True))
    s["artifact_check"] = bool(s.get("artifact_check"))
    s["subject"] = " ".join(str(s.get("subject") or "").split())
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
    # The renders of the previous identity go: a new reference means a new dataset,
    # and NNN_tK.png names would otherwise mix two people (seen on 2026-10-29).
    cards_dir = os.path.join(d, "cards")
    for f in os.listdir(cards_dir) if os.path.isdir(cards_dir) else []:
        try:
            os.remove(os.path.join(cards_dir, f))
        except OSError:
            pass
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

    def start(self, name, engines, ollama_cfg, force=False, only=None, engine_name=None,
              seed=None):
        """force on the WHOLE project (only=None) renders every card from scratch: its
        earlier tries are deleted. force on some cards (only=...) is the Re-roll: the
        card gets more tries and keeps the earlier ones, but only those rendered with
        the same engine prompt and model; a try made under another subject or model is
        dropped, it has no business in 'best of'. (The 25-card run of 2026-10-29 kept
        three tries of a previous subject per card.) seed = a base seed: try t of every
        card uses seed + t - 1, so two runs compare like for like (an A/B of models)."""
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
            fresh = bool(force) and only is None
            self.thread = threading.Thread(
                target=self._run, args=(p, engines, ollama_cfg, todo, engine_name, fresh,
                                        None if seed is None else int(seed)),
                daemon=True, name="ds-run")
            self.thread.start()
            return {"ok": True, "started": True, "total": len(todo)}

    def wait(self, timeout=None):
        if self.thread:
            self.thread.join(timeout)
        return not self.state["running"]

    # -- the loop -------------------------------------------------------------
    def _run(self, p, engines, ollama_cfg, todo, engine_name, fresh=False, base_seed=None):
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
                self._render_card(engine, p, card, ref, d, s, ollama_cfg, fresh, base_seed)
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

    def _drop_stale_tries(self, card, d, prompt, model, fresh):
        """Removes the tries (and their files) that cannot compete with the ones about
        to be rendered: all of them on a fresh run, those of another prompt or model
        otherwise. Tries from before this rule (no 'prompt' recorded) count as stale."""
        keep, gone = [], []
        for tr in card.get("tries") or []:
            same = (tr.get("prompt") == prompt and (tr.get("model") or "") == model)
            (gone if fresh or not same else keep).append(tr)
        for tr in gone:
            try:
                os.remove(os.path.join(d, tr["file"]))
            except OSError:
                pass
        card["tries"] = keep
        if gone:
            self._say(f"card {card['index']:03d}: {len(gone)} earlier try(ies) dropped "
                      f"({'fresh run' if fresh else 'another subject, prompt or model'})")
        # the file names follow the try count: with tries removed, a new name is picked
        # past every file still on disk, so nothing is overwritten
        return keep

    def _render_card(self, engine, p, card, ref, d, s, ollama_cfg, fresh=False,
                     base_seed=None):
        idx = card["index"]
        card.update({"status": "rendering", "error": "", "warnings": []})
        best = None
        text, scored, why = split_line(card["prompt"])
        card["scored"] = scored
        n_tries = (1 + int(s.get("max_rerolls") or 0)) if scored else 1
        if not scored:
            self._say(f"card {idx:03d}: unscored ({why}): one try, kept as is")
        prompt = engine_prompt(card["prompt"], s)
        model = (s.get("model") or "").strip()
        self._drop_stale_tries(card, d, prompt, model, fresh)
        for t in range(1, n_tries + 1):
            if self.cancel_flag and t > 1:
                break
            seed = (base_seed + t - 1) if base_seed is not None else random.randint(0, 2 ** 31 - 1)
            spec = {"prompt": prompt, "input": ref, "refs": [ref], "seed": seed,
                    "out_dir": os.path.join(d, "cards")}
            if s.get("size_mode") != "reference":
                spec["width"], spec["height"] = int(s["width"]), int(s["height"])
            if s.get("fast"):
                spec["fast"] = s["fast"]
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
            k = len(card["tries"]) + 1
            while os.path.exists(os.path.join(d, "cards", f"{idx:03d}_t{k}.png")):
                k += 1
            dst_name = f"{idx:03d}_t{k}.png"
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
            flags, qdetail = ds_quality.check(ref, dst, s, ollama_cfg)
            detail = dict(detail, **qdetail)
            tr = {"file": f"cards/{dst_name}", "seed": r.get("seed_used", seed),
                  "score": score, "detail": detail, "prompt": prompt, "model": model,
                  "flags": flags, "time_s": round(time.time() - t0, 1)}
            card["tries"].append(tr)
            self._say(f"card {idx:03d} try {t}: likeness {score} "
                      f"({detail.get('reason') or 'cos ' + str(detail.get('cosine'))}), "
                      f"{tr['time_s']} s" + (f"; FLAGGED: {'; '.join(flags)}" if flags else ""))
            # the best try: an unflagged one first, then the score
            if best is None or (not flags, score) > (not best.get("flags"), best["score"]):
                best = tr
            if not scored or (score >= int(s.get("threshold") or 0) and not flags):
                break
        if best is None:
            card["status"] = "failed"
            card["error"] = "cancelled before the first try"
            return
        card.update({"status": "done", "file": best["file"], "seed": best["seed"],
                     "score": best["score"], "detail": best["detail"]})
        if best.get("flags"):
            card["warnings"].append("kept try is flagged: " + "; ".join(best["flags"]))
        if not scored:
            card["detail"] = dict(best["detail"], unscored=why)
        elif best["score"] < int(s.get("threshold") or 0):
            card["warnings"].append(f"best of {len(card['tries'])} tries is below the "
                                    f"threshold ({best['score']} < {s.get('threshold')}): "
                                    f"review it")
        text = resolved_text(os.path.join(d, best["file"]), text, s)
        cap = ds_captions.caption_for(os.path.join(d, best["file"]), text, s, ollama_cfg)
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


def export(p, size=None, fmt=None, output_dir=None, date_subfolders=True):
    """The done cards as NNN.<ext> + NNN.txt, the kohya / ai-toolkit / OneTrainer
    pair layout, written to a new lot folder under the output root (see lot_dir)
    and zipped beside it. Images are resized so the LONG side = size (never
    upscaled). Returns {dir, zip, zip_rel, lot, count, skipped, size, format}."""
    from PIL import Image
    d = project_dir(p["name"])
    s = p["settings"]
    size = int(size or s.get("export_size") or 1024)
    fmt = (fmt or s.get("export_format") or "png").lower()
    ext = "jpg" if fmt == "jpg" else "png"
    out_dir, lot_rel = lot_dir(p["name"], output_dir, date_subfolders)
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
    zpath = out_dir + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out_dir)):
            z.write(os.path.join(out_dir, f), f)
    r = {"dir": out_dir, "zip": zpath, "zip_rel": lot_rel + ".zip", "lot": lot_rel,
         "count": count, "skipped": skipped, "size": size, "format": ext}
    p["last_export"] = dict(r, when=time.strftime("%Y-%m-%d %H:%M:%S"))
    save(p)
    return r
