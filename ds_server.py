"""crispz-dataset-studio - local server (stdlib): the SPA + a JSON API.

    python ds_server.py [--port 8780] [--no-browser]

Same shape as comics2crispz: http.server, zero server dependency, Pillow for the
reference and the export only. Dataset Studio renders nothing: the images come from
a crispz family engine (ds_engines), which it can start itself.

Routes:
  GET  /                      -> assets/studio.html
  GET  /assets/<f>            -> SPA files
  GET  /file/<project>/<rel>  -> project files (reference.png, cards/)
  GET  /out/<rel>             -> the export root (lot folders and their ZIPs)
  POST /api/<op>              -> JSON in / JSON out, {ok: true, ...} or {ok: false, error}

Ops:
  config_get {}                         the config (engines, ollama, port)
  config_set {config}                   writes config.json, reloads the engines
  engines {}                            status of every engine (probe)
  engine_launch {name}                  start the engine (blocks until cli_caps answers)
  engine_stop {name}                    stop the engine WE started
  ollama_models {}                      installed Ollama models
  promptsets {}                         bundled prompt sets
  projects {}                           the project list
  project_new {name, engine?, promptset?}
  project_get {name}
  project_settings {name, settings}     update settings / prompts
  set_reference {name, data (base64) | path (a local file)}
  run {name, force?, only?, engine?}    start the run thread
  cancel {}
  status {}                             the run state (poll)
  card_update {name, index, caption?, status?, use_try?}
  export {name, size?, format?}         -> {zip_rel, count}
  project_clone {name, new_name, with_renders?}  copy a project under a new name
  aitk_status {}                        ai-toolkit folder found?, presets
  aitk_create {name, preset?, steps?, lot?}  copy a lot into ai-toolkit's datasets
                                        and write config/<job>.yaml (lot = an
                                        export's lot path; none = a fresh export)
  aitk_train {name, job?, stop_engine?} start ai-toolkit on the job (default: the
                                        project's last one); the GPU must be free
  aitk_train_stop {}                    stop the training (whole process tree)
  lora_targets {name, job?}             the job's LoRA files + the engines' LoRA folders
  lora_copy {name, job?, file?, engine?} copy a LoRA of the job into an engine's folder
  lora_score {name, job?, files?, engine?, prompts?, weight?, baseline?, seed?,
              switch_model?}
                                        txt2img tests with / without the LoRA, scored
                                        against the reference (thread; see status)
  lora_score_cancel {}
  autopilot_start {name}                Run + export + job + training + score, hands off
                                        (the project's auto_* settings)
  autopilot_queue {names: [...]}        the autopilot on several projects, one after
                                        the other (each with its own auto_* settings)
  autopilot_cancel {all?}               all: also empty the queue
  status {} also carries 'training', 'lora_score' and 'autopilot' (job, progress, log tail, result)
"""

import os
import sys
import json
import time
import base64
import argparse
import threading
import mimetypes
import posixpath
import webbrowser
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ds_engines
import ds_project
import ds_aitoolkit
import ds_lorascore
import ds_autopilot
import ds_captions

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
CONFIG_PATH = os.path.join(HERE, "config.json")
SAMPLE_PATH = os.path.join(HERE, "config-sample.json")
DEFAULT_PORT = 8780
APP_VERSION = "0.1.0"

_LOG_LOCK = threading.Lock()


def log(msg):
    with _LOG_LOCK:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


ds_engines.set_log_hook(log)
ds_project.set_log_hook(log)
ds_aitoolkit.set_log_hook(log)
ds_lorascore.set_log_hook(log)
ds_autopilot.set_log_hook(log)


def load_config():
    path = CONFIG_PATH if os.path.isfile(CONFIG_PATH) else SAMPLE_PATH
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as e:
        log(f"config: {path} unreadable ({e}), using defaults")
        cfg = {}
    cfg.setdefault("engines", {})
    cfg.setdefault("ollama", {"url": ds_captions.DEFAULT_URL, "model": "", "keep_alive": 0})
    if path == SAMPLE_PATH:
        log("config: no config.json yet, using config-sample.json (Settings writes it)")
    return cfg


def save_config(cfg):
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, CONFIG_PATH)


class Studio:
    """The API. One instance per process; the handler calls studio.op(name, data)."""

    def __init__(self, config=None):
        self.config = config or load_config()
        self.engines = ds_engines.Engines(self.config)
        self.runner = ds_project.RUNNER
        self._resume_autopilot()

    def _resume_autopilot(self):
        """A chain cut by a restart during the training: the training is detached and
        went on, so the chain picks up there (then scores). Any other step: said."""
        ap = ds_autopilot.autopilot()
        st = ap.status()
        cut = st.get("interrupted")
        if not cut:
            return
        tr = ds_aitoolkit.trainer()
        if cut == "training" and st.get("job") and tr.state.get("job") == st["job"]:
            log(f"autopilot: resuming '{st.get('project')}' at the training ({st['job']})")
            try:
                ap.start(self, st["project"], resume_at="training")
            except Exception as e:
                log(f"autopilot: cannot resume: {e}")
        else:
            log(f"autopilot: the chain of '{st.get('project')}' was cut at the {cut} by a "
                f"restart; Run again to go on")
    # -- helpers --------------------------------------------------------------
    def _project(self, data):
        name = data.get("name")
        if not name:
            raise ValueError("missing project name")
        return ds_project.load(name)

    def _engine_or_none(self, name=None):
        try:
            e = self.engines.get(name)
        except KeyError:
            return None
        return e if e.caps(max_age=0) else None

    # -- ops ------------------------------------------------------------------
    def op(self, name, data):
        fn = getattr(self, f"op_{name}", None)
        if fn is None:
            return {"ok": False, "error": f"unknown op '{name}'"}
        try:
            out = fn(data or {})
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if isinstance(out, dict) and "ok" not in out:
            out["ok"] = True
        return out

    def op_config_get(self, data):
        cfg = {k: v for k, v in self.config.items() if not k.startswith("_")}
        return {"config": cfg, "config_file": CONFIG_PATH,
                "exists": os.path.isfile(CONFIG_PATH), "version": APP_VERSION}

    def op_config_set(self, data):
        cfg = data.get("config")
        if not isinstance(cfg, dict):
            raise ValueError("config must be an object")
        cfg.setdefault("engines", {})
        cfg.setdefault("ollama", self.config.get("ollama", {}))
        cfg.setdefault("port", self.config.get("port", DEFAULT_PORT))
        cfg.setdefault("output_dir", self.config.get("output_dir", ds_project.DEFAULT_OUTPUT_DIR))
        cfg.setdefault("date_subfolders", self.config.get("date_subfolders", True))
        cfg.setdefault("ai_toolkit_dir", self.config.get("ai_toolkit_dir", ""))
        save_config(cfg)
        self.config = cfg
        self.engines.reload(cfg)
        log(f"config: saved {CONFIG_PATH} ({len(cfg['engines'])} engine(s))")
        return {"config": cfg, "engines": self.engines.statuses()}

    def op_engines(self, data):
        return {"engines": self.engines.statuses(), "default": self.engines.default}

    def _training_guard(self, what):
        if ds_aitoolkit.trainer().running():
            raise RuntimeError(f"an ai-toolkit training ('{ds_aitoolkit.trainer().state['job']}') "
                               f"is using the GPU: wait for it or stop it before {what}")

    def op_engine_launch(self, data):
        self._training_guard("starting an engine")
        e = self.engines.get(data.get("name"))
        other = self.engines.alive_child()
        if other is not None and other is not e:
            return {"ok": False, "error": f"engine '{other.name}' started by Dataset Studio "
                    f"is still running: stop it first (one family app on the GPU at a time)"}
        r = e.launch(wait=True)
        r["status"] = e.status()
        return r

    def op_engine_stop(self, data):
        e = self.engines.get(data.get("name"))
        r = e.stop()
        r["status"] = e.status()
        return r

    def op_ollama_models(self, data):
        """The VISION models only (Ollama's own capability flag): the captioner has to
        look at the card. `all` = true lists every installed model for reference."""
        url = (self.config.get("ollama") or {}).get("url") or ds_captions.DEFAULT_URL
        allm = ds_captions.ollama_models(url)
        vis = ds_captions.ollama_vision_models(url) if allm else []
        return {"models": vis, "installed": len(allm),
                "message": (None if vis or not allm else
                            f"{len(allm)} Ollama model(s) installed, none with the vision "
                            f"capability: pull a vision model (llava, qwen3-vl, gemma3...)")}

    def op_promptsets(self, data):
        return {"promptsets": ds_project.list_promptsets()}

    def op_projects(self, data):
        return {"projects": ds_project.list_projects()}

    def op_project_new(self, data):
        p = ds_project.create(data.get("name"), engine=data.get("engine") or self.engines.default,
                              promptset=data.get("promptset") or "character-25")
        return {"project": p}

    def op_project_get(self, data):
        return {"project": self._project(data)}

    def op_project_settings(self, data):
        p = self._project(data)
        if self.runner.state["running"] and self.runner.state["project"] == p["name"]:
            raise RuntimeError("settings are locked while a run is in progress")
        if data.get("engine") is not None:
            p["engine"] = str(data["engine"])
        p = ds_project.update_settings(p, data.get("settings") or {})
        return {"project": p}

    def op_set_reference(self, data):
        p = self._project(data)
        path = str(data.get("path") or "").strip().strip('"')
        if path:
            # Everything is local: a path typed in the page is read by the server. This
            # is the route that works when the browser cannot open a file dialog (the
            # desktop app's embedded pane, 2026-10-29).
            if not os.path.isfile(path):
                raise FileNotFoundError(f"no such file: {path}")
            with open(path, "rb") as fh:
                blob = fh.read()
        else:
            raw = data.get("data") or ""
            if "," in raw[:64] and raw.startswith("data:"):
                raw = raw.split(",", 1)[1]
            blob = base64.b64decode(raw)
        if not blob:
            raise ValueError("empty image")
        engine = self._engine_or_none(p.get("engine") or None)
        p, warn = ds_project.set_reference(p, blob, engine)
        msg = None
        if engine is None:
            msg = ("reference saved; the engine is not running, its face will be "
                   "analysed at the start of the run")
        return {"project": p, "warning": warn, "message": msg}

    def op_run(self, data):
        self._training_guard("rendering")
        if ds_lorascore.SCORER.state["running"]:
            raise RuntimeError("a LoRA scoring is using the engine: wait for it or cancel it")
        name = data.get("name")
        only = data.get("only")
        if only is not None:
            only = {int(i) for i in only}
        return self.runner.start(name, self.engines, self.config.get("ollama") or {},
                                 force=bool(data.get("force")), only=only,
                                 engine_name=data.get("engine") or None,
                                 seed=data.get("seed"))

    def op_cancel(self, data):
        return {"cancelled": self.runner.cancel()}

    def op_status(self, data):
        return {"status": self.runner.status(), "training": ds_aitoolkit.trainer().status(),
                "lora_score": ds_lorascore.SCORER.status(),
                "autopilot": ds_autopilot.autopilot().status()}

    def op_card_update(self, data):
        p = self._project(data)
        card = ds_project.update_card(p, data.get("index"), caption=data.get("caption"),
                                      status=data.get("status"), use_try=data.get("use_try"))
        return {"card": card}

    def op_export(self, data):
        p = self._project(data)
        if self.runner.state["running"] and self.runner.state["project"] == p["name"]:
            raise RuntimeError("wait for the run to finish before exporting")
        r = ds_project.export(p, size=data.get("size"), fmt=data.get("format"),
                              output_dir=self.config.get("output_dir"),
                              date_subfolders=bool(self.config.get("date_subfolders", True)))
        r["out_url"] = "/out/" + urllib.request.quote(r["zip_rel"])
        return r

    def op_project_clone(self, data):
        p = self._project(data)
        if self.runner.state["running"] and self.runner.state["project"] == p["name"]:
            raise RuntimeError("wait for the run to finish before cloning this project")
        new = ds_project.clone(p["name"], data.get("new_name"),
                               with_renders=bool(data.get("with_renders", True)))
        return {"project": new}

    def op_aitk_status(self, data):
        return {"aitk": ds_aitoolkit.status(self.config)}

    def op_aitk_create(self, data):
        p = self._project(data)
        if self.runner.state["running"] and self.runner.state["project"] == p["name"]:
            raise RuntimeError("wait for the run to finish before creating the job")
        if not ds_aitoolkit.find_root(self.config.get("ai_toolkit_dir")):
            raise FileNotFoundError(ds_aitoolkit.status(self.config).get("error") or
                                    "set the AI-Toolkit folder in Settings first")
        root = ds_project.output_root(self.config.get("output_dir"))
        lot = str(data.get("lot") or "").strip()
        lot_path = _safe_join(root, lot) if lot else None
        exported = None
        if not lot_path or not os.path.isdir(lot_path):
            # no lot given (or it was moved): a fresh export, so the captions edited
            # since the last one are the ones trained on
            exported = self.op_export({"name": p["name"]})
            lot_path = exported["dir"]
            p = ds_project.load(p["name"])
        r = ds_aitoolkit.create_job(p, lot_path, self.config, preset=data.get("preset"),
                                    steps=data.get("steps"))
        p["last_aitk"] = {k: r[k] for k in ("job", "dataset_dir", "config_path", "command")}
        ds_project.save(p)
        if exported:
            r["exported"] = exported["lot"]
        return r

    def op_aitk_train(self, data):
        p = self._project(data)
        root = ds_aitoolkit.find_root(self.config.get("ai_toolkit_dir"))
        if not root:
            raise FileNotFoundError("set the AI-Toolkit folder in Settings first")
        job = str(data.get("job") or (p.get("last_aitk") or {}).get("job") or "").strip()
        if not job:
            raise ValueError("no AI-Toolkit job for this project yet: create it first")
        if self.runner.state["running"] or ds_lorascore.SCORER.state["running"]:
            raise RuntimeError("a dataset run or a LoRA scoring is in progress: the GPU is busy")
        # One GPU: the engine Dataset Studio started holds 15 to 30 GB. It is stopped
        # only when the user said so (stop_engine), never silently.
        busy = self.engines.alive_child()
        if busy is not None:
            if not data.get("stop_engine"):
                return {"ok": False, "engine_busy": busy.name,
                        "error": f"engine '{busy.name}' (started by Dataset Studio) holds the "
                                 f"GPU: stop it to train"}
            log(f"training: stopping engine '{busy.name}' first (GPU)")
            busy.stop()
            stopped = busy.name
        else:
            stopped = None
        warnings = []
        others = [e["name"] for e in self.engines.statuses()
                  if e.get("running") and not e.get("started_by_us")]
        if others:
            warnings.append(f"engine(s) {', '.join(others)} running outside Dataset Studio "
                            f"may hold GPU memory: close them if the training runs out of VRAM")
            for w in warnings:
                log(f"training: {w}")
        st = ds_aitoolkit.trainer().start(root, job, project=p["name"])
        return {"training": st, "warnings": warnings, "stopped_engine": stopped}

    def op_aitk_train_stop(self, data):
        return {"stopped": ds_aitoolkit.trainer().stop(),
                "training": ds_aitoolkit.trainer().status()}

    def _job_of(self, p, data):
        root = ds_aitoolkit.find_root(self.config.get("ai_toolkit_dir"))
        if not root:
            raise FileNotFoundError("set the AI-Toolkit folder in Settings first")
        job = str(data.get("job") or (p.get("last_aitk") or {}).get("job") or "").strip()
        if not job:
            raise ValueError("no AI-Toolkit job for this project yet")
        return root, job

    def _lora_engines(self):
        out = []
        for name, ec in (self.config.get("engines") or {}).items():
            d = ds_aitoolkit.engine_loras_dir(ec.get("dir"), ec.get("loras_dir"))
            if d:
                out.append({"name": name, "dir": ec.get("dir") or "", "loras_dir": d,
                            "exists": os.path.isdir(d)})
        return out

    @staticmethod
    def _arch_hint(arch):
        if arch == "flux2_klein_4b":
            return ("trained on Klein 4B: in crispz-klein pick the FLUX.2-klein-4B "
                    "checkpoint (on 9B the LoRA is skipped, hidden sizes differ)")
        if arch == "flux2_klein_9b":
            return ("trained on Klein 9B: in crispz-klein pick the FLUX.2-klein-9B "
                    "checkpoint (on 4B the LoRA is skipped)")
        return ""

    def op_lora_targets(self, data):
        p = self._project(data)
        root, job = self._job_of(p, data)
        engines = self._lora_engines()
        names = [e["name"] for e in engines]
        default = ("klein" if "klein" in names else
                   (p.get("engine") if p.get("engine") in names else (names[0] if names else "")))
        return {"job": job, "loras": ds_aitoolkit.job_loras(root, job), "engines": engines,
                "default_engine": default, "hint": self._arch_hint(ds_aitoolkit.job_arch(root, job)),
                "copied": (p.get("last_aitk") or {}).get("copied") or [],
                "default_prompts": ds_lorascore.DEFAULT_PROMPTS,
                "trigger": (p.get("settings") or {}).get("trigger_word") or "",
                "scores": [h for h in p.get("lora_scores") or [] if h.get("job") == job]}

    def op_lora_copy(self, data):
        p = self._project(data)
        root, job = self._job_of(p, data)
        loras = ds_aitoolkit.job_loras(root, job)
        if not loras:
            raise FileNotFoundError(f"no LoRA in {os.path.join(root, 'output', job)} yet: "
                                    f"train the job first")
        f = str(data.get("file") or "").strip()
        src = next((l for l in loras if l["file"] == f), None) if f else loras[0]
        if src is None:
            raise FileNotFoundError(f"no LoRA '{f}' in job '{job}'")
        eng = str(data.get("engine") or "").strip()
        target = next((e for e in self._lora_engines() if e["name"] == eng), None)
        if target is None:
            raise ValueError(f"engine '{eng}' has no install folder in Settings: "
                             f"its LoRA folder is unknown")
        r = ds_aitoolkit.copy_lora(src["path"], target["loras_dir"])
        hint = self._arch_hint(ds_aitoolkit.job_arch(root, job))
        la = p.setdefault("last_aitk", {"job": job})
        la.setdefault("copied", [])
        la["copied"] = [c for c in la["copied"] if c.get("path") != r["path"]] + \
                       [{"engine": eng, "path": r["path"], "file": r["file"]}]
        ds_project.save(p)
        e = self.engines.by_name.get(eng)
        running = bool(e and e.caps(max_age=0))
        r.update(engine=eng, source=src["file"], hint=hint,
                 message=(f"{r['file']} {'copied to' if r['copied'] else 'already in'} "
                          f"{target['loras_dir']}"
                          + ("; refresh the LoRA list in the engine" if running else "")))
        return r

    def op_lora_score(self, data):
        p = self._project(data)
        root, job = self._job_of(p, data)
        self._training_guard("scoring a LoRA")
        if self.runner.state["running"]:
            raise RuntimeError("a dataset run is in progress: wait for it")
        names = [e["name"] for e in self._lora_engines()]
        eng = str(data.get("engine") or ("klein" if "klein" in names else p.get("engine") or "")).strip()
        ec = (self.config.get("engines") or {}).get(eng) or {}
        prompts = data.get("prompts")
        if isinstance(prompts, str):
            prompts = [l for l in prompts.splitlines()]
        files = data.get("files")
        if isinstance(files, str):
            files = [files]
        st = ds_lorascore.SCORER.start(
            p["name"], self.engines, eng or None, root, job, files or [], prompts=prompts,
            weight=float(data.get("weight") or 1.0), baseline=bool(data.get("baseline", True)),
            seed=int(data.get("seed") if data.get("seed") not in (None, "") else ds_lorascore.DEFAULT_SEED),
            loras_dir=ds_aitoolkit.engine_loras_dir(ec.get("dir"), ec.get("loras_dir")) if ec else None,
            switch_model=bool(data.get("switch_model", True)))
        return {"lora_score": st}

    def op_lora_score_cancel(self, data):
        return {"cancelled": ds_lorascore.SCORER.cancel()}

    def _autopilot_opts(self, p, engine=None):
        s = p.get("settings") or {}
        if not p.get("reference"):
            raise ValueError(f"'{p['name']}': set a reference photo first")
        return {"engine": engine or p.get("engine") or None,
                "preset": s.get("auto_preset") or "klein4b",
                "steps": int(s.get("auto_steps") or 1500),
                "score": bool(s.get("auto_score", True)),
                "score_engine": s.get("auto_score_engine") or ""}

    def _autopilot_guard(self):
        if not ds_aitoolkit.find_root(self.config.get("ai_toolkit_dir")):
            raise FileNotFoundError("the autopilot needs the AI-Toolkit folder: set it in Settings")
        self._training_guard("starting the autopilot")
        if self.runner.state["running"] or ds_lorascore.SCORER.state["running"]:
            raise RuntimeError("a run or a LoRA scoring is in progress: wait for it")

    def op_autopilot_queue(self, data):
        names = [str(n) for n in (data.get("names") or []) if str(n).strip()]
        if not names:
            raise ValueError("pick at least one project")
        self._autopilot_guard()
        items = [(ds_project.load(n)["name"], self._autopilot_opts(ds_project.load(n)))
                 for n in names]
        st = ds_autopilot.autopilot().start_queue(self, items)
        log(f"autopilot: queue of {len(items)} project(s) started")
        return {"autopilot": st}

    def op_autopilot_start(self, data):
        p = self._project(data)
        self._autopilot_guard()
        opts = self._autopilot_opts(p, data.get("engine"))
        st = ds_autopilot.autopilot().start(self, p["name"], opts, queue=[], history=[])
        log(f"autopilot: '{p['name']}' started ({opts['preset']}, {opts['steps']} steps, "
            f"score on {opts['score_engine'] or 'none'})")
        return {"autopilot": st}

    def op_autopilot_cancel(self, data):
        return {"cancelled": ds_autopilot.autopilot().cancel(self, all_=bool(data.get("all")))}

    def shutdown(self):
        self.engines.stop_all()


STUDIO = None


def _safe_join(root, rel):
    rel = posixpath.normpath("/" + rel.replace("\\", "/")).lstrip("/")
    path = os.path.abspath(os.path.join(root, *rel.split("/")))
    if not path.startswith(os.path.abspath(root) + os.sep) and path != os.path.abspath(root):
        return None
    return path


class Handler(BaseHTTPRequestHandler):
    server_version = f"crispz-dataset-studio/{APP_VERSION}"

    def log_message(self, fmt, *args):      # quiet: the studio log is enough
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _file(self, path):
        if not path or not os.path.isfile(path):
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as fh:
            self._send(200, fh.read(), ctype)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._file(os.path.join(ASSETS, "studio.html"))
        if path.startswith("/assets/"):
            return self._file(_safe_join(ASSETS, path[len("/assets/"):]))
        if path.startswith("/file/"):
            rest = path[len("/file/"):]
            if "/" not in rest:
                return self._send(404, b"not found", "text/plain")
            proj, rel = rest.split("/", 1)
            try:
                root = ds_project.project_dir(urllib.request.unquote(proj))
            except ValueError:
                return self._send(404, b"not found", "text/plain")
            return self._file(_safe_join(root, urllib.request.unquote(rel)))
        if path.startswith("/out/"):
            root = ds_project.output_root(STUDIO.config.get("output_dir"))
            return self._file(_safe_join(root, urllib.request.unquote(path[len("/out/"):])))
        if path == "/api/ping":
            return self._json({"ok": True, "tool": "crispz-dataset-studio",
                               "version": APP_VERSION})
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if not path.startswith("/api/"):
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            return self._json({"ok": False, "error": "invalid JSON body"}, 400)
        self._json(STUDIO.op(path[len("/api/"):], data))


def already_running(port, timeout=1.5):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=timeout) as r:
            j = json.loads(r.read().decode("utf-8"))
        return j.get("tool") == "crispz-dataset-studio"
    except Exception:
        return False


def serve(port=DEFAULT_PORT, config=None, open_browser=True):
    global STUDIO
    STUDIO = Studio(config)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    log(f"crispz-dataset-studio {APP_VERSION}: http://127.0.0.1:{port}  "
        f"(engines: {', '.join(STUDIO.engines.by_name) or 'none configured'})")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        log("stopping: closing the engine we started, if any")
        STUDIO.shutdown()
        srv.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ds_server", description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    port = a.port or int(cfg.get("port") or DEFAULT_PORT)
    if already_running(port):
        log(f"already running at http://127.0.0.1:{port}, opening it")
        if not a.no_browser:
            webbrowser.open(f"http://127.0.0.1:{port}")
        return 0
    serve(port, cfg, open_browser=not a.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
