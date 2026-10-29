"""crispz-dataset-studio - local server (stdlib): the SPA + a JSON API.

    python ds_server.py [--port 8780] [--no-browser]

Same shape as comics2crispz: http.server, zero server dependency, Pillow for the
reference and the export only. Dataset Studio renders nothing: the images come from
a crispz family engine (ds_engines), which it can start itself.

Routes:
  GET  /                      -> assets/studio.html
  GET  /assets/<f>            -> SPA files
  GET  /file/<project>/<rel>  -> project files (reference.png, cards/, the ZIP)
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
        save_config(cfg)
        self.config = cfg
        self.engines.reload(cfg)
        log(f"config: saved {CONFIG_PATH} ({len(cfg['engines'])} engine(s))")
        return {"config": cfg, "engines": self.engines.statuses()}

    def op_engines(self, data):
        return {"engines": self.engines.statuses(), "default": self.engines.default}

    def op_engine_launch(self, data):
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
        name = data.get("name")
        only = data.get("only")
        if only is not None:
            only = {int(i) for i in only}
        return self.runner.start(name, self.engines, self.config.get("ollama") or {},
                                 force=bool(data.get("force")), only=only,
                                 engine_name=data.get("engine") or None)

    def op_cancel(self, data):
        return {"cancelled": self.runner.cancel()}

    def op_status(self, data):
        return {"status": self.runner.status()}

    def op_card_update(self, data):
        p = self._project(data)
        card = ds_project.update_card(p, data.get("index"), caption=data.get("caption"),
                                      status=data.get("status"), use_try=data.get("use_try"))
        return {"card": card}

    def op_export(self, data):
        p = self._project(data)
        if self.runner.state["running"] and self.runner.state["project"] == p["name"]:
            raise RuntimeError("wait for the run to finish before exporting")
        return ds_project.export(p, size=data.get("size"), fmt=data.get("format"))

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
