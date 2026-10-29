"""crispz-dataset-studio - client AND launcher of the crispz family engines.

Dataset Studio renders nothing itself: every image comes from a family tool
(crispz-klein, crispz-qwen-edit, crispz-flux3, crispz-studio...) through the CLI
protocol v1 (the hidden Gradio endpoints cli_caps / cli_edit / cli_gen / cli_faces).
The client part is the one comics2crispz uses (c2c_engines.py, same contract, same
`_gradio_call`). What is NEW here is the launcher: the engine's install folder is in
the config, so when nothing answers at its URL we start the app ourselves instead of
asking the user to open a terminal.

Config (config.json, see config-sample.json):

    "engines": {
        "klein": {"dir": "D:/Github/crispz-klein",
                  "url": "http://127.0.0.1:7861",
                  "launch": true}
    },
    "engine": "klein"

How the launch works (2026-10-29):
  - The interpreter is resolved like the family's run.bat does: `python` key of the
    engine if set, then <dir>/.venv/Scripts/python.exe (Windows) or
    <dir>/.venv/bin/python, then Pinokio's <dir>/env, then the `py -3` / `python3`
    launcher. Announced in the log line.
  - The port comes from the engine URL and is passed through GRADIO_SERVER_PORT:
    the family apps call `build_ui().launch()` without server_port, and Gradio reads
    that variable, so NO change is needed in the engines. They do not set
    inbrowser=True either, so launching them opens no browser tab.
  - stdout/stderr go to logs/<engine>.log (the user reads it when a boot fails).
  - We then poll cli_caps until it answers (default 120 s: a cold boot of a family
    app takes 10 to 25 s without the model; the model itself loads at the first
    image, which is why the first card of a run is slow and says so).
  - The child is kept so that stop_engine() can terminate it and so that the server
    closes it at exit. An engine we did NOT start is never killed.

VRAM rule of the family: Dataset Studio loads no model, so it adds nothing to the
GPU; but two family apps running at once (klein and flux3 for instance) do not fit
on 32 GB (incident of 2026-10-28). launch() refuses to start a second engine while
one it started is alive, unless stop_engine() was called first.

Everything is stdlib (urllib + subprocess). Failures come back as
{ok: false, error}: no exception crosses the HTTP server.
"""

import os
import re
import sys
import json
import time
import shutil
import base64
import subprocess
import threading
import urllib.request
import urllib.parse

PROTOCOL = 1
HERE = os.path.dirname(os.path.abspath(__file__))

# The base repos each family tool can switch to through spec.model (its own
# "checkpoint" dropdown). cli_caps lists only the checkpoint FILES of the tool's
# folder, not these HF repos, so the Model menu of a project merges both. On
# crispz-qwen-edit spec.model is the txt2img base; the edit model (Qwen-Image-Edit)
# is a separate setting of that app and is not switched from here (2026-10-29).
KNOWN_BASES = {
    "crispz-klein": ["black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-9B"],
    "crispz-flux3": ["black-forest-labs/flux-3-image-klein",
                     "black-forest-labs/flux-3-image-klein-base"],
    "crispz-qwen-edit": ["Qwen/Qwen-Image"],
}
LOG_DIR = os.path.join(HERE, "logs")

_LOG_HOOK = None


def set_log_hook(fn):
    """The server installs its logger here (one place prints to the console)."""
    global _LOG_HOOK
    _LOG_HOOK = fn


def _log(msg):
    line = f"[engines] {msg}"
    if _LOG_HOOK:
        _LOG_HOOK(line)
    else:
        print(line, flush=True)


# ----------------------------------------------------------------------------
# Protocol client (identical to comics2crispz / cz_protocol)
# ----------------------------------------------------------------------------
def _gradio_call(url, name, data, post_timeout=10, get_timeout=3600):
    """POST /gradio_api/call/<name> -> event_id, then GET the stream and return the
    first output, JSON-decoded when it is a JSON string."""
    req = urllib.request.Request(
        f"{url.rstrip('/')}/gradio_api/call/{name}",
        data=json.dumps({"data": data}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=post_timeout) as r:
        j = json.loads(r.read().decode("utf-8"))
    eid = j.get("event_id") or j.get("hash")
    if not eid:
        raise ValueError("no event id (endpoint missing? app too old?)")
    with urllib.request.urlopen(
            f"{url.rstrip('/')}/gradio_api/call/{name}/{eid}",
            timeout=get_timeout) as r:
        text = r.read().decode("utf-8")
    m = re.search(r"data:\s*(\[[\s\S]*?\])\s*(?:\n|$)", text)
    if not m:
        raise ValueError("empty API response")
    out = json.loads(m.group(1))[0]
    if isinstance(out, str):
        try:
            return json.loads(out)
        except ValueError:
            return out
    return out


def probe(url, timeout=4):
    """caps of the instance at `url`, None when nothing answers (or it is not a
    family tool)."""
    try:
        caps = _gradio_call(url, "cli_caps", [], post_timeout=timeout,
                            get_timeout=max(timeout, 8))
        return caps if isinstance(caps, dict) and caps.get("ok") else None
    except Exception:
        return None


def _port_of(url):
    try:
        p = urllib.parse.urlparse(url).port
        return int(p) if p else 7860
    except Exception:
        return 7860


def _pid_alive(pid):
    """Is a process with that pid running? stdlib only (no psutil)."""
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
                                 capture_output=True, text=True, timeout=10).stdout
            return str(int(pid)) in out
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _kill_pid(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                       capture_output=True, timeout=20)
    else:
        import signal
        os.kill(int(pid), signal.SIGTERM)


# ----------------------------------------------------------------------------
# Interpreter resolution (the family's run.bat rule)
# ----------------------------------------------------------------------------
def resolve_python(engine_dir, explicit=None):
    """The interpreter that runs <engine_dir>/app.py, and a label for the log.
    Order: explicit > .venv > Pinokio env > a system launcher. Returns
    (argv_prefix list, label) or (None, reason)."""
    if explicit:
        if os.path.isfile(explicit):
            return [explicit], f"python key ({explicit})"
        return None, f"python key points to a missing file: {explicit}"
    win = os.name == "nt"
    cands = [(".venv", os.path.join(engine_dir, ".venv", "Scripts" if win else "bin",
                                    "python.exe" if win else "python")),
             ("Pinokio env", os.path.join(engine_dir, "env", "Scripts" if win else "bin",
                                          "python.exe" if win else "python"))]
    for label, p in cands:
        if os.path.isfile(p):
            return [p], f"{label} ({p})"
    if win and shutil.which("py"):
        return ["py", "-3"], "py -3 launcher (no .venv in the engine folder)"
    for name in ("python3", "python"):
        p = shutil.which(name)
        if p:
            return [p], f"{name} on PATH (no .venv in the engine folder)"
    return None, "no Python found (no .venv, no env, no py/python3 on PATH)"


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
class Engine:
    """One configured family tool: how to reach it, and how to start it."""

    def __init__(self, name, cfg):
        self.name = name
        self.dir = str(cfg.get("dir") or "").strip()
        self.url = str(cfg.get("url") or "http://127.0.0.1:7860").strip()
        self.python = str(cfg.get("python") or "").strip() or None
        self.launch_allowed = bool(cfg.get("launch", True))
        self.boot_timeout = float(cfg.get("boot_timeout", 120))
        self.proc = None
        self.log_path = os.path.join(LOG_DIR, f"{name}.log")
        self.pid_path = os.path.join(LOG_DIR, f"{name}.pid")
        self._probe = (0.0, None)
        self._lock = threading.Lock()
        # A child started by a PREVIOUS Dataset Studio process (the server was
        # restarted, the engine kept running): its pid file lets us adopt it, so the
        # Stop button still works instead of the engine becoming an orphan holding
        # the VRAM (2026-10-29).
        self.adopted_pid = None
        try:
            with open(self.pid_path, encoding="utf-8") as fh:
                pid = int(fh.read().strip() or 0)
            if pid > 0 and _pid_alive(pid):
                self.adopted_pid = pid
            else:
                os.remove(self.pid_path)
        except (OSError, ValueError):
            pass

    # -- status ---------------------------------------------------------------
    def caps(self, max_age=5.0, timeout=4):
        """Cached probe: the caps dict when the instance answers, else None."""
        ts, caps = self._probe
        if time.time() - ts > max_age:
            caps = probe(self.url, timeout=timeout)
            self._probe = (time.time(), caps)
        return caps

    def forget(self):
        self._probe = (0.0, None)

    def started_by_us(self):
        if self.proc is not None:
            return self.proc.poll() is None
        if self.adopted_pid and _pid_alive(self.adopted_pid):
            return True
        self.adopted_pid = None
        return False

    def pid(self):
        return self.proc.pid if self.proc is not None else self.adopted_pid

    def status(self):
        caps = self.caps()
        return {"name": self.name, "url": self.url, "dir": self.dir,
                "running": caps is not None,
                "tool": (caps or {}).get("tool"),
                "version": (caps or {}).get("version"),
                "model_loaded": (caps or {}).get("model_loaded"),
                "faces": bool(((caps or {}).get("supports") or {}).get("faces")),
                "edit": bool(((caps or {}).get("supports") or {}).get("edit")),
                "max_refs": ((caps or {}).get("supports") or {}).get("max_refs"),
                # a list of checkpoint files in the family; anything else is ignored
                "models": [m for m in ((caps or {}).get("models") or [])
                           if isinstance(m, str)] if isinstance((caps or {}).get("models"), list) else [],
                "known_bases": list(KNOWN_BASES.get((caps or {}).get("tool") or "", [])),
                "started_by_us": self.started_by_us(), "pid": self.pid(),
                "launch_allowed": self.launch_allowed and bool(self.dir),
                "log": self.log_path if os.path.isfile(self.log_path) else None}

    # -- launch ---------------------------------------------------------------
    def launch(self, wait=True):
        """Starts <dir>/app.py when nothing answers at url. Returns
        {ok, started: bool, caps?} or {ok: false, error}."""
        with self._lock:
            caps = probe(self.url)
            if caps:
                self._probe = (time.time(), caps)
                return {"ok": True, "started": False, "caps": caps,
                        "message": f"{caps.get('tool')} already running at {self.url}"}
            if not self.launch_allowed:
                return {"ok": False, "error": f"engine '{self.name}' is not running at "
                        f"{self.url} and 'launch' is off in its config: start it by hand"}
            if not self.dir or not os.path.isdir(self.dir):
                return {"ok": False, "error": f"engine '{self.name}': install folder not "
                        f"found ({self.dir or 'empty'}); set 'dir' in Settings"}
            app = os.path.join(self.dir, "app.py")
            if not os.path.isfile(app):
                return {"ok": False, "error": f"engine '{self.name}': no app.py in {self.dir}"}
            py, label = resolve_python(self.dir, self.python)
            if not py:
                return {"ok": False, "error": f"engine '{self.name}': {label}"}
            if self.started_by_us():
                return {"ok": False, "error": f"engine '{self.name}' was started by us "
                        f"(pid {self.proc.pid}) but does not answer yet at {self.url}; "
                        f"see {self.log_path}"}
            os.makedirs(LOG_DIR, exist_ok=True)
            env = dict(os.environ)
            env["GRADIO_SERVER_PORT"] = str(_port_of(self.url))
            env["GRADIO_SERVER_NAME"] = "127.0.0.1"
            env.setdefault("PYTHONUTF8", "1")
            # The family apps print their boot log on stdout; it goes to the file.
            logf = open(self.log_path, "ab")
            logf.write(f"\n=== launched by crispz-dataset-studio {time.strftime('%Y-%m-%d %H:%M:%S')}"
                       f" :: {' '.join(py)} app.py (port {env['GRADIO_SERVER_PORT']})\n".encode())
            flags = 0
            if os.name == "nt":
                flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | \
                        getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                self.proc = subprocess.Popen(py + ["app.py"], cwd=self.dir, env=env,
                                             stdout=logf, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, creationflags=flags)
            except OSError as e:
                logf.close()
                return {"ok": False, "error": f"engine '{self.name}': cannot start: {e}"}
            try:
                with open(self.pid_path, "w", encoding="utf-8") as fh:
                    fh.write(str(self.proc.pid))
            except OSError:
                pass
            _log(f"{self.name}: started pid {self.proc.pid} with {label}, port "
                 f"{env['GRADIO_SERVER_PORT']}, log {self.log_path}")
        if not wait:
            return {"ok": True, "started": True, "pid": self.proc.pid}
        return self.wait_ready()

    def wait_ready(self, timeout=None):
        """Polls cli_caps until the app answers, or the child dies, or the timeout."""
        deadline = time.time() + float(timeout or self.boot_timeout)
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return {"ok": False, "error": f"engine '{self.name}' exited with code "
                        f"{self.proc.returncode} during boot; see {self.log_path}"}
            caps = probe(self.url, timeout=3)
            if caps:
                self._probe = (time.time(), caps)
                _log(f"{self.name}: {caps.get('tool')} {caps.get('version')} ready at {self.url}")
                return {"ok": True, "started": True, "caps": caps,
                        "pid": self.proc.pid if self.proc else None,
                        "message": "engine started; the model loads at the first image, "
                                   "so the first card takes longer"}
            time.sleep(1.0)
        return {"ok": False, "error": f"engine '{self.name}' did not answer at {self.url} "
                f"within {int(self.boot_timeout)} s; see {self.log_path}"}

    def stop(self, timeout=20):
        """Terminates the child WE started. An engine started by the user is left alone
        (we say so instead of killing someone else's session)."""
        if not self.started_by_us():
            self.proc = None
            self.forget()
            if probe(self.url, timeout=2):
                return {"ok": False, "error": f"engine '{self.name}' at {self.url} was not "
                        f"started by Dataset Studio: close it yourself"}
            return {"ok": True, "stopped": False, "message": "nothing to stop"}
        pid = self.pid()
        try:
            if self.proc is not None:
                p = self.proc
                p.terminate()
                try:
                    p.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)
            else:
                _kill_pid(pid)
                deadline = time.time() + timeout
                while _pid_alive(pid) and time.time() < deadline:
                    time.sleep(0.5)
        except Exception as e:
            return {"ok": False, "error": f"engine '{self.name}': stop failed: {e}"}
        _log(f"{self.name}: stopped pid {pid}")
        self.proc, self.adopted_pid = None, None
        try:
            os.remove(self.pid_path)
        except OSError:
            pass
        self.forget()
        return {"ok": True, "stopped": True}

    # -- protocol ops ---------------------------------------------------------
    def call(self, name, data, timeout=3600):
        """One endpoint call, {ok: false, error} on any failure."""
        try:
            out = _gradio_call(self.url, name, data, get_timeout=timeout)
        except Exception as e:
            return {"ok": False, "error": f"{self.name} ({name}): {type(e).__name__}: {e}"}
        if not isinstance(out, dict):
            return {"ok": False, "error": f"{self.name} ({name}): unexpected answer {out!r}"}
        return out

    def edit(self, spec, timeout=3600):
        """op edit: image + instruction -> image. `spec` is a v1 spec dict; protocol
        and op are filled in. Returns the tool's JSON ({images: [path], seed_used...})."""
        spec = dict(spec)
        spec["protocol"] = PROTOCOL
        spec["op"] = "edit"
        return self.call("cli_edit", [json.dumps(spec)], timeout=timeout)

    def gen(self, spec, timeout=3600):
        spec = dict(spec)
        spec["protocol"] = PROTOCOL
        spec["op"] = "gen"
        return self.call("cli_gen", [json.dumps(spec)], timeout=timeout)

    def faces(self, image_path, timeout=120):
        """cli_faces: boxes + mouths + ArcFace embeddings of a local image."""
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
        except OSError as e:
            return {"ok": False, "error": f"cannot read {image_path}: {e}"}
        return self.call("cli_faces", [b64], timeout=timeout)


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
class Engines:
    def __init__(self, config):
        self.by_name = {}
        self.reload(config)

    def reload(self, config):
        """Re-reads the 'engines' block. Children we started are kept across a
        settings change (same name)."""
        new = {}
        for name, cfg in (config.get("engines") or {}).items():
            if not isinstance(cfg, dict):
                continue
            e = Engine(name, cfg)
            old = self.by_name.get(name)
            if old is not None and old.started_by_us():
                e.proc, e.adopted_pid = old.proc, old.adopted_pid
            new[name] = e
        self.by_name = new
        self.default = str(config.get("engine") or "").strip() or \
            (next(iter(new)) if new else "")

    def get(self, name=None):
        name = name or self.default
        e = self.by_name.get(name)
        if e is None:
            raise KeyError(f"unknown engine '{name}' (configured: "
                           f"{', '.join(self.by_name) or 'none'})")
        return e

    def alive_child(self):
        """The engine we started that is still running, if any (one at a time)."""
        for e in self.by_name.values():
            if e.started_by_us():
                return e
        return None

    def ensure(self, name=None):
        """The engine, running: probes, launches when allowed, refuses a second
        child while one is alive (the family's VRAM rule)."""
        e = self.get(name)
        if e.caps(max_age=0):
            return e, {"ok": True, "started": False}
        other = self.alive_child()
        if other is not None and other is not e:
            return e, {"ok": False, "error": f"engine '{other.name}' (started by Dataset "
                       f"Studio) is still running: two family apps do not fit on one "
                       f"GPU. Stop it first (Settings > Engines)"}
        return e, e.launch(wait=True)

    def statuses(self):
        return [e.status() for e in self.by_name.values()]

    def stop_all(self):
        for e in self.by_name.values():
            if e.started_by_us():
                e.stop()
