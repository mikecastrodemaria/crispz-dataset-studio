"""crispz-dataset-studio - hand a finished dataset to ai-toolkit (Ostris).

Everything stays on this PC: the lot (NNN.png + NNN.txt pairs) is copied into
ai-toolkit's datasets folder and a training config is written in its config folder,
the layout that worked on 2026-10-31 (docs/lora-klein-2026-10-31.md). Nothing is
started: the result carries the command line, the user launches it (from ai-toolkit's
own UI, which lists the new dataset, or from a terminal).

    <root>/datasets/<job>/      the pairs
    <root>/config/<job>.yaml    the job, run with: <python> run.py config/<job>.yaml

The setting `ai_toolkit_dir` may name the ai-toolkit folder itself, a folder inside
it (output, datasets, config: the path a user copies from the explorer) or the
Easy-Install folder above it: find_root walks up and down to the folder holding
run.py. `ai_toolkit_datasets_dir` overrides the datasets folder (ai-toolkit's UI has
its own setting for it).
"""

import os
import re
import json
import shutil

_LOG_HOOK = None


def set_log_hook(fn):
    global _LOG_HOOK
    _LOG_HOOK = fn


def _log(msg):
    line = f"[ai-toolkit] {msg}"
    (_LOG_HOOK or (lambda s: print(s, flush=True)))(line)


# The recipes of the report of 2026-10-31 (rank 16, adamw8bit, lr 1e-4, sigmoid,
# 1500 steps, samples every 250 at 832x1216). 4B: 17 GB of VRAM, 24 min on a 5090.
# 9B: transformer in qfloat8, 24 GB, 50 min; its base repo is gated on Hugging Face.
PRESETS = {
    "klein4b": {"label": "FLUX.2 Klein 4B (base, free)", "arch": "flux2_klein_4b",
                "name_or_path": "black-forest-labs/FLUX.2-klein-base-4B",
                "quantize": False, "note": "about 17 GB of VRAM, 0.8 s per step on a 5090"},
    "klein9b": {"label": "FLUX.2 Klein 9B (base, gated on HF)", "arch": "flux2_klein_9b",
                "name_or_path": "black-forest-labs/FLUX.2-klein-base-9B",
                "quantize": True,
                "note": "about 24 GB of VRAM, 1.6 s per step; accept the licence on the "
                        "HF page with the account whose token is on this PC"},
}
DEFAULT_PRESET = "klein4b"
DEFAULT_STEPS = 1500
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _is_root(d):
    return os.path.isfile(os.path.join(d, "run.py")) and os.path.isdir(os.path.join(d, "toolkit"))


def find_root(path):
    """The ai-toolkit folder (the one holding run.py and toolkit/) from a setting that
    may point at it, inside it, or above it. None when nothing matches."""
    p = os.path.normpath(os.path.expanduser(str(path or "").strip().strip('"')))
    if not path or not str(path).strip() or not os.path.isdir(p):
        return None
    # the folder itself, then its parents (D:/.../AI-Toolkit/output -> AI-Toolkit)
    d = p
    while True:
        if _is_root(d):
            return d
        up = os.path.dirname(d)
        if up == d:
            break
        d = up
    # below it: the Easy-Install layout nests AI-Toolkit two levels down
    for sub in ("AI-Toolkit", "ai-toolkit", os.path.join("AI-Toolkit-Easy-Install", "AI-Toolkit")):
        c = os.path.join(p, sub)
        if _is_root(c):
            return c
    return None


def python_for(root):
    """ai-toolkit's own interpreter: its .venv (the Easy-Install python_embeded has no
    torch, report of 2026-10-31), else 'python'."""
    for rel in (os.path.join(".venv", "Scripts", "python.exe"), os.path.join(".venv", "bin", "python"),
                os.path.join("venv", "Scripts", "python.exe"), os.path.join("venv", "bin", "python")):
        c = os.path.join(root, rel)
        if os.path.isfile(c):
            return c
    return "python"


def status(cfg):
    """What the UI shows: configured or not, the root found, the presets."""
    raw = (cfg or {}).get("ai_toolkit_dir") or ""
    root = find_root(raw) if raw else None
    out = {"configured": bool(str(raw).strip()), "setting": raw, "root": root,
           "presets": [{"name": k, "label": v["label"], "note": v["note"]} for k, v in PRESETS.items()],
           "default_preset": DEFAULT_PRESET, "default_steps": DEFAULT_STEPS}
    if raw and not root:
        out["error"] = (f"no ai-toolkit found from '{raw}': point the setting at the folder "
                        f"that holds run.py")
    if root:
        out["datasets_dir"] = datasets_dir(root, cfg)
        out["python"] = python_for(root)
    return out


def datasets_dir(root, cfg=None):
    d = str((cfg or {}).get("ai_toolkit_datasets_dir") or "").strip().strip('"')
    return os.path.normpath(os.path.expanduser(d)) if d else os.path.join(root, "datasets")


def _slug(s):
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(s or "").strip()).strip("_-")
    return s[:60] or "dataset"


def job_name(root, ds_dir, base):
    """<base>_v1, or the next free vN: a job never overwrites a dataset, a config or
    an output folder of an earlier one."""
    base = _slug(base)
    n = 1
    while True:
        name = f"{base}_v{n}"
        if not (os.path.exists(os.path.join(ds_dir, name))
                or os.path.exists(os.path.join(root, "config", name + ".yaml"))
                or os.path.exists(os.path.join(root, "output", name))):
            return name
        n += 1


def _q(v):
    """A YAML scalar: JSON strings are valid YAML double-quoted strings."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v), ensure_ascii=False)


def _path(p):
    # forward slashes: a Windows path with backslashes in a quoted YAML string broke
    # ai-toolkit's parser (report of 2026-10-31, "unknown escape character")
    return str(p).replace("\\", "/")


def build_config(job, dataset_path, preset, steps, trigger, sample_prompts, resolutions):
    pr = PRESETS[preset]
    lines = [
        "# Written by crispz-dataset-studio. Run from the ai-toolkit folder:",
        f"#   python run.py config/{job}.yaml",
        "job: extension",
        "config:",
        f"  name: {_q(job)}",
        "  process:",
        "    - type: \"sd_trainer\"",
        "      training_folder: \"output\"",
        "      device: \"cuda:0\"",
        f"      trigger_word: {_q(trigger)}",
        "      network:",
        "        type: \"lora\"",
        "        linear: 16",
        "        linear_alpha: 16",
        "      save:",
        "        dtype: \"float16\"",
        "        save_every: 250",
        "        max_step_saves_to_keep: 6",
        "      datasets:",
        f"        - folder_path: {_q(_path(dataset_path))}",
        "          caption_ext: \"txt\"",
        "          caption_dropout_rate: 0.05",
        "          shuffle_tokens: false",
        "          cache_latents_to_disk: true",
        f"          resolution: [{', '.join(str(r) for r in resolutions)}]",
        "      train:",
        "        batch_size: 1",
        f"        steps: {int(steps)}",
        "        gradient_accumulation_steps: 1",
        "        train_unet: true",
        "        train_text_encoder: false",
        "        gradient_checkpointing: true",
        "        noise_scheduler: \"flowmatch\"",
        "        timestep_type: \"sigmoid\"",
        "        optimizer: \"adamw8bit\"",
        "        lr: 1e-4",
        "        dtype: \"bf16\"",
        "      model:",
        f"        name_or_path: {_q(pr['name_or_path'])}",
        f"        arch: {_q(pr['arch'])}",
        f"        quantize: {_q(pr['quantize'])}",
    ]
    if pr["quantize"]:
        lines.append("        qtype: \"qfloat8\"")
    lines += [
        "        quantize_te: true",
        "      sample:",
        "        sampler: \"flowmatch\"",
        "        sample_every: 250",
        "        width: 832",
        "        height: 1216",
        "        prompts:",
    ]
    lines += [f"          - {_q(s)}" for s in sample_prompts]
    lines += [
        "        neg: \"\"",
        "        seed: 42",
        "        walk_seed: true",
        "        guidance_scale: 4",
        "        sample_steps: 25",
        "meta:",
        f"  name: {_q(job)}",
        "  version: \"1.0\"",
        "",
    ]
    return "\n".join(lines)


def _sample_prompts(captions, trigger, n=4):
    caps = [c for c in captions if c]
    if len(caps) > n:
        step = len(caps) / float(n)
        caps = [caps[int(i * step)] for i in range(n)]
    if not caps:
        t = trigger or "the subject"
        caps = [f"{t}, close-up portrait, facing the camera, plain grey background",
                f"{t}, full body, walking in a city street"]
    return caps


def create_job(project, lot_dir, cfg, preset=None, steps=None, base_name=None):
    """Copies the lot into ai-toolkit's datasets folder and writes the job config.
    Returns {job, root, dataset_dir, config_path, command, pairs, preset, warnings}."""
    root = find_root((cfg or {}).get("ai_toolkit_dir"))
    if not root:
        raise FileNotFoundError("ai-toolkit folder not set or not found (Settings, "
                                "AI-Toolkit folder: the one that holds run.py)")
    preset = preset or DEFAULT_PRESET
    if preset not in PRESETS:
        raise ValueError(f"unknown preset '{preset}' (known: {', '.join(PRESETS)})")
    steps = int(steps or DEFAULT_STEPS)
    if not 50 <= steps <= 100000:
        raise ValueError("steps must be between 50 and 100000")
    if not lot_dir or not os.path.isdir(lot_dir):
        raise FileNotFoundError(f"export folder not found: {lot_dir}")
    images = sorted(f for f in os.listdir(lot_dir) if f.lower().endswith(IMAGE_EXTS))
    if not images:
        raise ValueError("the export holds no image: render and export the dataset first")
    warnings = []
    ds_root = datasets_dir(root, cfg)
    os.makedirs(ds_root, exist_ok=True)
    os.makedirs(os.path.join(root, "config"), exist_ok=True)
    s = project.get("settings") or {}
    job = job_name(root, ds_root, base_name or f"{project['name']}_{preset}")
    dst = os.path.join(ds_root, job)
    os.makedirs(dst)
    captions, missing = [], 0
    for f in images:
        shutil.copy2(os.path.join(lot_dir, f), os.path.join(dst, f))
        txt = os.path.splitext(f)[0] + ".txt"
        if os.path.isfile(os.path.join(lot_dir, txt)):
            shutil.copy2(os.path.join(lot_dir, txt), os.path.join(dst, txt))
            with open(os.path.join(lot_dir, txt), encoding="utf-8") as fh:
                captions.append(fh.read().strip())
        else:
            missing += 1
    if missing:
        warnings.append(f"{missing} image(s) without a caption file: ai-toolkit trains "
                        f"them on the trigger word alone")
    # The captions already start with the trigger word (ds_captions.compose): telling
    # ai-toolkit again would double it. Only when some caption lacks it is it passed.
    trig = (s.get("trigger_word") or "").strip()
    if not trig:
        warnings.append("no trigger word in the project: the LoRA has no keyword to call it")
        trigger = None
    elif all(c.lower().startswith(trig.lower()) for c in captions) and not missing:
        trigger = None
    else:
        trigger = trig
    size = int(s.get("export_size") or 1024)
    res = [r for r in (512, 768, 1024) if r <= size] or [512]
    text = build_config(job, dst, preset, steps, trigger, _sample_prompts(captions, trig), res)
    cfg_path = os.path.join(root, "config", job + ".yaml")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    py = python_for(root)
    if py.startswith(root + os.sep):
        py = os.path.relpath(py, root)
    if os.name == "nt":
        # cmd.exe: /d to change drive too; PYTHONUTF8 as in the report of 2026-10-31
        command = f'cd /d "{root}" && set PYTHONUTF8=1 && "{py}" run.py config/{job}.yaml'
    else:
        command = f'cd "{root}" && PYTHONUTF8=1 "{py}" run.py config/{job}.yaml'
    if PRESETS[preset]["name_or_path"].endswith("9B"):
        warnings.append("FLUX.2-klein-base-9B is gated: accept its licence on Hugging Face "
                        "with the account whose token is on this PC before training")
    _log(f"job '{job}': {len(images)} pair(s) copied to {dst}, config {cfg_path} "
         f"({PRESETS[preset]['label']}, {steps} steps)")
    for w in warnings:
        _log(f"job '{job}': {w}")
    return {"job": job, "root": root, "dataset_dir": dst, "config_path": cfg_path,
            "command": command, "pairs": len(images), "preset": preset, "steps": steps,
            "warnings": warnings}


# ----------------------------------------------------------------------------
# Training: run.py config/<job>.yaml in ai-toolkit's own interpreter
# ----------------------------------------------------------------------------
# One training at a time (one GPU). The process is detached from Dataset Studio:
# closing the studio does NOT stop a 50-minute training; its pid, log and job are
# kept in logs/training.json so the next start picks the status up again. Stop kills
# the whole process tree (the dataloader workers too).
#
# Progress: ai-toolkit prints a tqdm bar per step, "<job>:  17%|..| 250/1500 [03:20<..,
# 1.25it/s, lr: 1.0e-04 loss: 6.500e-01]"; the last bar whose total equals the job's
# steps is the training (the latent cache bars count images, not steps).
import time
import threading
import subprocess

import ds_engines

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(APP_DIR, "logs", "training.json")
_BAR = re.compile(r"(\d+)/(\d+)\s*\[([0-9:]+)?(?:<([0-9:?]+))?")
_LOSS = re.compile(r"loss:\s*([0-9.eE+-]+)")


def _hf_token():
    """HF_TOKEN for the gated 9B base: the environment, else the token file the
    huggingface CLI writes. ai-toolkit reads HF_TOKEN, and an empty value in its own
    settings was enough to lose the gated repo (report of 2026-10-31)."""
    t = (os.environ.get("HF_TOKEN") or "").strip()
    if t:
        return t
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    try:
        with open(os.path.join(home, "token"), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _steps_of(config_path):
    try:
        with open(config_path, encoding="utf-8") as fh:
            m = re.search(r"^\s*steps:\s*(\d+)", fh.read(), re.M)
        return int(m.group(1)) if m else None
    except OSError:
        return None


def parse_progress(text, total_steps=None):
    """{step, total, elapsed, eta, loss} from the tail of the log, or {}."""
    out = {}
    for chunk in re.split(r"[\r\n]+", text or ""):
        for m in _BAR.finditer(chunk):
            step, total = int(m.group(1)), int(m.group(2))
            if total_steps and total != total_steps:
                continue
            if not total_steps and total < 50:
                continue
            out = {"step": step, "total": total, "elapsed": m.group(3) or "",
                   "eta": m.group(4) or ""}
            lm = _LOSS.search(chunk)
            if lm:
                out["loss"] = lm.group(1)
    return out


def _tail(path, n=16000):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - n))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _last_lines(text, k=12):
    lines = [l.rstrip() for l in re.split(r"[\r\n]+", text) if l.strip()]
    return lines[-k:]


def _kill_tree(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                       capture_output=True, timeout=30)
    else:
        import signal
        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            ds_engines._kill_pid(pid)


class Trainer:
    def __init__(self, state_path=None):
        self.state_path = state_path or STATE_PATH
        self.proc = None
        self.lock = threading.Lock()
        self.state = self._load()

    def _load(self):
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self.state_path)

    def running(self):
        s = self.state
        if not s.get("pid") or s.get("finished"):
            return False
        if self.proc is not None:
            alive = self.proc.poll() is None
        else:
            alive = ds_engines._pid_alive(s["pid"])     # started by an earlier session
        if not alive:
            self._finish()
        return alive

    def _finish(self):
        s = self.state
        if s.get("finished"):
            return
        rc = self.proc.poll() if self.proc is not None else None
        s["finished"] = time.time()
        s["returncode"] = rc
        lora = os.path.join(s["root"], "output", s["job"], s["job"] + ".safetensors")
        s["lora"] = lora if os.path.isfile(lora) else ""
        if s.get("stopped"):
            s["result"] = "stopped"
        elif s["lora"] and rc in (0, None):
            s["result"] = "done"
        else:
            s["result"] = "failed"
        self._save()
        _log(f"training '{s['job']}': {s['result']}"
             + (f", LoRA {s['lora']}" if s["lora"] else "")
             + (f", exit code {rc}" if rc not in (0, None) else "")
             + f" (log {s['log']})")

    def start(self, root, job, project=None):
        with self.lock:
            if self.running():
                raise RuntimeError(f"training '{self.state['job']}' is already running")
            cfg_path = os.path.join(root, "config", job + ".yaml")
            if not os.path.isfile(cfg_path):
                raise FileNotFoundError(f"no config for job '{job}': {cfg_path}")
            os.makedirs(os.path.join(root, "output"), exist_ok=True)
            log_path = os.path.join(root, "output", job + ".dataset-studio.log")
            py = python_for(root)
            env = dict(os.environ)
            env["PYTHONUTF8"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            tok = _hf_token()
            if tok:
                env["HF_TOKEN"] = tok
            argv = [py, "run.py", f"config/{job}.yaml"]
            logf = open(log_path, "ab")
            logf.write(f"\n=== launched by crispz-dataset-studio {time.strftime('%Y-%m-%d %H:%M:%S')}"
                       f" :: {' '.join(argv)}\n".encode())
            logf.flush()
            kw = {}
            if os.name == "nt":
                kw["creationflags"] = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                                       | getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                kw["start_new_session"] = True
            try:
                self.proc = subprocess.Popen(argv, cwd=root, env=env, stdout=logf,
                                             stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, **kw)
            except OSError as e:
                raise RuntimeError(f"cannot start ai-toolkit with {py}: {e}")
            finally:
                logf.close()
            self.state = {"job": job, "root": root, "project": project, "pid": self.proc.pid,
                          "log": log_path, "config": cfg_path, "steps": _steps_of(cfg_path),
                          "started": time.time(), "finished": None, "result": "running",
                          "hf_token": bool(tok)}
            self._save()
            _log(f"training '{job}' started (pid {self.proc.pid}, {py}), log {log_path}"
                 + ("" if tok else "; no Hugging Face token found: a gated base will fail"))
            return self.status()

    def stop(self):
        with self.lock:
            if not self.running():
                return False
            self.state["stopped"] = True
            _log(f"training '{self.state['job']}': stop requested (pid {self.state['pid']})")
            _kill_tree(self.state["pid"])
            if self.proc is not None:
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            self._finish()
            return True

    def status(self):
        s = dict(self.state)
        if not s.get("job"):
            return {"active": False}
        run = self.running()
        s = dict(self.state)
        text = _tail(s.get("log") or "")
        s.update(running=run, active=True, progress=parse_progress(text, s.get("steps")),
                 tail=_last_lines(text))
        s["minutes"] = round(((s.get("finished") or time.time()) - s["started"]) / 60.0, 1)
        return s


TRAINER = None


def trainer():
    global TRAINER
    if TRAINER is None:
        TRAINER = Trainer()
    return TRAINER
