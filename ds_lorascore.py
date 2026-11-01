"""crispz-dataset-studio - the likeness of a trained LoRA, on the same scale as the cards.

The test of the report of 2026-10-31, automated: a few txt2img renders in a family
engine (crispz-klein) WITH the LoRA and, on the same prompt and seed, WITHOUT it, each
scored against the project's reference (ArcFace cosine x 100, ds_likeness.score).
The LoRA goes into the engine's LoRA folder first (ds_aitoolkit.copy_lora: the same
file is never copied twice). Several saves of one job (1000 / 1250 / 1500) are scored
one after the other to pick the best step.

Files: projects/<name>/lora_tests/<lora stem>/NN_lora.png, NN_base.png, scores.json.
project.json: "lora_scores": [{lora, job, engine, model, weight, mean_lora,
mean_base, scored, rows: [{prompt, seed, lora: {file, score}, base: {...}}], when}].

The engine's model must match the LoRA (a Klein 4B LoRA is skipped on the 9B, the
engine says so). The protocol's gen op does not switch the running instance's model
unless asked with spec.switch_model (crispz-klein, supports.switch_model in its caps):
when the engine runs the other variant, or nothing yet, the renders carry the right
base repo and switch_model, the swap is announced, and the model stays loaded. With an
older engine, or the switch refused, the run stops with the reason instead of scoring
renders that ignored the LoRA (checked before the first render and after it).
"""

import os
import json
import time
import shutil
import threading

import ds_project
import ds_likeness
import ds_aitoolkit

DEFAULT_PROMPTS = [
    "close-up portrait, facing the camera, neutral expression, soft studio light, plain grey background",
    "medium shot, black turtleneck, arms crossed, city street",
    "full body, walking on a beach at sunset, white dress, facing the camera",
    "sitting at a cafe table with a coffee cup, smiling",
]
DEFAULT_SEED = 42
DEFAULT_SIZE = (832, 1216)
# the checkpoint each ai-toolkit arch needs on the engine side
ARCH_MODEL = {"flux2_klein_4b": "4B", "flux2_klein_9b": "9B"}
# the distilled checkpoint the engine renders with, per variant (the report's tests)
VARIANT_REPO = {"4B": "black-forest-labs/FLUX.2-klein-4B",
                "9B": "black-forest-labs/FLUX.2-klein-9B"}

_LOG_HOOK = None


def set_log_hook(fn):
    global _LOG_HOOK
    _LOG_HOOK = fn


def _log(msg):
    (_LOG_HOOK or (lambda s: print(s, flush=True)))(f"[lora-score] {msg}")


def model_mismatch(arch, model_loaded):
    """A reason string when the engine's model cannot take this LoRA, else None.
    Unknown arch or no model loaded yet: None (checked again after a render)."""
    need = ARCH_MODEL.get(arch or "")
    m = str(model_loaded or "")
    if not need or not m or m in ("True", "False"):
        return None
    has4, has9 = "4B" in m.upper(), "9B" in m.upper()
    if (need == "4B" and has9 and not has4) or (need == "9B" and has4 and not has9):
        return (f"the engine runs {m} but the LoRA was trained on Klein {need}: it would be "
                f"skipped. Switch the Klein checkpoint to FLUX.2-klein-{need} in the engine "
                f"(menu 'Klein checkpoint'), then score again")
    return None


def _model_name(v):
    # model_loaded is a name on the instance side; an older tool answers a bool
    return v if isinstance(v, str) else ""


def _lora_skipped(warnings):
    for w in warnings or []:
        lw = str(w).lower()
        if "lora" in lw and any(k in lw for k in ("skip", "incompatib", "mismatch", "not loaded",
                                                  "not found", "ignored")):
            return str(w)
    return None


class Scorer:
    """One scoring at a time; it shares the GPU with the dataset run, so the server
    refuses one while the other works."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.cancel_flag = False
        self.state = {"running": False, "project": None, "done": 0, "total": 0,
                      "text": "", "ok": None, "error": "", "results": []}

    def status(self):
        return dict(self.state)

    def cancel(self):
        if self.state["running"]:
            self.cancel_flag = True
            return True
        return False

    def start(self, name, engines, engine_name, root, job, files, prompts=None,
              weight=1.0, baseline=True, seed=DEFAULT_SEED, size=DEFAULT_SIZE,
              steps=None, loras_dir=None, switch_model=True):
        with self.lock:
            if self.state["running"]:
                raise RuntimeError("a LoRA scoring is already running")
            p = ds_project.load(name)
            if not p.get("reference"):
                raise ValueError("the project has no reference photo to score against")
            s = p.get("settings") or {}
            trig = (s.get("trigger_word") or "").strip()
            lines = [l.strip() for l in (prompts or DEFAULT_PROMPTS) if l and l.strip()]
            if not lines:
                raise ValueError("no test prompt")
            full = [l if not trig or l.lower().startswith(trig.lower()) else f"{trig}, {l}"
                    for l in lines]
            all_loras = {l["file"]: l for l in ds_aitoolkit.job_loras(root, job)}
            files = [f for f in (files or []) if f]
            if not files and all_loras:
                files = [next(iter(sorted(all_loras.values(),
                                          key=lambda x: (not x["final"], -(x["step"] or 0)))))["file"]]
            missing = [f for f in files if f not in all_loras]
            if not files or missing:
                raise FileNotFoundError(f"LoRA not found in job '{job}': "
                                        f"{', '.join(missing) or 'none trained yet'}")
            total = len(files) * len(full) + (len(full) if baseline else 0)
            self.cancel_flag = False
            self.state = {"running": True, "project": p["name"], "done": 0, "total": total,
                          "text": "starting", "ok": None, "error": "", "results": [],
                          "job": job, "started": time.time()}
            args = (p["name"], engines, engine_name, root, job,
                    [all_loras[f] for f in files], full, float(weight), bool(baseline),
                    int(seed), size, steps, loras_dir, bool(switch_model))
            self.thread = threading.Thread(target=self._run, args=args, daemon=True,
                                           name="ds-lora-score")
            self.thread.start()
            return self.status()

    def wait(self, timeout=None):
        if self.thread:
            self.thread.join(timeout)
        return not self.state["running"]

    def _say(self, t):
        self.state["text"] = t
        _log(t)

    def _render(self, engine, spec, out_path):
        r = engine.gen(spec)
        if not r.get("ok"):
            raise RuntimeError(r.get("error") or "engine error")
        src = (r.get("images") or [None])[0]
        if not src or not os.path.isfile(src):
            raise RuntimeError(f"engine returned no image file ({src})")
        shutil.move(src, out_path)
        for ext in (".json", ".txt"):
            side = os.path.splitext(src)[0] + ext
            if os.path.isfile(side):
                try:
                    os.remove(side)
                except OSError:
                    pass
        return r

    def _score(self, engine, ref_emb, path):
        fr = engine.faces(path)
        if not fr.get("ok"):
            return 0, {"reason": f"faces: {fr.get('error')}"}
        faces = fr.get("faces") or []
        if not faces:
            return None, {"reason": "no face (rear view?): not scored"}
        return ds_likeness.score(ref_emb, faces)

    def _run(self, name, engines, engine_name, root, job, loras, prompts, weight,
             baseline, seed, size, steps, loras_dir, switch_model=True):
        try:
            p = ds_project.load(name)
            self._say(f"engine: checking {engine_name or engines.default}")
            engine, r = engines.ensure(engine_name or None)
            if not r.get("ok"):
                raise RuntimeError(r.get("error"))
            caps = engine.caps(max_age=0) or {}
            if not (caps.get("supports") or {}).get("faces", True):
                raise RuntimeError(f"{engine.name} has no face service: nothing to score with")
            arch = ds_aitoolkit.job_arch(root, job)
            loaded = _model_name(caps.get("model_loaded"))
            why = model_mismatch(arch, loaded)
            switch = None
            need = ARCH_MODEL.get(arch or "")
            can = bool((caps.get("supports") or {}).get("switch_model"))
            if need and (why or not loaded) and switch_model and can:
                switch = VARIANT_REPO[need]
                self._say(f"engine on {loaded or 'no model yet'}: switching it to {switch} "
                          f"for the LoRA (it stays loaded afterwards)")
            elif why:
                raise RuntimeError(why + ("" if can or not switch_model else
                                          " (this engine cannot switch it from here: "
                                          "update it for the protocol's switch_model)"))
            if not p.get("ref_embedding"):
                w = ds_project.refresh_reference_embedding(p, engine)
                ds_project.save(p)
                if not p.get("ref_embedding"):
                    raise RuntimeError(f"reference: {w}")
            ref_emb = p["ref_embedding"]
            pdir = ds_project.project_dir(name)
            dest = loras_dir or ds_aitoolkit.engine_loras_dir(getattr(engine, "dir", ""))
            if not dest:
                raise RuntimeError(f"engine '{engine.name}' has no install folder in Settings: "
                                   f"its LoRA folder is unknown")
            spec0 = {"seed": seed, "width": int(size[0]), "height": int(size[1])}
            if steps:
                spec0["steps"] = int(steps)
            if switch:
                # every render, base included: the comparison is on the same model
                spec0["model"], spec0["switch_model"] = switch, True
            base_rows = None
            if baseline:
                base_rows = []
                bdir = os.path.join(pdir, "lora_tests", "_base")
                os.makedirs(bdir, exist_ok=True)
                for i, pr in enumerate(prompts, 1):
                    if self.cancel_flag:
                        raise InterruptedError()
                    self._say(f"base {i}/{len(prompts)}: rendering without LoRA")
                    out = os.path.join(bdir, f"{i:02d}_base.png")
                    rr = self._render(engine, dict(spec0, prompt=pr, loras=[],
                                                   out_dir=bdir), out)
                    sc, det = self._score(engine, ref_emb, out)
                    base_rows.append({"file": os.path.relpath(out, pdir).replace(os.sep, "/"),
                                      "score": sc, "detail": det,
                                      "warnings": rr.get("warnings") or []})
                    self.state["done"] += 1
            history = p.get("lora_scores") or []
            for lo in loras:
                if self.cancel_flag:
                    raise InterruptedError()
                cp = ds_aitoolkit.copy_lora(lo["path"], dest,
                                            meta=ds_aitoolkit.lora_meta_for(p, job))
                stem = os.path.splitext(cp["file"])[0]
                tdir = os.path.join(pdir, "lora_tests", stem)
                os.makedirs(tdir, exist_ok=True)
                rows = []
                for i, pr in enumerate(prompts, 1):
                    if self.cancel_flag:
                        raise InterruptedError()
                    self._say(f"{lo['file']} {i}/{len(prompts)}: rendering with the LoRA "
                              f"at {weight:g}")
                    out = os.path.join(tdir, f"{i:02d}_lora.png")
                    rr = self._render(engine, dict(spec0, prompt=pr, out_dir=tdir,
                                                   loras=[f"{cp['file']}:{weight:g}"]), out)
                    skipped = _lora_skipped(rr.get("warnings"))
                    if skipped:
                        raise RuntimeError(f"the engine did not apply the LoRA: {skipped}")
                    if i == 1:
                        why = model_mismatch(arch, (engine.caps(max_age=0) or {}).get("model_loaded"))
                        if why:
                            raise RuntimeError(why)
                    sc, det = self._score(engine, ref_emb, out)
                    row = {"prompt": pr, "seed": seed,
                           "lora": {"file": os.path.relpath(out, pdir).replace(os.sep, "/"),
                                    "score": sc, "detail": det}}
                    if base_rows:
                        row["base"] = base_rows[i - 1]
                    rows.append(row)
                    self.state["done"] += 1
                    self._say(f"{lo['file']} {i}/{len(prompts)}: likeness "
                              f"{'n/s' if sc is None else sc}"
                              + (f" (without: {base_rows[i - 1]['score']})" if base_rows else ""))
                both = [r for r in rows if r["lora"]["score"] is not None
                        and (not base_rows or r["base"]["score"] is not None)]
                mean = lambda k: (round(sum(r[k]["score"] for r in both) / len(both), 1)
                                  if both else None)
                res = {"lora": lo["file"], "job": job, "step": lo.get("step"),
                       "final": lo.get("final"), "engine": engine.name,
                       "model": _model_name((engine.caps(max_age=0) or {}).get("model_loaded")),
                       "weight": weight, "seed": seed, "size": list(size),
                       "scored": len(both), "mean_lora": mean("lora"),
                       "mean_base": mean("base") if base_rows else None,
                       "rows": rows, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
                with open(os.path.join(tdir, "scores.json"), "w", encoding="utf-8") as fh:
                    json.dump(res, fh, ensure_ascii=False, indent=1)
                self.state["results"].append(res)
                history = [h for h in history if not (h.get("lora") == lo["file"]
                                                      and h.get("job") == job
                                                      and h.get("weight") == weight)]
                history.append(res)
                p = ds_project.load(name)
                p["lora_scores"] = history
                ds_project.save(p)
                self._say(f"{lo['file']}: mean likeness {res['mean_lora']} on {res['scored']} "
                          f"shot(s)" + (f", {res['mean_base']} without the LoRA"
                                        if base_rows else ""))
            self.state["ok"] = True
        except InterruptedError:
            self.state["ok"] = False
            self.state["error"] = "cancelled"
            self._say("cancelled")
        except Exception as e:
            self.state["ok"] = False
            self.state["error"] = f"{type(e).__name__}: {e}" if not isinstance(e, RuntimeError) else str(e)
            self._say(f"failed: {self.state['error']}")
        finally:
            self.state["running"] = False
            self.state["finished"] = time.time()


SCORER = Scorer()
