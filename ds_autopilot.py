"""crispz-dataset-studio - Autopilot: from the reference photo to a scored LoRA, hands off.

One Run with the project's Autopilot box ticked chains the steps the user otherwise
clicks one by one, each checked before the next:

    1. dataset   the run (ds_project.Runner); it must end complete: no pending or
                 failed card (a rejected one is fine), at least MIN_CARDS kept
    2. export    a fresh lot (the captions as they are now)
    3. job       ai-toolkit dataset + config (ds_aitoolkit.create_job), the project's
                 auto_preset / auto_steps
    4. training  the engine Dataset Studio started is stopped first (one GPU: ticking
                 Autopilot is the consent the manual button asks for), then run.py
    5. score     when a scoring engine is set (auto_score_engine, default klein): the
                 LoRA is copied into it and scored with and without (ds_lorascore),
                 the final LoRA and the two latest step saves, switching the engine's
                 Klein checkpoint when it can

A QUEUE runs several projects end to end, one after the other, each with its own
auto_* settings (a night batch). A project that fails or is cancelled does not stop
the queue: its result and reason are recorded and the next one starts. "Cancel all"
stops the project in flight and empties the queue.

Any step that fails stops the chain with its reason, said in the status and the log;
nothing is skipped silently. Cancel stops the step in flight (the run, the training,
the scoring). The chain is written to logs/autopilot.json: when Dataset Studio is
restarted during the training (which is detached and keeps going), the chain picks up
at the training and goes on to the score; any other step interrupted that way is
reported as such.
"""

import os
import json
import time
import threading

import ds_project
import ds_aitoolkit
import ds_lorascore

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(APP_DIR, "logs", "autopilot.json")
MIN_CARDS = 5
STEPS = ("dataset", "export", "job", "training", "score")

_LOG_HOOK = None


def set_log_hook(fn):
    global _LOG_HOOK
    _LOG_HOOK = fn


def _log(msg):
    (_LOG_HOOK or (lambda s: print(s, flush=True)))(f"[autopilot] {msg}")


class Cancelled(Exception):
    pass


class Autopilot:
    def __init__(self, state_path=None, poll=2.0):
        self.state_path = state_path or STATE_PATH
        self.poll = poll
        self.lock = threading.Lock()
        self.thread = None
        self.cancel_flag = False
        self.cancel_all = False
        self.save_lock = threading.RLock()
        self.state = self._load()
        if self.state.get("running"):
            # the previous process died with the chain in flight
            self.state["running"] = False
            self.state["interrupted"] = self.state.get("step")

    # -- persistence ------------------------------------------------------------
    def _load(self):
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _save(self):
        # the request thread (queue start, cancel) and the chain thread both save
        with self.save_lock:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.state, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, self.state_path)

    def status(self):
        return dict(self.state)

    def _say(self, text):
        self.state["text"] = text
        self.state["log"] = (self.state.get("log") or [])[-40:] + [time.strftime("%H:%M:%S ") + text]
        self._save()
        _log(text)

    def _step(self, name):
        self.state["step"] = name
        self.state["steps"][name] = "running"
        self._save()

    def _done(self, name, note=""):
        self.state["steps"][name] = "done"
        if note:
            self._say(f"{name}: {note}")
        self._save()

    def _check(self):
        if self.cancel_flag:
            raise Cancelled()

    # -- control ----------------------------------------------------------------
    # -- queue ------------------------------------------------------------------
    def start_queue(self, studio, items):
        """items = [(name, opts), ...]: runs them one after the other."""
        if not items:
            raise ValueError("the queue is empty")
        with self.lock:
            if self.state.get("running"):
                raise RuntimeError("the autopilot is already running: wait or cancel it")
        first, rest = items[0], items[1:]
        _log(f"queue: {len(items)} project(s): " + ", ".join(n for n, _ in items))
        return self.start(studio, first[0], first[1], queue=[list(x) for x in rest], history=[])

    def _next(self, studio):
        q = self.state.get("queue") or []
        if not q or self.cancel_all:
            if q:
                self._say(f"queue: cancelled, {len(q)} project(s) not started: "
                          + ", ".join(n for n, _ in q))
                self.state["queue"] = []
                self._save()
            return
        name, opts = q[0]
        try:
            self.start(studio, name, opts, queue=q[1:], history=self.state.get("history") or [])
        except Exception as e:
            self.state.setdefault("history", []).append(
                {"project": name, "ok": False, "error": f"not started: {e}"})
            self.state["queue"] = q[1:]
            self._save()
            self._next(studio)

    def start(self, studio, name, opts=None, resume_at=None, queue=None, history=None):
        with self.lock:
            if self.state.get("running"):
                raise RuntimeError(f"the autopilot is already running on "
                                   f"'{self.state.get('project')}'")
            self.cancel_flag = False
            if queue is not None or not resume_at:
                self.cancel_all = False
            prev = self.state if resume_at else {}
            self.state = {"running": True, "project": name, "opts": dict(opts or prev.get("opts") or {}),
                          "step": resume_at or "dataset", "text": "", "ok": None, "error": "",
                          "steps": dict(prev.get("steps") or {k: "pending" for k in STEPS}),
                          "log": list(prev.get("log") or []), "started": prev.get("started") or time.time(),
                          "finished": None, "job": prev.get("job"), "result": prev.get("result"),
                          "queue": queue if queue is not None else list(prev.get("queue") or []),
                          "history": history if history is not None else list(prev.get("history") or [])}
            if resume_at:
                self.state["log"].append(time.strftime("%H:%M:%S ") + f"resumed at the {resume_at}")
            self._save()
            self.thread = threading.Thread(target=self._run, args=(studio, name, resume_at),
                                           daemon=True, name="ds-autopilot")
            self.thread.start()
            return self.status()

    def cancel(self, studio, all_=False):
        if not self.state.get("running"):
            if all_ and self.state.get("queue"):
                self.state["queue"] = []
                self._save()
                return True
            return False
        self.cancel_flag = True
        self.cancel_all = bool(all_)
        self._say("cancel requested: stopping the step in flight")
        step = self.state.get("step")
        if step == "dataset":
            studio.runner.cancel()
        elif step == "training":
            ds_aitoolkit.trainer().stop()
        elif step == "score":
            ds_lorascore.SCORER.cancel()
        return True

    def wait(self, timeout=None):
        if self.thread:
            self.thread.join(timeout)
        return not self.state.get("running")

    # -- the chain ----------------------------------------------------------------
    def _run(self, studio, name, resume_at):
        try:
            o = self.state["opts"]
            order = list(STEPS)
            start = order.index(resume_at) if resume_at in order else 0
            for step in order[start:]:
                self._check()
                getattr(self, "_do_" + step)(studio, name, o)
            self.state["ok"] = True
            r = self.state.get("result") or {}
            self._say("finished: " + (f"LoRA {r.get('lora')}" if r.get("lora") else "LoRA trained")
                      + (f", likeness {r['mean_lora']} with it, {r['mean_base']} without"
                         if r.get("mean_lora") is not None else ""))
        except Cancelled:
            self.state["ok"] = False
            self.state["error"] = "cancelled"
            self.state["steps"][self.state["step"]] = "cancelled"
            self._say(f"cancelled at the {self.state['step']}")
        except Exception as e:
            self.state["ok"] = False
            self.state["error"] = str(e) if isinstance(e, (RuntimeError, ValueError, FileNotFoundError)) \
                else f"{type(e).__name__}: {e}"
            self.state["steps"][self.state.get("step") or "dataset"] = "failed"
            self._say(f"stopped at the {self.state.get('step')}: {self.state['error']}")
        finally:
            self.state["running"] = False
            self.state["finished"] = time.time()
            self.state.setdefault("history", []).append(
                {"project": name, "ok": self.state.get("ok"), "error": self.state.get("error"),
                 "job": self.state.get("job"), "result": self.state.get("result"),
                 "steps": dict(self.state.get("steps") or {})})
            self._save()
        self._next(studio)

    def _do_dataset(self, studio, name, o):
        self._step("dataset")
        r = studio.op("run", {"name": name, "engine": o.get("engine")})
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        if r.get("started"):
            self._say(f"dataset: rendering {r.get('total')} card(s)")
            while not studio.runner.wait(self.poll):
                pass
            st = studio.runner.status()
            if self.cancel_flag:
                raise Cancelled()
            if not st.get("ok"):
                raise RuntimeError(f"the run failed: {st.get('error') or st.get('text')}")
        p = ds_project.load(name)
        cards = p.get("cards") or []
        left = [c["index"] for c in cards if c.get("status") in ("pending", "failed", "rendering")]
        done = [c for c in cards if c.get("status") == "done"]
        if left:
            raise RuntimeError(f"the dataset is not complete: card(s) {', '.join(f'{i:03d}' for i in left)} "
                               f"failed or not rendered; fix them and run again")
        if len(done) < MIN_CARDS:
            raise RuntimeError(f"only {len(done)} card(s) kept: at least {MIN_CARDS} are needed "
                               f"to train a LoRA")
        low = [c for c in done if c.get("scored") is not False
               and (c.get("score") or 0) < int((p.get("settings") or {}).get("threshold") or 0)]
        self._done("dataset", f"{len(done)} card(s) kept"
                   + (f", {len(low)} below the likeness threshold (kept, review them later)"
                      if low else ""))

    def _do_export(self, studio, name, o):
        self._step("export")
        r = studio.op("export", {"name": name})
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        self.state["lot"] = r["lot"]
        self._done("export", f"{r['count']} pair(s) in {r['dir']}")

    def _do_job(self, studio, name, o):
        self._step("job")
        r = studio.op("aitk_create", {"name": name, "lot": self.state.get("lot"),
                                      "preset": o.get("preset"), "steps": o.get("steps")})
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        self.state["job"] = r["job"]
        for w in r.get("warnings") or []:
            self._say(f"job: {w}")
        self._done("job", f"{r['job']} ({r['pairs']} pair(s), {r['steps']} steps)")

    def _do_training(self, studio, name, o):
        self._step("training")
        tr = ds_aitoolkit.trainer()
        job = self.state.get("job")
        if not (tr.running() and tr.state.get("job") == job):
            if tr.state.get("job") == job and tr.state.get("result") == "done":
                pass                                  # resumed after it ended
            else:
                r = studio.op("aitk_train", {"name": name, "job": job, "stop_engine": True})
                if not r.get("ok"):
                    raise RuntimeError(r.get("error"))
                self._say(f"training: {job} started (the engine was stopped to free the GPU)"
                          if r.get("stopped_engine") else f"training: {job} started")
        while tr.running():
            self._check()
            pg = tr.status().get("progress") or {}
            if pg.get("total"):
                self.state["text"] = (f"training {job}: step {pg['step']}/{pg['total']}"
                                      + (f", {pg['eta']} left" if pg.get("eta") else ""))
            time.sleep(self.poll)
        self._check()
        st = tr.status()
        if st.get("result") != "done":
            raise RuntimeError(f"the training ended '{st.get('result')}'"
                               + (f" (exit {st.get('returncode')})" if st.get("returncode") is not None else "")
                               + f": see {st.get('log')}")
        self.state["result"] = {"lora": st.get("lora")}
        self._done("training", f"LoRA {st.get('lora')} in {st.get('minutes')} min")

    def _do_score(self, studio, name, o):
        self._step("score")
        eng = (o.get("score_engine") or "").strip()
        if not o.get("score", True) or not eng:
            self.state["steps"]["score"] = "skipped"
            self._say("score: skipped (" + ("turned off" if not o.get("score", True)
                                            else "no scoring engine set") + ")")
            return
        job = self.state.get("job")
        t = studio.op("lora_targets", {"name": name, "job": job})
        if not t.get("ok"):
            raise RuntimeError(t.get("error"))
        if eng not in [e["name"] for e in t.get("engines") or []]:
            self.state["steps"]["score"] = "skipped"
            self._say(f"score: skipped (engine '{eng}' has no install folder in Settings)")
            return
        files = [l["file"] for l in (t.get("loras") or [])][:3]   # final + 2 latest saves
        r = studio.op("lora_score", {"name": name, "job": job, "engine": eng, "files": files,
                                     "switch_model": True})
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        self._say(f"score: {len(files)} LoRA file(s) on {eng}")
        while not ds_lorascore.SCORER.wait(self.poll):
            self._check()
            s = ds_lorascore.SCORER.status()
            self.state["text"] = f"score: {s.get('done')}/{s.get('total')} {s.get('text')}"
        self._check()
        s = ds_lorascore.SCORER.status()
        if not s.get("ok"):
            raise RuntimeError(f"the score failed: {s.get('error')}")
        res = s.get("results") or []
        final = next((x for x in res if x.get("final")), res[0] if res else None)
        best = max((x for x in res if x.get("mean_lora") is not None),
                   key=lambda x: x["mean_lora"], default=None)
        if final:
            self.state["result"] = dict(self.state.get("result") or {},
                                        mean_lora=final.get("mean_lora"),
                                        mean_base=final.get("mean_base"),
                                        best=best and best["lora"],
                                        best_mean=best and best["mean_lora"])
        self._done("score", (f"final {final.get('mean_lora')} vs {final.get('mean_base')} without"
                             if final else "done")
                   + (f"; best: {best['lora']} ({best['mean_lora']})"
                      if best and final and best is not final else ""))


AUTOPILOT = None


def autopilot():
    global AUTOPILOT
    if AUTOPILOT is None:
        AUTOPILOT = Autopilot()
    return AUTOPILOT
