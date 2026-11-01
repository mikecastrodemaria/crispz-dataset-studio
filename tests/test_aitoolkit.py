"""ds_project.clone + ds_aitoolkit: the project copy, the ai-toolkit folder search
and the job (dataset copy + config), then both over HTTP."""

import os
import json
import time

import pytest
from PIL import Image

import ds_project
import ds_aitoolkit
from test_server import studio, post, data_url  # noqa: F401  (fixture reuse)


def fake_aitk(base):
    """An ai-toolkit install the way Easy-Install lays it out."""
    root = base / "AI-Toolkit-Easy-Install" / "AI-Toolkit-Easy-Install" / "AI-Toolkit"
    (root / "toolkit").mkdir(parents=True)
    (root / "output").mkdir()
    (root / "run.py").write_text("# ai-toolkit\n")
    return root


def make_lot(d, n=3, trigger="ohwx character", caption=True):
    d.mkdir(parents=True)
    for i in range(1, n + 1):
        Image.new("RGB", (24, 32), (i * 40, 20, 20)).save(d / f"{i:03d}.png")
        if caption:
            (d / f"{i:03d}.txt").write_text(f"{trigger}, shot {i}\n", encoding="utf-8")
    return d


# ------------------------------------------------------------------ find_root
def test_find_root_from_inside_above_and_itself(tmp_path):
    root = fake_aitk(tmp_path)
    assert ds_aitoolkit.find_root(str(root)) == str(root)
    # the path the user copies from the explorer: ...\AI-Toolkit\output
    assert ds_aitoolkit.find_root(str(root / "output")) == str(root)
    # the Easy-Install folder above
    assert ds_aitoolkit.find_root(str(tmp_path / "AI-Toolkit-Easy-Install")) == str(root)
    assert ds_aitoolkit.find_root(str(tmp_path / "nope")) is None
    assert ds_aitoolkit.find_root("") is None
    st = ds_aitoolkit.status({"ai_toolkit_dir": str(tmp_path / "elsewhere")})
    assert st["configured"] and st["root"] is None and "run.py" in st["error"]


# ------------------------------------------------------------------ create_job
def test_create_job_copies_pairs_and_writes_config(tmp_path):
    root = fake_aitk(tmp_path)
    lot = make_lot(tmp_path / "out" / "lot1")
    p = {"name": "trial-flux3", "settings": {"trigger_word": "ohwx character",
                                             "export_size": 768}}
    cfg = {"ai_toolkit_dir": str(root / "output")}
    r = ds_aitoolkit.create_job(p, str(lot), cfg, preset="klein9b", steps=1200)
    assert r["job"] == "trial-flux3_klein9b_v1" and r["pairs"] == 3
    ds = root / "datasets" / r["job"]
    assert sorted(os.listdir(ds)) == ["001.png", "001.txt", "002.png", "002.txt",
                                      "003.png", "003.txt"]
    text = (root / "config" / (r["job"] + ".yaml")).read_text(encoding="utf-8")
    assert '"flux2_klein_9b"' in text and "FLUX.2-klein-base-9B" in text
    assert 'qtype: "qfloat8"' in text and "steps: 1200" in text
    assert "resolution: [512, 768]" in text
    # captions already start with the trigger: not passed a second time
    assert "trigger_word: null" in text
    assert "\\" not in text.split("folder_path:")[1].splitlines()[0]
    assert any("gated" in w for w in r["warnings"])
    assert "run.py config/trial-flux3_klein9b_v1.yaml" in r["command"]
    # a second job never overwrites the first
    r2 = ds_aitoolkit.create_job(p, str(lot), cfg, preset="klein9b")
    assert r2["job"] == "trial-flux3_klein9b_v2"


def test_create_job_trigger_and_errors(tmp_path):
    root = fake_aitk(tmp_path)
    lot = make_lot(tmp_path / "lot", trigger="someone else")
    cfg = {"ai_toolkit_dir": str(root)}
    p = {"name": "x", "settings": {"trigger_word": "tok"}}
    r = ds_aitoolkit.create_job(p, str(lot), cfg)
    text = open(r["config_path"], encoding="utf-8").read()
    assert 'trigger_word: "tok"' in text and '"flux2_klein_4b"' in text
    assert "quantize: false" in text and "qtype" not in text
    with pytest.raises(ValueError):
        ds_aitoolkit.create_job(p, str(lot), cfg, preset="sdxl")
    with pytest.raises(FileNotFoundError):
        ds_aitoolkit.create_job(p, str(lot), {"ai_toolkit_dir": ""})
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        ds_aitoolkit.create_job(p, str(empty), cfg)


# ------------------------------------------------------------------ over HTTP
def _run_to_end(base, name):
    r = post(base, "run", {"name": name})
    assert r["ok"] and r["started"]
    for _ in range(100):
        s = post(base, "status")["status"]
        if not s["running"]:
            return s
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def test_clone_and_aitk_over_http(studio, tmp_path):  # noqa: F811
    post(studio, "project_new", {"name": "src"})
    post(studio, "project_settings", {"name": "src", "settings": {
        "prompts": ["same person, one", "same person, two"], "width": 32, "height": 32,
        "trigger_word": "tok", "max_rerolls": 0}})
    post(studio, "set_reference", {"name": "src", "data": data_url()})
    assert _run_to_end(studio, "src")["ok"]

    # clone with the renders: same files, a new name, no export carried over
    r = post(studio, "project_clone", {"name": "src", "new_name": "src copy"})
    assert r["ok"] and r["project"]["name"] == "src-copy"
    assert r["project"]["cloned_from"] == "src"
    c = r["project"]["cards"][0]
    assert c["status"] == "done"
    assert os.path.isfile(os.path.join(ds_project.project_dir("src-copy"), c["file"]))
    # clone without: reference and settings kept, every card pending, cards/ empty
    r = post(studio, "project_clone", {"name": "src", "new_name": "fresh",
                                       "with_renders": False})
    p = r["project"]
    assert all(c["status"] == "pending" for c in p["cards"]) and p["ref_embedding"]
    assert p["settings"]["trigger_word"] == "tok"
    assert os.listdir(os.path.join(ds_project.project_dir("fresh"), "cards")) == []
    assert os.path.isfile(os.path.join(ds_project.project_dir("fresh"), "reference.png"))
    r = post(studio, "project_clone", {"name": "src", "new_name": "fresh"})
    assert not r["ok"] and "already exists" in r["error"]

    # ai-toolkit not set: status says so, create refuses
    st = post(studio, "aitk_status")["aitk"]
    assert not st["configured"] and st["root"] is None
    assert not post(studio, "aitk_create", {"name": "src"})["ok"]

    # set it to the output folder, the path the user has at hand
    root = fake_aitk(tmp_path / "tools")
    cfg = post(studio, "config_get")["config"]
    cfg["ai_toolkit_dir"] = str(root / "output")
    assert post(studio, "config_set", {"config": cfg})["ok"]
    st = post(studio, "aitk_status")["aitk"]
    assert st["root"] == str(root) and st["datasets_dir"] == str(root / "datasets")

    # no lot given: a fresh export first
    r = post(studio, "aitk_create", {"name": "src", "preset": "klein4b", "steps": 500})
    assert r["ok"] and r["pairs"] == 2 and r["exported"] and r["job"] == "src_klein4b_v1"
    assert os.path.isfile(os.path.join(r["dataset_dir"], "001.txt"))
    # with the lot of an export made before
    ex = post(studio, "export", {"name": "src"})
    r = post(studio, "aitk_create", {"name": "src", "lot": ex["lot"]})
    assert r["ok"] and "exported" not in r and r["job"] == "src_klein4b_v2"
    assert post(studio, "project_get", {"name": "src"})["project"]["last_aitk"]["job"] == r["job"]


# ------------------------------------------------------------------ training
FAKE_RUN = r'''
import os, sys, time
cfg = sys.argv[1]
job = os.path.splitext(os.path.basename(cfg))[0]
mode = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mode.txt")).read().strip()
print("Caching latents:", "100%|##########| 3/3 [00:01<00:00]", flush=True)
for i in (0, 50, 100):
    sys.stdout.write(f"\r{job}:  {i}%|###| {i * 15}/1500 [00:0{i // 50}<00:02,  1.2it/s, lr: 1.0e-04 loss: 6.5{i}e-01]")
    sys.stdout.flush()
    time.sleep(0.05 if mode != "slow" else 0.5)
if mode == "slow":
    time.sleep(60)
if mode == "fail":
    print("\nRuntimeError: CUDA out of memory", flush=True)
    sys.exit(3)
out = os.path.join("output", job)
os.makedirs(out, exist_ok=True)
open(os.path.join(out, job + ".safetensors"), "wb").write(b"lora")
print("\nDone", flush=True)
'''


def fake_aitk_trainable(base, mode="ok"):
    import sys
    root = fake_aitk(base)
    (root / "run.py").write_text(FAKE_RUN)
    (root / "mode.txt").write_text(mode)
    vbin = root / ".venv" / "bin"
    vbin.mkdir(parents=True)
    os.symlink(sys.executable, vbin / "python")
    (root / "config").mkdir()
    (root / "config" / "job_v1.yaml").write_text("config:\n  process:\n    - train:\n        steps: 1500\n")
    return root


def wait_done(tr, secs=20):
    for _ in range(int(secs * 10)):
        if not tr.running():
            return tr.status()
        time.sleep(0.1)
    raise AssertionError("training did not end")


def test_parse_progress_keeps_the_training_bar():
    text = ("Caching latents: 100%|###| 25/25 [00:10<00:00]\r"
            "job:  17%|#  | 250/1500 [03:20<16:40,  1.25it/s, lr: 1.0e-04 loss: 6.500e-01]\r"
            "job:  17%|#  | 251/1500 [03:21<16:39,  1.25it/s, lr: 1.0e-04 loss: 5.900e-01]")
    pg = ds_aitoolkit.parse_progress(text, 1500)
    assert pg == {"step": 251, "total": 1500, "elapsed": "03:21", "eta": "16:39",
                  "loss": "5.900e-01"}
    assert ds_aitoolkit.parse_progress("Caching latents: 25/25 [00:10<00:00]", None) == {}


@pytest.mark.skipif(os.name == "nt", reason="the fake interpreter is a symlink")
def test_trainer_runs_to_the_lora_and_survives_a_restart(tmp_path):
    root = fake_aitk_trainable(tmp_path)
    state = str(tmp_path / "logs" / "training.json")
    tr = ds_aitoolkit.Trainer(state)
    st = tr.start(str(root), "job_v1", project="p")
    assert st["running"] and st["steps"] == 1500
    with pytest.raises(RuntimeError):
        tr.start(str(root), "job_v1")
    st = wait_done(tr)
    assert st["result"] == "done" and st["returncode"] == 0
    assert st["lora"].endswith(os.path.join("job_v1", "job_v1.safetensors"))
    assert st["progress"]["step"] == 1500 and st["progress"]["loss"] == "6.5100e-01"
    # a new Dataset Studio session reads the state back
    st2 = ds_aitoolkit.Trainer(state).status()
    assert st2["job"] == "job_v1" and st2["result"] == "done" and not st2["running"]
    with pytest.raises(FileNotFoundError):
        tr.start(str(root), "nope_v1")


@pytest.mark.skipif(os.name == "nt", reason="the fake interpreter is a symlink")
def test_trainer_failure_and_stop(tmp_path):
    root = fake_aitk_trainable(tmp_path / "a", mode="fail")
    tr = ds_aitoolkit.Trainer(str(tmp_path / "s1.json"))
    tr.start(str(root), "job_v1")
    st = wait_done(tr)
    assert st["result"] == "failed" and st["returncode"] == 3 and not st["lora"]
    assert any("out of memory" in l for l in st["tail"])
    root = fake_aitk_trainable(tmp_path / "b", mode="slow")
    tr = ds_aitoolkit.Trainer(str(tmp_path / "s2.json"))
    tr.start(str(root), "job_v1")
    time.sleep(0.5)
    assert tr.stop() is True
    st = tr.status()
    assert st["result"] == "stopped" and not st["running"]
    assert tr.stop() is False


@pytest.mark.skipif(os.name == "nt", reason="the fake interpreter is a symlink")
def test_train_over_http_blocks_the_gpu(studio, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ds_aitoolkit, "TRAINER", ds_aitoolkit.Trainer(str(tmp_path / "t.json")))
    root = fake_aitk_trainable(tmp_path / "tk", mode="slow")
    cfg = post(studio, "config_get")["config"]
    cfg["ai_toolkit_dir"] = str(root)
    post(studio, "config_set", {"config": cfg})
    post(studio, "project_new", {"name": "p"})
    r = post(studio, "aitk_train", {"name": "p"})
    assert not r["ok"] and "create it first" in r["error"]
    r = post(studio, "aitk_train", {"name": "p", "job": "job_v1"})
    assert r["ok"] and r["training"]["running"]
    assert post(studio, "status")["training"]["job"] == "job_v1"
    r = post(studio, "run", {"name": "p"})
    assert not r["ok"] and "GPU" in r["error"]
    r = post(studio, "aitk_train_stop")
    assert r["ok"] and r["stopped"] and r["training"]["result"] == "stopped"


# ------------------------------------------------------------------ LoRA -> engine
def test_engine_loras_dir_follows_the_app_rule(tmp_path):
    app = tmp_path / "crispz-klein"
    app.mkdir()
    assert ds_aitoolkit.engine_loras_dir(str(app)) == str(app / "loras")
    (app / "config.txt").write_text('{"loras_dir": "models/loras"}')
    assert ds_aitoolkit.engine_loras_dir(str(app)) == str(app / "models" / "loras")
    (app / "preferences.json").write_text('{"loras_dir": "%s"}' % str(tmp_path / "L").replace("\\", "/"))
    assert ds_aitoolkit.engine_loras_dir(str(app)) == os.path.normpath(str(tmp_path / "L"))
    assert ds_aitoolkit.engine_loras_dir(str(app), str(tmp_path / "X")) == str(tmp_path / "X")
    assert ds_aitoolkit.engine_loras_dir("") is None


def test_job_loras_order_and_copy_never_overwrites(tmp_path):
    root = fake_aitk(tmp_path)
    out = root / "output" / "j_v1"
    out.mkdir(parents=True)
    for f, b in (("j_v1_000001000.safetensors", b"a"), ("j_v1.safetensors", b"final"),
                 ("j_v1_000001250.safetensors", b"b")):
        (out / f).write_bytes(b)
    ls = ds_aitoolkit.job_loras(str(root), "j_v1")
    assert [l["file"] for l in ls] == ["j_v1.safetensors", "j_v1_000001250.safetensors",
                                       "j_v1_000001000.safetensors"]
    assert ls[0]["final"] and ls[1]["step"] == 1250
    dst = tmp_path / "loras"
    r = ds_aitoolkit.copy_lora(ls[0]["path"], str(dst))
    assert r["copied"] and (dst / "j_v1.safetensors").read_bytes() == b"final"
    r = ds_aitoolkit.copy_lora(ls[0]["path"], str(dst))        # same file: left as is
    assert not r["copied"] and r["file"] == "j_v1.safetensors"
    (out / "j_v1.safetensors").write_bytes(b"retrained")
    r = ds_aitoolkit.copy_lora(ls[0]["path"], str(dst))        # another file: _2
    assert r["copied"] and r["file"] == "j_v1_2.safetensors"
    assert (dst / "j_v1.safetensors").read_bytes() == b"final"


def test_lora_copy_over_http(studio, tmp_path):  # noqa: F811
    root = fake_aitk(tmp_path / "tk")
    klein = tmp_path / "crispz-klein"
    klein.mkdir()
    cfg = post(studio, "config_get")["config"]
    cfg["ai_toolkit_dir"] = str(root)
    cfg["engines"]["klein"] = {"url": "http://127.0.0.1:1", "dir": str(klein), "launch": False}
    post(studio, "config_set", {"config": cfg})
    post(studio, "project_new", {"name": "p"})
    assert not post(studio, "lora_targets", {"name": "p"})["ok"]        # no job yet
    lot = make_lot(tmp_path / "lot")
    p = ds_project.load("p")
    p["settings"]["trigger_word"] = "ohwx character"
    job = ds_aitoolkit.create_job(p, str(lot), cfg, preset="klein4b")["job"]
    p["last_aitk"] = {"job": job}
    ds_project.save(p)
    r = post(studio, "lora_copy", {"name": "p", "engine": "klein"})
    assert not r["ok"] and "train the job first" in r["error"]
    (root / "output" / job).mkdir(parents=True)
    (root / "output" / job / (job + ".safetensors")).write_bytes(b"lora")
    t = post(studio, "lora_targets", {"name": "p"})
    assert t["ok"] and t["default_engine"] == "klein" and t["loras"][0]["final"]
    assert t["engines"][0]["loras_dir"] == str(klein / "loras") and "4B" in t["hint"]
    r = post(studio, "lora_copy", {"name": "p", "engine": "klein"})
    assert r["ok"] and r["copied"] and os.path.isfile(klein / "loras" / (job + ".safetensors"))
    assert "4B" in r["hint"]
    r = post(studio, "lora_copy", {"name": "p", "engine": "fake"})       # no dir
    assert not r["ok"] and "install folder" in r["error"]
    assert post(studio, "lora_targets", {"name": "p"})["copied"][0]["engine"] == "klein"


# ------------------------------------------------------------------ LoRA likeness
import fake_engine            # noqa: E402
import ds_lorascore           # noqa: E402


def _score_setup(studio, tmp_path):  # noqa: F811
    fake_engine.STATE.update({"gens": [], "model": None, "lora_skip": False,
                              "lora_color": (10, 200, 30), "can_switch": False})
    root = fake_aitk(tmp_path / "tk")
    klein = tmp_path / "crispz-klein"
    klein.mkdir()
    cfg = post(studio, "config_get")["config"]
    cfg["ai_toolkit_dir"] = str(root)
    cfg["engines"]["fake"]["dir"] = str(klein)
    post(studio, "config_set", {"config": cfg})
    post(studio, "project_new", {"name": "p"})
    post(studio, "project_settings", {"name": "p", "settings": {"trigger_word": "ohwx character"}})
    assert post(studio, "set_reference", {"name": "p", "data": data_url((10, 200, 30))})["ok"]
    p = ds_project.load("p")
    job = ds_aitoolkit.create_job(p, str(make_lot(tmp_path / "lot")), cfg, preset="klein4b")["job"]
    p = ds_project.load("p")
    p["last_aitk"] = {"job": job}
    ds_project.save(p)
    out = root / "output" / job
    out.mkdir(parents=True)
    (out / (job + ".safetensors")).write_bytes(b"final")
    (out / (job + "_000001000.safetensors")).write_bytes(b"step1000")
    return root, klein, job


def _wait_score():
    for _ in range(200):
        s = ds_lorascore.SCORER.status()
        if not s["running"]:
            return s
        time.sleep(0.05)
    raise AssertionError("scoring did not end")


def test_model_mismatch_reasons():
    assert ds_lorascore.model_mismatch("flux2_klein_4b", "black-forest-labs/FLUX.2-klein-9B")
    assert ds_lorascore.model_mismatch("flux2_klein_9b", "black-forest-labs/FLUX.2-klein-4B")
    assert ds_lorascore.model_mismatch("flux2_klein_4b", "black-forest-labs/FLUX.2-klein-4B") is None
    assert ds_lorascore.model_mismatch("flux2_klein_4b", "") is None
    assert ds_lorascore.model_mismatch("", "black-forest-labs/FLUX.2-klein-9B") is None


def test_lora_score_over_http(studio, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ds_lorascore, "SCORER", ds_lorascore.Scorer())
    root, klein, job = _score_setup(studio, tmp_path)
    t = post(studio, "lora_targets", {"name": "p"})
    assert t["default_prompts"] == ds_lorascore.DEFAULT_PROMPTS and t["scores"] == []
    r = post(studio, "lora_score", {"name": "p", "engine": "fake",
                                    "files": [job + ".safetensors", job + "_000001000.safetensors"],
                                    "prompts": "close-up portrait\nmedium shot", "weight": 1})
    assert r["ok"] and r["lora_score"]["total"] == 2 * 2 + 2
    s = _wait_score()
    assert s["ok"] is True, s
    # same seed and size with and without; the trigger word in front; the LoRA by name
    specs = fake_engine.STATE["gens"]
    assert len(specs) == 6 and {sp["seed"] for sp in specs} == {42}
    assert specs[0]["prompt"] == "ohwx character, close-up portrait" and specs[0]["loras"] == []
    assert specs[2]["loras"] == [job + ".safetensors:1"]
    assert os.path.isfile(klein / "loras" / (job + ".safetensors"))
    assert os.path.isfile(klein / "loras" / (job + "_000001000.safetensors"))
    t = post(studio, "lora_targets", {"name": "p"})
    assert len(t["scores"]) == 2
    final = next(h for h in t["scores"] if h["final"])
    assert final["mean_lora"] >= 99 and final["mean_base"] < final["mean_lora"]
    assert final["scored"] == 2 and final["rows"][0]["base"]["score"] is not None
    assert os.path.isfile(os.path.join(ds_project.project_dir("p"), final["rows"][0]["lora"]["file"]))
    # scoring again replaces the entry of that LoRA and weight, keeps the others
    post(studio, "lora_score", {"name": "p", "engine": "fake", "files": [job + ".safetensors"],
                                "baseline": False, "prompts": "close-up portrait"})
    assert _wait_score()["ok"]
    t = post(studio, "lora_targets", {"name": "p"})
    assert len(t["scores"]) == 2
    assert next(h for h in t["scores"] if h["final"])["mean_base"] is None


def test_lora_score_refuses_a_wrong_model_or_a_skipped_lora(studio, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ds_lorascore, "SCORER", ds_lorascore.Scorer())
    root, klein, job = _score_setup(studio, tmp_path)
    fake_engine.STATE["model"] = "black-forest-labs/FLUX.2-klein-9B"
    assert post(studio, "lora_score", {"name": "p", "engine": "fake"})["ok"]
    s = _wait_score()
    assert s["ok"] is False and "Klein 4B" in s["error"] and fake_engine.STATE["gens"] == []
    assert "switch_model" in s["error"]           # an older engine: says how to fix it
    # switch refused by the user: the same stop
    post(studio, "lora_score", {"name": "p", "engine": "fake", "switch_model": False})
    s = _wait_score()
    assert s["ok"] is False and "Klein 4B" in s["error"] and "update it" not in s["error"]
    fake_engine.STATE.update({"model": None, "lora_skip": True})
    post(studio, "lora_score", {"name": "p", "engine": "fake", "baseline": False})
    s = _wait_score()
    assert s["ok"] is False and "did not apply the LoRA" in s["error"]
    assert ds_project.load("p").get("lora_scores") in (None, [])
    fake_engine.STATE.update({"lora_skip": False})
    r = post(studio, "lora_score", {"name": "p", "engine": "fake", "files": ["nope.safetensors"]})
    assert not r["ok"] and "not found" in r["error"]


def test_lora_score_switches_a_klein_that_can(studio, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ds_lorascore, "SCORER", ds_lorascore.Scorer())
    root, klein, job = _score_setup(studio, tmp_path)
    fake_engine.STATE.update({"model": "black-forest-labs/FLUX.2-klein-9B", "can_switch": True})
    post(studio, "lora_score", {"name": "p", "engine": "fake", "prompts": "close-up"})
    s = _wait_score()
    assert s["ok"] is True, s
    specs = fake_engine.STATE["gens"]
    # every render, the base one included, asks for the 4B with switch_model
    assert len(specs) == 2 and all(sp["model"] == "black-forest-labs/FLUX.2-klein-4B"
                                   and sp["switch_model"] for sp in specs)
    assert fake_engine.STATE["model"] == "black-forest-labs/FLUX.2-klein-4B"
    h = post(studio, "lora_targets", {"name": "p"})["scores"][0]
    assert h["model"] == "black-forest-labs/FLUX.2-klein-4B" and h["mean_lora"] >= 99
    # already on the right model: nothing is asked
    fake_engine.STATE["gens"] = []
    post(studio, "lora_score", {"name": "p", "engine": "fake", "prompts": "close-up"})
    assert _wait_score()["ok"]
    assert all("model" not in sp for sp in fake_engine.STATE["gens"])
    fake_engine.STATE.update({"model": None, "can_switch": False})


# ------------------------------------------------------------------ autopilot
import ds_autopilot           # noqa: E402


def _auto_setup(studio, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ds_lorascore, "SCORER", ds_lorascore.Scorer())
    monkeypatch.setattr(ds_aitoolkit, "TRAINER", ds_aitoolkit.Trainer(str(tmp_path / "t.json")))
    monkeypatch.setattr(ds_autopilot, "AUTOPILOT",
                        ds_autopilot.Autopilot(str(tmp_path / "ap.json"), poll=0.05))
    fake_engine.STATE.update({"gens": [], "model": None, "lora_skip": False,
                              "lora_color": (10, 200, 30), "can_switch": False})
    root = fake_aitk_trainable(tmp_path / "tk", mode="ok")
    klein = tmp_path / "crispz-klein"
    klein.mkdir()
    cfg = post(studio, "config_get")["config"]
    cfg["ai_toolkit_dir"] = str(root)
    cfg["engines"]["fake"]["dir"] = str(klein)
    post(studio, "config_set", {"config": cfg})
    return root, klein


def _new_auto_project(studio, name, prompts):  # noqa: F811
    post(studio, "project_new", {"name": name})
    r = post(studio, "project_settings", {"name": name, "settings": {
        "prompts": prompts, "width": 32, "height": 32, "trigger_word": "tok",
        "max_rerolls": 0, "autopilot": True, "auto_steps": 1500,
        "auto_score_engine": "fake"}})
    assert r["ok"] and r["project"]["settings"]["autopilot"] is True
    assert post(studio, "set_reference", {"name": name, "data": data_url((10, 200, 30))})["ok"]


def _wait_auto(secs=60):
    ap = ds_autopilot.autopilot()
    for _ in range(secs * 20):
        st = ap.status()
        if not st.get("running") and not st.get("queue"):
            time.sleep(0.1)
            st = ap.status()
            if not st.get("running") and not st.get("queue"):
                return st
        time.sleep(0.05)
    raise AssertionError(f"autopilot did not end: {ap.status().get('text')}")


@pytest.mark.skipif(os.name == "nt", reason="the fake interpreter is a symlink")
def test_autopilot_goes_from_the_reference_to_a_scored_lora(studio, tmp_path, monkeypatch):  # noqa: F811
    root, klein = _auto_setup(studio, tmp_path, monkeypatch)
    _new_auto_project(studio, "a", ["one", "two", "three", "four", "five"])
    r = post(studio, "autopilot_start", {"name": "a"})
    assert r["ok"], r
    st = _wait_auto()
    assert st["ok"] is True, st.get("log")
    assert st["steps"] == {"dataset": "done", "export": "done", "job": "done",
                           "training": "done", "score": "done"}
    job = st["job"]
    assert job == "a_klein4b_v1"
    # only the best save stays in the engine, under its stable name
    assert sorted(os.listdir(klein / "loras")) == ["a_klein4b_best.safetensors"]
    assert st["result"]["lora"].endswith("a_klein4b_best.safetensors")
    assert st["result"]["mean_lora"] >= 99 and st["result"]["mean_base"] < 99
    p = ds_project.load("a")
    assert p["last_aitk"]["job"] == job and p["lora_scores"]
    assert post(studio, "status")["autopilot"]["project"] == "a"


@pytest.mark.skipif(os.name == "nt", reason="the fake interpreter is a symlink")
def test_autopilot_queue_goes_on_after_a_failed_project(studio, tmp_path, monkeypatch):  # noqa: F811
    _auto_setup(studio, tmp_path, monkeypatch)
    _new_auto_project(studio, "bad", ["one", "FAIL two", "three", "four", "five"])
    _new_auto_project(studio, "good", ["one", "two", "three", "four", "five"])
    post(studio, "project_new", {"name": "noref"})
    r = post(studio, "autopilot_queue", {"names": ["bad", "noref"]})
    assert not r["ok"] and "reference" in r["error"]
    r = post(studio, "autopilot_queue", {"names": ["bad", "good"]})
    assert r["ok"], r
    st = _wait_auto()
    h = {x["project"]: x for x in st["history"]}
    assert h["bad"]["ok"] is False and "002" in h["bad"]["error"]
    assert h["bad"]["steps"]["dataset"] == "failed" and h["bad"]["steps"]["training"] == "pending"
    assert h["good"]["ok"] is True and h["good"]["steps"]["score"] == "done"
    assert st["project"] == "good" and st["queue"] == []


def test_autopilot_needs_the_ai_toolkit_folder(studio, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ds_autopilot, "AUTOPILOT",
                        ds_autopilot.Autopilot(str(tmp_path / "ap.json"), poll=0.05))
    post(studio, "project_new", {"name": "x"})
    post(studio, "set_reference", {"name": "x", "data": data_url()})
    r = post(studio, "autopilot_start", {"name": "x"})
    assert not r["ok"] and "AI-Toolkit" in r["error"]


def test_autopilot_resumes_a_training_cut_by_a_restart(tmp_path):
    path = str(tmp_path / "ap.json")
    with open(path, "w") as fh:
        json.dump({"running": True, "project": "p", "step": "training", "job": "j_v1",
                   "steps": {"dataset": "done", "export": "done", "job": "done",
                             "training": "running", "score": "pending"}}, fh)
    ap = ds_autopilot.Autopilot(path)
    st = ap.status()
    assert st["running"] is False and st["interrupted"] == "training"


# ------------------------------------------------------------------ gated check, LoRA metadata
def _real_lora(path, meta=None):
    """A real safetensors file, written by hand (no dependency): two float32 tensors."""
    import struct
    a = struct.pack("<4f", 1, 2, 3, 4)
    b = struct.pack("<2f", 5, 6)
    h = {"lora_A.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
         "lora_B.weight": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]}}
    if meta:
        h["__metadata__"] = meta
    hb = json.dumps(h).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)) + hb + a + b)


def test_copy_lora_writes_the_trigger_into_the_copy_only(tmp_path):
    src = tmp_path / "out" / "j_v1.safetensors"
    src.parent.mkdir()
    _real_lora(src, {"name": "j_v1", "software": "ai-toolkit"})
    before = src.read_bytes()
    dst = tmp_path / "loras"
    meta = {"trigger_words": "ohwx character", "crispz_dataset_studio": '{"project": "p"}'}
    r = ds_aitoolkit.copy_lora(str(src), str(dst), meta=meta)
    assert r["copied"] and r["meta"]
    assert src.read_bytes() == before                          # ai-toolkit's file untouched
    got = ds_aitoolkit.read_lora_meta(r["path"])
    assert got["trigger_words"] == "ohwx character" and got["software"] == "ai-toolkit"
    assert ds_aitoolkit._same_tensors(str(src), r["path"])
    try:                                                        # the real loader agrees
        from safetensors.numpy import load_file
        t = load_file(r["path"])
        assert t["lora_A.weight"].tolist() == [[1, 2], [3, 4]] and t["lora_B.weight"].tolist() == [5, 6]
    except ImportError:
        pass
    # again: the same tensors, same metadata -> left as is, no _2
    r2 = ds_aitoolkit.copy_lora(str(src), str(dst), meta=meta)
    assert not r2["copied"] and r2["file"] == "j_v1.safetensors"
    # a new trigger word: the copy is updated in place, still no _2
    r3 = ds_aitoolkit.copy_lora(str(src), str(dst), meta=dict(meta, trigger_words="tok"))
    assert not r3["copied"] and ds_aitoolkit.read_lora_meta(r3["path"])["trigger_words"] == "tok"
    assert sorted(os.listdir(dst)) == ["j_v1.safetensors"]
    # a retrained LoRA under the same name: _2, the first one kept
    _real_lora(src)
    with open(src, "r+b") as fh:
        fh.seek(-4, 2)
        fh.write(b"\x00\x00\x80\x3f")
    r4 = ds_aitoolkit.copy_lora(str(src), str(dst), meta=meta)
    assert r4["copied"] and r4["file"] == "j_v1_2.safetensors"


def test_gated_check(tmp_path, monkeypatch):
    root = fake_aitk(tmp_path)
    repo = "black-forest-labs/FLUX.2-klein-base-9B"
    monkeypatch.setattr(ds_aitoolkit, "_AUTH_CACHE", {})
    answers = {"r": {"ok": False, "kind": "GatedRepoError", "error": "403 Client Error"}}
    monkeypatch.setattr(ds_aitoolkit, "_auth_probe", lambda py, rp, env, timeout=90: answers["r"])
    r = ds_aitoolkit.check_hf_access(str(root), repo)
    assert r["ok"] is False and "accept its licence" in r["error"] and repo in r["error"]
    answers["r"] = {"ok": False, "kind": "ImportError", "error": "no huggingface_hub"}
    r = ds_aitoolkit.check_hf_access(str(root), repo)
    assert r["ok"] is None and "going on" in r["warning"]
    answers["r"] = {"ok": True}
    assert ds_aitoolkit.check_hf_access(str(root), repo)["ok"] is True
    answers["r"] = {"ok": False, "kind": "GatedRepoError", "error": "403"}
    assert ds_aitoolkit.check_hf_access(str(root), repo).get("cached")      # 10 min cache
    assert ds_aitoolkit.gated_repo_of("klein9b") == repo
    assert ds_aitoolkit.gated_repo_of("klein4b") is None


def test_autopilot_refuses_a_9b_queue_without_access(studio, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ds_autopilot, "AUTOPILOT",
                        ds_autopilot.Autopilot(str(tmp_path / "ap.json"), poll=0.05))
    monkeypatch.setattr(ds_aitoolkit, "_AUTH_CACHE", {})
    monkeypatch.setattr(ds_aitoolkit, "_auth_probe",
                        lambda py, rp, env, timeout=90: {"ok": False, "kind": "GatedRepoError",
                                                         "error": "403"})
    root = fake_aitk(tmp_path / "tk")
    cfg = post(studio, "config_get")["config"]
    cfg["ai_toolkit_dir"] = str(root)
    post(studio, "config_set", {"config": cfg})
    post(studio, "project_new", {"name": "nine"})
    post(studio, "project_settings", {"name": "nine", "settings": {"auto_preset": "klein9b"}})
    post(studio, "set_reference", {"name": "nine", "data": data_url()})
    r = post(studio, "autopilot_queue", {"names": ["nine"]})
    assert not r["ok"] and "FLUX.2-klein-base-9B" in r["error"]
    assert not ds_autopilot.autopilot().status().get("running")
    p = ds_project.load("nine")                           # nothing rendered by the queue
    assert all(c["status"] == "pending" for c in p["cards"])


def test_keep_best_keeps_the_best_save_and_only_removes_our_copies(tmp_path, monkeypatch):
    monkeypatch.setattr(ds_project, "PROJECTS_DIR", str(tmp_path / "projects"))
    root = fake_aitk(tmp_path / "tk")
    job = "p_klein4b_v1"
    out = root / "output" / job
    out.mkdir(parents=True)
    saves = {}
    for f, v in ((job + ".safetensors", 1.0), (job + "_000001250.safetensors", 2.0),
                 (job + "_000001000.safetensors", 3.0)):
        _real_lora(out / f)
        with open(out / f, "r+b") as fh:            # distinct tensors per save
            fh.seek(-4, 2)
            fh.write(__import__("struct").pack("<f", v))
        saves[f] = out / f
    loras = tmp_path / "crispz-klein" / "loras"
    p = ds_project.create("p")
    p["settings"]["trigger_word"] = "tok"
    copies = {f: ds_aitoolkit.copy_lora(str(sp), str(loras), meta=ds_aitoolkit.lora_meta_for(p, job))["path"]
              for f, sp in saves.items()}
    (loras / "my_style.safetensors").write_bytes(b"the user's own LoRA")
    p["lora_scores"] = [
        {"job": job, "lora": job + ".safetensors", "final": True, "step": None, "scored": 4,
         "mean_lora": 58.0, "mean_base": 2.0, "copy": copies[job + ".safetensors"]},
        {"job": job, "lora": job + "_000001250.safetensors", "final": False, "step": 1250,
         "scored": 4, "mean_lora": 63.5, "mean_base": 2.0, "copy": copies[job + "_000001250.safetensors"]},
        {"job": job, "lora": job + "_000001000.safetensors", "final": False, "step": 1000,
         "scored": 0, "mean_lora": None, "mean_base": None, "copy": copies[job + "_000001000.safetensors"]}]
    ds_project.save(p)
    r = ds_lorascore.keep_best("p", str(root), job, str(loras))
    assert r["best"] == job + "_000001250.safetensors" and r["file"] == "p_klein4b_best.safetensors"
    assert sorted(os.listdir(loras)) == ["my_style.safetensors", "p_klein4b_best.safetensors"]
    assert ds_aitoolkit._same_tensors(str(saves[job + "_000001250.safetensors"]), r["path"])
    meta = ds_aitoolkit.read_lora_meta(r["path"])
    assert meta["trigger_words"] == "tok" and "63.5" in meta["crispz_best"]
    assert len(os.listdir(out)) == 3                          # ai-toolkit keeps every save
    assert "warning" not in r and not r["replaced"]
    la = ds_project.load("p")["last_aitk"]
    assert la["best"]["lora"] == job + "_000001250.safetensors"
    assert [c["file"] for c in la["copied"]] == ["p_klein4b_best.safetensors"]
    # a weak LoRA is said; a second keep replaces the _best file
    p = ds_project.load("p")
    for h in p["lora_scores"]:
        h["mean_base"] = 60.0
    ds_project.save(p)
    r = ds_lorascore.keep_best("p", str(root), job, str(loras))
    assert r["replaced"] and "weak LoRA" in r["warning"]
    with pytest.raises(ValueError):
        ds_lorascore.keep_best("p", str(root), "other_v1", str(loras))
