"""ds_project.clone + ds_aitoolkit: the project copy, the ai-toolkit folder search
and the job (dataset copy + config), then both over HTTP."""

import os
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
