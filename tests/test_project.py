"""ds_project + ds_likeness + ds_captions: a whole run against the fake engine, on
CPU, in a temporary projects folder."""

import os
import io
import json
import socket
import zipfile

import pytest
from PIL import Image

import ds_engines
import ds_project
import ds_likeness
import ds_captions
import fake_engine


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def projects(tmp_path, monkeypatch):
    monkeypatch.setattr(ds_project, "PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(ds_project, "RUNNER", ds_project.Runner())
    return tmp_path


@pytest.fixture
def engines():
    port = free_port()
    srv = fake_engine.start_in_thread(port)
    fake_engine.STATE.update({"edits": 0, "faces": 0, "delay": 0.0})
    reg = ds_engines.Engines({"engine": "fake", "engines": {
        "fake": {"url": f"http://127.0.0.1:{port}", "launch": False}}})
    yield reg
    srv.shutdown()


def png_bytes(color=(200, 60, 60), size=(48, 48)):
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "PNG")
    return b.getvalue()


# ---------------------------------------------------------------- likeness
def test_likeness_scale_and_reasons():
    a = [1.0] * 512
    assert ds_likeness.score(a, [{"box": [0, 0, 10, 10], "embedding": a}])[0] == 100
    s, d = ds_likeness.score(a, [{"box": [0, 0, 10, 10], "embedding": [-1.0] * 512}])
    assert s == 0 and d["cosine"] == -1.0
    s, d = ds_likeness.score(a, [])
    assert s == 0 and "no face detected" in d["reason"]
    s, d = ds_likeness.score(None, [{"box": [0, 0, 1, 1], "embedding": a}])
    assert s == 0 and "reference" in d["reason"]
    big = {"box": [0, 0, 100, 100], "embedding": a}
    small = {"box": [0, 0, 5, 5], "embedding": a}
    assert ds_likeness.main_face([small, big]) is big


# ---------------------------------------------------------------- captions
def test_caption_compose_and_prompt_fallback(tmp_path):
    assert ds_captions.compose("a, b", "ohwx man", "prompt") == "ohwx man, a, b"
    assert ds_captions.compose("a", "", "both", "desc") == "a, desc"
    assert ds_captions.compose("a", "", "ollama", None) == "a"
    img = tmp_path / "x.png"
    img.write_bytes(png_bytes())
    # Ollama at an unreachable port: falls back to the prompt line and says so
    r = ds_captions.caption_for(str(img), "same person, smiling", {
        "caption_source": "ollama", "trigger_word": "zzz"},
        {"url": f"http://127.0.0.1:{free_port()}", "model": "x"})
    assert r["caption"] == "zzz, same person, smiling"
    assert r["source_used"] == "prompt" and "fell back" in r["warning"]


def test_split_line_and_engine_prompt():
    assert ds_project.split_line("a, b #noscore") == ("a, b", False, "marked #noscore")
    t, scored, why = ds_project.split_line("same person, rear view, walking away")
    assert not scored and "rear view" in why
    assert ds_project.split_line("same person, smiling") == ("same person, smiling", True, None)
    assert ds_project.engine_prompt("close-up #noscore", {"subject": " the same  woman, "}) \
        == "the same woman, close-up"
    assert ds_project.engine_prompt("close-up", {}) == "close-up"


def test_ollama_vision_filter(monkeypatch):
    monkeypatch.setattr(ds_captions, "ollama_models",
                        lambda url=None, timeout=4: ["qwen3.6:latest", "gemma4:26b",
                                                     "llava:13b", "old-vl:7b", "bert"])
    show = {"qwen3.6:latest": {"capabilities": ["completion", "tools"]},
            "gemma4:26b": {"capabilities": ["completion", "vision"]},
            "llava:13b": {"capabilities": ["completion", "vision"]},
            "old-vl:7b": {},                       # an old Ollama: no capabilities field
            "bert": {"capabilities": ["embedding"]}}
    assert ds_captions.ollama_vision_models(show=show) == ["llava:13b", "old-vl:7b", "gemma4:26b"]


# ---------------------------------------------------------------- project files
def test_create_load_settings_and_card_rebuild(projects):
    p = ds_project.create("My Hero!", engine="fake")
    assert p["name"] == "My-Hero" and len(p["cards"]) == 25
    assert p["cards"][0]["status"] == "pending"
    p = ds_project.update_settings(p, {"prompts": "one\n# c\n\ntwo", "threshold": 150,
                                       "width": 1000, "steps": ""})
    assert [c["prompt"] for c in p["cards"]] == ["one", "two"]
    assert p["settings"]["threshold"] == 100 and p["settings"]["width"] == 1000
    assert p["settings"]["steps"] is None
    p = ds_project.update_settings(p, {"model": " black-forest-labs/FLUX.2-klein-4B "})
    assert p["settings"]["model"] == "black-forest-labs/FLUX.2-klein-4B"
    p["cards"][0]["status"] = "done"
    p["cards"][0]["file"] = "cards/001_t1.png"
    ds_project.save(p)
    # an unchanged line keeps its render, a changed one resets
    p = ds_project.update_settings(ds_project.load("My-Hero"), {"prompts": ["one", "three"]})
    assert p["cards"][0]["status"] == "done" and p["cards"][1]["status"] == "pending"
    assert ds_project.list_projects()[0]["done"] == 1
    # picking the bundled set again under the SAME name restores its 25 lines
    p = ds_project.update_settings(ds_project.load("My-Hero"), {"promptset": "character-25"})
    assert len(p["cards"]) == 25
    with pytest.raises(FileExistsError):
        ds_project.create("My Hero")
    assert ds_project.list_promptsets()[0] == {"name": "character-25", "count": 25}


def test_set_reference_without_engine_then_embedding_at_run(projects, engines):
    p = ds_project.create("ref", engine="fake")
    p, warn = ds_project.set_reference(p, png_bytes(), engine=None)
    assert p["reference"] == "reference.png" and p["ref_embedding"] is None
    assert warn is None
    assert os.path.isfile(os.path.join(ds_project.project_dir("ref"), "reference.png"))
    stale = os.path.join(ds_project.project_dir("ref"), "cards", "001_t1.png")
    open(stale, "wb").write(png_bytes())
    p, warn = ds_project.set_reference(p, png_bytes(), engine=engines.get("fake"))
    assert len(p["ref_embedding"]) == 512 and p["ref_faces"] == 1
    assert not os.path.exists(stale)          # the old identity's renders are gone
    # a reference with no face says so
    p, warn = ds_project.set_reference(p, png_bytes((0, 0, 0)), engine=engines.get("fake"))
    assert p["ref_embedding"] is None and "no face" in warn


# ---------------------------------------------------------------- the run
def test_full_run_rerolls_and_resume(projects, engines):
    p = ds_project.create("run", engine="fake")
    ds_project.update_settings(p, {"prompts": ["same person, close-up",
                                               "same person, DRIFT far away",
                                               "same person, FAIL this one",
                                               "same person, DRIFT rear view #noscore"],
                                   "threshold": 90, "max_rerolls": 2, "width": 64, "height": 64,
                                   "trigger_word": "ohwx", "subject": "the hero"})
    p, _ = ds_project.set_reference(ds_project.load("run"), png_bytes(), engine=None)
    runner = ds_project.RUNNER
    r = runner.start("run", engines, {}, force=False)
    assert r["ok"] and r["started"] and r["total"] == 4
    assert runner.wait(60)
    st = runner.status()
    assert st["ok"] is True and st["done"] == 4, st
    p = ds_project.load("run")
    c1, c2, c3, c4 = p["cards"]
    # card 4: unscored -> one try only although it drifts, no threshold warning,
    # the caption carries the line without the mark and without the subject
    assert c4["status"] == "done" and len(c4["tries"]) == 1 and c4["scored"] is False
    assert not any("below" in w for w in c4["warnings"]) and c4["detail"]["unscored"]
    assert c4["caption"] == "ohwx, same person, DRIFT rear view"
    # the reference face was analysed at run start
    assert len(p["ref_embedding"]) == 512
    # card 1: same colour -> high likeness on the first try, caption = trigger + prompt
    assert c1["status"] == "done" and c1["score"] >= 90 and len(c1["tries"]) == 1
    assert c1["caption"] == "ohwx, same person, close-up"
    assert os.path.isfile(os.path.join(ds_project.project_dir("run"), c1["file"]))
    # card 2: drift -> all 3 tries below the threshold, best kept, a warning says so
    assert c2["status"] == "done" and len(c2["tries"]) == 3 and c2["score"] < 90
    assert any("below the threshold" in w for w in c2["warnings"])
    # card 3: the engine refused -> failed with its message, no silent skip
    assert c3["status"] == "failed" and "FAIL" in c3["error"]
    # resume: only the failed card is rendered again (still fails)
    n = fake_engine.STATE["edits"]
    r = runner.start("run", engines, {}, force=False)
    assert r["total"] == 1
    runner.wait(30)
    assert fake_engine.STATE["edits"] == n + 1
    # nothing to do once everything is done / only failed left? reject card 1 -> rendered again
    ds_project.update_card(ds_project.load("run"), 1, status="rejected")
    r = runner.start("run", engines, {}, force=False)
    assert r["total"] == 2          # the rejected one + the failed one (card 4 is done)
    runner.wait(30)
    p = ds_project.load("run")
    assert p["cards"][0]["status"] == "done" and len(p["cards"][0]["tries"]) == 2
    # the user can pick an earlier try back
    c = ds_project.update_card(p, 1, use_try=0)
    assert c["file"].endswith("001_t1.png")
    # only=[2] with force renders exactly card 2
    r = runner.start("run", engines, {}, force=True, only={2})
    assert r["total"] == 1
    runner.wait(30)
    # a re-roll after a subject change drops the tries of the old subject (their files
    # too), a re-roll with the same settings keeps them (the user may pick one back)
    p = ds_project.load("run")
    old_files = [t["file"] for t in p["cards"][0]["tries"]]
    assert len(old_files) == 2
    ds_project.update_settings(p, {"subject": "another hero"})
    runner.start("run", engines, {}, force=True, only={1})
    runner.wait(30)
    p = ds_project.load("run")
    assert len(p["cards"][0]["tries"]) == 1
    cards_dir = os.path.join(ds_project.project_dir("run"), "cards")
    assert [f for f in os.listdir(cards_dir) if f.startswith("001_")] == ["001_t1.png"]
    assert p["cards"][0]["tries"][0]["prompt"].startswith("another hero")
    runner.start("run", engines, {}, force=True, only={1})
    runner.wait(30)
    assert len(ds_project.load("run")["cards"][0]["tries"]) == 2
    # a forced run of the whole project starts every card from scratch, with the base
    # seed the tries are replayable: try t = seed + t - 1
    runner.start("run", engines, {}, force=True, seed=1000)
    runner.wait(60)
    p = ds_project.load("run")
    assert len(p["cards"][0]["tries"]) == 1 and p["cards"][0]["tries"][0]["seed"] == 1000
    assert [t["seed"] for t in p["cards"][1]["tries"]] == [1000, 1001, 1002]


def test_run_refuses_without_reference_and_when_busy(projects, engines):
    ds_project.create("empty", engine="fake")
    r = ds_project.RUNNER.start("empty", engines, {})
    assert not r["ok"] and "reference" in r["error"]


def test_run_engine_down_is_reported(projects):
    reg = ds_engines.Engines({"engine": "down", "engines": {
        "down": {"url": f"http://127.0.0.1:{free_port()}", "launch": False}}})
    p = ds_project.create("down", engine="down")
    ds_project.set_reference(p, png_bytes(), engine=None)
    r = ds_project.RUNNER.start("down", reg, {})
    assert r["ok"] and r["started"]
    ds_project.RUNNER.wait(30)
    st = ds_project.RUNNER.status()
    assert st["ok"] is False and "start it by hand" in st["error"]


def test_cancel_stops_between_cards(projects, engines):
    fake_engine.STATE["delay"] = 0.4
    p = ds_project.create("cancel", engine="fake")
    ds_project.update_settings(p, {"prompts": [f"same person {i}" for i in range(6)],
                                   "threshold": 0, "max_rerolls": 0, "width": 16, "height": 16})
    ds_project.set_reference(ds_project.load("cancel"), png_bytes(), engine=None)
    runner = ds_project.RUNNER
    runner.start("cancel", engines, {})
    import time
    time.sleep(0.6)
    assert runner.cancel()
    runner.wait(30)
    st = runner.status()
    assert st["ok"] is False and st["done"] < 6 and any("cancelled" in l for l in st["log"])
    fake_engine.STATE["delay"] = 0.0


# ---------------------------------------------------------------- export
def test_export_pairs_resized_and_skips_rejected(projects, engines):
    p = ds_project.create("exp", engine="fake")
    ds_project.update_settings(p, {"prompts": ["a", "b", "c"], "threshold": 0,
                                   "max_rerolls": 0, "width": 128, "height": 64})
    ds_project.set_reference(ds_project.load("exp"), png_bytes(), engine=None)
    ds_project.RUNNER.start("exp", engines, {})
    ds_project.RUNNER.wait(30)
    p = ds_project.load("exp")
    ds_project.update_card(p, 2, status="rejected")
    ds_project.update_card(ds_project.load("exp"), 3, caption="  custom caption ")
    r = ds_project.export(ds_project.load("exp"), size=32, fmt="jpg",
                          output_dir=str(projects / "out"))
    assert r["count"] == 2 and r["skipped"] == [2]
    # the family layout: <out>/<date>/<project>_<stamp>/ + <lot>.zip beside it
    import time as _t
    day = _t.strftime("%Y-%m-%d")
    assert r["lot"].startswith(f"{day}/exp_") and r["zip_rel"] == r["lot"] + ".zip"
    assert os.path.isdir(r["dir"]) and r["zip"] == r["dir"] + ".zip"
    assert sorted(os.listdir(r["dir"])) == ["001.jpg", "001.txt", "003.jpg", "003.txt"]
    assert ds_project.load("exp")["last_export"]["lot"] == r["lot"]
    # a second export the same second is a new lot, the first one is untouched
    r2 = ds_project.export(ds_project.load("exp"), size=32, fmt="jpg",
                           output_dir=str(projects / "out"))
    assert r2["dir"] != r["dir"] and os.path.isdir(r["dir"]) and os.path.isfile(r["zip"])
    flat = ds_project.export(ds_project.load("exp"), output_dir=str(projects / "flat"),
                             date_subfolders=False)
    assert "/" not in flat["lot"] and flat["lot"].startswith("exp_")
    with zipfile.ZipFile(r["zip"]) as z:
        names = sorted(z.namelist())
        assert names == ["001.jpg", "001.txt", "003.jpg", "003.txt"]
        assert z.read("003.txt").decode().strip() == "custom caption"
        img = Image.open(io.BytesIO(z.read("001.jpg")))
        assert max(img.size) == 32 and img.size == (32, 16)
