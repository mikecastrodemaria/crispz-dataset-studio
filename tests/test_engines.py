"""ds_engines: the protocol client against the fake engine, and the launcher against
a fake app.py (a real child process, started with THIS interpreter)."""

import os
import sys
import socket
import time

import pytest
from PIL import Image

import ds_engines
import fake_engine

HERE = os.path.dirname(os.path.abspath(__file__))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def running_engine():
    port = free_port()
    srv = fake_engine.start_in_thread(port)
    e = ds_engines.Engine("fake", {"url": f"http://127.0.0.1:{port}", "launch": False})
    yield e
    srv.shutdown()


def test_probe_nothing_there():
    assert ds_engines.probe(f"http://127.0.0.1:{free_port()}", timeout=1) is None


def test_probe_and_status(running_engine):
    caps = running_engine.caps(max_age=0)
    assert caps and caps["tool"] == "fake-klein"
    st = running_engine.status()
    assert st["running"] and st["faces"] and st["edit"] and not st["started_by_us"]


def test_edit_and_faces_round_trip(running_engine, tmp_path):
    ref = tmp_path / "ref.png"
    Image.new("RGB", (32, 32), (200, 40, 40)).save(ref)
    r = running_engine.edit({"prompt": "same person, smiling", "input": str(ref),
                             "width": 64, "height": 64, "seed": 3, "out_dir": str(tmp_path)})
    assert r["ok"], r
    assert os.path.isfile(r["images"][0]) and r["seed_used"] == 3
    f = running_engine.faces(r["images"][0])
    assert f["ok"] and len(f["faces"]) == 1 and len(f["faces"][0]["embedding"]) == 512


def test_edit_failure_is_a_dict_not_an_exception(running_engine, tmp_path):
    ref = tmp_path / "ref.png"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(ref)
    r = running_engine.edit({"prompt": "FAIL please", "input": str(ref)})
    assert r["ok"] is False and "FAIL" in r["error"]
    bad = ds_engines.Engine("down", {"url": f"http://127.0.0.1:{free_port()}"})
    r = bad.edit({"prompt": "x", "input": str(ref)})
    assert r["ok"] is False and "down" in r["error"]


def test_launch_refused_without_dir_or_when_disabled():
    e = ds_engines.Engine("nodir", {"url": f"http://127.0.0.1:{free_port()}"})
    r = e.launch()
    assert not r["ok"] and "install folder" in r["error"]
    e = ds_engines.Engine("off", {"url": f"http://127.0.0.1:{free_port()}",
                                  "dir": HERE, "launch": False})
    r = e.launch()
    assert not r["ok"] and "start it by hand" in r["error"]


def test_resolve_python_order(tmp_path):
    argv, label = ds_engines.resolve_python(str(tmp_path), explicit=sys.executable)
    assert argv == [sys.executable] and "python key" in label
    argv, label = ds_engines.resolve_python(str(tmp_path), explicit=str(tmp_path / "nope.exe"))
    assert argv is None and "missing" in label
    win = os.name == "nt"
    venv = tmp_path / ".venv" / ("Scripts" if win else "bin")
    venv.mkdir(parents=True)
    py = venv / ("python.exe" if win else "python")
    py.write_bytes(b"")
    argv, label = ds_engines.resolve_python(str(tmp_path))
    assert argv == [str(py)] and label.startswith(".venv")


def test_launch_starts_the_app_waits_for_caps_and_stops_it():
    port = free_port()
    e = ds_engines.Engine("fakeapp", {"url": f"http://127.0.0.1:{port}",
                                      "dir": os.path.join(HERE, "fake_app_dir"),
                                      "python": sys.executable, "boot_timeout": 30})
    try:
        r = e.launch(wait=True)
        assert r["ok"] and r["started"], r
        assert r["caps"]["tool"] == "fake-klein"
        assert e.started_by_us() and e.status()["started_by_us"]
        # a second launch call is a no-op: it is already running
        r2 = e.launch(wait=True)
        assert r2["ok"] and r2["started"] is False
        assert os.path.isfile(e.log_path)
        with open(e.log_path, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        assert f"port {port}" in txt and "fake app booting" in txt
    finally:
        r = e.stop()
    assert r["ok"] and r["stopped"]
    assert not e.started_by_us()
    time.sleep(0.3)
    assert ds_engines.probe(e.url, timeout=1) is None


def test_registry_ensure_refuses_a_second_child():
    p1, p2 = free_port(), free_port()
    cfg = {"engine": "a", "engines": {
        "a": {"url": f"http://127.0.0.1:{p1}", "dir": os.path.join(HERE, "fake_app_dir"),
              "python": sys.executable, "boot_timeout": 30},
        "b": {"url": f"http://127.0.0.1:{p2}", "dir": os.path.join(HERE, "fake_app_dir"),
              "python": sys.executable, "boot_timeout": 30}}}
    reg = ds_engines.Engines(cfg)
    try:
        e, r = reg.ensure()
        assert e.name == "a" and r["ok"] and r["started"]
        e2, r2 = reg.ensure("b")
        assert not r2["ok"] and "still running" in r2["error"]
        # a settings reload keeps the child we started
        reg.reload(cfg)
        assert reg.alive_child().name == "a"
    finally:
        reg.stop_all()
    assert reg.alive_child() is None


def test_stop_never_kills_an_engine_we_did_not_start(running_engine):
    r = running_engine.stop()
    assert not r["ok"] and "not started by Dataset Studio" in r["error"]
    assert running_engine.caps(max_age=0) is not None
