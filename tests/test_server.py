"""ds_server: the JSON API and the file routes, over real HTTP, with the fake engine
behind it and a temporary config / projects folder."""

import os
import io
import json
import time
import socket
import base64
import threading
import urllib.request

import pytest
from PIL import Image

import ds_server
import ds_project
import fake_engine


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setattr(ds_project, "PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(ds_project, "RUNNER", ds_project.Runner())
    monkeypatch.setattr(ds_server, "CONFIG_PATH", str(tmp_path / "config.json"))
    eport = free_port()
    esrv = fake_engine.start_in_thread(eport)
    cfg = {"engine": "fake", "engines": {"fake": {"url": f"http://127.0.0.1:{eport}",
                                                   "launch": False}},
           "ollama": {"url": f"http://127.0.0.1:{free_port()}", "model": ""}, "port": 0}
    port = free_port()
    ds_server.STUDIO = ds_server.Studio(cfg)
    srv = ds_server.ThreadingHTTPServer(("127.0.0.1", port), ds_server.Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()
    esrv.shutdown()


def post(base, op, data=None):
    req = urllib.request.Request(f"{base}/api/{op}", data=json.dumps(data or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def get(base, path):
    with urllib.request.urlopen(f"{base}{path}", timeout=10) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def data_url(color=(10, 200, 30)):
    b = io.BytesIO()
    Image.new("RGB", (40, 40), color).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def test_spa_and_ping(studio):
    st, ct, body = get(studio, "/")
    assert st == 200 and "text/html" in ct and b"crispz Dataset Studio" in body
    st, _, body = get(studio, "/api/ping")
    assert json.loads(body)["tool"] == "crispz-dataset-studio"
    with pytest.raises(urllib.error.HTTPError):
        get(studio, "/assets/../ds_server.py")


def test_engines_and_config_round_trip(studio, tmp_path):
    r = post(studio, "engines")
    assert r["ok"] and r["engines"][0]["running"] and r["default"] == "fake"
    r = post(studio, "config_get")
    assert not r["exists"]
    cfg = r["config"]
    cfg["engines"]["second"] = {"url": "http://127.0.0.1:1", "dir": "", "launch": True}
    r = post(studio, "config_set", {"config": cfg})
    assert r["ok"] and {e["name"] for e in r["engines"]} == {"fake", "second"}
    assert os.path.isfile(str(tmp_path / "config.json"))
    r = post(studio, "engine_launch", {"name": "second"})
    assert not r["ok"] and "install folder" in r["error"]
    r = post(studio, "unknown_op")
    assert not r["ok"]


def test_project_flow_over_http(studio):
    r = post(studio, "project_new", {"name": "http test"})
    assert r["ok"] and r["project"]["name"] == "http-test" and r["project"]["engine"] == "fake"
    r = post(studio, "project_settings", {"name": "http-test", "settings": {
        "prompts": ["same person, one", "same person, two"], "threshold": 50,
        "max_rerolls": 1, "width": 32, "height": 32, "caption_source": "both",
        "trigger_word": "tok"}})
    assert r["ok"] and len(r["project"]["cards"]) == 2
    # the run refuses without a reference
    r = post(studio, "run", {"name": "http-test"})
    assert not r["ok"] and "reference" in r["error"]
    r = post(studio, "set_reference", {"name": "http-test", "data": data_url()})
    assert r["ok"] and r["project"]["ref_embedding"] and r["warning"] is None
    st, ct, body = get(studio, "/file/http-test/reference.png")
    assert st == 200 and ct == "image/png"
    r = post(studio, "run", {"name": "http-test"})
    assert r["ok"] and r["started"] and r["total"] == 2
    for _ in range(100):
        s = post(studio, "status")["status"]
        if not s["running"]:
            break
        time.sleep(0.1)
    assert s["ok"] is True and s["done"] == 2
    p = post(studio, "project_get", {"name": "http-test"})["project"]
    c = p["cards"][0]
    assert c["status"] == "done" and c["score"] >= 50
    # Ollama is unreachable in the tests: 'both' fell back to the prompt and said so
    assert c["caption"] == "tok, same person, one"
    assert any("fell back" in w for w in c["warnings"])
    st, ct, body = get(studio, f"/file/http-test/{c['file']}")
    assert st == 200 and ct == "image/png"
    r = post(studio, "card_update", {"name": "http-test", "index": 2, "status": "rejected",
                                     "caption": "edited"})
    assert r["ok"] and r["card"]["status"] == "rejected" and r["card"]["caption"] == "edited"
    r = post(studio, "export", {"name": "http-test", "size": 16, "format": "png"})
    assert r["ok"] and r["count"] == 1 and r["skipped"] == [2]
    st, ct, body = get(studio, f"/file/http-test/{r['zip_rel']}")
    assert st == 200 and body[:2] == b"PK"
    assert post(studio, "projects")["projects"][0]["done"] == 1
    assert post(studio, "promptsets")["promptsets"][0]["count"] == 25
