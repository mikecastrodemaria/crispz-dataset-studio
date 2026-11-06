"""ds_similar (cards too close), image_meta (viewer Info / Compare) and
tools/send_to_studio.py (the IrfanView key), over real HTTP."""

import os
import sys
import time

from PIL import Image, ImageDraw

import ds_project
import ds_similar
from test_server import studio, post, data_url  # noqa: F401  (fixture reuse)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import send_to_studio  # noqa: E402


def _scene(path, x, colour=(200, 60, 60), bg=(30, 30, 30), size=(96, 96)):
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    d.ellipse([x, 20, x + 40, 80], fill=colour)
    d.rectangle([0, 80, size[0], size[1]], fill=(90, 90, 150))
    im.save(path)
    return str(path)


def test_dhash_finds_the_same_composition_not_another(tmp_path):
    a = _scene(tmp_path / "a.png", 10)
    b = _scene(tmp_path / "b.png", 10, colour=(210, 70, 50))      # same scene, other colour
    c = _scene(tmp_path / "c.png", 50)                              # the figure moved
    d = tmp_path / "d.png"
    d.write_bytes(open(a, "rb").read())                             # a copy
    assert ds_similar.distance(ds_similar.dhash(a), ds_similar.dhash(b)) <= 2
    assert ds_similar.distance(ds_similar.dhash(a), ds_similar.dhash(c)) > 6
    pairs = ds_similar.find_similar([(1, a), (2, b), (3, c), (4, str(d)), (5, "missing.png")])
    keys = {(p["a"], p["b"]): p for p in pairs}
    assert keys[(1, 4)]["identical"] and keys[(1, 4)]["distance"] == 0
    assert (1, 2) in keys and (2, 4) in keys
    assert not any(3 in k for k in keys)
    assert pairs[0]["identical"]                                    # copies first
    s = ds_similar.summary(pairs)
    assert "001=004" in s and "too close" in s
    assert ds_similar.summary([]) == ""


def _rendered(base, name, prompts):
    post(base, "project_new", {"name": name})
    post(base, "project_settings", {"name": name, "settings": {
        "prompts": prompts, "width": 32, "height": 32, "trigger_word": "tok", "max_rerolls": 1,
        "threshold": 50}})
    post(base, "set_reference", {"name": name, "data": data_url()})
    post(base, "run", {"name": name})
    for _ in range(100):
        if not post(base, "status")["status"]["running"]:
            break
        time.sleep(0.05)


def test_similar_and_meta_over_http(studio):  # noqa: F811
    # the fake engine renders flat colours: every card looks like the others
    _rendered(studio, "sim", ["one {red|blue} dress DRIFT", "two"])
    r = post(studio, "similar", {"name": "sim"})
    assert r["ok"] and r["pairs"] and r["pairs"][0]["a"] == 1 and "too close" in r["summary"]
    ex = post(studio, "export", {"name": "sim"})
    assert ex["ok"] and any("too close" in w for w in ex["warnings"])
    r = post(studio, "project_settings", {"name": "sim", "settings": {"similar_max": 99}})
    assert r["project"]["settings"]["similar_max"] == 32                   # clamped
    p = ds_project.load("sim")
    c = p["cards"][0]
    assert len(c["tries"]) == 2                                            # DRIFT scores low: re-rolled
    m = post(studio, "image_meta", {"name": "sim", "file": c["tries"][1]["file"]})
    assert m["ok"] and m["card"]["index"] == 1 and m["card"]["try"] == 2 and m["card"]["tries"] == 2
    assert m["crispz"]["seed"] == c["tries"][1]["seed"] and "dress" in m["crispz"]["prompt"]
    assert m["width"] == 64 and m["engine"] == "fake"          # sizes are 64 px at least
    assert not post(studio, "image_meta", {"name": "sim", "file": "../../config.json"})["ok"]
    assert not post(studio, "image_meta", {"name": "sim", "file": "cards/nope.png"})["ok"]


def test_send_to_studio(studio, tmp_path):  # noqa: F811
    port = int(studio.rsplit(":", 1)[1])
    photo = tmp_path / "IMG 0042 (portrait).jpg"
    Image.new("RGB", (40, 40), (10, 200, 30)).save(photo)
    msgs = []
    r = send_to_studio.send(str(photo), port=port, browser=False, out=msgs.append)
    assert r["project"] == "IMG-0042-portrait" and r["url"].endswith("?project=IMG-0042-portrait")
    p = ds_project.load("IMG-0042-portrait")
    assert p["reference"] == "reference.png" and p["ref_embedding"]
    r = send_to_studio.send(str(photo), port=port, browser=False, out=msgs.append)
    assert r["project"] == "IMG-0042-portrait-2"                           # never overwritten
    r = send_to_studio.send(str(photo), project="lea", port=port, browser=False, out=msgs.append)
    assert r["project"] == "lea"
    assert send_to_studio.main([str(tmp_path / "missing.jpg"), "--port", str(port),
                                "--no-browser"]) == 1
