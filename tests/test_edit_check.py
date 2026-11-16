"""The queue refuses, before anything starts, a project that still has cards to render
with a text-to-image engine (A/B queue of 2026-11-16: krea2 then ideogram as the
DATASET engine, each project died after the engine swap)."""

import pytest

import ds_engines
import ds_project
import ds_server
from test_server import studio, free_port  # noqa: F401


def _studio_with(tmp_path, monkeypatch, engines):
    monkeypatch.setattr(ds_engines, "CAPS_SEEN", str(tmp_path / "engine_caps.json"))
    st = ds_server.STUDIO
    cfg = dict(st.config)
    cfg["engines"] = dict(cfg["engines"], **engines)
    st.engines.reload(cfg)
    return st


def _dead(tmp_path, app):
    (tmp_path / app).mkdir(exist_ok=True)
    return {"url": f"http://127.0.0.1:{free_port()}", "launch": False,
            "dir": str(tmp_path / app)}


def test_a_text_to_image_engine_is_refused_when_cards_are_left(studio, tmp_path, monkeypatch):  # noqa: F811
    st = _studio_with(tmp_path, monkeypatch, {"ideo": _dead(tmp_path, "crispz-ideogram"),
                                              "odd": _dead(tmp_path, "my-own-app")})
    assert st.engines.get("fake").can_edit() is True        # live caps, remembered
    assert st.engines.get("ideo").can_edit() is False       # app folder
    assert st.engines.get("odd").can_edit() is None         # unknown: let the run ask
    ds_project.create("ab-ideo", engine="ideo")
    assert ds_project.cards_to_render(ds_project.load("ab-ideo"))
    with pytest.raises(ValueError) as ei:
        st._dataset_engine_check([("ab-ideo", {"engine": "ideo"})])
    msg = str(ei.value)
    assert "'ab-ideo'" in msg and "engine 'ideo'" in msg
    assert "fake" in msg and "keep the" in msg              # who can, and the clone way
    st._dataset_engine_check([("ab-ideo", {"engine": "odd"})])
    st._dataset_engine_check([("ab-ideo", {"engine": "fake"})])


def test_a_finished_clone_passes_whatever_the_engine(studio, tmp_path, monkeypatch):  # noqa: F811
    st = _studio_with(tmp_path, monkeypatch, {"ideo": _dead(tmp_path, "crispz-ideogram")})
    p = ds_project.create("done", engine="ideo")
    for c in p["cards"]:
        c["status"] = "done"
    ds_project.save(p)
    st._dataset_engine_check([("done", {"engine": "ideo"})])


def test_what_an_engine_said_last_beats_its_folder(studio, tmp_path, monkeypatch):  # noqa: F811
    st = _studio_with(tmp_path, monkeypatch, {"k": _dead(tmp_path, "crispz-klein")})
    assert st.engines.get("k").can_edit() is True
    ds_engines._seen_save("k", {"tool": "crispz-klein", "supports": {"edit": False}})
    assert st.engines.get("k").can_edit() is False
    assert "k" not in st.engines.edit_capable()
