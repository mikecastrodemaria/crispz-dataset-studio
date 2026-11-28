"""ds_term: the start banner and the log lines (plain when not a terminal)."""

import ds_term


def test_plain_lines_keep_the_old_format_when_colours_are_off(monkeypatch):
    monkeypatch.setattr(ds_term, "_COLOR", False)
    l = ds_term.line("[engines] minimax: ready at http://127.0.0.1:7866")
    assert l.endswith("[engines] minimax: ready at http://127.0.0.1:7866") and "\033" not in l
    assert ds_term.line("stopping").endswith(" stopping")


def test_coloured_lines_mark_the_source_errors_and_likeness(monkeypatch):
    monkeypatch.setattr(ds_term, "_COLOR", True)
    monkeypatch.setattr(ds_term, "_UNI", True)
    ok = ds_term.line("[project] card 001 try 1: likeness 82 (cos 0.81)")
    assert "project" in ok and "\033[38;5;120m" in ok and "82" in ok
    bad = ds_term.line("[project] card 003: FAILED: KeyError")
    assert "\033[38;5;203m" in bad
    assert "\033[38;5;177m" in ds_term.line("[autopilot] finished")


def test_banner_lists_the_engines_and_fits_its_box(monkeypatch):
    monkeypatch.setattr(ds_term, "_COLOR", False)
    monkeypatch.setattr(ds_term, "_UNI", True)
    b = ds_term.banner("0.1.0", "http://127.0.0.1:8780",
                       [("klein", 7861, True), ("minimax", 7866, False)], None, 3)
    assert "http://127.0.0.1:8780" in b and "klein" in b and "running" in b
    assert "minimax" in b and "stopped" in b and "not set" in b
    boxed = [l for l in b.splitlines() if l.strip().startswith(("│", "╭", "╰"))]
    assert len({len(l) for l in boxed}) == 1                  # every row the same width
    monkeypatch.setattr(ds_term, "_UNI", False)
    assert "│" not in ds_term.banner("0.1.0", "u", [], None, 0)
