"""The terminal face of Dataset Studio: a start banner and coloured log lines.

Plain text when the output is not a terminal (a file, a pipe), when NO_COLOR is
set, or when the console cannot show colours; box drawing falls back to ASCII when
the console is not UTF-8. No emoji (house rule): bullets and boxes only.
"""

import os
import re
import sys
import time

_COLOR = False
_UNI = False


def _setup():
    """Colours on when they can be shown; UTF-8 output where possible."""
    global _COLOR, _UNI
    out = sys.stdout
    try:
        out.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    enc = (getattr(out, "encoding", "") or "").lower().replace("-", "")
    _UNI = enc in ("utf8", "utf8sig")
    tty = hasattr(out, "isatty") and out.isatty()
    if os.environ.get("NO_COLOR") or not tty:
        _COLOR = False
        return
    if os.name == "nt":
        # Windows 10+: ask the console for VT sequences (Windows Terminal has them)
        try:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if k.GetConsoleMode(h, ctypes.byref(mode)):
                k.SetConsoleMode(h, mode.value | 0x0004)
                _COLOR = True
        except Exception:
            _COLOR = bool(os.environ.get("WT_SESSION"))
    else:
        _COLOR = os.environ.get("TERM", "") != "dumb"


_setup()


def c(code, s):
    """ANSI 256 colour (an int) or a raw SGR string; plain when colours are off."""
    if not _COLOR:
        return s
    sgr = f"38;5;{code}" if isinstance(code, int) else code
    return f"\033[{sgr}m{s}\033[0m"


def dim(s):
    return c("2", s)


def bold(s):
    return c("1", s)


def ch(uni, ascii_):
    return uni if _UNI else ascii_


# -- the logo -----------------------------------------------------------------
_LOGO = [
    " ___  ___  ___  ___  ___  ____",
    "/ __|| _ \\|_ _|/ __|| _ \\|_  /",
    "| (__|   / | | \\__ \\|  _/ / / ",
    " \\___|_|_\\|___||___/|_|  /___|",
]
# cyan to violet, one colour per line
_GRAD = [51, 45, 39, 99, 135]

# the source of a log line: colour and short label
TAGS = {
    "studio": (75, "studio"),
    "engines": (214, "engines"),
    "project": (114, "project"),
    "autopilot": (177, "autopilot"),
    "ai-toolkit": (39, "toolkit"),
    "lora-score": (219, "score"),
}
_TAG_RE = re.compile(r"^\[([a-z\-]+)\]\s*(.*)$", re.S)
_BAD = re.compile(r"\b(FAILED|failed|error|Error|refused|cannot|exited with code|Traceback)\b")
_WARN = re.compile(r"\b(WARNING|stopped:|skipped|ignored)\b")
_GOOD = re.compile(r"\b(ready|finished|done|kept|saved|started|merged|copied)\b")
_LIKE = re.compile(r"likeness (\d+)")
_URL = re.compile(r"(https?://\S+)")


def _paint(msg):
    if not _COLOR:
        return msg
    if _BAD.search(msg):
        return c(203, msg)
    msg = _URL.sub(lambda m: c("4;38;5;75", m.group(1)), msg)
    msg = _LIKE.sub(lambda m: "likeness " + c(_like_col(int(m.group(1))),
                                              bold(m.group(1))), msg)
    if _WARN.search(msg):
        return c(221, msg)
    if msg.startswith("card ") and ": rendering" in msg:
        return dim(msg)
    return _GOOD.sub(lambda m: c(120, m.group(1)), msg)


def _like_col(v):
    return 120 if v >= 70 else (221 if v >= 40 else 203)


def line(msg):
    """One log line: time, source, message."""
    t = time.strftime("%H:%M:%S")
    m = _TAG_RE.match(msg)
    tag, body = (m.group(1), m.group(2)) if m else ("studio", msg)
    col, label = TAGS.get(tag, (250, tag))
    if not _COLOR:
        return f"{t} [{tag}] {body}" if m else f"{t} {body}"
    bar = ch("│", "|")
    return f"{dim(t)} {c(col, ch(chr(0x25CF), '*'))} {c(col, label.ljust(10))}{dim(bar)} {_paint(body)}"


# -- the banner ---------------------------------------------------------------
def _box(rows, width):
    tl, tr, bl, br, h, v = (ch("╭", "+"), ch("╮", "+"), ch("╰", "+"),
                            ch("╯", "+"), ch("─", "-"), ch("│", "|"))
    out = [dim(tl + h * (width + 2) + tr)]
    for r, plain_len in rows:
        out.append(dim(v) + " " + r + " " * max(0, width - plain_len) + " " + dim(v))
    out.append(dim(bl + h * (width + 2) + br))
    return out


def banner(version, url, engines, aitk, projects, python=None):
    """engines: [(name, port, running: bool|None)]; aitk: path or None."""
    lines = [""]
    for i, l in enumerate(_LOGO):
        lines.append("  " + c(_GRAD[i % len(_GRAD)], bold(l)))
    lines.append("  " + c(_GRAD[-1], bold("  dataset studio")) + "  " + dim(f"v{version}"))
    lines.append("")
    arrow = ch("▸", ">")
    on, off = ch("●", "*"), ch("○", "o")
    rows = []

    def row(label, value_col, value_plain):
        p = f"{arrow} {label.ljust(11)}{value_plain}"
        rows.append((c(75, arrow) + " " + bold(label.ljust(11)) + value_col, len(p)))

    row("open", c("4;38;5;75", url), url)
    if engines:
        first = True
        for name, port, running in engines:
            st = ("running" if running else "stopped") if running is not None else "?"
            dot = c(120, on) if running else dim(off)
            plain = f"{on} {name.ljust(12)} :{port}  {st}"
            colored = f"{dot} {name.ljust(12)} {dim(':' + str(port))}  " + \
                (c(120, st) if running else dim(st))
            if first:
                row("engines", colored, plain)
                first = False
            else:
                rows.append((" " * 13 + colored, 13 + len(plain)))
    else:
        row("engines", c(221, "none configured (Settings)"), "none configured (Settings)")
    if aitk:
        row("ai-toolkit", c(120, aitk), aitk)
    else:
        row("ai-toolkit", dim("not set (Settings)"), "not set (Settings)")
    row("projects", bold(str(projects)), str(projects))
    if python:
        row("python", dim(python), python)
    width = max(p for _, p in rows)
    lines += ["  " + l for l in _box(rows, width)]
    lines.append("  " + dim("Ctrl+C to stop (the engines Dataset Studio started are closed)"))
    lines.append("")
    return "\n".join(lines)
