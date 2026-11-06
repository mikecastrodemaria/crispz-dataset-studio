"""crispz-dataset-studio - send an image to Dataset Studio as a new project's reference.

    python tools/send_to_studio.py <image> [--project NAME] [--port N] [--no-browser]

Made for IrfanView's external editor (Shift+E, 2026-11-06): browse photos there, press
the key on a face, and Dataset Studio opens on a new project with that photo as its
reference. IrfanView has no public plugin SDK, but it launches an external program on
the current file (%1): send_to_studio.bat is that program. Works from any other viewer
or from Explorer's "Send to" the same way.

What it does, through Dataset Studio's local API only (127.0.0.1, nothing leaves the PC):
  1. Dataset Studio answering? If not, it is started (ds_server.py --no-browser, its
     log in logs/send_to_studio.log) and awaited up to 30 s.
  2. project_new: --project, else the photo's file name; an existing name gets -2, -3.
  3. set_reference with the photo's path (read by the server, no upload).
  4. The browser opens http://127.0.0.1:<port>/?project=<name>.
Exit code 0 on success, 1 on any failure with the reason printed.
"""

import os
import re
import sys
import json
import time
import argparse
import subprocess
import webbrowser
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
DEFAULT_PORT = 8780


def _port():
    for name in ("config.json", "config-sample.json"):
        try:
            with open(os.path.join(APP, name), encoding="utf-8") as fh:
                return int(json.load(fh).get("port") or DEFAULT_PORT)
        except (OSError, ValueError, TypeError):
            continue
    return DEFAULT_PORT


def _opener():
    # a local service: the system proxy is bypassed (the family rule)
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def ping(port, timeout=1.5):
    try:
        with _opener().open(f"http://127.0.0.1:{port}/api/ping", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")).get("tool") == "crispz-dataset-studio"
    except Exception:
        return False


def api(port, op, data, timeout=120):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/{op}",
                                 data=json.dumps(data).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with _opener().open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def start_server(port, wait=30.0):
    os.makedirs(os.path.join(APP, "logs"), exist_ok=True)
    logf = open(os.path.join(APP, "logs", "send_to_studio.log"), "ab")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                               | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        kw["start_new_session"] = True
    subprocess.Popen([sys.executable, os.path.join(APP, "ds_server.py"), "--no-browser",
                      "--port", str(port)], cwd=APP, stdout=logf, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, **kw)
    t0 = time.time()
    while time.time() - t0 < wait:
        if ping(port):
            return True
        time.sleep(0.5)
    return False


def project_name(image, wanted=None):
    base = wanted or os.path.splitext(os.path.basename(image))[0]
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", base.strip()).strip("-.")
    return (s or "photo")[:56]


def send(image, project=None, port=None, browser=True, out=print):
    port = port or _port()
    image = os.path.abspath(image)
    if not os.path.isfile(image):
        raise FileNotFoundError(f"no such file: {image}")
    if not ping(port):
        out(f"Dataset Studio is not running on port {port}: starting it...")
        if not start_server(port):
            raise RuntimeError(f"Dataset Studio did not answer on port {port} after 30 s "
                               f"(see logs/send_to_studio.log)")
    base = project_name(image, project)
    name, n = base, 2
    while True:
        r = api(port, "project_new", {"name": name})
        if r.get("ok"):
            name = r["project"]["name"]
            break
        if "already exists" not in (r.get("error") or "") or n > 99:
            raise RuntimeError(f"project_new: {r.get('error')}")
        name, n = f"{base}-{n}", n + 1
    r = api(port, "set_reference", {"name": name, "path": image}, timeout=300)
    if not r.get("ok"):
        raise RuntimeError(f"set_reference: {r.get('error')}")
    out(f"project '{name}' created with {image} as its reference"
        + (f" ({r['warning']})" if r.get("warning") else ""))
    url = f"http://127.0.0.1:{port}/?project={urllib.request.quote(name)}"
    if browser:
        webbrowser.open(url)
    return {"project": name, "url": url, "warning": r.get("warning")}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Send an image to crispz Dataset Studio as a "
                                             "new project's reference photo.")
    ap.add_argument("image")
    ap.add_argument("--project", help="project name (default: the photo's file name)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    try:
        send(a.image, a.project, a.port, browser=not a.no_browser)
        return 0
    except Exception as e:
        print(f"send to Dataset Studio failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
