#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
[ ! -x .venv/bin/python ] && [ -x ../crispz-klein/.venv/bin/python ] && PY=../crispz-klein/.venv/bin/python
$PY -c "import PIL" 2>/dev/null || { echo "Pillow missing for $PY: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
exec $PY ds_server.py "$@"
