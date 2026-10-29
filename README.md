# crispz-dataset-studio

> One reference photo in, a captioned, likeness-checked character LoRA dataset out.
> A standalone local app that drives the crispz family engines (crispz-klein,
> crispz-qwen-edit, crispz-flux3) through their CLI protocol v1, and starts them itself.
> Version 0.1.0 (2026-10-29), local work in progress.

## What it does

1. Drop a reference photo (only its main face matters).
2. Pick a prompt set (25 shots: framing, angles, expressions, poses, places, light), or
   edit the lines.
3. Run. For every line the engine edits the reference with that instruction, the
   engine's face service (insightface, the same as its Face Swap) gives the ArcFace
   embedding of the result, and the likeness against the reference is scored 0-100.
   Below the threshold the card is re-rolled with a new seed (up to N times); the best
   try is kept and the card says when it stayed below.
4. Captions: the prompt line (trigger word first), a local Ollama vision model, or both.
5. Review the grid: reject, restore, re-roll a card, pick another try, edit a caption.
6. Export a ZIP of `001.png` + `001.txt` pairs (kohya, ai-toolkit, OneTrainer), long
   side 1024 / 1536 / 2048, PNG or JPG.

Everything is on disk under `projects/<name>/` (`project.json`, `reference.png`,
`cards/NNN_tK.png`): a run resumes where it stopped and re-renders only the pending,
rejected or failed cards.

## The engine is started for you

Settings lists the family engines with their install folder and URL. When nothing
answers at the URL, Dataset Studio starts `<folder>/app.py` with the folder's `.venv`
(then Pinokio's `env`, then `py -3`), passes the URL's port through
`GRADIO_SERVER_PORT`, logs to `logs/<engine>.log`, and waits for the `cli_caps`
endpoint. No change is needed in the engines: they already read that variable and do
not open a browser. Give each engine its own port (7861, 7862, 7863 in
`config-sample.json`). One engine at a time: two family apps do not fit on one GPU,
so a second launch is refused until the first is stopped from Settings. An engine you
started by hand is used as is and never killed.

The model loads at the first image, so the first card of a session is slow.

## Install and run

Server: Python stdlib. Dependency: Pillow. `run.bat` reuses a family venv when it
finds one next door (`../crispz-klein/.venv`), else `install.bat` makes a `.venv`.

```bash
run.bat                 # http://127.0.0.1:8780
python ds_server.py --port 8780 --no-browser
```

Tests (CPU, a fake engine, no model): `python -m pytest tests -q` (21 tests).

## Likeness scale

score = ArcFace cosine x 100, clamped. Same person photos usually sit above 50,
different people below 30; a generated image of the same identity lands in between.
The default threshold 35 is a floor for clear drift, to calibrate after a first batch
on the real engine (the raw cosine is kept per try in `project.json`).

## Status (2026-10-29)

- Done on CPU: engine client and launcher, project state, run loop with re-rolls and
  resume, captions (prompt / Ollama / both), export, SPA, 21 tests.
- Not yet measured on the GPU: a real run on crispz-klein (identity fidelity of the
  edits, timing per card, the threshold), then crispz-qwen-edit.
- Later: a `dataset` op in the family protocol so the same run can be driven without
  the UI; a bounding-box / outfit lock; per-card prompt overrides.
