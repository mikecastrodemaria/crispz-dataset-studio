"""Visual checks next to the likeness score.

ArcFace alone is not enough (qwen-edit lots of 2026-10-30): a try that is the
reference re-rendered without the asked edit scored 79, and a ghosted, doubled
face scored 54. Two checks flag a try; a flagged try is re-rolled like a low score
and the card says so when the kept try is still flagged.

  copy check (always cheap, no model): the reference and the try are reduced to a
  32x32 grayscale and their Pearson correlation is taken. Real edits of the same
  face measured 0.69 at most (a close-up of the same pose), copies 0.95 and 0.98,
  so 0.9 separates them with margin. A flat image (zero variance) gives 0.

  artifact check (Ollama vision, off by default): the caption model is asked one
  closed question (ghosting, double exposure, doubled or smeared face, tiling);
  an answer starting with "yes" flags the try. Costs one vision call per try.
"""
import os
from PIL import Image, ImageOps

import ds_captions

COPY_THRESHOLD = 0.9
_SIDE = 32
ARTIFACT_PROMPT = (
    "Look only for rendering defects: a ghosted or double-exposed face, a doubled or "
    "smeared body part, tiled or repeated patches, a face melting into the "
    "background. Answer with YES or NO first, then at most ten words. If the image is "
    "clean, answer NO.")


def _vector(path):
    im = ImageOps.fit(Image.open(path).convert("L"), (_SIDE, _SIDE), Image.LANCZOS)
    v = [float(x) for x in im.tobytes()]      # L mode: one byte per pixel
    m = sum(v) / len(v)
    v = [x - m for x in v]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n > 1e-6 else None


def copy_similarity(ref_path, try_path):
    """Pearson correlation of the two 32x32 grayscale reductions, 0 when one image
    is flat (the tests' plain colours), -1 to 1 otherwise."""
    a, b = _vector(ref_path), _vector(try_path)
    if a is None or b is None:
        return 0.0
    return round(sum(x * y for x, y in zip(a, b)), 3)


def artifact_answer(image_path, ollama_cfg):
    """{"ok", "flag", "answer", "model"} or {"ok": False, "error"}."""
    r = ds_captions.describe(image_path, ollama_cfg, prompt=ARTIFACT_PROMPT)
    if not r.get("ok"):
        return r
    ans = r["caption"].strip()
    return {"ok": True, "flag": ans.lower().startswith("yes"), "answer": ans,
            "model": r.get("model")}


def check(ref_path, try_path, settings, ollama_cfg):
    """(flags: [str], detail: dict) for one try. detail is merged into the try's
    detail (copy_sim, artifact answer or the reason the check could not run)."""
    flags, detail = [], {}
    s = settings or {}
    if s.get("copy_check", True) and ref_path and os.path.isfile(ref_path):
        sim = copy_similarity(ref_path, try_path)
        detail["copy_sim"] = sim
        if sim >= COPY_THRESHOLD:
            flags.append(f"looks like the reference copied back (similarity {sim:.2f})")
    if s.get("artifact_check"):
        r = artifact_answer(try_path, ollama_cfg)
        if r.get("ok"):
            detail["artifact"] = r["answer"]
            if r["flag"]:
                flags.append(f"artifacts seen by {r.get('model')}: {r['answer']}")
        else:
            detail["artifact"] = f"not checked: {r.get('error')}"
    return flags, detail
