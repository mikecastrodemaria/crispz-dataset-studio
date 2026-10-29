"""crispz-dataset-studio - the likeness score.

The engine's cli_faces endpoint returns, for every face, the ArcFace embedding that
insightface (buffalo_l) computes for the Face Swap. The likeness of a generated card
is the cosine similarity between the reference's main face and the closest face of
the card, shown on a 0-100 scale.

Scale (2026-10-29, to calibrate on real renders): score = cosine x 100, clamped to
0-100. With buffalo_l, two photos of the same person usually sit above 0.5, two
different people below 0.3; a generated image of the same identity lands in between.
The default threshold (35) is therefore a floor that catches a clear drift, not a
guarantee. The run page shows the raw score so the user picks their own threshold
after a first batch. No face on the card = score 0 with a reason (never a silent 0).

Pure Python (no numpy): the vectors have 512 floats, a dot product is nothing.
"""

import math


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else None


def cosine(a, b):
    a, b = _norm(a), _norm(b)
    if a is None or b is None or len(a) != len(b):
        return None
    return sum(x * y for x, y in zip(a, b))


def main_face(faces):
    """The largest face of a cli_faces answer (the reference's subject), or None."""
    best, area = None, -1
    for f in faces or []:
        box = f.get("box") or []
        if len(box) != 4 or not f.get("embedding"):
            continue
        a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        if a > area:
            best, area = f, a
    return best


def score(ref_embedding, faces):
    """(score 0-100, detail dict). detail.reason explains a 0."""
    if not ref_embedding:
        return 0, {"reason": "no face embedding on the reference"}
    best, best_cos = None, None
    for f in faces or []:
        emb = f.get("embedding")
        if not emb:
            continue
        c = cosine(ref_embedding, emb)
        if c is not None and (best_cos is None or c > best_cos):
            best, best_cos = f, c
    if best is None:
        return 0, {"reason": "no face detected on the card", "faces": 0}
    s = int(round(max(0.0, min(1.0, best_cos)) * 100))
    return s, {"cosine": round(best_cos, 4), "faces": len(faces or []),
               "box": best.get("box")}
