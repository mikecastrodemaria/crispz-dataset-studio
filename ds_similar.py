"""crispz-dataset-studio - cards too close to each other, before they are trained on.

A LoRA learns badly from a dataset that repeats itself: two cards with the same
framing, pose and background weigh double and teach the model that composition
instead of the identity. Idea taken from DTInspector's duplicate finder
(qxm.de DT Helper, 2026-11-06), cut down to what a dataset needs:

  - identical: the same file content (SHA-256), a copy kept twice;
  - similar:   a perceptual difference hash (dHash, 64 bits: the image in grey,
               9x8 pixels, each bit = "brighter than the right neighbour") whose
               Hamming distance is at most `max_distance`. Same composition and
               light give close hashes even when colours or small details change.

Nothing is removed: the pairs are reported (the grid marks them, the export, the
AI-Toolkit job and the autopilot say so), the user rejects what should go.
"""

import os
import hashlib

DEFAULT_MAX_DISTANCE = 6        # of 64 bits; 0 = only near-identical images


def dhash(path, size=8):
    from PIL import Image
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = g.tobytes()                 # one byte per pixel (mode L), any Pillow
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def distance(a, b):
    return bin(a ^ b).count("1")


def find_similar(items, max_distance=DEFAULT_MAX_DISTANCE):
    """items = [(key, path)] -> [{a, b, distance, identical}], closest first.
    A file that cannot be read is skipped."""
    sig = []
    for key, path in items:
        try:
            sig.append((key, dhash(path), sha256(path)))
        except Exception:
            continue
    pairs = []
    for i in range(len(sig)):
        for j in range(i + 1, len(sig)):
            ka, ha, sa = sig[i]
            kb, hb, sb = sig[j]
            same = sa == sb
            d = 0 if same else distance(ha, hb)
            if same or d <= max_distance:
                pairs.append({"a": ka, "b": kb, "distance": d, "identical": same})
    pairs.sort(key=lambda p: (not p["identical"], p["distance"], str(p["a"]), str(p["b"])))
    return pairs


def project_pairs(project, project_dir, max_distance=None):
    """The kept cards of a project that are too close to each other."""
    s = project.get("settings") or {}
    md = DEFAULT_MAX_DISTANCE if max_distance is None else int(max_distance)
    if max_distance is None and s.get("similar_max") not in (None, ""):
        md = int(s["similar_max"])
    items = [(c["index"], os.path.join(project_dir, c["file"]))
             for c in project.get("cards") or []
             if c.get("status") == "done" and c.get("file")]
    return find_similar(items, md)


def summary(pairs, limit=6):
    """One line for a warning: '3 pair(s) of cards too close: 004~007 (2), ...'."""
    if not pairs:
        return ""
    shown = ", ".join(f"{p['a']:03d}{'=' if p['identical'] else '~'}{p['b']:03d}"
                      + ("" if p["identical"] else f" ({p['distance']})")
                      for p in pairs[:limit])
    more = f" and {len(pairs) - limit} more" if len(pairs) > limit else ""
    return (f"{len(pairs)} pair(s) of cards too close to each other: {shown}{more}; "
            f"reject one of each pair or re-roll it, a LoRA learns the repeated "
            f"composition instead of the identity")
