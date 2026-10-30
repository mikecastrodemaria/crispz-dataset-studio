"""crispz-dataset-studio - captions for the training pairs (NNN.txt).

Three sources, chosen per project ("caption_source"):
  "prompt"  : the prompt line itself, trigger word first. No model, always available.
              This is what the enzino 25-shot packs ship: the caption IS the prompt.
  "ollama"  : a local vision model (Ollama /api/generate with the image), the same
              service the family's Describe uses. keep_alive 0 by default: the GPU is
              shared with the drawing engine, the LLM must not stay resident.
  "both"    : the prompt line, then the Ollama description after a comma.

Trigger word: prepended to every caption when set ("ohwx woman, ..."), the kohya /
ai-toolkit convention. Never raises: {ok, caption} or {ok: false, error, caption}
with the prompt fallback, so a dead Ollama never blocks a run (announced in the log).
"""

import json
import base64
import urllib.request

DEFAULT_URL = "http://127.0.0.1:11434"
DESCRIBE = ("Describe this image for an image-generation training caption: one line, "
            "comma-separated phrases, concrete and visual (framing, pose, expression, "
            "clothing, setting, lighting). No names, no opinions, no preamble.")


def _strip_thinking(text):
    """Some models leak <think>...</think> even with think=false."""
    import re
    return re.sub(r"<think>[\s\S]*?</think>", "", text or "").strip()


def _ollama(url, payload, timeout):
    req = urllib.request.Request(f"{url.rstrip('/')}/api/generate",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    # A local service: system proxies are bypassed on purpose (the family rule).
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_json(url, path, payload=None, timeout=5):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    full = f"{url.rstrip('/')}{path}"
    if payload is None:
        with opener.open(full, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    req = urllib.request.Request(full, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ollama_models(url=DEFAULT_URL, timeout=4):
    """Every installed model (names)."""
    try:
        j = _get_json(url, "/api/tags", timeout=timeout)
        return [m.get("name") for m in j.get("models", []) if m.get("name")]
    except Exception:
        return []


# Names that are multimodal for sure, used when /api/show gives no capabilities (an
# old Ollama) and to put the reliable choices first. Same rule as the family's
# cz_ollama._ollama_vision_models: we trust Ollama's 'vision' capability, never the
# 'families' field (clip in a text model's family is a false positive).
_VISION_NAME = ("llava", "-vl", "vl:", "moondream", "minicpm-v", "bakllava",
                "llama3.2-vision", "llama-3.2-vision")


def ollama_vision_models(url=DEFAULT_URL, timeout=4, show=None):
    """The models that can look at an image, reliable ones first. `show` lets the
    tests inject /api/show answers ({name: info})."""
    names = ollama_models(url, timeout)
    vision = []
    for n in names:
        try:
            info = show[n] if show is not None else _get_json(url, "/api/show", {"model": n},
                                                                timeout=8)
            caps = [c.lower() for c in (info.get("capabilities") or [])]
            if "vision" in caps:
                vision.append(n)
            elif not info.get("capabilities") and any(k in n.lower() for k in _VISION_NAME):
                vision.append(n)
        except Exception:
            if any(k in n.lower() for k in _VISION_NAME):
                vision.append(n)
    vision.sort(key=lambda n: 0 if any(k in n.lower() for k in _VISION_NAME) else 1)
    return vision


def describe(image_path, cfg, timeout=180, prompt=None):
    """Ollama vision description of one image. cfg = {"url", "model", "keep_alive"}.
    prompt: another question for the same model (ds_quality's artifact check)."""
    url = (cfg or {}).get("url") or DEFAULT_URL
    model = (cfg or {}).get("model") or ""
    if not model:
        models = ollama_vision_models(url)
        if not models:
            return {"ok": False, "error": "Ollama not reachable or no vision model installed"}
        model = models[0]
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
    except OSError as e:
        return {"ok": False, "error": f"cannot read {image_path}: {e}"}
    payload = {"model": model, "prompt": prompt or DESCRIBE, "images": [b64], "stream": False,
               "think": False, "keep_alive": (cfg or {}).get("keep_alive", 0)}
    try:
        j = _ollama(url, payload, timeout)
    except urllib.error.HTTPError as e:
        if e.code == 400:
            payload.pop("think", None)
            try:
                j = _ollama(url, payload, timeout)
            except Exception as e2:
                return {"ok": False, "error": f"Ollama: {e2}"}
        else:
            return {"ok": False, "error": f"Ollama: HTTP {e.code}"}
    except Exception as e:
        return {"ok": False, "error": f"Ollama: {type(e).__name__}: {e}"}
    text = _strip_thinking(j.get("response", ""))
    text = " ".join(text.split())
    if not text:
        return {"ok": False, "error": f"Ollama ({model}): empty answer"}
    return {"ok": True, "caption": text, "model": model}


def compose(prompt, trigger="", source="prompt", described=None):
    """The final caption line from its parts."""
    parts = []
    t = (trigger or "").strip()
    if t:
        parts.append(t)
    p = " ".join((prompt or "").split())
    if source in ("prompt", "both") and p:
        parts.append(p)
    if source in ("ollama", "both") and described:
        parts.append(described.strip())
    if source == "ollama" and not described and p:
        parts.append(p)          # fallback, the caller logs why
    return ", ".join(parts)


def caption_for(image_path, prompt, project_cfg, ollama_cfg):
    """{caption, source_used, warning?} for one card."""
    source = (project_cfg or {}).get("caption_source") or "prompt"
    trigger = (project_cfg or {}).get("trigger_word") or ""
    described, warning = None, None
    if source in ("ollama", "both"):
        r = describe(image_path, ollama_cfg)
        if r.get("ok"):
            described = r["caption"]
        else:
            warning = f"caption fell back to the prompt line: {r.get('error')}"
    return {"caption": compose(prompt, trigger, source, described),
            "source_used": "ollama" if described else "prompt",
            "warning": warning}
