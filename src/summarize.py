import json
import re
import urllib.error
import urllib.request

from . import config
from .data_loader import load_label_catalog

_COMPONENT = re.compile(r"\[([^\]]+)\]")
_STATUS = re.compile(r"\b([45]\d\d)\b")
_LATENCY = re.compile(r"(\d+)\s?ms", re.I)
_ENTITY = re.compile(r"\b(?:client|usr|user|txn|rec|org)_\w+", re.I)
_PROVIDER = re.compile(r"(?:provider|API|from)\s+([A-Z][\w\-]+)")

_CATALOG = None


def _catalog():
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = load_label_catalog()
    return _CATALOG


def _extract_entities(message: str) -> dict:
    ent = {}
    if m := _COMPONENT.search(message):
        ent["component"] = m.group(1)
    if m := _STATUS.search(message):
        ent["status_code"] = m.group(1)
    if m := _LATENCY.search(message):
        ent["latency_ms"] = int(m.group(1))
    if m := _PROVIDER.search(message):
        ent["provider"] = m.group(1)
    if found := _ENTITY.findall(message):
        ent["affected_entities"] = sorted(set(found))
    return ent


def summarize(message: str, predicted_label: str, confidence: float) -> dict:
    meta = _catalog().get(predicted_label, {})
    return {
        "root_cause_id": predicted_label,
        "root_cause": meta.get("label", "Unknown"),
        "severity": meta.get("severity", "Unknown"),
        "confidence": round(float(confidence), 4),
        "what_happened": meta.get("description", ""),
        "details": _extract_entities(message),
        "recommended_action": meta.get("typical_resolution", ""),
        "raw_log": message,
    }


def summarize_with_llm(message: str, predicted_label: str, confidence: float) -> dict:
    base = summarize(message, predicted_label, confidence)
    meta = _catalog().get(predicted_label, {})
    prompt = (
        "You are an SRE assistant. Write a 2-sentence incident summary for an on-call "
        "engineer. Be concrete and reference the log. Do not invent facts.\n\n"
        f"Log: {message}\n"
        f"Predicted root cause: {meta.get('label')} ({predicted_label})\n"
        f"Category definition: {meta.get('description')}\n"
        f"Typical resolution: {meta.get('typical_resolution')}\n\n"
        "Summary:"
    )
    payload = json.dumps(
        {"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False}
    ).encode()
    req = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = json.loads(resp.read())["response"].strip()
        base["llm_summary"] = text
    except (urllib.error.URLError, KeyError, TimeoutError) as e:
        base["llm_summary"] = None
        base["llm_error"] = f"Ollama unavailable ({type(e).__name__}); using structured summary only."
    return base
