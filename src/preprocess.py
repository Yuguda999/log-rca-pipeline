import re

_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_ID = re.compile(r"\b(?:client|usr|user|txn|rec|org|req|trace)_\w+", re.I)
_KEY = re.compile(r"\b\w*\.\.\.")
_MEAS = re.compile(r"\b\d+(?:\.\d+)?\s?(ms|mb|gb|kb|rps|rpm|%)\b", re.I)
_BIGNUM = re.compile(r"\b\d{4,}\b")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    t = str(text)
    t = _IP.sub(" IP ", t)
    t = _ID.sub(" ENTITY_ID ", t)
    t = _KEY.sub(" ENTITY_ID ", t)
    t = _MEAS.sub(lambda m: f" NUM{m.group(1).lower()} ", t)
    t = _BIGNUM.sub(" NUM ", t)
    t = t.lower()
    t = _WS.sub(" ", t).strip()
    return t
