import re
try:
    from rapidfuzz.fuzz import ratio as rf_ratio
    def similarity(a, b):
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return rf_ratio(a, b)/100.0
except Exception:
    from difflib import SequenceMatcher
    def similarity(a,b):
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

def normalize_text(s: str):
    if s is None:
        return ""
    s = s.replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()
