import re
from rapidfuzz import fuzz


def normalize_text(s: str) -> str:
    """
    Нормализует текст: убирает переносы, лишние пробелы,
    приводит к нижнему регистру.
    """
    s = s.replace("\r", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def similarity(a: str, b: str) -> float:
    """
    Сравнение двух строк с помощью RapidFuzz (быстрее и точнее, чем difflib).
    Возвращает число от 0.0 до 1.0
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0
