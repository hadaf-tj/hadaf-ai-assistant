"""Language detection and localization helpers for Tajik, Russian, and English."""

import re
from typing import Literal

SupportedLanguage = Literal["ru", "tg", "en"]

# Distinctive Tajik characters
TAJIK_SPECIFIC_CHARS = set("ҷҳқӯӣғҶҲҚӮӢҒ")

# Common Tajik vocabulary indicators
TAJIK_KEYWORDS = {
    "аст", "ҳаст", "нест", "барои", "хайрия", "кумак", "ёрӣ", "шумо", "мо", "ин",
    "он", "дар", "бо", "аз", "ба", "салом", "маблағ", "ҷамъоварӣ", "фонд", "ниёз",
    "муассиса", "чорабинӣ", "савол", "чӣ", "чанд", "кай", "куҷо", "ташаккур",
}

# Common Russian vocabulary indicators
RUSSIAN_KEYWORDS = {
    "что", "как", "где", "когда", "почему", "зачем", "кто", "сколько", "это",
    "для", "сбор", "фонд", "помощь", "пожертвование", "комиссия", "процент", "учреждение",
    "мероприятие", "вопрос", "здравствуйте", "привет", "спасибо", "какие", "какая",
    "какой", "каком", "какую", "устав", "устава", "уставе", "платформа", "платформы", "платформе",
}

# Common English vocabulary indicators
ENGLISH_KEYWORDS = {
    "what", "how", "where", "when", "why", "who", "which", "is", "are", "the",
    "for", "charity", "foundation", "donation", "donations", "campaign", "commission", "fee",
    "institution", "event", "faq", "hello", "hi", "thank", "thanks", "help",
    "problem", "charter", "solve", "trying", "according", "platform", "policy",
}


def detect_language(text: str, default: SupportedLanguage = "ru") -> SupportedLanguage:
    """Detect whether input text is Tajik ('tg'), Russian ('ru'), or English ('en')."""
    if not text or not text.strip():
        return default

    cleaned = text.strip()

    cyrillic_chars = sum(1 for ch in cleaned if "\u0400" <= ch <= "\u04FF")
    latin_chars = sum(1 for ch in cleaned if "a" <= ch.lower() <= "z")

    words = [w.lower() for w in re.findall(r"\b[\w']+\b", cleaned)]
    if not words:
        return default

    # If no alphabet letters found, return default
    if latin_chars == 0 and cyrillic_chars == 0:
        return default

    # 1. Predominantly Latin text -> English
    if latin_chars > cyrillic_chars:
        return "en"

    # 2. Predominantly Cyrillic text -> Tajik vs Russian
    has_tajik_chars = any(ch in TAJIK_SPECIFIC_CHARS for ch in cleaned)
    tg_score = sum(1 for w in words if w in TAJIK_KEYWORDS)
    ru_score = sum(1 for w in words if w in RUSSIAN_KEYWORDS)

    if has_tajik_chars and tg_score > 0:
        return "tg"

    if tg_score > ru_score:
        return "tg"

    if ru_score > 0:
        return "ru"

    if has_tajik_chars:
        return "tg"

    return default
