"""
Runs after content is captured and before chunking
(Readability extraction, raw browser selection, later — OCR output)

source_type drives the only source-specific behaviour (OCR homoglyph repair)
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from lingua import Language, LanguageDetectorBuilder

SourceType = Literal["readability", "selection", "ocr"]

# min length below which the result is treated as a likely extraction failure
MIN_VALID_LENGTH = 20

# languages the detector considers
# restricting the set measurably improves accuracy (fewer competing candidates)
# and keeps lingua's memory footprint modest
SUPPORTED_LANGUAGES = [
    Language.UKRAINIAN,
    Language.ENGLISH,
    Language.POLISH,
    Language.GERMAN,
    Language.FRENCH,
    Language.SPANISH,
]

_DETECTOR = LanguageDetectorBuilder.from_languages(*SUPPORTED_LANGUAGES).build()

# min confidence before we report a language at all
_MIN_CONFIDENCE = 0.60

# zero-width / invisible characters
# leak in from copy-pasted or DOM-derived text.
_ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\ufeff\u2060"

# ukrainian text uses several visually-distinct apostrophe characters (об'єкт / обʼєкт / об’єкт)
_APOSTROPHE_VARIANTS = "\u2019\u02bc\u0027"
_CANONICAL_APOSTROPHE = "\u2019"

# Cyrillic <-> Latin lookalikes, ONLY for OCR-sourced text
_LATIN_TO_CYRILLIC = {
    "a": "а",
    "c": "с",
    "e": "е",
    "i": "і",
    "o": "о",
    "p": "р",
    "x": "х",
    "y": "у",
    "s": "ѕ",
    "j": "ј",
    "A": "А",
    "B": "В",
    "C": "С",
    "E": "Е",
    "H": "Н",
    "I": "І",
    "K": "К",
    "M": "М",
    "O": "О",
    "P": "Р",
    "T": "Т",
    "X": "Х",
    "Y": "У",
}
_CYRILLIC_TO_LATIN = {v: k for k, v in _LATIN_TO_CYRILLIC.items()}

_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass
class NormalizedText:
    text: str
    original_char_count: int
    normalized_char_count: int
    source_type: SourceType
    is_likely_valid: bool  # false if result looks like an extraction failure
    detected_language: str | None  # ("uk", "en", ...) or None if unknown
    language_confidence: float | None


def normalize_text(raw_text: str, source_type: SourceType = "readability") -> NormalizedText:
    """
    Clean raw captured text into a consistent form ready for chunking

    Args:
        raw_text: text from Readability extraction (raw browser selection or OCR)
        source_type: where this text came from. Only "ocr" changes behaviour (homoglyph)
    Returns:
        NormalizedText with detected language and basic diagnostics.
    """
    original_char_count = len(raw_text)
    text = raw_text

    # residual HTML entities ("selection" path, which bypasses Readability's own handling).
    text = _decode_html_entities(text)

    # unicode NFKC (applies uniformly to Latin, Cyrillic, and any other script)
    text = unicodedata.normalize("NFKC", text)

    # invisible characters.
    text = _strip_zero_width(text)

    # apostrophe variants
    text = _normalize_apostrophes(text)

    # line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # non-breaking spaces
    text = text.replace("\u00a0", " ")

    # OCR only: repair words that mix Cyrillic and Latin characters
    if source_type == "ocr":
        text = _fold_mixed_script_words(text)

    # whitespace, preserving paragraph boundaries for the chunker
    text = _collapse_whitespace(text)

    normalized_char_count = len(text)
    is_likely_valid = normalized_char_count >= MIN_VALID_LENGTH

    if is_likely_valid:
        detected_language, language_confidence = _detect_language(text)
    else:
        detected_language, language_confidence = None, None

    return NormalizedText(
        text=text,
        original_char_count=original_char_count,
        normalized_char_count=normalized_char_count,
        source_type=source_type,
        is_likely_valid=is_likely_valid,
        detected_language=detected_language,
        language_confidence=language_confidence,
    )


def _decode_html_entities(text: str) -> str:
    import html

    return html.unescape(text)


def _strip_zero_width(text: str) -> str:
    return text.translate({ord(c): None for c in _ZERO_WIDTH_CHARS})


def _normalize_apostrophes(text: str) -> str:
    return text.translate({ord(c): _CANONICAL_APOSTROPHE for c in _APOSTROPHE_VARIANTS})


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)  # runs of spaces/tabs -> one space
    text = re.sub(r"[ \t]+\n", "\n", text)  # trailing whitespace per line
    text = re.sub(r"\n{3,}", "\n\n", text)  # 3+ newlines -> paragraph break
    return text.strip()


def _fold_mixed_script_words(text: str) -> str:
    """
    Repair OCR homoglyph contamination.

    only words containing BOTH Cyrillic and Latin characters are
    touched
    within a mixed word, the minority script is folded into the
    majority script
    """

    def fold(match: re.Match[str]) -> str:
        word = match.group(0)
        cyrillic_count = len(_CYRILLIC_RE.findall(word))
        latin_count = len(_LATIN_RE.findall(word))

        if cyrillic_count == 0 or latin_count == 0:
            return word  # single-script word: leave alone

        if cyrillic_count >= latin_count:
            return "".join(_LATIN_TO_CYRILLIC.get(ch, ch) for ch in word)
        return "".join(_CYRILLIC_TO_LATIN.get(ch, ch) for ch in word)

    return re.sub(r"\w+", fold, text)


def _detect_language(text: str) -> tuple[str | None, float | None]:
    """
    SUPPORTED_LANGUAGES list for wide detection

    Returns:
    (iso_639_1_code, confidence) -- evrything alright
    (None, None) -- no candidate clears _MIN_CONFIDENCE
    """
    try:
        confidences = _DETECTOR.compute_language_confidence_values(text)
    except Exception:
        return None, None

    if not confidences:
        return None, None

    best = confidences[0]
    if best.value < _MIN_CONFIDENCE:
        return None, None

    return best.language.iso_code_639_1.name.lower(), round(best.value, 4)
