from app.services.ingestion.normalization import normalize_text

# --- whitespace ---------------------------------------------------------

def test_collapses_multiple_spaces():
    assert normalize_text("This   has     extra   spaces.").text == "This has extra spaces."


def test_preserves_paragraph_breaks():
    result = normalize_text("First paragraph.\n\nSecond paragraph.")
    assert result.text == "First paragraph.\n\nSecond paragraph."


def test_collapses_excessive_newlines():
    result = normalize_text("First paragraph.\n\n\n\n\nSecond paragraph.")
    assert result.text == "First paragraph.\n\nSecond paragraph."


def test_normalizes_windows_line_endings():
    result = normalize_text("Line one.\r\nLine two.\r\n\r\nLine three.")
    assert "\r" not in result.text
    assert result.text == "Line one.\nLine two.\n\nLine three."


def test_strips_non_breaking_spaces():
    assert normalize_text("Text\u00a0with\u00a0nbsp").text == "Text with nbsp"


# --- unicode / entities -------------------------------------------------

def test_decodes_html_entities():
    raw = "Tom &amp; Jerry said &quot;hello&quot;"
    assert normalize_text(raw).text == 'Tom & Jerry said "hello"'


def test_strips_zero_width_characters():
    assert normalize_text("Hidden\u200bzero\u200cwidth\ufeffchars").text == "Hiddenzerowidthchars"


def test_unicode_nfkc_normalization():
    assert normalize_text("Ｈｅｌｌｏ").text == "Hello"


def test_normalizes_apostrophe_variants_in_ukrainian_text():
    straight = normalize_text("об'єкт").text
    modifier = normalize_text("об\u02bcєкт").text
    curly = normalize_text("об\u2019єкт").text
    assert straight == modifier == curly


# --- validity guardrails ------------------------------------------------

def test_flags_very_short_text_as_invalid():
    assert normalize_text("Hi.").is_likely_valid is False


def test_flags_normal_length_text_as_valid():
    result = normalize_text("This is a reasonably long sentence with real content in it.")
    assert result.is_likely_valid is True


def test_handles_empty_string():
    result = normalize_text("")
    assert result.text == ""
    assert result.is_likely_valid is False
    assert result.original_char_count == 0
    assert result.detected_language is None


def test_tracks_char_counts():
    raw = "Text   with   extra   spaces"
    result = normalize_text(raw)
    assert result.original_char_count == len(raw)
    assert result.normalized_char_count == len(result.text)
    assert result.normalized_char_count < result.original_char_count


# --- source type --------------------------------------------------------

def test_source_type_is_recorded():
    assert normalize_text("Some text.", source_type="selection").source_type == "selection"
    assert normalize_text("Some text.").source_type == "readability"


# --- language detection -------------------------------------------------

def test_detects_ukrainian():
    raw = (
        "Сьогодні в Києві відбулася важлива зустріч представників "
        "різних галузей економіки для обговорення подальшого розвитку."
    )
    result = normalize_text(raw)
    assert result.detected_language == "uk"
    assert result.language_confidence is not None


def test_detects_english():
    raw = (
        "Today in the city center, representatives from various "
        "industries met to discuss further economic development plans."
    )
    assert normalize_text(raw).detected_language == "en"


def test_detects_short_ukrainian_selection():
    # lingua's main advantage over langdetect: short snippets, which is
    # exactly what the Selection API capture path produces.
    result = normalize_text("Зустріч відбулася вчора ввечері.", source_type="selection")
    assert result.detected_language == "uk"


def test_no_language_for_too_short_text():
    result = normalize_text("Hi.")
    assert result.detected_language is None
    assert result.language_confidence is None


# --- OCR homoglyph folding ---------------------------------------------

def test_ocr_folds_mixed_script_word():
    # "Київ" with a Latin 'y' and 'i' — classic OCR contamination.
    raw = "Зустріч відбулася у місті Ки\u0457в минулого тижня у центрі."
    contaminated = raw.replace("Ки", "K\u0438")  # Latin K into a Cyrillic word
    result = normalize_text(contaminated, source_type="ocr")
    assert "K" not in result.text  # Latin K folded to Cyrillic К


def test_ocr_leaves_pure_latin_words_alone():
    # a legitimate brand name inside Ukrainian text must survive.
    raw = "Користувач купив новий iPhone у магазині на минулому тижні."
    result = normalize_text(raw, source_type="ocr")
    assert "iPhone" in result.text


def test_non_ocr_sources_skip_homoglyph_folding():
    contaminated = "Ки\u0457в"  # would be folded under ocr
    result = normalize_text(
        "Зустріч відбулася у місті " + contaminated + " минулого тижня у центрі міста.",
        source_type="readability",
    )
    # readability path must not mutate characters at all
    assert contaminated in result.text