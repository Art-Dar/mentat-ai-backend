import io
import random
import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.services.ingestion.extractors.ocr import (
    LOW_CONFIDENCE,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MIN_CHARS,
    ImageTooLargeError,
    OcrEngineUnavailable,
    OcrExtraction,
    OcrWord,
    UnreadableImageError,
    extract_image_text,
    extract_image_text_async,
)

needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="the tesseract binary is not installed on this host",
)

RECEIPT = ["ATB Market", "Chek 10482", "Razom: 349.50 UAH", "Data: 16.10.2026"]


def _font(size: int = 34) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    # The bitmap fallback is tiny and OCRs poorly, so a host without DejaVu
    # would fail the recognition tests for reasons that have nothing to do
    # with this module.
    pytest.skip("no scalable font available to render a test scan")


def scan(
    lines: list[str],
    *,
    size: tuple[int, int] = (1000, 500),
    noise: int = 0,
    fmt: str = "PNG",
    exif_orientation: int | None = None,
) -> bytes:
    """A rendered 'photo' of a receipt: real pixels, no text layer."""
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black", font=font)
        y += 52

    if noise:
        random.seed(0)  # a flaky OCR test is worse than no OCR test
        pixels = image.load()
        for _ in range(noise):
            x, yy = random.randrange(size[0]), random.randrange(size[1])
            shade = random.randrange(256)
            pixels[x, yy] = (shade, shade, shade)

    buf = io.BytesIO()
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[274] = exif_orientation
        # A portrait phone shot: pixels stored rotated 90 CCW from the
        # intended view, with the tag telling the viewer to turn them back.
        image.rotate(90, expand=True).save(buf, fmt, exif=exif)
    else:
        image.save(buf, fmt)
    return buf.getvalue()


def test_empty_bytes_are_refused():
    with pytest.raises(UnreadableImageError, match="empty"):
        extract_image_text(b"")


def test_a_non_image_is_refused():
    with pytest.raises(UnreadableImageError, match="recognisable"):
        extract_image_text(b"this is a text file, not a photograph")


def test_a_pdf_is_refused_here():
    """The PDF path is pdf.py's. Sending one here is a caller mistake."""
    with pytest.raises(UnreadableImageError):
        extract_image_text(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")


def test_a_truncated_png_is_refused():
    whole = scan(["anything"])
    with pytest.raises(UnreadableImageError):
        extract_image_text(whole[: len(whole) // 3])


def test_an_oversized_file_is_refused_before_decoding():
    """
    The check is on the byte length, so the payload is never parsed. If it
    were parsed first, this test would be the attack it is guarding against.
    """
    with pytest.raises(ImageTooLargeError, match="byte limit"):
        extract_image_text(b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_IMAGE_BYTES)


def test_a_decompression_bomb_is_refused_on_its_header():
    """
    A huge, highly compressible image: small on disk, enormous in memory.
    Pillow's own guard only trips at twice its 89 MP default, which is
    already far past what this process should allocate.
    """
    side = 8000
    assert side * side > MAX_IMAGE_PIXELS
    buf = io.BytesIO()
    Image.new("L", (side, side), "white").save(buf, "PNG")
    bomb = buf.getvalue()
    assert len(bomb) < MAX_IMAGE_BYTES, "the bomb must pass the byte check to test the pixel one"

    with pytest.raises(ImageTooLargeError, match="pixel limit"):
        extract_image_text(bomb)


def test_mean_confidence_is_none_when_nothing_was_read():
    """
    None rather than 0.0. "Not read" and "read badly" are different states
    and the API reports them differently.
    """
    empty = OcrExtraction(text="", words=(), language="eng")
    assert empty.mean_confidence is None
    assert empty.has_text is False
    assert empty.is_low_confidence is False


def test_mean_confidence_averages_only_real_words():
    extraction = OcrExtraction(
        text="Razom 349.50",
        words=(
            OcrWord("Razom", 90.0, 0, 0, 10, 10),
            OcrWord("349.50", 50.0, 0, 0, 10, 10),
        ),
        language="eng",
    )
    assert extraction.mean_confidence == 70.0


def test_low_confidence_needs_text_as_well_as_a_low_score():
    """
    A blank page is not 'low confidence', it is 'no text'. Conflating them
    would tell the user to retake a photo of a page that is genuinely blank.
    """
    blank = OcrExtraction(text="", words=(OcrWord("", 5.0, 0, 0, 1, 1),), language="eng")
    assert blank.is_low_confidence is False

    poor = OcrExtraction(
        text="x" * MIN_CHARS,
        words=(OcrWord("x", LOW_CONFIDENCE - 10, 0, 0, 1, 1),),
        language="eng",
    )
    assert poor.is_low_confidence is True


def test_has_text_is_the_min_chars_threshold():
    assert OcrExtraction("x" * (MIN_CHARS - 1), (), "eng").has_text is False
    assert OcrExtraction("x" * MIN_CHARS, (), "eng").has_text is True


@needs_tesseract
def test_a_clean_scan_is_read():
    result = extract_image_text(scan(RECEIPT))

    assert result.has_text
    assert "ATB" in result.text
    assert "349.50" in result.text
    assert result.word_count >= 6
    assert result.mean_confidence is not None and result.mean_confidence > 80
    assert result.is_low_confidence is False


@needs_tesseract
def test_word_boxes_land_inside_the_image():
    """
    The boxes are what a citation will point at, so coordinates that fall
    outside the page would be worse than no coordinates at all.
    """
    width, height = 1000, 500
    result = extract_image_text(scan(RECEIPT, size=(width, height)))

    assert result.words
    for word in result.words:
        assert 0 <= word.left <= width
        assert 0 <= word.top <= height
        assert word.left + word.width <= width
        assert word.top + word.height <= height
        assert word.text.strip() == word.text != ""
        assert 0 <= word.confidence <= 100


@needs_tesseract
def test_a_blank_page_is_an_answer_not_an_error():
    """
    Returning an empty extraction rather than raising: the caller decides
    what an empty page means, and for the endpoint it is a 422 with advice.
    """
    result = extract_image_text(scan([]))

    assert result.has_text is False
    assert result.word_count == 0
    assert result.mean_confidence is None


@needs_tesseract
def test_pure_noise_does_not_report_high_confidence():
    """
    The regression this module was designed around. Averaging tesseract's
    raw confidence column scores an image of pure noise at 95, because it
    emits one high-confidence row whose text is empty. Pairing each score
    with its own word is what stops that.
    """
    result = extract_image_text(scan([], noise=120_000))

    assert result.has_text is False
    assert result.mean_confidence is None


@needs_tesseract
def test_degraded_text_is_read_but_flagged():
    result = extract_image_text(scan(["Razom: 349.50 UAH"], noise=40_000))

    if result.has_text:
        assert result.mean_confidence is not None
        assert result.mean_confidence < 80, "noisy text should not score like a clean scan"


@needs_tesseract
def test_a_rotated_phone_photo_is_straightened_first():
    """
    Without exif_transpose this returns confident garbage — measured:
    "HVN OS'6veE -WwozeY" for "Razom: 349.50 UAH". Confident, so no
    threshold would catch it; the only defence is honouring the tag.
    """
    upright = extract_image_text(scan(["Razom: 349.50 UAH"], fmt="JPEG"))
    rotated = extract_image_text(
        scan(["Razom: 349.50 UAH"], fmt="JPEG", exif_orientation=6)
    )

    assert "349.50" in upright.text
    assert "349.50" in rotated.text, "the EXIF orientation tag was ignored"


@needs_tesseract
@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "TIFF"])
def test_every_accepted_format_is_decodable(fmt):
    result = extract_image_text(scan(RECEIPT, fmt=fmt))
    assert "349.50" in result.text


@needs_tesseract
def test_an_unknown_language_pack_fails_as_an_engine_problem():
    """
    A language the host has no training data for is an operational fault
    (the pack is not installed), not a bad upload — so it must not surface
    as a 4xx telling the user their photo is wrong.
    """
    with pytest.raises(OcrEngineUnavailable):
        extract_image_text(scan(RECEIPT), language="zzz")


@needs_tesseract
async def test_the_async_wrapper_returns_the_same_result(anyio_backend):
    data = scan(RECEIPT)
    assert (await extract_image_text_async(data)).text == extract_image_text(data).text