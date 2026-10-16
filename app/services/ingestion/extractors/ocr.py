import io
import logging
from dataclasses import dataclass

import anyio.to_thread
import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError
from pytesseract import Output, TesseractNotFoundError

logger = logging.getLogger(__name__)

# TESSERACT_REQUIRED
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MIN_CHARS = 16
LOW_CONFIDENCE = 60.0
OCR_TIMEOUT_SECONDS = 30

DEFAULT_LANGUAGE = "eng"


class OcrError(RuntimeError):
    """The image could not be turned into text."""


class UnreadableImageError(OcrError):
    """Not a decodable image: wrong bytes, truncated, or an unknown format."""


class ImageTooLargeError(OcrError):
    """Over the byte or pixel budget."""


class OcrEngineUnavailable(OcrError):
    """
    The tesseract binary is missing or did not finish in time.
    """


@dataclass(frozen=True)
class OcrWord:
    """
    One recognised word and where it sits in the image.
    """

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class OcrExtraction:
    text: str
    words: tuple[OcrWord, ...]
    language: str

    @property
    def char_count(self) -> int:
        return len(self.text.strip())

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def mean_confidence(self) -> float | None:
        """
        Average confidence over words that actually produced text.

        None, not 0.0, when nothing was recognised — "no reading" and "a
        bad reading" are different states and the caller must be able to
        tell them apart.
        """
        if not self.words:
            return None
        return sum(word.confidence for word in self.words) / len(self.words)

    @property
    def has_text(self) -> bool:
        """Enough characters to be worth storing and embedding."""
        return self.char_count >= MIN_CHARS

    @property
    def is_low_confidence(self) -> bool:
        """
        Text was recovered, but poorly. Worth surfacing: the user can
        retake the photo, which is cheaper than correcting the extraction.
        """
        confidence = self.mean_confidence
        return self.has_text and confidence is not None and confidence < LOW_CONFIDENCE


def _open(data: bytes) -> Image.Image:
    """Decode and normalise, refusing anything oversized before decoding."""
    if not data:
        raise UnreadableImageError("image is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageTooLargeError(
            f"image is {len(data)} bytes, over the {MAX_IMAGE_BYTES} byte limit"
        )

    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as exc:
        raise UnreadableImageError("not a recognisable image format") from exc
    except Exception as exc:
        raise UnreadableImageError(f"{type(exc).__name__}: {exc}") from exc

    # Image.open only reads the header, so .size is known before any pixel
    # data is allocated. This is the only point at which a decompression
    # bomb can be refused cheaply.
    pixels = image.size[0] * image.size[1]
    if pixels > MAX_IMAGE_PIXELS:
        raise ImageTooLargeError(
            f"image is {image.size[0]}x{image.size[1]} ({pixels} pixels), "
            f"over the {MAX_IMAGE_PIXELS} pixel limit"
        )

    try:
        # A portrait phone photo is stored as landscape pixels plus an EXIF
        # Orientation tag. Skipping this does not merely lower quality.
        image = ImageOps.exif_transpose(image)
        # Greyscale drops JPEG colour noise that tesseract would otherwise
        # try to read, and makes the result independent of the sensor.
        image = image.convert("L")
    except OSError as exc:
        # "image file is truncated" lands here, on load rather than on open.
        raise UnreadableImageError(f"image data is incomplete: {exc}") from exc

    return image


def _engine_failure(exc: RuntimeError, language: str) -> OcrError:
    """
    Classify a tesseract failure as the host's fault or the file's.
    """
    message = str(exc).lower()

    if "timeout" in message:
        return OcrEngineUnavailable(f"OCR did not finish within {OCR_TIMEOUT_SECONDS}s")

    # "Error opening data file .../ukr.traineddata ... Failed loading
    # language 'ukr' ... Could not initialize tesseract." The pack is not
    # installed; the upload was never the problem.
    if "failed loading language" in message or "could not initialize tesseract" in message:
        return OcrEngineUnavailable(f"the '{language}' language pack is not installed on this host")

    return OcrError(f"OCR failed: {exc}")


def extract_image_text(data: bytes, *, language: str = DEFAULT_LANGUAGE) -> OcrExtraction:
    """
    Recognise text in an image.
    Returns an empty-but-valid extraction for a blank page
    """
    image = _open(data)

    try:
        data_dict = pytesseract.image_to_data(
            image,
            lang=language,
            output_type=Output.DICT,
            timeout=OCR_TIMEOUT_SECONDS,
        )
    except TesseractNotFoundError as exc:
        raise OcrEngineUnavailable("the tesseract binary is not installed on this host") from exc
    except RuntimeError as exc:
        # Both failures arrive as RuntimeError — pytesseract raises a bare
        # one on timeout, and TesseractError (also a RuntimeError) for
        # anything the binary itself reported. Neither carries a code, so
        # the message is the only discriminator there is.
        raise _engine_failure(exc, language) from exc
    except Exception as exc:
        raise OcrError(f"{type(exc).__name__}: {exc}") from exc

    words: list[OcrWord] = []
    for text, conf, left, top, width, height in zip(
        data_dict["text"],
        data_dict["conf"],
        data_dict["left"],
        data_dict["top"],
        data_dict["width"],
        data_dict["height"],
        strict=False,
    ):
        stripped = text.strip()
        confidence = float(conf)
        # Tesseract emits a row per layout block, most of which are
        # structural: empty text, or confidence -1. Only rows with both a
        # word and a real confidence describe something it actually read.
        if not stripped or confidence < 0:
            continue
        words.append(
            OcrWord(
                text=stripped,
                confidence=confidence,
                left=int(left),
                top=int(top),
                width=int(width),
                height=int(height),
            )
        )

    try:
        text = pytesseract.image_to_string(image, lang=language, timeout=OCR_TIMEOUT_SECONDS)
    except RuntimeError as exc:
        raise _engine_failure(exc, language) from exc

    # image_to_string keeps tesseract's line and paragraph breaks, which the
    # chunker uses the same way it uses PAGE_SEPARATOR in pdf.py. Rebuilding
    # the text by joining `words` would throw that layout away.
    return OcrExtraction(text=text.strip(), words=tuple(words), language=language)


async def extract_image_text_async(
    data: bytes, *, language: str = DEFAULT_LANGUAGE
) -> OcrExtraction:
    """
    Called from the API and the pipeline.
    tesseract is a subprocess doing CPU-bound work awaiting on the event loop
    """
    return await anyio.to_thread.run_sync(lambda: extract_image_text(data, language=language))
