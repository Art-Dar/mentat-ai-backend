import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "app/schemas/files.py": [
        ("ocr_pages", "the PDF response reports which pages OCR recovered"),
        ("STRICT = ConfigDict", "schema/handler drift fails loudly"),
        ("DocumentStatusResponse", "the status endpoint's response model"),
    ],
    "app/api/v1/files.py": [
        ("extract_pdf_text_with_ocr_async", "scanned PDFs get an OCR pass"),
        ("ocr_result.ocr_available", "a missing engine degrades instead of 503-ing"),
        ("MIN_VALID_LENGTH", "the empty-PDF guard uses the pipeline's threshold"),
    ],
    "app/api/v1/documents.py": [("chunk_count", "GET /documents/{id}")],
    "app/api/v1/router.py": [("documents.router", "the documents router is registered")],
    "app/core/deps.py": [("get_current_user_id", "the shared caller-id dependency")],
    "app/services/ingestion/extractors/scanned_pdf.py": [
        ("ocr_available", "graceful degradation when tesseract is missing"),
        ("OCR_RENDER_DPI = 200", "the measured render resolution"),
    ],
    "app/services/ingestion/extractors/ocr.py": [("_engine_failure", "503 vs 422 classification")],
    "app/services/tagging/delete.py": [("TagInUse", "DELETE /tags/{id} guard")],
    "app/api/v1/tags.py": [("delete_vocabulary_tag", "DELETE /tags/{id}")],
    "app/main.py": [("CORSMiddleware", "the frontend can reach the API")],
    "tests/test_scanned_pdf.py": [("assert_offsets_hold", "")],
    "tests/test_document_status.py": [("chunk_count", "")],
    "tests/test_tag_delete.py": [("force", "")],
    "tests/test_cors.py": [("access-control-allow-origin", "")],
    "tests/test_ocr.py": [("pure_noise", "")],
    "tests/test_ingest_image.py": [("lying_client_content_type", "")],
}

missing_files: list[str] = []
stale_files: list[tuple[str, str, str]] = []

for relative, markers in EXPECTED.items():
    path = ROOT / relative
    if not path.is_file():
        missing_files.append(relative)
        continue
    text = path.read_text(encoding="utf-8")
    for marker, why in markers:
        if marker not in text:
            stale_files.append((relative, marker, why))

if not missing_files and not stale_files:
    print("All expected files are present and up to date.")
    sys.exit(0)

for relative in missing_files:
    print(f"MISSING  {relative}")
for relative, marker, why in stale_files:
    print(f"STALE    {relative}")
    print(f"         has no {marker!r}{' — ' + why if why else ''}")

print(f"\n{len(missing_files)} missing, {len(stale_files)} stale marker(s). "
      "Copy those files in and re-run.")
sys.exit(1)