# mentat-ai-backend

## Requirements
* Python 3.12+
* Poetry
* Docker Desktop (running)

## Contributing

* `main` is protected, no direct pushes.
* Work on a branch: `git checkout -b feat/your-feature`
* Open a PR against `main`. CI (ruff + migrations + pytest) must pass.
* One approval from the other team member is required before merge.

## Check before creating PR
```angular2html
poetry check --lock          # lock file matches pyproject.toml
poetry run ruff check .      # lint
poetry run ruff format .     # format
poetry run pytest            # tests
```


### Poetry (venv & deps)
* Establish venv with poetry: `` poetry config virtualenvs.in-project true ``
* Install all recorded dependencies: `` poetry install ``
* Add new package: `` poetry add <package> ``

### ENV files
`` cp .env.example .env`` (copy defaults form the example)

### Server

* Install all dependencies ``poetry install``
* Start server `` poetry run uvicorn app.main:app --reload``
* Check connection ``http://localhost:8000/api/v1/health``
* Documentation `` http://127.0.0.1:8000/docs ``

### Database container
* Start container: `` docker compose up -d   ``
* Check db status: `` docker compose ps`` _Note: wait for 'healthy' status_
* Close connection to container: `` docker compose down  ``
* Check db container extensions: ``  docker exec -it mentat-ai-db psql -U mentat_user -d mentat_ai -c "\dx" ``

### Alembic migrations instructions
1. Define schema inside `` app/models/ ``
2. Import schema inside the `` alembic/env.py `` inside:
    ```
    from app.core.db import Base
    from app.core.config import settings
     ***HERE***
    
    target_metadata = Base.metadata
    ```
3. Create migration with message `` poetry run alembic revision -m "message" ``
4. Stage migration and update head ``  poetry run alembic upgrade head ``
* Roll back one migration  `` poetry run alembic downgrade -1``
* Show current migration heads	`` poetry run alembic heads ``

### Auth
* Seed the two Mentat users (reads `SEED_USER_1_EMAIL`/`SEED_USER_1_PASSWORD`, `SEED_USER_2_EMAIL`/`SEED_USER_2_PASSWORD` from `.env`): `` poetry run python scripts/seed_users.py ``
* Register: `` POST /api/v1/auth/register `` with body `{"email": "...", "password": "..."}` (password: 8+ chars) → `201` + `{"access_token": "...", "token_type": "bearer"}`. `409` if that email is already registered, `422` if the password is too short.
* Login: `` POST /api/v1/auth/login `` with body `{"email": "...", "password": "..."}` → returns `{"access_token": "...", "token_type": "bearer"}`
* Use the token on protected routes: `Authorization: Bearer <access_token>` header

#### Google sign-in
Requires `GOOGLE_CLIENT_ID` in `.env` (a Google Cloud OAuth 2.0 Client ID — "Chrome extension" type for the real frontend).
* `` POST /api/v1/auth/google `` — body is exactly one of `{"id_token": "..."}` (a Google Identity Services / OAuth Playground credential) or `{"access_token": "..."}` (what `chrome.identity.getAuthToken` returns). Logs in if that Google identity is already linked; creates a new user on first sign-in. `409` if the email is already registered under a different identity (password, most likely) — log in that way first, then connect Google explicitly. `401` on an invalid/expired token.
* `` POST /api/v1/auth/google/connect `` — same body shape, requires `Authorization: Bearer <access_token>`. Links that Google identity to the signed-in user. `201` on success, `409` if that Google identity is already connected to any account.
* `` POST /api/v1/auth/set-password `` — the reverse: lets a Google-only account add password sign-in. Requires `Authorization: Bearer <access_token>`, body `{"password": "..."}` (8+ chars). Optional — nothing requires it. `201` on success, `409` if the account already has a password, `422` if too short.

### Test
Docker container up **_NOT_** required:
* Testing text normalization: `` poetry run pytest tests/test_normalizer.py -v ``
* Testing chunking: `` poetry run pytest tests/test_chunking.py -v ``
* Testing error handling: ``poetry run pytest tests/test_errors.py -v``
* Testing embeddings: ``poetry run test/test_embedder.py -v``
* Testing scheduler: `` poetry run pytest tests/test_scheduler.py -v  ``
* Testing pdf extraction: ``poetry run pytest tests/test_pdf_extraction.py``
* Test storage script (fake): ``poetry run pytest tests/test_storage.py``
* Test pdf file ingestion: ``poetry run pytest tests/test_ingest_file.py``
* Testing auto tagging: `` poetry run pytest tests/test_auto_tagger.py -v``
* Testing image (OCR) extractor: `` poetry run pytest tests/test_ocr.py -v``
* Testing scanned pdfs: ``poetry run pytest tests/test_scanned_pdf.py -v``
* Testing citation retrieval script: ``poetry run pytest tests/test_citation.py -v``
* Testing answering prompt (+injection): ``poetry run pytest tests/test_prompt.py -v ``
* Testing passage script: ``poetry run pytest tests/test_passage.py -v``
* Testing markers: ``poetry run pytest tests/test_markers.py -v``
* Testing llm client (conversational): ``poetry run pytest tests/test_llm_client.py -v``
* Testing condense: ``poetry run pytest tests/test_condense.py -v``

Docker container up required:
* Testing ingest: `` poetry run pytest tests/test_ingest.py -v``
* Testing auth (login + token validation): `` poetry run pytest tests/test_auth.py tests/test_security.py -v ``
* Testing integration pipeline: `` poetry run pytest test/integration -v``
* Testing integration pipeline (save records): ``poetry run pytest tests/integration -q --keep-rows``
* Testing vector search: `` poetry run pytest tests/integration/test_search.py -v ``
* Testing background work of the ingest pipeline: ``poetry run pytest tests/test_ingest_background.py -v``
* Testing tags pipeline tagging (save records): `` poetry run pytest tests/integration/test_pipeline_tagging.py -q --keep-rows``
* Testing tag rename: ``poetry run pytest tests/integration/test_tag_rename.py -v``
* Testing image ingestion (tesseract engine needed): `` poetry run pytest tests/test_ingest_image.py -v ``
* Testing document status: ``poetry run pytest tests/test_document_status.py -v``
* Testing tagging delete form vocabulary: `` poetry run pytest tests/test_tag_delete.py -v``
* Testing integrational citation: ``poetry run pytest tests/integration/test_citations.py -v``
* Testing search api: ``poetry run pytest tests/test_search_api.py -v``
* Testing integration of the search service: ``poetry run pytest tests/integration/test_search.py -v``
* Testing integration answering retrieval context: `` poetry run pytest tests/integration/test_retrieval_context.py``
* Testing passage api: `` poetry run pytest tests/test_passage.py -v  ``
* Testing document file link: ``poetry run pytest tests/test_markers.py -v``
* Testing frontend contract: ``poetry run pytest tests/test_frontend_contract.py -v`` 
* Testing answerer integration: ``poetry run pytest tests/ingestion/test_answerer.py -v``
* Testing conversation api: ``poetry run pytest tests/test_conversations_api.py -v``

Manual run of the pdf processing and storing: ``poetry run python -m scripts.check_pdf_storage ingest docs\sample-invoice.pdf``
Manual run of the image processing and storing: ``poetry run python -m scripts.check_image_ocr ingest docs/receipt.jpg``

### Ruff
* Lint  `` poetry run ruff check . ``
* Lint + autofix `` poetry run ruff check . --fix ``
* Format `` poetry run ruff format . ``

### Text Embeddings
Model: voyage-4
Plan: 0.06$ / million tokens 
Free start: basic 200 million free tokens on register

| Document | Chunks | Tokens | Cost if paid |
| --- | --- | --- | --- |
| Short note (2k chars) | 3 | ~900 | $0.00005 |
| Blog post (8k) | 10 | ~3,000 | $0.0002 |
