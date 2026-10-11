import logging
import uuid

from fastapi import BackgroundTasks

from app.services.ingestion.pipeline import process_document

logger = logging.getLogger(__name__)


async def schedule_processing(
        document_id: uuid.UUID,
        background_tasks: BackgroundTasks | None = None,
) -> None:
    """
    Arrange for `document_id` to be processed after the response is sent.

    Returns immediately -> COMMITTED the document first ->
    BackgroundTask starts when response is sent & pipeline opens own session

    `background_tasks` is optional so non-HTTP callers — a reprocess
    script, a test — can use the same entry point. Without it the work is
    awaited inline, which is the honest behaviour: there is no response to
    get out of the way of.
    """
    if background_tasks is None:
        logger.debug("no request context, processing %s inline", document_id)
        await process_document(document_id)
        return

    # Only the id crosses this line
    # Never an ORM object & never the request's AsyncSession:
    # Depends(get_db) closes that session when the
    # response is sent, which happens BEFORE the task runs. Using it raises
    # "Session is closed" long after the endpoint already returned 201 —
    # the hardest version of this bug to trace back to its cause.
    background_tasks.add_task(process_document, document_id)
