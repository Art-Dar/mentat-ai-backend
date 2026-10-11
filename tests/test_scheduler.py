import uuid

import pytest
from fastapi import BackgroundTasks

from app.services.ingestion import scheduler
from app.services.ingestion.scheduler import schedule_processing


@pytest.fixture
def recorded(monkeypatch):
    """Replace the pipeline with a recorder so nothing touches a database."""
    calls: list[uuid.UUID] = []

    async def fake_process(document_id):
        calls.append(document_id)

    monkeypatch.setattr(scheduler, "process_document", fake_process)
    return calls


@pytest.mark.anyio
async def test_the_task_is_registered_not_run(recorded):
    """
    The whole point: scheduling returns before the work happens, so the
    endpoint can answer in milliseconds.
    """
    background_tasks = BackgroundTasks()
    document_id = uuid.uuid4()

    await schedule_processing(document_id, background_tasks)

    assert recorded == []  # nothing has run yet
    assert len(background_tasks.tasks) == 1


@pytest.mark.anyio
async def test_the_registered_task_runs_the_pipeline(recorded):
    background_tasks = BackgroundTasks()
    document_id = uuid.uuid4()
    await schedule_processing(document_id, background_tasks)

    await background_tasks()  # what Starlette does after sending the response

    assert recorded == [document_id]


@pytest.mark.anyio
async def test_only_the_id_crosses_the_boundary(recorded):
    """
    Never an ORM object and never the request's AsyncSession: Depends(get_db)
    closes that session when the response is sent, which happens BEFORE the
    task runs. Using it raises "Session is closed" long after the endpoint
    already returned 201.
    """
    background_tasks = BackgroundTasks()
    document_id = uuid.uuid4()

    await schedule_processing(document_id, background_tasks)

    task = background_tasks.tasks[0]
    assert task.args == (document_id,)
    assert task.kwargs == {}


@pytest.mark.anyio
async def test_without_a_request_context_the_work_runs_inline(recorded):
    """
    A reprocess script or a test has no response to get out of the way of,
    so awaiting is the honest behaviour rather than silently doing nothing.
    """
    document_id = uuid.uuid4()

    await schedule_processing(document_id)

    assert recorded == [document_id]
