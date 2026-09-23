# mentat-ai-backend

#### Requirements
* Python 3.12+
* Poetry
* Docker Desktop (running)

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

### Database container
* Start container: `` docker compose up -d   ``
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