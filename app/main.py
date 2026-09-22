from fastapi import FastAPI
from app.core.config import settings

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
    )
    return app

app = create_app()