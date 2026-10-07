from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.errors import register_error_handling


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
    )
    register_error_handling(app)
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
