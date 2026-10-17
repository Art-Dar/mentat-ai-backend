from fastapi import APIRouter

from app.api.v1 import ask, auth, conversations, documents, files, health, ingest, search, tags

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, tags=["auth"])
api_router.include_router(ingest.router, tags=["ingest"])
api_router.include_router(tags.router, tags=["tags"])
api_router.include_router(files.router, tags=["files"])
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(search.router, tags=["search"])
api_router.include_router(ask.router, tags=["ask"])
api_router.include_router(conversations.router, tags=["conversations"])
