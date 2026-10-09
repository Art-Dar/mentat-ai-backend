"""
Every error leaving the API — validation, domain, database, or a bug — comes
back in one shape:

    {
      "error": {
        "code": "not_found",
        "message": "Document not found",
        "details": [...],          # optional, omitted when empty
        "request_id": "3f9c1e2a"
      }
    }
"""

import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class AppError(Exception):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    message: str = "Something went wrong"

    def __init__(
        self,
        message: str | None = None,
        *,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details or []
        super().__init__(self.message)


class NotFound(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"
    message = "Resource not found"


class Conflict(AppError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"
    message = "Resource already exists"


class Unauthorized(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"
    message = "Authentication required"


class Forbidden(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "forbidden"
    message = "Not permitted"


class InvalidRequest(AppError):
    # Input that is well-formed but semantically wrong for this operation.
    status_code = 422
    code = "invalid_request"
    message = "Request could not be processed"


class ServiceUnavailable(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "service_unavailable"
    message = "A required service is temporarily unavailable"


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": _request_id(request),
        }
    }
    if details:
        body["error"]["details"] = details

    response_headers = {REQUEST_ID_HEADER: _request_id(request)}
    if headers:
        response_headers.update(headers)

    return JSONResponse(status_code=status_code, content=body, headers=response_headers)


def _safe_validation_details(exc: RequestValidationError) -> list[dict[str, Any]]:
    details = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        details.append({"field": location, "message": error.get("msg", "Invalid value")})
    return details


async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    if exc.status_code >= 500:
        logger.error("app error: %s", exc.message, extra={"request_id": _request_id(request)})
    return _error_response(request, exc.status_code, exc.code, exc.message, exc.details)


async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    # FastAPI's own HTTPException, so existing raises keep working
    code = _HTTP_CODES.get(exc.status_code, "http_error")
    message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return _error_response(
        request, exc.status_code, code, message, headers=getattr(exc, "headers", None)
    )


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _error_response(
        request,
        422,
        "validation_error",
        "Request payload is invalid",
        _safe_validation_details(exc),
    )


async def handle_integrity_error(request: Request, exc: IntegrityError) -> JSONResponse:
    # constraint rejected the write — usually a duplicate.
    # Reported as 409 rather than 500 because the caller can act on it
    logger.warning("integrity error: %s", exc.orig, extra={"request_id": _request_id(request)})
    return _error_response(
        request,
        status.HTTP_409_CONFLICT,
        "conflict",
        "That conflicts with something already stored",
    )


async def handle_database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.exception("database error", extra={"request_id": _request_id(request)})
    return _error_response(
        request,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "database_unavailable",
        "The database is temporarily unavailable",
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error", extra={"request_id": _request_id(request)})
    return _error_response(
        request,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "Something went wrong on our side",
    )


_HTTP_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "invalid_request",
    429: "rate_limited",
}


def register_error_handling(app: FastAPI) -> None:
    # attach the request-id middleware and every exception handler

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request.state.request_id = incoming or uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return response

    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(IntegrityError, handle_integrity_error)
    app.add_exception_handler(SQLAlchemyError, handle_database_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
