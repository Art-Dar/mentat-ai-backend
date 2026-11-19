"""
One JSON log line per request
logs every request — success or failure
(time, route, status, duration_ms, and the same request_id the error)
"""

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

request_logger = logging.getLogger("app.requests")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs one JSON line per request. Reuses request.state.request_id if
    the error-handling middleware (app.core.errors) already set one on this
    request; otherwise generates its own, so this works standalone too."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()

        request_id = getattr(request.state, "request_id", None)
        if request_id is None:
            request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
            request.state.request_id = request_id

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            # Let the real exception handlers in app.core.errors deal with
            # the response; still record that this request happened and
            # failed, with the 500 it will end up as.
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self._write(request, 500, duration_ms, request_id)
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        self._write(request, status_code, duration_ms, request_id)
        response.headers.setdefault(REQUEST_ID_HEADER, request_id)
        return response

    @staticmethod
    def _write(request: Request, status_code: int, duration_ms: float, request_id: str) -> None:
        request_logger.info(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "method": request.method,
                    "route": request.url.path,
                    "status": status_code,
                    "duration_ms": duration_ms,
                    "request_id": request_id,
                }
            )
        )


def configure_request_log(log_path: str) -> None:
    """Routes app.requests to its own file, one JSON object per line, and
    stops it from also bubbling up to the root logger's handlers."""
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    request_logger.handlers = [handler]
    request_logger.setLevel(logging.INFO)
    request_logger.propagate = False
