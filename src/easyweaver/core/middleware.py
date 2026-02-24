import time
from collections import defaultdict

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from easyweaver.core.exceptions import EasyWeaverError
from easyweaver.settings import settings

logger = structlog.get_logger()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter per client IP."""

    def __init__(self, app, max_requests: int = 120, window_seconds: int = 60):
        super().__init__(app)
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        # Clean old entries
        window_start = now - self._window
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > window_start
        ]

        if len(self._requests[client_ip]) >= self._max_requests:
            logger.warning("rate_limit_exceeded", client_ip=client_ip)
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please try again later.",
                    }
                },
            )

        self._requests[client_ip].append(now)
        return await call_next(request)


def setup_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.rate_limit_per_minute > 0:
        app.add_middleware(
            RateLimitMiddleware,
            max_requests=settings.rate_limit_per_minute,
        )

    @app.exception_handler(EasyWeaverError)
    async def easyweaver_error_handler(request: Request, exc: EasyWeaverError):
        status_map = {
            "NOT_FOUND": 404,
            "VALIDATION_ERROR": 422,
            "AUTHENTICATION_FAILED": 401,
            "CONNECTION_TEST_FAILED": 400,
            "QUERY_EXECUTION_FAILED": 400,
            "PROCESS_EXECUTION_FAILED": 400,
            "STORAGE_ERROR": 500,
        }
        status_code = status_map.get(exc.code, 500)
        logger.error("request_error", code=exc.code, message=exc.message)
        return JSONResponse(
            status_code=status_code,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception("unhandled_error", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}},
        )
