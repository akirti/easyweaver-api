import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from easyweaver.core.exceptions import EasyWeaverError
from easyweaver.settings import settings

logger = structlog.get_logger()


def setup_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(EasyWeaverError)
    async def easyweaver_error_handler(request: Request, exc: EasyWeaverError):
        status_map = {
            "NOT_FOUND": 404,
            "VALIDATION_ERROR": 422,
            "AUTHENTICATION_FAILED": 401,
            "CONNECTION_TEST_FAILED": 400,
            "QUERY_EXECUTION_FAILED": 400,
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
