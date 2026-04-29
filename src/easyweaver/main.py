from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from easyweaver.core.middleware import setup_middleware
from easyweaver.dependencies import init_db, shutdown_db, init_redis, shutdown_redis
from easyweaver.settings import settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting EasyWeaver", debug=settings.debug)
    try:
        await init_db()
        logger.info("MongoDB connected (easyweaver_meta)")
    except Exception as e:
        logger.warning("MongoDB not available at startup", error=str(e))
    try:
        await init_redis()
    except Exception as e:
        logger.warning("Redis not available at startup", error=str(e))
    yield
    await shutdown_redis()
    shutdown_db()
    logger.info("EasyWeaver shut down")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    setup_middleware(app)
    _register_routes(app)
    return app


def _register_routes(app: FastAPI):
    from easyweaver.sources.router import router as sources_router
    from easyweaver.queries.router import router as queries_router
    from easyweaver.auth.router import router as auth_router

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/v1/settings")
    async def public_settings():
        return {"max_result_rows": settings.max_result_rows}

    app.include_router(sources_router, prefix="/api/v1/sources", tags=["sources"])
    app.include_router(queries_router, prefix="/api/v1/queries", tags=["queries"])
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])

    from easyweaver.processes.router import router as processes_router

    app.include_router(processes_router, prefix="/api/v1/processes", tags=["processes"])

    from easyweaver.dashboard.router import router as dashboard_router

    app.include_router(dashboard_router, prefix="/api/v1/dashboard", tags=["dashboard"])

    from easyweaver.lookups.router import router as lookups_router

    app.include_router(lookups_router, prefix="/api/v1/lookups", tags=["lookups"])


app = create_app()
