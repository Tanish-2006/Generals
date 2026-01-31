import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from .core.config import settings
from .core.inference import inference_manager
from .api import game

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    try:
        inference_manager.load_model()
        await inference_manager.start()
        logger.info("System ready.")
    except Exception as e:
        logger.critical(f"Startup failure: {e}", exc_info=True)
        # Should we exit? For a service, maybe yes. But let's allow inspection.

    yield

    # Shutdown
    logger.info("Shutting down...")
    await inference_manager.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        lifespan=lifespan,
        description="Production-grade AI Backend for Generals Game",
    )

    # Middleware
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.BACKEND_CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Routers
    app.include_router(game.router, prefix=settings.API_PREFIX, tags=["game"])

    @app.get("/health", tags=["system"])
    async def health_check():
        return {
            "status": "active",
            "version": settings.VERSION,
            "inference": "ready" if inference_manager._server else "not_loaded",
        }

    @app.exception_handler(500)
    async def internal_exception_handler(request, exc):
        logger.error(f"Global 500: {exc}")
        return JSONResponse(
            status_code=500, content={"message": "Internal Server Error"}
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
    )
