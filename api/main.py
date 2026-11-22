"""
FastAPI application for Greek Law Case RAG API.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.config.settings import Settings
from api.routes import search
from api.services.search_service import set_search_service, SearchService
from api.middleware.logging_middleware import (
    RequestLoggingMiddleware,
    MetricsMiddleware,
    setup_logging,
)

# Configure logging
setup_logging(log_level="INFO", json_logs=False)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown."""
    logger.info("Starting Greek Law Case RAG API...")

    # Load settings
    settings = Settings()

    # Initialize search service
    search_service = SearchService(settings)
    set_search_service(search_service)

    # Verify health on startup
    health = await search_service.health_check()
    if health["status"] == "unhealthy":
        logger.error("Health check failed on startup", extra={"checks": health["checks"]})
    else:
        logger.info("Startup health check passed", extra={"status": health["status"]})

    logger.info("API startup complete")
    yield

    # Shutdown
    logger.info("Shutting down API...")
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Greek Law Case RAG API",
        description="""
        **Greek Law Case RAG** - Semantic search for Greek legal cases.

        This API provides hybrid search capabilities combining vector similarity
        search with keyword matching for finding relevant Greek law cases.

        ## Features
        - **Hybrid Search**: Combines semantic vector search (70%) with keyword matching (30%)
        - **Greek Language Support**: Optimized for Greek legal terminology
        - **Configurable Weights**: Adjust vector/keyword weights per query
        - **Chunk-level Results**: Retrieve relevant document sections
        - **Health Monitoring**: Built-in health checks for all components
        - **Metrics**: Prometheus-compatible metrics endpoint

        ## Authentication
        Currently no authentication required. For production, add API key or OAuth2.
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(MetricsMiddleware)

    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "message": "Invalid request parameters",
                "detail": exc.errors(),
                "timestamp": datetime.now().isoformat(),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Error",
                "message": exc.detail if isinstance(exc.detail, str) else "Request failed",
                "detail": exc.detail if isinstance(exc.detail, dict) else None,
                "timestamp": datetime.now().isoformat(),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            extra={
                "request_id": getattr(request.state, "request_id", None),
                "path": request.url.path,
                "method": request.method,
            },
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now().isoformat(),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    # Include routers
    app.include_router(search.router)

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": "Greek Law Case RAG API",
            "version": "1.0.0",
            "description": "Semantic search for Greek legal cases",
            "docs": "/docs",
            "health": "/api/v1/health",
            "metrics": "/api/v1/metrics",
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )