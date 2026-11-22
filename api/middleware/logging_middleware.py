"""
Logging and tracing middleware for FastAPI.
"""

import time
import uuid
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

# Configure structured logging
structlog = None
try:
    import structlog
except ImportError:
    pass

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging with structured logging."""

    def __init__(self, app: ASGIApp, logger_name: str = "api"):
        super().__init__(app)
        self.logger = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Add request ID to request state for downstream use
        request.state.request_id = request_id

        # Log request
        self.logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client": request.client.host if request.client else "unknown",
            }
        )

        # Process request
        response = None
        try:
            response = await call_next(request)
        except Exception as e:
            process_time = time.time() - start_time
            self.logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "process_time_ms": round(process_time * 1000, 2),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True
            )
            raise

        process_time = time.time() - start_time

        # Log response
        self.logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
            }
        )

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting metrics."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.request_count = 0
        self.request_duration_sum = 0.0
        self.error_count = 0
        self.status_codes = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        try:
            response = await call_next(request)
        except Exception:
            self.error_count += 1
            raise

        duration = time.time() - start_time
        self.request_count += 1
        self.request_duration_sum += duration

        # Track status codes
        status = response.status_code
        self.status_codes[status] = self.status_codes.get(status, 0) + 1

        return response

    def get_metrics(self) -> dict:
        """Get collected metrics."""
        return {
            "total_requests": self.request_count,
            "avg_response_time_ms": (
                self.request_duration_sum / self.request_count * 1000
                if self.request_count > 0 else 0
            ),
            "error_count": self.error_count,
            "status_codes": self.status_codes,
        }


def get_logger(name: str = "api") -> logging.Logger:
    """Get configured logger instance."""
    return logging.getLogger(name)


def setup_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    """Configure application logging."""
    import sys

    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Configure structlog if available
    if structlog and json_logs:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )