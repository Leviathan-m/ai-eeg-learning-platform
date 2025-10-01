"""
Logging configuration for AI-EEG Learning Platform.

Provides structured logging with JSON format for production and
human-readable format for development.
"""

import logging
import sys
from typing import Any, Dict, Callable, Iterable, MutableMapping, Mapping, Sequence

import structlog
from pythonjsonlogger import jsonlogger

from .config import settings


def setup_logging() -> None:
    """
    Configure structured logging for the application.

    Sets up different log formats for development vs production:
    - Development: Human-readable console output
    - Production: JSON structured logging
    """
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    # Configure structlog
    shared_processors: list[Callable[[Any, str, MutableMapping[str, Any]], Mapping[str, Any] | str | bytes | bytearray | tuple[Any, ...]]] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.ENVIRONMENT == "development":
        # Development configuration
        processors: list[Callable[[Any, str, MutableMapping[str, Any]], Mapping[str, Any] | str | bytes | bytearray | tuple[Any, ...]]] = shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]

        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(colors=True),
                foreign_pre_chain=shared_processors,
            )
        )

        root_logger = logging.getLogger()
        root_logger.addHandler(console_handler)

    else:
        # Production configuration with JSON logging
        processors = shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]

        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # JSON handler for production
        json_handler = logging.StreamHandler(sys.stdout)
        json_formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        json_handler.setFormatter(json_formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(json_handler)

    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # Suppress noisy third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Configure Sentry integration if DSN is provided
    if settings.SENTRY_DSN:
        import sentry_sdk
        try:
            from sentry_sdk.integrations.fastapi import FastApiIntegration as FastAPIIntegration  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - fallback for older/newer SDKs
            from sentry_sdk.integrations.fastapi import FastApiIntegration as FastAPIIntegration  # type: ignore[no-redef]
        from sentry_sdk.integrations.redis import RedisIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            environment=settings.ENVIRONMENT,
            integrations=[
                FastAPIIntegration(),
                RedisIntegration(),
                SqlalchemyIntegration(),
            ],
            traces_sample_rate=0.1,
            send_default_pii=False,
        )


def get_request_logger(request_id: str) -> structlog.BoundLogger:
    """
    Get a logger with request context.

    Args:
        request_id: Unique identifier for the request

    Returns:
        Logger with request context
    """
    return structlog.get_logger().bind(request_id=request_id)


class RequestLoggingMiddleware:
    """
    Middleware for logging HTTP requests with structured format.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Generate request ID
        request_id = (
            structlog.get_logger()
            .bind(request_id=id(scope))
            ._context.get("request_id", "unknown")
        )

        logger = get_request_logger(request_id)

        # Log request start
        logger.info(
            "Request started",
            method=scope["method"],
            path=scope["path"],
            query_string=scope.get("query_string", b"").decode(),
        )

        start_time = structlog.processors.TimeStamper(fmt="iso")

        # Process request
        response_status = 200
        response_length = 0

        async def send_wrapper(message):
            nonlocal response_status, response_length

            if message["type"] == "http.response.start":
                response_status = message["status"]

            elif message["type"] == "http.response.body":
                response_length += len(message.get("body", b""))

            return await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            logger.error(
                "Request failed",
                error=str(e),
                exc_info=True,
            )
            raise
        finally:
            # Log request completion
            logger.info(
                "Request completed",
                status=response_status,
                response_length=response_length,
            )


# Custom log processors for domain-specific logging
def add_user_context(
    logger, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Add user context to log entries if available.
    """
    user_id = event_dict.get("user_id")
    if user_id:
        event_dict["user_id"] = user_id
    return event_dict


def add_eeg_context(
    logger, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Add EEG processing context to log entries if available.
    """
    session_id = event_dict.get("session_id")
    if session_id:
        event_dict["eeg_session_id"] = session_id

    processing_time = event_dict.get("processing_time")
    if processing_time:
        event_dict["eeg_processing_time_ms"] = processing_time

    return event_dict
