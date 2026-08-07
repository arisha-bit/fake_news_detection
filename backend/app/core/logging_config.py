"""
Structured logging configuration.

Sets up JSON-friendly log formatting for production.
Call configure_logging() once at app startup in main.py.
"""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """
    Configure root logger with a structured format.

    Format: [LEVEL] timestamp | logger_name | message
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove duplicate handlers if called multiple times
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ["httpx", "httpcore", "uvicorn.access"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("app").setLevel(log_level)
    logging.info("Logging configured at level: %s", level)
