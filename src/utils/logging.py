"""
Structured logging utility for runtime baseline observability.
"""

from __future__ import annotations

import logging
import sys


def setup_logger(name: str = "split_runtime", level: int = logging.INFO) -> logging.Logger:
    """Set up and configure a structured console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
