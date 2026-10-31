"""Logging utilities for the credit scoring project."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(funcName)s:%(lineno)d | %(message)s"
)


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: str = DEFAULT_FORMAT,
) -> logging.Logger:
    """Create or return a configured logger.

    Parameters
    ----------
    name : str
        Logger name. Usually ``__name__`` of the calling module.
    level : int, default ``logging.INFO``
        Logging level.
    log_file : pathlib.Path, optional
        If provided, also write logs to this file.
    format_string : str, default ``DEFAULT_FORMAT``
        Log message format.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
