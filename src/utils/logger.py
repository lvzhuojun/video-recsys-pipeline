"""Unified logging utility for the recsys project.

All modules should use get_logger(__name__) instead of bare print().
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Get a configured logger that writes to stdout and optionally a file.

    Args:
        name: Logger name, typically ``__name__``.
        log_file: Path to log file. If None, console-only output.
        level: Logging verbosity level.

    Returns:
        Configured :class:`logging.Logger` instance.

    Example:
        >>> logger = get_logger(__name__, log_file="experiments/logs/train.log")
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)

    # Guard against adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Always write to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Optionally mirror output to a file
    if log_file is not None:
        log_path = Path(log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except OSError as e:
            logger.warning(f"Could not create log file {log_file}: {e}")

    return logger
