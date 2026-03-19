"""Centralized logging configuration for the arena package."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_FMT_PLAIN = logging.Formatter("%(message)s")
_FMT_LEVEL = logging.Formatter("%(levelname)s: %(message)s")


class _ArenaHandler(logging.StreamHandler):
    """Handler that uses a plain format for INFO and a prefixed format for WARNING+."""

    def emit(self, record: logging.LogRecord) -> None:
        self.setFormatter(_FMT_PLAIN if record.levelno <= logging.INFO else _FMT_LEVEL)
        super().emit(record)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the ``arena`` root logger with a console handler.

    Safe to call multiple times; only the first call attaches a handler.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger("arena")
    root.setLevel(level)
    root.addHandler(_ArenaHandler(sys.stderr))


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``arena`` namespace."""
    setup_logging()
    return logging.getLogger(name)


def get_script_logger(name: str) -> logging.Logger:
    """Return a logger for pipeline scripts that writes to **stdout**.

    This keeps output captured in ``stdout_tail`` by the pipeline runner,
    matching the previous ``print()`` behaviour.  INFO messages use a plain
    format; WARNING and above get a level prefix.
    """
    logger = logging.getLogger(f"arena.scripts.{name}")
    if not logger.handlers:
        handler = _ArenaHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
