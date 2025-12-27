from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from typing import Iterator, Sequence

DEFAULT_NOISY_LOGGERS: tuple[str, ...] = (
    "cobraCoach",
    "butler",
    "ics.cobraCharmer",
    "ics.cobraOps",
)


@contextmanager
def suppress_stderr_fd(enabled: bool = True) -> Iterator[None]:
    """Redirect process-level stderr (fd=2) to /dev/null.

    Notes
    -----
    - This affects the whole process (and thus all threads) during the context.
    - If loguru (or anything else) is configured to emit to stderr, that output will
      also be suppressed while this context is active.
    """

    if not enabled:
        yield
        return

    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        try:
            os.dup2(saved_stderr_fd, stderr_fd)
        finally:
            os.close(devnull_fd)
            os.close(saved_stderr_fd)


@contextmanager
def suppress_loggers(
    logger_names: Sequence[str] = (),
    *,
    level: int = logging.WARNING,
    include_root: bool = True,
    suppress_handlers: bool = True,
    enabled: bool = True,
) -> Iterator[None]:
    """Temporarily raise stdlib logging levels for the given loggers.

    This saves and restores both logger levels and (optionally) each handler level.

    Parameters
    ----------
    logger_names
        Names passed to ``logging.getLogger(name)``.
    level
        Temporary level to set.
    include_root
        If True, also applies to the root logger ``logging.getLogger()``.
    suppress_handlers
        If True, also raises handler levels under each affected logger.
    enabled
        If False, this is a no-op.
    """

    if not enabled:
        yield
        return

    saved: dict[str, tuple[int, list[int]]] = {}

    def snapshot_and_set(name: str, log: logging.Logger) -> None:
        saved[name] = (log.level, [h.level for h in log.handlers])
        log.setLevel(level)
        if suppress_handlers:
            for h in log.handlers:
                h.setLevel(level)

    try:
        if include_root:
            snapshot_and_set("__root__", logging.getLogger())

        for name in logger_names:
            snapshot_and_set(name, logging.getLogger(name))

        yield
    finally:
        for name, (saved_level, saved_handler_levels) in saved.items():
            log = logging.getLogger() if name == "__root__" else logging.getLogger(name)
            log.setLevel(saved_level)
            if suppress_handlers:
                for h, hl in zip(log.handlers, saved_handler_levels):
                    h.setLevel(hl)


@contextmanager
def suppress_root_logger(
    level: int = logging.WARNING,
    *,
    suppress_handlers: bool = True,
    enabled: bool = True,
) -> Iterator[None]:
    """Convenience wrapper to suppress only the root logger."""

    with suppress_loggers(
        (),
        level=level,
        include_root=True,
        suppress_handlers=suppress_handlers,
        enabled=enabled,
    ):
        yield


@contextmanager
def suppress_stdout(enabled: bool = True) -> Iterator[None]:
    """Redirect stdout to /dev/null when enabled."""

    if not enabled:
        yield
        return

    with open(os.devnull, "w") as f, redirect_stdout(f):
        yield


@contextmanager
def suppress_third_party_logging(
    enabled: bool = True,
    *,
    logger_names: Sequence[str] = DEFAULT_NOISY_LOGGERS,
    level: int = logging.WARNING,
    suppress_root: bool = True,
    suppress_handlers: bool = True,
    redirect_stderr: bool = True,
) -> Iterator[None]:
    """Suppress third-party logging (stdlib logging + low-level stderr writes).

    Intended for wrapping known-noisy initialization code paths.
    """

    if not enabled:
        yield
        return

    with suppress_loggers(
        logger_names,
        level=level,
        include_root=suppress_root,
        suppress_handlers=suppress_handlers,
        enabled=True,
    ):
        with suppress_stderr_fd(enabled=redirect_stderr):
            yield
