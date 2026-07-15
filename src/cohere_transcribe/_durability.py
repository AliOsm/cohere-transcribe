"""Small cross-cutting helpers for durable filesystem publication."""

from __future__ import annotations

import errno
import os
from collections.abc import Iterable
from pathlib import Path

_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = {
    errno.EACCES,
    errno.EBADF,
    errno.EINVAL,
    errno.EISDIR,
    errno.ENOTSUP,
    errno.EPERM,
}


def fsync_directories(directories: Iterable[Path]) -> None:
    """Persist directory entries when supported and propagate real I/O failures."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    for directory in dict.fromkeys(path.resolve() for path in directories):
        try:
            descriptor = os.open(directory, flags)
        except OSError as exc:
            if exc.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                continue
            raise
        failed = False
        try:
            try:
                os.fsync(descriptor)
            except OSError as exc:
                if exc.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                    raise
        except BaseException:
            failed = True
            raise
        finally:
            try:
                os.close(descriptor)
            except OSError:
                if not failed:
                    raise
