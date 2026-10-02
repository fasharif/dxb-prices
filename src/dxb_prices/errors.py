"""Errors whose message is written for the person running a command.

The command line prints these without a traceback and exits with status 1.
Each subclass also keeps the built-in base it had before (``ValueError``,
``FileNotFoundError`` or ``RuntimeError``), so callers that catch those still
work. This module imports nothing, so the command line can use it without
loading the heavier modules.
"""

from __future__ import annotations


class UserFacingError(Exception):
    """Something the user can fix: missing data, a bad argument, an unusable download."""


class MissingInputError(UserFacingError, FileNotFoundError):
    """A file or directory the command needs does not exist yet."""
