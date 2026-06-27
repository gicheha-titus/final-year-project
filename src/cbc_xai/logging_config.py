"""Application-wide logging configuration.

Configures a rotating file handler that writes structured log lines to
``APP_LOG_PATH`` in the artifacts directory.  The application logger is
named ``cbc_xai`` — every module should use::

    import logging
    log = logging.getLogger(__name__)

rather than creating module-specific configuration.

Level policy:
  - DEBUG   developer detail: feature values, model selection steps
  - INFO    normal operation: import count, prediction result, report path
  - WARNING operational concern: re-hashing a legacy password, missing file
  - ERROR   recoverable failure: validation error, DB write failed
  - CRITICAL unrecoverable failure: cannot open database, no model bundle

The rotating handler keeps at most three 5 MB log files on disk — cheap
enough for a 4 GB Windows desktop and sufficient to diagnose a session
that happened yesterday.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def configure_logging(log_path: Path) -> None:
    """Attach a rotating file handler to the root ``cbc_xai`` logger.

    Safe to call multiple times — guard prevents duplicate handlers from
    accumulating across test runs or re-import.
    """
    app_logger = logging.getLogger("cbc_xai")

    # Idempotent: only configure if no handlers exist yet.
    if app_logger.handlers:
        return

    app_logger.setLevel(logging.DEBUG)

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    app_logger.addHandler(handler)
