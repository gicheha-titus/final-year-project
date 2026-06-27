"""Custom exception hierarchy for cbc_xai.

Distinct exception types let callers catch exactly what they care about
without swallowing unrelated failures.  All exceptions carry a plain,
teacher-readable message alongside any technical detail so that the UI
can surface an appropriate error without having to parse exception text.

The hierarchy is intentionally shallow — three levels is enough for a
single-application, single-process desktop tool.
"""

from __future__ import annotations


class CbcXaiError(Exception):
    """Base class for all cbc_xai application errors."""


class IngestionError(CbcXaiError):
    """Raised when a CSV file fails structural or value validation.

    Carries ``row_errors`` so callers can display every bad row at once
    rather than making the user fix one problem per import attempt.
    """

    def __init__(self, message: str, row_errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.row_errors: list[str] = row_errors or []


class ModelError(CbcXaiError):
    """Raised when model training, loading, or prediction fails."""


class ReportError(CbcXaiError):
    """Raised when PDF report generation fails."""


class StorageError(CbcXaiError):
    """Raised when a database read or write operation fails."""


class AuthenticationError(CbcXaiError):
    """Raised when a password verification or account operation fails.

    Note: ``authenticate_user`` intentionally does not raise this —
    returning ``None`` on failed login avoids leaking account-existence
    information.  This exception is used for *unexpected* auth failures
    (e.g. corrupted hash format).
    """
