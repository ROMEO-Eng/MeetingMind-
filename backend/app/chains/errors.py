"""Model runtime failures surfaced to API and UI callers."""


class ModelUnavailableError(RuntimeError):
    """Raised when the local model cannot run in the current environment."""
