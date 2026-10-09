"""Model runtime failures surfaced to API and UI callers."""


class ModelUnavailableError(RuntimeError):
    """Raised when the configured remote inference model cannot run."""


class RemoteInferenceError(ModelUnavailableError):
    """Raised when the configured remote inference service cannot be used."""


class RemoteResponseError(RuntimeError):
    """Raised when the remote service returns an invalid response contract."""
