class UnsupportedDatasetError(ValueError):
    """Raised when dataset year is not supported"""

    pass


class UnavailableDatasetError(ValueError):
    """Raised when a requested resource is not available"""

    pass


class TrainingError(ValueError):
    """Raised when a training error occurs"""

    pass


class DatasetLoadingError(ValueError):
    """Raised when a dataset error occurs"""

    pass


class DataPreprocessingError(ValueError):
    """Raised when a data preprocessing error occurs"""

    pass


class ExportError(AssertionError):
    """Raised when an export error occurs"""

    pass


class LoadingConfigFailed(ValueError):
    """Raised when a loading error occurs"""

    pass


class PipelineError(ValueError):
    """Raised when a pipeline error occurs"""

    pass


class ReportError(ValueError):
    """Raised when a report cannot be built with the given settings"""

    pass


class SettingsError(Exception):
    pass


class ApiError(Exception):
    status_code: int = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnauthorizedError(ApiError):
    status_code = 401


class InvalidNameError(ApiError):
    status_code = 400


class InvalidParameterError(ApiError):
    status_code = 422


class ResourceNotFoundError(ApiError):
    status_code = 404


class JobConflictError(ApiError):
    status_code = 409


class PayloadTooLargeError(ApiError):
    status_code = 413


class StorageError(ApiError):
    status_code = 507
