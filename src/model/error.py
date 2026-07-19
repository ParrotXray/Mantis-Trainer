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
