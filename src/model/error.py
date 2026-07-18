class UnsupportedDatasetError(ValueError):
    """Raised when dataset year is not supported"""

    pass

class UnavailableDatasetError(ValueError):
    """Raised when a requested resource is not available"""

    pass