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
