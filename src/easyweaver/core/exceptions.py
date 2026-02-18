from typing import Any


class EasyWeaverError(Exception):
    """Base exception for all EasyWeaver errors."""

    def __init__(self, message: str, code: str = "INTERNAL_ERROR", details: Any = None):
        self.message = message
        self.code = code
        self.details = details
        super().__init__(message)


class NotFoundError(EasyWeaverError):
    def __init__(self, resource: str, id: Any):
        super().__init__(
            message=f"{resource} not found: {id}",
            code="NOT_FOUND",
            details={"resource": resource, "id": str(id)},
        )


class ConnectionTestError(EasyWeaverError):
    def __init__(self, message: str):
        super().__init__(message=message, code="CONNECTION_TEST_FAILED")


class QueryExecutionError(EasyWeaverError):
    def __init__(self, message: str, details: Any = None):
        super().__init__(message=message, code="QUERY_EXECUTION_FAILED", details=details)


class ValidationError(EasyWeaverError):
    def __init__(self, message: str, details: Any = None):
        super().__init__(message=message, code="VALIDATION_ERROR", details=details)


class AuthenticationError(EasyWeaverError):
    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(message=message, code="AUTHENTICATION_FAILED")
