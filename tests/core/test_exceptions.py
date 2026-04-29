"""Tests for easyweaver.core.exceptions"""
import pytest

from easyweaver.core.exceptions import (
    AuthenticationError,
    ConnectionTestError,
    EasyWeaverError,
    NotFoundError,
    ProcessExecutionError,
    QueryExecutionError,
    StorageError,
    ValidationError,
)


# ---------------------------------------------------------------------------
# EasyWeaverError (base)
# ---------------------------------------------------------------------------


class TestEasyWeaverError:
    def test_basic_instantiation(self):
        err = EasyWeaverError("something failed")
        assert err.message == "something failed"
        assert err.code == "INTERNAL_ERROR"
        assert err.details is None

    def test_custom_code(self):
        err = EasyWeaverError("msg", code="CUSTOM_CODE")
        assert err.code == "CUSTOM_CODE"

    def test_custom_details(self):
        err = EasyWeaverError("msg", details={"field": "value"})
        assert err.details == {"field": "value"}

    def test_is_exception(self):
        err = EasyWeaverError("msg")
        assert isinstance(err, Exception)

    def test_str_representation(self):
        err = EasyWeaverError("error message")
        assert str(err) == "error message"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(EasyWeaverError) as exc_info:
            raise EasyWeaverError("test", code="TEST_CODE", details={"x": 1})
        assert exc_info.value.code == "TEST_CODE"
        assert exc_info.value.details == {"x": 1}

    def test_details_can_be_list(self):
        err = EasyWeaverError("msg", details=[1, 2, 3])
        assert err.details == [1, 2, 3]

    def test_details_can_be_string(self):
        err = EasyWeaverError("msg", details="extra info")
        assert err.details == "extra info"


# ---------------------------------------------------------------------------
# NotFoundError
# ---------------------------------------------------------------------------


class TestNotFoundError:
    def test_message_format(self):
        err = NotFoundError("User", 42)
        assert err.message == "User not found: 42"

    def test_code(self):
        assert NotFoundError("X", 1).code == "NOT_FOUND"

    def test_details_resource(self):
        err = NotFoundError("Dataset", "abc-123")
        assert err.details["resource"] == "Dataset"

    def test_details_id_as_string(self):
        err = NotFoundError("Dataset", 999)
        assert err.details["id"] == "999"

    def test_is_easyweaver_error(self):
        assert isinstance(NotFoundError("R", 1), EasyWeaverError)

    def test_can_be_raised(self):
        with pytest.raises(NotFoundError):
            raise NotFoundError("Query", "q-1")

    def test_caught_as_easyweaver_error(self):
        with pytest.raises(EasyWeaverError):
            raise NotFoundError("Source", 7)


# ---------------------------------------------------------------------------
# ConnectionTestError
# ---------------------------------------------------------------------------


class TestConnectionTestError:
    def test_message(self):
        err = ConnectionTestError("timeout connecting to db")
        assert err.message == "timeout connecting to db"

    def test_code(self):
        assert ConnectionTestError("msg").code == "CONNECTION_TEST_FAILED"

    def test_details_none(self):
        assert ConnectionTestError("msg").details is None

    def test_is_easyweaver_error(self):
        assert isinstance(ConnectionTestError("x"), EasyWeaverError)


# ---------------------------------------------------------------------------
# QueryExecutionError
# ---------------------------------------------------------------------------


class TestQueryExecutionError:
    def test_message_and_code(self):
        err = QueryExecutionError("syntax error")
        assert err.message == "syntax error"
        assert err.code == "QUERY_EXECUTION_FAILED"

    def test_details_default_none(self):
        assert QueryExecutionError("msg").details is None

    def test_details_passed(self):
        err = QueryExecutionError("msg", details={"line": 10})
        assert err.details == {"line": 10}

    def test_is_easyweaver_error(self):
        assert isinstance(QueryExecutionError("x"), EasyWeaverError)


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------


class TestValidationError:
    def test_message_and_code(self):
        err = ValidationError("invalid field")
        assert err.message == "invalid field"
        assert err.code == "VALIDATION_ERROR"

    def test_details_default_none(self):
        assert ValidationError("msg").details is None

    def test_details_passed(self):
        err = ValidationError("msg", details={"field": "name"})
        assert err.details == {"field": "name"}

    def test_is_easyweaver_error(self):
        assert isinstance(ValidationError("x"), EasyWeaverError)


# ---------------------------------------------------------------------------
# AuthenticationError
# ---------------------------------------------------------------------------


class TestAuthenticationError:
    def test_default_message(self):
        err = AuthenticationError()
        assert err.message == "Invalid credentials"

    def test_custom_message(self):
        err = AuthenticationError("token expired")
        assert err.message == "token expired"

    def test_code(self):
        assert AuthenticationError().code == "AUTHENTICATION_FAILED"

    def test_is_easyweaver_error(self):
        assert isinstance(AuthenticationError(), EasyWeaverError)


# ---------------------------------------------------------------------------
# ProcessExecutionError
# ---------------------------------------------------------------------------


class TestProcessExecutionError:
    def test_message_and_code(self):
        err = ProcessExecutionError("process crashed")
        assert err.message == "process crashed"
        assert err.code == "PROCESS_EXECUTION_FAILED"

    def test_details_default_none(self):
        assert ProcessExecutionError("msg").details is None

    def test_details_passed(self):
        err = ProcessExecutionError("msg", details={"step": "transform"})
        assert err.details == {"step": "transform"}

    def test_is_easyweaver_error(self):
        assert isinstance(ProcessExecutionError("x"), EasyWeaverError)


# ---------------------------------------------------------------------------
# StorageError
# ---------------------------------------------------------------------------


class TestStorageError:
    def test_message_and_code(self):
        err = StorageError("disk full")
        assert err.message == "disk full"
        assert err.code == "STORAGE_ERROR"

    def test_details_default_none(self):
        assert StorageError("msg").details is None

    def test_details_passed(self):
        err = StorageError("msg", details={"backend": "redis"})
        assert err.details == {"backend": "redis"}

    def test_is_easyweaver_error(self):
        assert isinstance(StorageError("x"), EasyWeaverError)


# ---------------------------------------------------------------------------
# Exception hierarchy — caught as base class
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    @pytest.mark.parametrize("exc_cls,kwargs", [
        (NotFoundError, {"resource": "R", "id": 1}),
        (ConnectionTestError, {"message": "m"}),
        (QueryExecutionError, {"message": "m"}),
        (ValidationError, {"message": "m"}),
        (AuthenticationError, {}),
        (ProcessExecutionError, {"message": "m"}),
        (StorageError, {"message": "m"}),
    ])
    def test_all_subclasses_caught_as_easyweaver_error(self, exc_cls, kwargs):
        with pytest.raises(EasyWeaverError):
            raise exc_cls(**kwargs)
