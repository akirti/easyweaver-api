"""Unit tests for BaseConnector batching interface."""

import pytest

from easyweaver.connectors.base import BaseConnector


class _StubConnector(BaseConnector):
    """Minimal concrete subclass that implements all abstract methods."""

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def test_connection(self):
        return {"success": True, "latency_ms": 0, "message": "ok"}

    async def get_schema(self):
        return []

    async def get_table_schema(self, table_name):
        return {}

    async def preview_table(self, table_name, limit=50):
        return {}

    async def execute_query(self, table, columns=None, filters=None, sort=None, limit=None, filter_logic="and"):
        return []


class _BatchableStubConnector(_StubConnector):
    """Stub that declares batching support."""

    @property
    def supports_batching(self):
        return True

    async def execute_query_batched(self, table, columns=None, filters=None, filter_logic="and", batch_size=10_000, offset=0, last_key=None):
        return [{"id": 1}], False, 1


class TestSupportsBatching:
    def test_default_is_false(self):
        connector = _StubConnector(credentials={})
        assert connector.supports_batching is False

    def test_overridden_to_true(self):
        connector = _BatchableStubConnector(credentials={})
        assert connector.supports_batching is True


class TestExecuteQueryBatched:
    @pytest.mark.anyio
    async def test_raises_not_implemented_by_default(self):
        connector = _StubConnector(credentials={})
        with pytest.raises(NotImplementedError, match="does not support batched queries"):
            await connector.execute_query_batched(table="test_table")

    @pytest.mark.anyio
    async def test_error_message_includes_class_name(self):
        connector = _StubConnector(credentials={})
        with pytest.raises(NotImplementedError, match="_StubConnector"):
            await connector.execute_query_batched(table="t")

    @pytest.mark.anyio
    async def test_batchable_connector_returns_3_tuple(self):
        connector = _BatchableStubConnector(credentials={})
        rows, has_more, last_pk = await connector.execute_query_batched(table="t")
        assert rows == [{"id": 1}]
        assert has_more is False
        assert last_pk == 1


class TestSettings:
    def test_batch_settings_fields_exist_with_correct_defaults(self):
        """Verify batch setting fields are declared with correct defaults.

        We read the source file and parse the class to avoid module-level
        Settings() instantiation which can fail due to .env file contents.
        """
        import ast
        from pathlib import Path

        settings_path = Path(__file__).resolve().parents[2] / "src" / "easyweaver" / "settings.py"
        tree = ast.parse(settings_path.read_text())

        # Find the Settings class
        settings_cls = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Settings":
                settings_cls = node
                break
        assert settings_cls is not None, "Settings class not found"

        # Extract annotated assignments with defaults
        field_defaults = {}
        for item in settings_cls.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value is not None:
                try:
                    field_defaults[item.target.id] = ast.literal_eval(item.value)
                except (ValueError, KeyError):
                    pass  # Skip fields with dynamic defaults (e.g. _get() calls)

        assert field_defaults["batch_default_target_seconds"] == 10.0
        assert field_defaults["batch_min_size"] == 1_000
        assert field_defaults["batch_max_size"] == 100_000
        assert field_defaults["batch_intermediate_ttl_seconds"] == 3600
