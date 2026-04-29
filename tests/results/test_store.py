"""Tests for easyweaver.results.store (ResultStore abstract base class)"""
import inspect

import polars as pl
import pytest

from easyweaver.results.store import ResultStore


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class TestResultStoreAbstractInterface:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ResultStore()  # type: ignore[abstract]

    def test_has_store_result_abstract_method(self):
        assert hasattr(ResultStore, "store_result")
        assert getattr(ResultStore.store_result, "__isabstractmethod__", False)

    def test_has_get_result_abstract_method(self):
        assert hasattr(ResultStore, "get_result")
        assert getattr(ResultStore.get_result, "__isabstractmethod__", False)

    def test_has_delete_result_abstract_method(self):
        assert hasattr(ResultStore, "delete_result")
        assert getattr(ResultStore.delete_result, "__isabstractmethod__", False)

    def test_store_result_signature(self):
        sig = inspect.signature(ResultStore.store_result)
        params = list(sig.parameters.keys())
        assert "run_id" in params
        assert "df" in params
        assert "ttl" in params

    def test_get_result_signature(self):
        sig = inspect.signature(ResultStore.get_result)
        params = list(sig.parameters.keys())
        assert "run_id" in params

    def test_delete_result_signature(self):
        sig = inspect.signature(ResultStore.delete_result)
        params = list(sig.parameters.keys())
        assert "run_id" in params

    def test_store_result_ttl_has_default(self):
        sig = inspect.signature(ResultStore.store_result)
        ttl_param = sig.parameters["ttl"]
        assert ttl_param.default != inspect.Parameter.empty

    def test_is_abc(self):
        from abc import ABC
        assert issubclass(ResultStore, ABC)


# ---------------------------------------------------------------------------
# Concrete implementation satisfies the interface
# ---------------------------------------------------------------------------


class ConcreteStore(ResultStore):
    """Minimal in-memory implementation for interface verification."""

    def __init__(self):
        self._store: dict[str, pl.DataFrame] = {}

    async def store_result(self, run_id: str, df: pl.DataFrame, ttl: int = 3600) -> None:
        self._store[run_id] = df

    async def get_result(self, run_id: str) -> pl.DataFrame | None:
        return self._store.get(run_id)

    async def delete_result(self, run_id: str) -> None:
        self._store.pop(run_id, None)


class TestConcreteStoreImplementation:
    def test_concrete_can_be_instantiated(self):
        store = ConcreteStore()
        assert store is not None

    def test_concrete_is_result_store(self):
        store = ConcreteStore()
        assert isinstance(store, ResultStore)

    async def test_store_and_get_roundtrip(self):
        store = ConcreteStore()
        df = pl.DataFrame({"x": [1, 2, 3]})
        await store.store_result("run-1", df)
        result = await store.get_result("run-1")
        assert result is not None
        assert result.to_dicts() == df.to_dicts()

    async def test_get_missing_returns_none(self):
        store = ConcreteStore()
        result = await store.get_result("nonexistent")
        assert result is None

    async def test_delete_removes_result(self):
        store = ConcreteStore()
        df = pl.DataFrame({"a": [10]})
        await store.store_result("run-del", df)
        await store.delete_result("run-del")
        result = await store.get_result("run-del")
        assert result is None

    async def test_delete_nonexistent_does_not_raise(self):
        store = ConcreteStore()
        await store.delete_result("ghost-run")  # should not raise

    async def test_multiple_run_ids_independent(self):
        store = ConcreteStore()
        df1 = pl.DataFrame({"n": [1]})
        df2 = pl.DataFrame({"n": [2]})
        await store.store_result("run-a", df1)
        await store.store_result("run-b", df2)
        assert (await store.get_result("run-a")).to_dicts() == df1.to_dicts()
        assert (await store.get_result("run-b")).to_dicts() == df2.to_dicts()

    async def test_overwrite_run_id(self):
        store = ConcreteStore()
        df_old = pl.DataFrame({"v": [1]})
        df_new = pl.DataFrame({"v": [99]})
        await store.store_result("run-ow", df_old)
        await store.store_result("run-ow", df_new)
        result = await store.get_result("run-ow")
        assert result.to_dicts() == df_new.to_dicts()


# ---------------------------------------------------------------------------
# Incomplete implementation raises TypeError
# ---------------------------------------------------------------------------


class TestIncompleteImplementation:
    def test_missing_all_methods_raises(self):
        class BadStore(ResultStore):
            pass

        with pytest.raises(TypeError):
            BadStore()

    def test_missing_one_method_raises(self):
        class PartialStore(ResultStore):
            async def store_result(self, run_id, df, ttl=3600):
                pass

            async def get_result(self, run_id):
                return None

            # missing delete_result

        with pytest.raises(TypeError):
            PartialStore()
