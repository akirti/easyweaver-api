from abc import ABC, abstractmethod
from typing import Any


class BaseConnector(ABC):
    """Abstract base for all data source connectors."""

    def __init__(self, credentials: dict[str, Any]):
        self.credentials = credentials

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection and verify it works."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection."""

    @abstractmethod
    async def test_connection(self) -> dict[str, Any]:
        """Test connection, return {success, latency_ms, message}."""

    @abstractmethod
    async def get_schema(self) -> list[dict[str, Any]]:
        """Return list of tables/collections with column info."""

    @abstractmethod
    async def get_table_schema(self, table_name: str) -> dict[str, Any]:
        """Return detailed schema for a single table/collection."""

    @abstractmethod
    async def preview_table(self, table_name: str, limit: int = 50) -> dict[str, Any]:
        """Return sample rows from a table/collection."""

    @abstractmethod
    async def execute_query(
        self,
        table: str,
        columns: list[str] | None = None,
        filters: list[dict] | None = None,
        sort: list[dict] | None = None,
        limit: int | None = None,
        filter_logic: str = "and",
    ) -> list[dict[str, Any]]:
        """Execute a query and return rows as dicts."""

    @property
    def supports_batching(self) -> bool:
        """Whether this connector supports batched fetching.

        Connectors that support batching (SQL, MongoDB) should override
        this to return True and implement execute_query_batched().
        File and REST connectors return False (the default) and use
        single-shot execute_query() instead.
        """
        return False

    async def execute_query_batched(
        self,
        table: str,
        columns: list[str] | None = None,
        filters: list[dict] | None = None,
        filter_logic: str = "and",
        batch_size: int = 10_000,
        offset: int = 0,
        last_key: Any = None,
    ) -> tuple[list[dict[str, Any]], bool, Any]:
        """Fetch a batch of rows with keyset or OFFSET/LIMIT pagination.

        Uses PK-based ordering internally for stable pagination.
        User-specified sort is applied post-fetch in Polars.

        Args:
            table: Table or collection name.
            columns: Columns to select (None = all).
            filters: Filter conditions.
            filter_logic: How to combine filters ("and" / "or").
            batch_size: Number of rows to fetch in this batch.
            offset: Row offset for pagination.
            last_key: Last PK value from the previous batch for keyset
                pagination.  When provided, the connector uses
                ``WHERE pk > last_key`` instead of OFFSET.  When *None*
                and *offset* > 0, falls back to OFFSET pagination.

        Returns:
            Tuple of (rows, has_more, last_key_for_next_batch) where
            *has_more* indicates whether additional rows exist beyond
            this batch and *last_key_for_next_batch* is the PK value
            of the last returned row (or *None* if empty).

        Raises:
            NotImplementedError: If the connector does not support batching.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support batched queries. "
            f"Check supports_batching before calling execute_query_batched()."
        )

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
