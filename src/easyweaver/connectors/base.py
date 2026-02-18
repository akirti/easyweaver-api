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
    ) -> list[dict[str, Any]]:
        """Execute a query and return rows as dicts."""

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
