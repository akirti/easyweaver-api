from abc import ABC, abstractmethod
from typing import Any

import polars as pl


class ResultStore(ABC):
    @abstractmethod
    async def store_result(self, run_id: str, df: pl.DataFrame, ttl: int = 3600) -> None:
        """Store a result DataFrame."""

    @abstractmethod
    async def get_result(self, run_id: str) -> pl.DataFrame | None:
        """Retrieve a stored result DataFrame."""

    @abstractmethod
    async def delete_result(self, run_id: str) -> None:
        """Delete a stored result."""
