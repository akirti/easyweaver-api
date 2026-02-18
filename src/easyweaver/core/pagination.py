from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class PaginationParams(BaseModel):
    page: int = 1
    page_size: int = 50

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


@dataclass
class PaginatedResult(Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        return (self.total + self.page_size - 1) // self.page_size

    def to_dict(self, item_serializer=None) -> dict[str, Any]:
        items = [item_serializer(i) if item_serializer else i for i in self.items]
        return {
            "items": items,
            "total": self.total,
            "page": self.page,
            "page_size": self.page_size,
            "total_pages": self.total_pages,
        }
