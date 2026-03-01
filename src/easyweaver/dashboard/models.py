import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class DashboardConfig:
    id: uuid.UUID
    user_id: str
    name: str
    source_id: str
    tables: list[dict] = field(default_factory=list)
    refresh_interval_minutes: int = 60
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_doc(cls, doc: dict) -> "DashboardConfig":
        return cls(
            id=uuid.UUID(doc["_id"]),
            user_id=doc["user_id"],
            name=doc["name"],
            source_id=doc["source_id"],
            tables=doc.get("tables", []),
            refresh_interval_minutes=doc.get("refresh_interval_minutes", 60),
            is_active=doc.get("is_active", True),
            created_at=doc.get("created_at", datetime.now(timezone.utc)),
            updated_at=doc.get("updated_at", datetime.now(timezone.utc)),
        )

    def to_doc(self) -> dict:
        return {
            "_id": str(self.id),
            "user_id": self.user_id,
            "name": self.name,
            "source_id": self.source_id,
            "tables": self.tables,
            "refresh_interval_minutes": self.refresh_interval_minutes,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class DataSnapshot:
    id: uuid.UUID
    config_id: str
    source_id: str
    table_name: str
    row_count: int = 0
    stats: dict[str, Any] = field(default_factory=dict)
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_doc(cls, doc: dict) -> "DataSnapshot":
        return cls(
            id=uuid.UUID(doc["_id"]),
            config_id=doc["config_id"],
            source_id=doc["source_id"],
            table_name=doc["table_name"],
            row_count=doc.get("row_count", 0),
            stats=doc.get("stats", {}),
            captured_at=doc.get("captured_at", datetime.now(timezone.utc)),
        )

    def to_doc(self) -> dict:
        return {
            "_id": str(self.id),
            "config_id": self.config_id,
            "source_id": self.source_id,
            "table_name": self.table_name,
            "row_count": self.row_count,
            "stats": self.stats,
            "captured_at": self.captured_at,
        }
