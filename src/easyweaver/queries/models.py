import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class QueryRun:
    id: uuid.UUID
    config: str
    status: str = "pending"
    row_count: int | None = None
    error: str | None = None
    progress: dict | None = None
    control: dict | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_doc(cls, doc: dict) -> "QueryRun":
        """Create a QueryRun from a MongoDB document."""
        return cls(
            id=uuid.UUID(doc["_id"]),
            config=doc["config"],
            status=doc.get("status", "pending"),
            row_count=doc.get("row_count"),
            error=doc.get("error"),
            progress=doc.get("progress"),
            control=doc.get("control"),
            created_at=doc.get("created_at", datetime.now(timezone.utc)),
            updated_at=doc.get("updated_at", datetime.now(timezone.utc)),
        )

    def to_doc(self) -> dict:
        """Convert to a MongoDB document."""
        return {
            "_id": str(self.id),
            "config": self.config,
            "status": self.status,
            "row_count": self.row_count,
            "error": self.error,
            "progress": self.progress,
            "control": self.control,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
