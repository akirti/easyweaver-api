import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ProcessConfiguration:
    id: uuid.UUID
    user_id: str
    name: str
    description: str = ""
    version: int = 1
    config: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    save_destination: str = "redis"
    gcp_path: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_doc(cls, doc: dict) -> "ProcessConfiguration":
        """Create a ProcessConfiguration from a MongoDB document."""
        return cls(
            id=uuid.UUID(doc["_id"]),
            user_id=doc["user_id"],
            name=doc["name"],
            description=doc.get("description", ""),
            version=doc.get("version", 1),
            config=doc.get("config", {}),
            params=doc.get("params", {}),
            save_destination=doc.get("save_destination", "redis"),
            gcp_path=doc.get("gcp_path", ""),
            tags=doc.get("tags", []),
            created_at=doc.get("created_at", datetime.now(timezone.utc)),
            updated_at=doc.get("updated_at", datetime.now(timezone.utc)),
        )

    def to_doc(self) -> dict:
        """Convert to a MongoDB document."""
        return {
            "_id": str(self.id),
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "config": self.config,
            "params": self.params,
            "save_destination": self.save_destination,
            "gcp_path": self.gcp_path,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ProcessRun:
    id: uuid.UUID
    process_id: str
    user_id: str
    param_values: dict = field(default_factory=dict)
    status: str = "pending"
    row_count: int | None = None
    error: str | None = None
    result_gcp_path: str = ""
    result_run_id: str = ""
    progress: dict | None = None
    control: dict | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_doc(cls, doc: dict) -> "ProcessRun":
        """Create a ProcessRun from a MongoDB document."""
        return cls(
            id=uuid.UUID(doc["_id"]),
            process_id=doc["process_id"],
            user_id=doc["user_id"],
            param_values=doc.get("param_values", {}),
            status=doc.get("status", "pending"),
            row_count=doc.get("row_count"),
            error=doc.get("error"),
            result_gcp_path=doc.get("result_gcp_path", ""),
            result_run_id=doc.get("result_run_id", ""),
            progress=doc.get("progress"),
            control=doc.get("control"),
            created_at=doc.get("created_at", datetime.now(timezone.utc)),
            updated_at=doc.get("updated_at", datetime.now(timezone.utc)),
        )

    def to_doc(self) -> dict:
        """Convert to a MongoDB document."""
        doc = {
            "_id": str(self.id),
            "process_id": self.process_id,
            "user_id": self.user_id,
            "param_values": self.param_values,
            "status": self.status,
            "row_count": self.row_count,
            "error": self.error,
            "result_gcp_path": self.result_gcp_path,
            "result_run_id": self.result_run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.progress is not None:
            doc["progress"] = self.progress
        if self.control is not None:
            doc["control"] = self.control
        return doc
