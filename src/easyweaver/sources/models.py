import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class DataSource:
    id: uuid.UUID
    name: str
    source_type: str
    encrypted_credentials: str
    metadata_: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_doc(cls, doc: dict) -> "DataSource":
        """Create a DataSource from a MongoDB document."""
        return cls(
            id=uuid.UUID(doc["_id"]),
            name=doc["name"],
            source_type=doc["source_type"],
            encrypted_credentials=doc["encrypted_credentials"],
            metadata_=doc.get("metadata"),
            created_at=doc.get("created_at", datetime.now(timezone.utc)),
            updated_at=doc.get("updated_at", datetime.now(timezone.utc)),
        )

    def to_doc(self) -> dict:
        """Convert to a MongoDB document."""
        return {
            "_id": str(self.id),
            "name": self.name,
            "source_type": self.source_type,
            "encrypted_credentials": self.encrypted_credentials,
            "metadata": self.metadata_,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
