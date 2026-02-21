import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class User:
    id: uuid.UUID
    email: str
    hashed_password: str
    display_name: str
    role: str = "admin"
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_doc(cls, doc: dict) -> "User":
        """Create a User from a MongoDB document."""
        return cls(
            id=uuid.UUID(doc["_id"]),
            email=doc["email"],
            hashed_password=doc["hashed_password"],
            display_name=doc["display_name"],
            role=doc.get("role", "admin"),
            is_active=doc.get("is_active", True),
            created_at=doc.get("created_at", datetime.now(timezone.utc)),
            updated_at=doc.get("updated_at", datetime.now(timezone.utc)),
        )

    def to_doc(self) -> dict:
        """Convert to a MongoDB document."""
        return {
            "_id": str(self.id),
            "email": self.email,
            "hashed_password": self.hashed_password,
            "display_name": self.display_name,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
