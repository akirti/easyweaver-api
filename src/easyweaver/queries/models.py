import uuid
from datetime import datetime

from sqlalchemy import String, Integer, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from easyweaver.sources.models import Base


class QueryRun(Base):
    __tablename__ = "query_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    config: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
