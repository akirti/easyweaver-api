from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TableMonitorConfig(BaseModel):
    table_name: str
    timestamp_column: str | None = None
    modified_by_column: str | None = None


class DashboardConfigCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    source_id: str
    tables: list[TableMonitorConfig] = Field(min_length=1)
    refresh_interval_minutes: int = Field(default=60, ge=1, le=1440)


class DashboardConfigUpdate(BaseModel):
    name: str | None = None
    tables: list[TableMonitorConfig] | None = None
    refresh_interval_minutes: int | None = Field(default=None, ge=1, le=1440)
    is_active: bool | None = None


class DashboardConfigResponse(BaseModel):
    id: str
    user_id: str
    name: str
    source_id: str
    tables: list[TableMonitorConfig]
    refresh_interval_minutes: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TableStats(BaseModel):
    table_name: str
    current_row_count: int = 0
    changes_1h: int | None = None
    changes_3h: int | None = None
    changes_24h: int | None = None
    last_modified_at: str | None = None
    last_modified_by: str | None = None
    column_count: int = 0
    size_bytes: int | None = None


class DashboardStatsResponse(BaseModel):
    config_id: str
    source_name: str
    source_type: str
    tables: list[TableStats]
    captured_at: datetime
    connection_healthy: bool


class SnapshotRecord(BaseModel):
    id: str
    config_id: str
    source_id: str
    table_name: str
    row_count: int
    stats: dict[str, Any] = Field(default_factory=dict)
    captured_at: datetime

    model_config = {"from_attributes": True}


class SnapshotHistoryResponse(BaseModel):
    snapshots: list[SnapshotRecord]
    total: int
