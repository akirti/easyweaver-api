from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class FilterCondition(BaseModel):
    column: str
    operator: Literal["eq", "neq", "gt", "lt", "gte", "lte", "like", "is_null", "is_not_null"]
    value: Any = None


class SortSpec(BaseModel):
    column: str
    direction: Literal["asc", "desc"] = "asc"


class QuerySourceConfig(BaseModel):
    source_id: uuid.UUID
    table: str
    columns: list[str] | None = None
    filters: list[FilterCondition] = Field(default_factory=list)


class JoinConfig(BaseModel):
    join_type: Literal["inner", "left", "right", "outer"] = "inner"
    left_on: str
    right_on: str


class QueryRequest(BaseModel):
    type: Literal["single", "join"] = "single"
    left: QuerySourceConfig
    right: QuerySourceConfig | None = None
    join: JoinConfig | None = None
    sort: list[SortSpec] = Field(default_factory=list)
    page: int = 1
    page_size: int = 50


class QueryRunResponse(BaseModel):
    id: uuid.UUID
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    row_count: int | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class QueryResultsResponse(BaseModel):
    columns: list[dict[str, str]]
    rows: list[dict[str, Any]]
    total: int
    page: int
    page_size: int
    total_pages: int
