from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ValueFromDataset(BaseModel):
    """Reference to a column in another dataset's results, resolved at execution time."""
    run_id: uuid.UUID
    column: str


class FilterCondition(BaseModel):
    column: str
    operator: Literal["eq", "neq", "gt", "lt", "gte", "lte", "like", "is_null", "is_not_null", "in", "not_in", "between"]
    value: Any = None
    value2: Any = None
    value_from: ValueFromDataset | None = None

    @model_validator(mode="after")
    def check_value_source(self):
        if self.value_from is not None and self.operator not in ("in", "not_in"):
            raise ValueError("value_from is only supported with 'in' or 'not_in' operators")
        if self.operator in ("in", "not_in") and self.value is None and self.value_from is None:
            raise ValueError("'in'/'not_in' requires either 'value' (list) or 'value_from'")
        # Normalize comma-separated string to list for in/not_in
        if self.operator in ("in", "not_in") and isinstance(self.value, str):
            self.value = [v.strip() for v in self.value.split(",") if v.strip()]
        if self.operator == "between" and (self.value is None or self.value2 is None):
            raise ValueError("'between' requires both 'value' and 'value2'")
        return self


class SortSpec(BaseModel):
    column: str
    direction: Literal["asc", "desc"] = "asc"


class TransformSpec(BaseModel):
    column: str
    type: Literal[
        "rename", "format_date", "round", "uppercase", "lowercase", "trim", "cast",
        "strip_leading_zeros", "pad_left", "replace", "substring",
    ]
    new_name: str | None = None
    date_format: str | None = None
    decimals: int | None = None
    target_type: Literal["string", "integer", "float", "boolean", "date", "datetime"] | None = None
    # pad_left params
    pad_char: str | None = None
    pad_length: int | None = None
    # replace params
    find_str: str | None = None
    replace_str: str | None = None
    # substring params
    start: int | None = None
    length: int | None = None

    @model_validator(mode="after")
    def check_params(self):
        if self.type == "rename" and not self.new_name:
            raise ValueError("'rename' transform requires 'new_name'")
        if self.type == "format_date" and not self.date_format:
            raise ValueError("'format_date' transform requires 'date_format'")
        if self.type == "round" and self.decimals is None:
            raise ValueError("'round' transform requires 'decimals'")
        if self.type == "cast" and not self.target_type:
            raise ValueError("'cast' transform requires 'target_type'")
        if self.type == "pad_left" and (not self.pad_length or self.pad_length < 1):
            raise ValueError("'pad_left' transform requires 'pad_length' >= 1")
        if self.type == "replace" and self.find_str is None:
            raise ValueError("'replace' transform requires 'find_str'")
        return self


class BindingMapping(BaseModel):
    source_column: str
    target_column: str


class DataBindingSpec(BaseModel):
    source_run_id: uuid.UUID
    mode: Literal["distinct", "row_pair"]
    mappings: list[BindingMapping]

    @model_validator(mode="after")
    def check_mappings(self):
        if len(self.mappings) == 0:
            raise ValueError("At least one mapping is required")
        return self


class DerivedColumnSpec(BaseModel):
    name: str
    expression_type: Literal["concat", "math", "date_part", "conditional", "literal"]

    # For concat
    columns: list[str] | None = None
    separator: str = ""

    # For math
    expression: str | None = None

    # For date_part
    source_column: str | None = None
    part: Literal[
        "year", "month", "day", "hour", "minute", "second", "day_of_week", "quarter"
    ] | None = None

    # For conditional
    condition_column: str | None = None
    condition_operator: str | None = None
    condition_value: Any = None
    then_value: Any = None
    else_value: Any = None

    # For literal
    value: Any = None


class AggregationSpec(BaseModel):
    column: str
    function: Literal["count", "sum", "avg", "min", "max", "count_distinct"]
    alias: str | None = None


class GroupBySpec(BaseModel):
    group_columns: list[str]
    aggregations: list[AggregationSpec]


class DistinctSpec(BaseModel):
    enabled: bool = True
    columns: list[str] | None = None
    keep: Literal["first", "last", "any", "none"] = "first"


class QuerySourceConfig(BaseModel):
    source_id: uuid.UUID
    table: str
    columns: list[str] | None = None
    filters: list[FilterCondition] = Field(default_factory=list)
    filter_logic: Literal["and", "or"] = "and"


class JoinConfig(BaseModel):
    join_type: Literal["inner", "left", "right", "outer"] = "inner"
    left_on: str | list[str]
    right_on: str | list[str]

    @model_validator(mode="after")
    def check_join_keys(self):
        left = self.left_on if isinstance(self.left_on, list) else [self.left_on]
        right = self.right_on if isinstance(self.right_on, list) else [self.right_on]
        if len(left) != len(right):
            raise ValueError("left_on and right_on must have the same number of columns")
        if len(left) == 0:
            raise ValueError("At least one join key pair is required")
        return self


class QueryRequest(BaseModel):
    type: Literal["single", "join"] = "single"
    left: QuerySourceConfig
    right: QuerySourceConfig | None = None
    join: JoinConfig | None = None
    sort: list[SortSpec] = Field(default_factory=list)
    transforms: list[TransformSpec] = Field(default_factory=list)
    bindings: list[DataBindingSpec] = Field(default_factory=list)
    group_by: GroupBySpec | None = None
    distinct: DistinctSpec | None = None
    page: int = 1
    page_size: int = 50


class JoinResultsRequest(BaseModel):
    """Join two previously-executed result sets by their run IDs."""
    left_run_id: uuid.UUID
    right_run_id: uuid.UUID
    join: JoinConfig
    select_columns: list[str] | None = None
    derived_columns: list[DerivedColumnSpec] = Field(default_factory=list)
    filters: list[FilterCondition] = Field(default_factory=list)
    filter_logic: Literal["and", "or"] = "and"
    group_by: GroupBySpec | None = None
    distinct: DistinctSpec | None = None
    sort: list[SortSpec] = Field(default_factory=list)
    transforms: list[TransformSpec] = Field(default_factory=list)
    bindings: list[DataBindingSpec] = Field(default_factory=list)


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
