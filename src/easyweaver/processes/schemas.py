from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from easyweaver.queries.schemas import DerivedColumnSpec, DistinctSpec, GroupBySpec, SortSpec


class ParamDefinition(BaseModel):
    type: Literal["string", "number", "boolean", "date", "datetime"]
    default: Any = None
    label: str = ""


class ProcessFilterConfig(BaseModel):
    column: str
    operator: str
    value: Any = None
    value2: Any = None


class BindingMapping(BaseModel):
    source_column: str
    target_column: str


class ProcessQueryBinding(BaseModel):
    source_dataset: str  # e.g., "schema1.orders"
    mode: Literal["distinct", "row_pair"]
    mappings: list[BindingMapping]


class ProcessQueryConfig(BaseModel):
    source_id: str
    table: str
    columns: list[str] | None = None
    filters: list[ProcessFilterConfig] = Field(default_factory=list)
    filter_logic: Literal["and", "or"] = "and"
    bindings: list[ProcessQueryBinding] = Field(default_factory=list)
    # Embedded source details for self-sufficient configs (v2+)
    source_name: str | None = None
    source_type: str | None = None
    encrypted_credentials: str | None = None


class ProcessLogicStep(BaseModel):
    key: str
    type: Literal["join"]
    left: str
    right: str
    join_type: Literal["inner", "left", "right", "outer"] = "inner"
    left_on: list[str]
    right_on: list[str]
    select_columns: list[str] | None = None


class ProcessOperations(BaseModel):
    filters: list[ProcessFilterConfig] = Field(default_factory=list)
    filter_logic: Literal["and", "or"] = "and"
    group_by: GroupBySpec | None = None
    distinct: DistinctSpec | None = None
    sorts: list[SortSpec] = Field(default_factory=list)


class ProcessTransformation(BaseModel):
    column: str
    type: str
    new_name: str | None = None
    date_format: str | None = None
    decimals: int | None = None
    target_type: str | None = None


class ProcessConfig(BaseModel):
    queries: dict[str, dict[str, ProcessQueryConfig]]
    logics: list[ProcessLogicStep] = Field(default_factory=list)
    derived_columns: list[DerivedColumnSpec] = Field(default_factory=list)
    operations: ProcessOperations | None = None
    transformations: list[ProcessTransformation] = Field(default_factory=list)
    config_version: int = 1


class ProcessConfigurationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    params: dict[str, ParamDefinition] = Field(default_factory=dict)
    config: ProcessConfig
    save_destination: Literal["redis", "gcp", "both"] = "redis"
    gcp_path: str = ""
    tags: list[str] = Field(default_factory=list)


class ProcessConfigurationUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    params: dict[str, ParamDefinition] | None = None
    config: ProcessConfig | None = None
    save_destination: str | None = None
    gcp_path: str | None = None
    tags: list[str] | None = None


class ProcessConfigurationResponse(BaseModel):
    id: str
    user_id: str
    name: str
    description: str
    version: int
    config: ProcessConfig
    params: dict[str, ParamDefinition]
    save_destination: str
    gcp_path: str
    tags: list[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProcessRunRequest(BaseModel):
    param_values: dict[str, Any] = Field(default_factory=dict)
    max_rows: int = Field(default=1000, gt=0, description="Maximum rows to return (mandatory, must be ≤ system max_result_rows)")
    save_results_to_gcp: bool = False
    config_source: Literal["auto", "mongodb", "gcp"] = "auto"


class ProcessRunResponse(BaseModel):
    id: str
    process_id: str
    status: str
    param_values: dict
    row_count: int | None = None
    error: str | None = None
    result_gcp_path: str = ""
    result_run_id: str = ""
    progress: dict | None = None  # Progress state (phases, datasets, operations)
    control: dict | None = None   # Control state (paused, batch_size_override, etc.)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProcessRunHistoryResponse(BaseModel):
    runs: list[ProcessRunResponse]
    total: int
