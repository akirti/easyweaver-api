from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class PostgresCredentials(BaseModel):
    type: Literal["postgres"] = "postgres"
    host: str
    port: int = 5432
    database: str
    user: str
    password: str


class MongoCredentials(BaseModel):
    type: Literal["mongodb"] = "mongodb"
    host: str
    port: int = 27017
    database: str
    user: str = ""
    password: str = ""
    auth_database: str = "admin"


SourceCredentials = Annotated[
    PostgresCredentials | MongoCredentials,
    Field(discriminator="type"),
]


class SourceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    source_type: Literal["postgres", "mongodb"]
    credentials: SourceCredentials


class SourceUpdate(BaseModel):
    name: str | None = None
    credentials: SourceCredentials | None = None


class SourceResponse(BaseModel):
    id: uuid.UUID
    name: str
    source_type: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConnectionTestResponse(BaseModel):
    success: bool
    latency_ms: float | None = None
    message: str
