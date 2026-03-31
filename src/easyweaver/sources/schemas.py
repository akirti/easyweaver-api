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
    ssl_mode: str = "disable"
    ssl_client_cert: str = ""
    ssl_client_key: str = ""
    ssl_ca_cert: str = ""


class MongoCredentials(BaseModel):
    type: Literal["mongodb"] = "mongodb"
    connection_string: str = ""
    scheme: Literal["mongodb", "mongodb+srv"] = "mongodb"
    host: str = ""
    port: int | None = None
    database: str = ""
    user: str = ""
    password: str = ""
    auth_database: str = "admin"


class MySQLCredentials(BaseModel):
    type: Literal["mysql"] = "mysql"
    host: str
    port: int = 3306
    database: str
    user: str
    password: str


class DB2Credentials(BaseModel):
    type: Literal["db2"] = "db2"
    host: str
    port: int = 50000
    database: str
    user: str
    password: str


class FileCredentials(BaseModel):
    type: Literal["file"] = "file"
    gcp_path: str
    file_format: Literal["csv", "json", "xlsx", "xls"]
    original_filename: str


class RestAPICredentials(BaseModel):
    type: Literal["rest_api"] = "rest_api"
    base_url: str
    auth_type: Literal[
        "none", "bearer", "basic", "api_key", "oauth2_client_credentials", "login"
    ] = "none"
    bearer_token: str = ""
    basic_user: str = ""
    basic_password: str = ""
    api_key_header: str = "X-API-Key"
    api_key_value: str = ""
    oauth2_token_url: str = ""
    oauth2_client_id: str = ""
    oauth2_client_secret: str = ""
    oauth2_scope: str = ""
    login_url: str = ""
    login_body: dict = Field(default_factory=dict)
    login_token_path: str = "access_token"
    headers: dict[str, str] = Field(default_factory=dict)
    endpoints: dict[str, dict] = Field(default_factory=dict)


SourceCredentials = Annotated[
    PostgresCredentials
    | MongoCredentials
    | MySQLCredentials
    | DB2Credentials
    | FileCredentials
    | RestAPICredentials,
    Field(discriminator="type"),
]


class SourceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    source_type: Literal["postgres", "mongodb", "mysql", "db2", "file", "rest_api"]
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
