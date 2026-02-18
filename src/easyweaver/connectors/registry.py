from typing import Any

from easyweaver.connectors.base import BaseConnector
from easyweaver.connectors.implementations.postgres import PostgresConnector
from easyweaver.connectors.implementations.mongodb import MongoDBConnector


_CONNECTORS: dict[str, type[BaseConnector]] = {
    "postgres": PostgresConnector,
    "mongodb": MongoDBConnector,
}


def get_connector(source_type: str, credentials: dict[str, Any]) -> BaseConnector:
    cls = _CONNECTORS.get(source_type)
    if cls is None:
        raise ValueError(f"Unsupported source type: {source_type}")
    return cls(credentials)
