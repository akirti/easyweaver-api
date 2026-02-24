from typing import Any

from easyweaver.connectors.base import BaseConnector
from easyweaver.connectors.implementations.postgres import PostgresConnector
from easyweaver.connectors.implementations.mongodb import MongoDBConnector
from easyweaver.connectors.implementations.mysql import MySQLConnector
from easyweaver.connectors.implementations.db2 import DB2Connector
from easyweaver.connectors.implementations.file_source import FileConnector
from easyweaver.connectors.implementations.rest_api import RestAPIConnector


_CONNECTORS: dict[str, type[BaseConnector]] = {
    "postgres": PostgresConnector,
    "mongodb": MongoDBConnector,
    "mysql": MySQLConnector,
    "db2": DB2Connector,
    "file": FileConnector,
    "rest_api": RestAPIConnector,
}


def get_connector(source_type: str, credentials: dict[str, Any]) -> BaseConnector:
    cls = _CONNECTORS.get(source_type)
    if cls is None:
        raise ValueError(f"Unsupported source type: {source_type}")
    return cls(credentials)
