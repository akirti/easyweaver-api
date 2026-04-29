"""Tests for connectors/registry.py — get_connector function."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock ibm_db_dbi to prevent ImportError when registry imports DB2Connector
if "ibm_db_dbi" not in sys.modules:
    sys.modules["ibm_db_dbi"] = MagicMock()


class TestGetConnector:
    def test_returns_postgres_connector(self):
        from easyweaver.connectors.registry import get_connector
        from easyweaver.connectors.implementations.postgres import PostgresConnector

        creds = {"user": "u", "password": "p", "host": "h", "port": 5432, "database": "d"}
        connector = get_connector("postgres", creds)
        assert isinstance(connector, PostgresConnector)

    def test_returns_mysql_connector(self):
        from easyweaver.connectors.registry import get_connector
        from easyweaver.connectors.implementations.mysql import MySQLConnector

        creds = {"user": "u", "password": "p", "host": "h", "port": 3306, "database": "d"}
        connector = get_connector("mysql", creds)
        assert isinstance(connector, MySQLConnector)

    def test_returns_mongodb_connector(self):
        from easyweaver.connectors.registry import get_connector
        from easyweaver.connectors.implementations.mongodb import MongoDBConnector

        creds = {"host": "localhost", "database": "testdb"}
        connector = get_connector("mongodb", creds)
        assert isinstance(connector, MongoDBConnector)

    def test_returns_file_connector(self):
        from easyweaver.connectors.registry import get_connector
        from easyweaver.connectors.implementations.file_source import FileConnector

        creds = {"gcp_path": "bucket/file.csv", "file_format": "csv", "original_filename": "file.csv"}
        connector = get_connector("file", creds)
        assert isinstance(connector, FileConnector)

    def test_returns_rest_api_connector(self):
        from easyweaver.connectors.registry import get_connector
        from easyweaver.connectors.implementations.rest_api import RestAPIConnector

        creds = {"base_url": "https://api.example.com"}
        connector = get_connector("rest_api", creds)
        assert isinstance(connector, RestAPIConnector)

    def test_unknown_source_type_raises_value_error(self):
        from easyweaver.connectors.registry import get_connector

        with pytest.raises(ValueError, match="Unsupported source type"):
            get_connector("oracle", {"user": "u"})

    def test_empty_source_type_raises_value_error(self):
        from easyweaver.connectors.registry import get_connector

        with pytest.raises(ValueError, match="Unsupported source type"):
            get_connector("", {})

    def test_db2_connector_with_mocked_ibm_db(self):
        """DB2Connector should be returned when ibm_db_dbi is available."""
        from easyweaver.connectors.registry import get_connector

        creds = {"user": "u", "password": "p", "host": "h", "port": 50000, "database": "d"}
        with patch("easyweaver.connectors.implementations.db2._require_ibm_db"):
            connector = get_connector("db2", creds)

        from easyweaver.connectors.implementations.db2 import DB2Connector
        assert isinstance(connector, DB2Connector)
