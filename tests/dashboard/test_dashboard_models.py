"""Tests for easyweaver.dashboard.models — DashboardConfig and DataSnapshot."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from easyweaver.dashboard.models import DashboardConfig, DataSnapshot

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_fixture(name: str) -> dict:
    with open(FIXTURES_DIR / f"{name}.json") as f:
        return json.load(f)


@pytest.fixture
def dashboard_fixtures():
    return load_fixture("dashboard")


@pytest.fixture
def config_doc(dashboard_fixtures):
    return dashboard_fixtures["config_doc"]


@pytest.fixture
def snapshot_doc(dashboard_fixtures):
    return dashboard_fixtures["snapshot_doc"]


# ---------------------------------------------------------------------------
# DashboardConfig
# ---------------------------------------------------------------------------


class TestDashboardConfig:
    def test_instantiation_with_required_fields(self):
        cfg = DashboardConfig(
            id=uuid.uuid4(),
            user_id="user-1",
            name="My Dashboard",
            source_id="src-1",
        )
        assert cfg.user_id == "user-1"
        assert cfg.name == "My Dashboard"
        assert cfg.source_id == "src-1"
        assert cfg.tables == []
        assert cfg.refresh_interval_minutes == 60
        assert cfg.is_active is True

    def test_defaults_are_applied(self):
        cfg = DashboardConfig(id=uuid.uuid4(), user_id="u", name="n", source_id="s")
        assert cfg.tables == []
        assert cfg.refresh_interval_minutes == 60
        assert cfg.is_active is True
        assert isinstance(cfg.created_at, datetime)
        assert isinstance(cfg.updated_at, datetime)

    def test_from_doc_round_trip(self, config_doc):
        cfg = DashboardConfig.from_doc(config_doc)
        assert str(cfg.id) == config_doc["_id"]
        assert cfg.user_id == config_doc["user_id"]
        assert cfg.name == config_doc["name"]
        assert cfg.source_id == config_doc["source_id"]
        assert cfg.refresh_interval_minutes == config_doc["refresh_interval_minutes"]
        assert cfg.is_active == config_doc["is_active"]
        assert len(cfg.tables) == len(config_doc["tables"])

    def test_to_doc_contains_expected_keys(self):
        cfg = DashboardConfig(
            id=uuid.UUID("550e8400-e29b-41d4-a716-446655440000"),
            user_id="user-001",
            name="Test",
            source_id="src-001",
            tables=[{"table_name": "orders"}],
            refresh_interval_minutes=30,
            is_active=True,
        )
        doc = cfg.to_doc()
        assert doc["_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert doc["user_id"] == "user-001"
        assert doc["name"] == "Test"
        assert doc["source_id"] == "src-001"
        assert doc["refresh_interval_minutes"] == 30
        assert doc["is_active"] is True
        assert isinstance(doc["tables"], list)

    def test_to_doc_id_is_string(self):
        cfg = DashboardConfig(id=uuid.uuid4(), user_id="u", name="n", source_id="s")
        doc = cfg.to_doc()
        assert isinstance(doc["_id"], str)

    def test_from_doc_uses_defaults_for_missing_optional_fields(self):
        minimal = {
            "_id": str(uuid.uuid4()),
            "user_id": "u",
            "name": "n",
            "source_id": "s",
        }
        cfg = DashboardConfig.from_doc(minimal)
        assert cfg.tables == []
        assert cfg.refresh_interval_minutes == 60
        assert cfg.is_active is True

    def test_from_doc_to_doc_round_trip_preserves_data(self, config_doc):
        cfg = DashboardConfig.from_doc(config_doc)
        doc = cfg.to_doc()
        assert doc["_id"] == config_doc["_id"]
        assert doc["user_id"] == config_doc["user_id"]
        assert doc["name"] == config_doc["name"]
        assert doc["source_id"] == config_doc["source_id"]

    def test_custom_tables_and_interval(self):
        tables = [{"table_name": "orders"}, {"table_name": "users"}]
        cfg = DashboardConfig(
            id=uuid.uuid4(),
            user_id="u",
            name="n",
            source_id="s",
            tables=tables,
            refresh_interval_minutes=15,
        )
        assert len(cfg.tables) == 2
        assert cfg.refresh_interval_minutes == 15

    def test_created_at_and_updated_at_are_timezone_aware(self):
        cfg = DashboardConfig(id=uuid.uuid4(), user_id="u", name="n", source_id="s")
        assert cfg.created_at.tzinfo is not None
        assert cfg.updated_at.tzinfo is not None


# ---------------------------------------------------------------------------
# DataSnapshot
# ---------------------------------------------------------------------------


class TestDataSnapshot:
    def test_instantiation_with_required_fields(self):
        snap = DataSnapshot(
            id=uuid.uuid4(),
            config_id="cfg-1",
            source_id="src-1",
            table_name="orders",
        )
        assert snap.config_id == "cfg-1"
        assert snap.source_id == "src-1"
        assert snap.table_name == "orders"
        assert snap.row_count == 0
        assert snap.stats == {}

    def test_from_doc_round_trip(self, snapshot_doc):
        snap = DataSnapshot.from_doc(snapshot_doc)
        assert str(snap.id) == snapshot_doc["_id"]
        assert snap.config_id == snapshot_doc["config_id"]
        assert snap.source_id == snapshot_doc["source_id"]
        assert snap.table_name == snapshot_doc["table_name"]
        assert snap.row_count == snapshot_doc["row_count"]
        assert snap.stats == snapshot_doc["stats"]

    def test_to_doc_contains_expected_keys(self):
        snap = DataSnapshot(
            id=uuid.UUID("770e8400-e29b-41d4-a716-446655440002"),
            config_id="cfg-1",
            source_id="src-1",
            table_name="orders",
            row_count=500,
            stats={"size_bytes": 1000},
        )
        doc = snap.to_doc()
        assert doc["_id"] == "770e8400-e29b-41d4-a716-446655440002"
        assert doc["config_id"] == "cfg-1"
        assert doc["source_id"] == "src-1"
        assert doc["table_name"] == "orders"
        assert doc["row_count"] == 500
        assert doc["stats"] == {"size_bytes": 1000}

    def test_to_doc_id_is_string(self):
        snap = DataSnapshot(
            id=uuid.uuid4(), config_id="c", source_id="s", table_name="t"
        )
        doc = snap.to_doc()
        assert isinstance(doc["_id"], str)

    def test_from_doc_uses_defaults_for_missing_fields(self):
        minimal = {
            "_id": str(uuid.uuid4()),
            "config_id": "c",
            "source_id": "s",
            "table_name": "t",
        }
        snap = DataSnapshot.from_doc(minimal)
        assert snap.row_count == 0
        assert snap.stats == {}

    def test_from_doc_to_doc_round_trip(self, snapshot_doc):
        snap = DataSnapshot.from_doc(snapshot_doc)
        doc = snap.to_doc()
        assert doc["_id"] == snapshot_doc["_id"]
        assert doc["config_id"] == snapshot_doc["config_id"]
        assert doc["table_name"] == snapshot_doc["table_name"]
        assert doc["row_count"] == snapshot_doc["row_count"]

    def test_stats_dict_preserved(self):
        stats = {"changes_1h": 5, "changes_24h": 100, "size_bytes": 999}
        snap = DataSnapshot(
            id=uuid.uuid4(),
            config_id="c",
            source_id="s",
            table_name="t",
            stats=stats,
        )
        assert snap.stats == stats

    def test_captured_at_has_default(self):
        snap = DataSnapshot(
            id=uuid.uuid4(), config_id="c", source_id="s", table_name="t"
        )
        assert isinstance(snap.captured_at, datetime)
