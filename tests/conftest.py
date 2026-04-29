import json
from pathlib import Path

import polars as pl
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def load_fixture(name: str) -> dict:
    with open(FIXTURES_DIR / f"{name}.json") as f:
        return json.load(f)


@pytest.fixture
def operations_data():
    return load_fixture("operations")


@pytest.fixture
def sample_df(operations_data):
    return pl.DataFrame(operations_data["filter_data"]["sample_rows"])
