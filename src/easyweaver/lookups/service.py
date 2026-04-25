"""Lookup data service — caches distinct column values in the configurations collection.

Document structure in ``configurations`` collection::

    {
        "_id": "<process_id>_lookups",
        "key": "<process_id>_lookups",
        "type": "lookup-data",
        "process_id": "<process_id>",
        "lookups": {
            "<param_name>": [val1, val2, ...],
        },
        "references": [
            {"source_id": "...", "table": "...", "column": "..."},
        ],
        "created_at": <datetime>,
        "updated_at": <datetime>,
    }
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

from easyweaver.connectors.registry import get_connector
from easyweaver.sources.service import get_source, get_source_credentials

logger = structlog.get_logger()

COLLECTION = "configurations"


def _doc_id(process_id: str) -> str:
    return f"{process_id}_lookups"


async def get_lookups(db: AsyncIOMotorDatabase, process_id: str) -> dict | None:
    """Return the lookup document for a process, or None."""
    doc = await db[COLLECTION].find_one({"_id": _doc_id(process_id)})
    if doc:
        doc.pop("_id", None)
    return doc


async def upsert_lookups(
    db: AsyncIOMotorDatabase,
    process_id: str,
    lookups: dict[str, list],
    references: list[dict],
) -> dict:
    """Create or replace the lookup document for a process."""
    now = datetime.now(timezone.utc)
    doc = {
        "_id": _doc_id(process_id),
        "key": _doc_id(process_id),
        "type": "lookup-data",
        "process_id": process_id,
        "lookups": lookups,
        "references": references,
        "updated_at": now,
    }
    existing = await db[COLLECTION].find_one({"_id": _doc_id(process_id)})
    if existing:
        await db[COLLECTION].update_one(
            {"_id": _doc_id(process_id)},
            {"$set": {
                "lookups": lookups,
                "references": references,
                "updated_at": now,
            }},
        )
    else:
        doc["created_at"] = now
        await db[COLLECTION].insert_one(doc)
    logger.info("lookups_upserted", process_id=process_id, params=list(lookups.keys()))
    result = doc.copy()
    result.pop("_id", None)
    return result


async def delete_lookups(db: AsyncIOMotorDatabase, process_id: str) -> None:
    await db[COLLECTION].delete_one({"_id": _doc_id(process_id)})


async def build_lookups_from_params(
    db: AsyncIOMotorDatabase,
    params: dict[str, dict],
) -> tuple[dict[str, list], list[dict]]:
    """Fetch distinct values for all select/multi_select params that have options_source.

    Returns (lookups_dict, references_list).
    """
    lookups: dict[str, list] = {}
    references: list[dict] = []
    seen_refs: set[tuple] = set()

    for param_name, param_def in params.items():
        ptype = param_def.get("type", "")
        if ptype not in ("select", "multi_select"):
            continue
        opts_src = param_def.get("options_source")
        if not opts_src:
            continue

        source_id = opts_src.get("source_id")
        table = opts_src.get("table")
        column = opts_src.get("column")
        max_options = param_def.get("max_options", 500)

        if not (source_id and table and column):
            continue

        ref_key = (source_id, table, column)
        if ref_key not in seen_refs:
            seen_refs.add(ref_key)
            references.append({"source_id": source_id, "table": table, "column": column})

        try:
            source = await get_source(db, source_id)
            creds = get_source_credentials(source)
            connector = get_connector(source.source_type, creds)
            async with connector:
                result = await connector.get_distinct_values(table, column, max_options)
            lookups[param_name] = result.get("values", [])
            logger.info(
                "lookup_values_fetched",
                param=param_name,
                count=len(lookups[param_name]),
            )
        except Exception as e:
            logger.warning(
                "lookup_fetch_failed",
                param=param_name,
                source_id=source_id,
                error=str(e),
            )
            lookups[param_name] = []

    return lookups, references
