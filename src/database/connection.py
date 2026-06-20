"""
src/database/connection.py
MongoDB singleton connection manager.
Usage:
    from src.database.connection import get_db
    db = get_db()
    db.document_chunks.find_one(...)
"""
from __future__ import annotations

import threading
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from loguru import logger

from src.config import MONGO_URI, MONGO_DB_NAME, MONGO_TIMEOUT

# ── Thread-safe singleton ─────────────────────────────────────────────────────
_lock: threading.Lock = threading.Lock()
_client: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    """Return (or create) the global MongoClient singleton."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:   # double-checked locking
                logger.info(f"Connecting to MongoDB at {MONGO_URI} …")
                _client = MongoClient(
                    MONGO_URI,
                    serverSelectionTimeoutMS=MONGO_TIMEOUT * 1000,
                    maxPoolSize=10,
                    connectTimeoutMS=10_000,
                    socketTimeoutMS=30_000,
                )
                # Ping to fail fast on bad URI
                _client.admin.command("ping")
                logger.success("MongoDB connection established.")
    return _client


def get_db() -> Database:
    """Return the insurance_rag database handle."""
    return _get_client()[MONGO_DB_NAME]


def close_connection() -> None:
    """Cleanly close the MongoClient (call on app shutdown)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB connection closed.")


def check_health() -> dict:
    """Return connection health info. Used by verify_setup.py."""
    try:
        result = _get_client().admin.command("ping")
        db = get_db()
        collections = db.list_collection_names()
        return {
            "status": "ok",
            "ping": result,
            "database": MONGO_DB_NAME,
            "collections": collections,
        }
    except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
        return {"status": "error", "message": str(exc)}
