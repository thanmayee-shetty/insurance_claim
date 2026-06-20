"""
src/memory/session.py
Session lifecycle management:
  - generate_session_id()
  - get_or_create_session()
  - session_expiry_check()
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta

from loguru import logger

from src.database.connection import get_db
from src.config import SESSION_TTL_HOURS

SESSION_ID_RE = re.compile(r"^sess_[a-f0-9]{12}$")


def generate_session_id() -> str:
    """Create a unique, URL-safe session identifier: sess_<12 hex chars>."""
    return f"sess_{uuid.uuid4().hex[:12]}"


def get_or_create_session(session_id: str) -> dict:
    """
    Validate format, check 24h expiry, return or create session document.
    Raises ValueError for invalid format.
    """
    if not SESSION_ID_RE.match(session_id):
        raise ValueError(
            f"Invalid session_id format: {session_id!r}. "
            f"Expected: sess_<12 hex chars>"
        )

    db  = get_db()
    doc = db.conversation_history.find_one({"session_id": session_id})

    if doc:
        age = datetime.utcnow() - doc["last_updated"]
        if age > timedelta(hours=SESSION_TTL_HOURS):
            # Session expired — delete and start fresh
            db.conversation_history.delete_one({"session_id": session_id})
            logger.info(f"Session {session_id} expired ({age.total_seconds()/3600:.1f}h) — reset.")
            doc = None

    if doc is None:
        now = datetime.utcnow()
        doc = {
            "session_id":      session_id,
            "messages":        [],
            "context_summary": "",
            "last_updated":    now,
        }
        db.conversation_history.insert_one(doc)
        logger.info(f"Created new session: {session_id}")

    return doc


def session_expiry_check() -> int:
    """
    Bulk-delete all sessions older than SESSION_TTL_HOURS.
    Call periodically (e.g., on app startup or via cron).
    Returns count of deleted sessions.
    """
    cutoff = datetime.utcnow() - timedelta(hours=SESSION_TTL_HOURS)
    result = get_db().conversation_history.delete_many(
        {"last_updated": {"$lt": cutoff}}
    )
    if result.deleted_count:
        logger.info(f"Expired {result.deleted_count} stale session(s).")
    return result.deleted_count
