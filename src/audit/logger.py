"""
src/audit/logger.py
Writes immutable audit log entries to the audit_logs MongoDB collection.
Called by process_query() after every query cycle.
"""
from __future__ import annotations

from datetime import datetime
from loguru import logger as log

from src.database.connection import get_db


def write_audit_log(
    session_id: str,
    query_text: str,
    query_intent: str | None,
    retrieved_chunk_ids: list[str],
    answer_text: str | None,
    confidence_score: float | None,
    reasoning_chain: list[str],
    response_time_ms: int,
    reflection_notes: str | None = None,
) -> None:
    """Insert one audit log record. Failures are logged but do not raise."""
    try:
        get_db().audit_logs.insert_one({
            "session_id":          session_id,
            "query_text":          query_text,
            "query_intent":        query_intent,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "answer_text":         answer_text,
            "confidence_score":    confidence_score,
            "reflection_notes":    reflection_notes,
            "response_time_ms":    response_time_ms,
            "reasoning_chain":     reasoning_chain,
            "created_at":          datetime.utcnow(),
        })
    except Exception as exc:
        log.warning(f"Audit log write failed: {exc}")
