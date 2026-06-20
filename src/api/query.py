"""
src/api/query.py
Main entry point called by the Streamlit UI.
Orchestrates: session → memory → agent → persist → audit → response.
"""
from __future__ import annotations

import time
import logging
from datetime import datetime

from loguru import logger

from src.api.models import QueryRequest, QueryResponse, state_to_response
from src.agents.graph import run_agent
from src.memory.conversation import ConversationManager
from src.memory.session import get_or_create_session
from src.audit.logger import write_audit_log

_memory = ConversationManager()


def process_query(request: QueryRequest) -> QueryResponse:
    """
    Full query pipeline:
      1. Validate / create session
      2. Load conversation history from MongoDB
      3. Invoke LangGraph agent workflow
      4. Persist new QA pair to MongoDB
      5. Write audit log
      6. Return typed QueryResponse
    """
    t_start = time.time()

    # ── 1. Session guard ──────────────────────────────────────────────────────
    try:
        get_or_create_session(request.session_id)
    except ValueError as exc:
        logger.error(f"Invalid session_id: {exc}")
        return generate_fallback_response(str(exc), 0.0)

    # ── 2. Load conversation history ──────────────────────────────────────────
    try:
        history     = _memory.get_conversation_history(request.session_id)
        ctx_summary = _memory.get_context_summary(request.session_id) or ""
    except Exception as exc:
        logger.warning(f"MongoDB history load failed for {request.session_id}: {exc}")
        history, ctx_summary = [], ""    # degrade gracefully — proceed without history

    # ── 3. Run LangGraph agent ────────────────────────────────────────────────
    try:
        final_state = run_agent(
            query                = request.query,
            session_id           = request.session_id,
            user_role            = request.user_role,
            conversation_history = history,
            context_summary      = ctx_summary,
        )
    except Exception as exc:
        logger.error(f"Agent graph failure for {request.session_id}: {exc}")
        return generate_fallback_response(
            "The AI agent encountered an unexpected error. Please try again.", 0.0
        )

    # ── 4. Build response ─────────────────────────────────────────────────────
    response = state_to_response(final_state, request.include_reasoning)

    # ── 5. Persist conversation history ───────────────────────────────────────
    answer_to_save = (
        response.clarification_question
        if response.status == "needs_clarification"
        else response.answer
    )
    try:
        _memory.update_conversation_history(
            session_id         = request.session_id,
            user_query         = request.query,
            assistant_response = answer_to_save or "",
        )
    except Exception as exc:
        logger.warning(f"Failed to persist history for {request.session_id}: {exc}")

    # ── 6. Write audit log ────────────────────────────────────────────────────
    elapsed_ms = int((time.time() - t_start) * 1000)
    write_audit_log(
        session_id           = request.session_id,
        query_text           = request.query,
        query_intent         = final_state.get("router_decision", {}).get("intent"),
        retrieved_chunk_ids  = [c.chunk_id for c in response.citations],
        answer_text          = response.answer[:500] if response.answer else None,
        confidence_score     = response.confidence,
        reasoning_chain      = response.reasoning_chain,
        response_time_ms     = elapsed_ms,
        reflection_notes     = str(final_state.get("reflection_result", {})
                                   .get("gaps", [])),
    )

    logger.info(
        f"[process_query] session={request.session_id!r} "
        f"status={response.status} confidence={response.confidence:.2f} "
        f"recommendation={response.recommendation} time={elapsed_ms}ms"
    )
    return response


def extract_recommendation(final_answer: str) -> str:
    """Parse recommendation label from free-text LLM output."""
    import re
    match = re.search(
        r"\b(APPROVE|REJECT|ESCALATE|NEEDS_MORE_INFO|NEEDS MORE INFO)\b",
        final_answer, re.IGNORECASE,
    )
    if match:
        return match.group(1).upper().replace(" ", "_")
    return "ESCALATE"


def generate_fallback_response(reason: str, confidence: float) -> QueryResponse:
    """Return a safe, structured fallback when the agent cannot complete."""
    return QueryResponse(
        status         = "fallback",
        answer         = (
            f"⚠️ **Insufficient Information Retrieved**\n\n"
            f"{reason}\n\n"
            f"**Recommended next steps:**\n"
            f"1. Upload the relevant policy document via the Documents tab\n"
            f"2. Rephrase your query with more specific policy/procedure details\n"
            f"3. Contact your TPA representative directly for this query"
        ),
        confidence     = confidence,
        citations      = [],
        reasoning_chain= [f"Fallback triggered: {reason}"],
        recommendation = "ESCALATE",
    )
