"""
src/api/models.py
Pydantic request/response models for the Streamlit ↔ Agent interface.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query:             str
    session_id:        str
    user_role:         str  = "insurance_staff"
    include_reasoning: bool = True


class Citation(BaseModel):
    chunk_id:         str
    doc_name:         str
    section:          str
    similarity_score: float
    snippet:          str


class QueryResponse(BaseModel):
    status:                 Literal["success", "needs_clarification", "fallback"]
    answer:                 str
    confidence:             float
    citations:              list[Citation] = []
    reasoning_chain:        list[str]      = []
    recommendation:         Literal["APPROVE", "REJECT", "ESCALATE", "NEEDS_MORE_INFO"]
    clarification_question: Optional[str]  = None


# ── Converter: AgentState dict → QueryResponse ────────────────────────────────

def state_to_response(
    state: dict,
    include_reasoning: bool = True,
) -> QueryResponse:
    """Convert final AgentState dict into a typed QueryResponse."""
    needs_clarification = state.get("needs_clarification", False)

    if needs_clarification:
        return QueryResponse(
            status                 = "needs_clarification",
            answer                 = "",
            confidence             = 0.0,
            citations              = [],
            reasoning_chain        = [],
            recommendation         = "ESCALATE",
            clarification_question = state.get("clarification_question"),
        )

    confidence  = state.get("reflection_result", {}).get("confidence", 0.0)
    final_answer= state.get("final_answer", "")
    is_fallback = (
        confidence < 0.4
        or final_answer.startswith("⚠️")
        or state.get("recommendation") == "ESCALATE"
        and "Insufficient Information" in final_answer
    )

    citations = [
        Citation(
            chunk_id         = c.get("chunk_id", ""),
            doc_name         = c.get("doc_name", ""),
            section          = c.get("section", ""),
            similarity_score = float(c.get("similarity_score", 0.0)),
            snippet          = c.get("snippet", ""),
        )
        for c in state.get("citations", [])
    ]

    rec = state.get("recommendation", "ESCALATE")
    if rec not in ("APPROVE", "REJECT", "ESCALATE", "NEEDS_MORE_INFO"):
        rec = "ESCALATE"

    return QueryResponse(
        status          = "fallback" if is_fallback else "success",
        answer          = final_answer,
        confidence      = confidence,
        citations       = citations,
        reasoning_chain = state.get("reasoning_chain", []) if include_reasoning else [],
        recommendation  = rec,
    )
