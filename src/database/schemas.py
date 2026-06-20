"""
src/database/schemas.py
Pydantic models that mirror MongoDB collection shapes.
Used for validation before insert and type-safe reads.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field
from bson import ObjectId


# ── Helper: allow ObjectId in Pydantic v2 ────────────────────────────────────
class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return str(v)
        if ObjectId.is_valid(str(v)):
            return str(v)
        raise ValueError(f"Not a valid ObjectId: {v!r}")


# ── document_metadata ─────────────────────────────────────────────────────────
class DocumentMetadata(BaseModel):
    filename:       str
    doc_type:       Literal["policy", "agreement", "claim", "regulation"]
    source_name:    str
    effective_date: Optional[datetime] = None
    policy_number:  Optional[str]      = None
    insurer_name:   Optional[str]      = None
    hospital_name:  Optional[str]      = None   # agreements only
    ingested_at:    datetime           = Field(default_factory=datetime.utcnow)
    checksum:       str                          # SHA-256 hex
    total_chunks:   int                = 0


# ── document_chunks ───────────────────────────────────────────────────────────
class DocumentChunk(BaseModel):
    document_id:    str               # ObjectId as str
    chunk_index:    int
    chunk_text:     str
    embedding:      list[float]       # 768-dim
    page_number:    Optional[int]     = None
    section_header: Optional[str]     = None
    token_count:    Optional[int]     = None
    created_at:     datetime          = Field(default_factory=datetime.utcnow)


# ── historical_claims ─────────────────────────────────────────────────────────
class HistoricalClaim(BaseModel):
    claim_id:         str
    patient_name:     str
    age:              int
    diagnosis_code:   str
    diagnosis_desc:   str
    procedure_code:   str
    procedure_desc:   str
    claimed_amount:   float
    approved_amount:  float
    outcome:          Literal["approved", "rejected", "partially_approved"]
    rejection_reason: Optional[str]  = None
    policy_number:    str
    insurer_name:     str
    claim_date:       datetime
    decision_date:    datetime
    decision_notes:   Optional[str]  = None
    document_id:      Optional[str]  = None   # ref to document_metadata


# ── audit_logs ────────────────────────────────────────────────────────────────
class AuditLog(BaseModel):
    session_id:          str
    query_text:          str
    query_intent:        Optional[str]       = None
    retrieved_chunk_ids: list[str]           = []
    answer_text:         Optional[str]       = None
    confidence_score:    Optional[float]     = None
    reflection_notes:    Optional[str]       = None
    response_time_ms:    Optional[int]       = None
    created_at:          datetime            = Field(default_factory=datetime.utcnow)
    reasoning_chain:     list[str]           = []


# ── conversation_history ──────────────────────────────────────────────────────
class ConversationMessage(BaseModel):
    role:      Literal["user", "assistant"]
    content:   str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationHistory(BaseModel):
    session_id:      str
    messages:        list[ConversationMessage] = []
    context_summary: str                       = ""
    last_updated:    datetime                  = Field(default_factory=datetime.utcnow)
