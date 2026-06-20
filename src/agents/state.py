"""
src/agents/state.py
LangGraph shared state TypedDict — passed through every node in the graph.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, TypedDict


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    query:      str           # Raw user query text
    session_id: str           # UUID / sess_xxx identifier
    user_role:  str           # 'insurance_staff' | 'supervisor' | 'auditor'

    # ── Conversation Memory ────────────────────────────────────────────────
    conversation_history: list[dict]   # [{query, answer, timestamp}, ...]
    context_summary:      str          # LLM-compressed session summary

    # ── Clarification ─────────────────────────────────────────────────────
    needs_clarification:  bool
    clarification_question: Optional[str]

    # ── Stage 1: Router Output ─────────────────────────────────────────────
    router_decision: dict
    # router_decision keys:
    #   intent:                  'coverage_check' | 'claims_precedent' |
    #                            'regulatory_compliance' | 'general'
    #   doc_types:               list[str]  e.g. ['policy', 'claim']
    #   entities:                {policy_number, diagnosis, procedure,
    #                             patient_age, amount}
    #   reasoning:               str
    #   is_followup:             bool
    #   expanded:                bool  (set True on retry)
    #   num_candidates_multiplier: int (doubled on retry)
    #   add_synonyms:            bool

    # ── Stage 2: Retrieval Output ─────────────────────────────────────────
    retrieved_chunks:   list[dict]   # Top-k merged chunks with metadata
    retrieval_sources:  list[str]    # Which collections were searched
    retrieval_strategy: str          # 'parallel' | 'fallback_keyword'

    # ── Stage 3: Reflection Output ────────────────────────────────────────
    reflection_result: dict
    # reflection_result keys:
    #   is_sufficient:    bool
    #   confidence:       float 0.0–1.0
    #   chunk_scores:     list[{chunk_id, relevance, reason}]
    #   gaps:             list[str]
    #   recommendation:   'answer' | 'expand_search' | 'fallback'
    #   validated_chunks: list[str]  (chunk ObjectId strings)

    retry_count: int   # Incremented on each retrieval retry
    max_retries: int   # Default 3

    # ── Stage 4: Answer Output ────────────────────────────────────────────
    final_answer:    str
    citations:       list[dict]   # [{chunk_id, doc_name, section,
                                  #   similarity_score, snippet}]
    reasoning_chain: list[str]    # Step-by-step transparency log
    recommendation:  str          # 'APPROVE' | 'REJECT' | 'ESCALATE'
                                  # | 'NEEDS_MORE_INFO'

    # ── Audit ─────────────────────────────────────────────────────────────
    response_time_ms: int
    created_at:       datetime
