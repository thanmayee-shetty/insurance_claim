"""
src/agents/answer_agent.py
Stage 4: Answer Synthesis Node
  - Builds a structured answer from validated chunks
  - Extracts citations
  - Parses recommendation label (APPROVE/REJECT/ESCALATE/NEEDS_MORE_INFO)
  - Formats answer for Streamlit display
"""
from __future__ import annotations

import re
from loguru import logger
from langchain_ollama import OllamaLLM

from src.config import OLLAMA_BASE_URL, LLM_MODEL
from src.agents.state import AgentState

_llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)

_ANSWER_PROMPT = """\
You are a senior medical insurance analyst for an Indian hospital system.
Using ONLY the provided evidence below, write a structured insurance case analysis.
Do NOT hallucinate. If evidence is insufficient, say so explicitly.

Conversation context: {context_summary}
{followup_note}

User query: {query}
Router intent: {intent}
Extracted entities: {entities}

Validated evidence chunks:
{chunks_formatted}

Write your response using EXACTLY these sections (include all headers):

## Summary
[2-3 sentence executive summary of the coverage situation]

## Policy Analysis
[Detailed analysis citing relevant policy clauses. Format citations as: (Source: <doc_name> — <section>)]

## Historical Precedent
[Summarize similar past claim decisions if claims data was retrieved. Write "No historical claims data retrieved." if not.]

## Regulatory Compliance
[Note applicable IRDA guidelines if regulation data was retrieved. Write "No regulatory data retrieved." if not.]

## Recommendation
[Write ONE of: APPROVE | REJECT | ESCALATE | NEEDS_MORE_INFO]
[2-3 sentence justification]

## Next Steps
[Numbered list of 3-5 concrete actions for the insurance staff member]
"""

_FOLLOWUP_NOTE = (
    "Note: This is a follow-up question to the previous conversation. "
    "Reference the previous context when answering and maintain consistency."
)

_RECOMMENDATION_RE = re.compile(
    r"\b(APPROVE|REJECT|ESCALATE|NEEDS_MORE_INFO|NEEDS MORE INFO)\b",
    re.IGNORECASE,
)


def _format_validated_chunks(
    chunks: list[dict],
    validated_ids: list[str],
    max_chunks: int = 6,
) -> str:
    """Format top validated chunks for the LLM prompt."""
    valid_set = set(validated_ids) if validated_ids else None
    lines = []
    count = 0
    for i, c in enumerate(chunks):
        if valid_set and str(i) not in valid_set and str(c.get("_id", "")) not in valid_set:
            continue
        score = c.get("score", c.get("rrf_score", 0))
        lines.append(
            f"[{count + 1}] Score={score:.3f} | {c.get('source_name', 'Unknown')} "
            f"— {c.get('section_header', 'Unknown Section')}\n"
            f"    {c.get('chunk_text', '')[:400]}"
        )
        count += 1
        if count >= max_chunks:
            break
    return "\n\n".join(lines) if lines else "No relevant chunks available."


def _build_citations(chunks: list[dict], validated_ids: list[str]) -> list[dict]:
    """Build citation dicts from validated chunks."""
    valid_set = set(validated_ids) if validated_ids else None
    citations = []
    for i, c in enumerate(chunks):
        if valid_set and str(i) not in valid_set:
            continue
        citations.append({
            "chunk_id":         str(c.get("_id", f"chunk_{i}")),
            "doc_name":         c.get("source_name", "Unknown Document"),
            "section":          c.get("section_header") or "General",
            "similarity_score": round(c.get("score", c.get("rrf_score", 0.0)), 3),
            "snippet":          c.get("chunk_text", "")[:200] + "...",
        })
    return citations[:6]  # cap at 6 citations


def _extract_recommendation(answer_text: str) -> str:
    """Parse the recommendation label from answer text."""
    match = _RECOMMENDATION_RE.search(answer_text)
    if match:
        label = match.group(1).upper().replace(" ", "_")
        if label in ("APPROVE", "REJECT", "ESCALATE", "NEEDS_MORE_INFO"):
            return label
    return "ESCALATE"   # safe default


def answer_agent_node(state: AgentState) -> AgentState:
    """LangGraph node: Stage 4 — answer synthesis."""
    query      = state["query"]
    chunks     = state.get("retrieved_chunks", [])
    decision   = state.get("router_decision", {})
    reflection = state.get("reflection_result", {})
    ctx_sum    = state.get("context_summary", "")
    is_followup= decision.get("is_followup", False)

    validated_ids = reflection.get("validated_chunks", [])
    confidence    = reflection.get("confidence", 0.0)

    formatted = _format_validated_chunks(chunks, validated_ids)
    citations  = _build_citations(chunks, validated_ids)

    prompt = _ANSWER_PROMPT.format(
        context_summary  = ctx_sum or "No prior conversation.",
        followup_note    = _FOLLOWUP_NOTE if is_followup else "",
        query            = query,
        intent           = decision.get("intent", "general"),
        entities         = str(decision.get("entities", {})),
        chunks_formatted = formatted,
    )

    logger.info(f"[Answer] Generating answer (confidence={confidence:.2f}) …")
    answer_text = _llm.invoke(prompt).strip()

    recommendation = _extract_recommendation(answer_text)

    state["final_answer"]  = answer_text
    state["citations"]     = citations
    state["recommendation"]= recommendation

    state.setdefault("reasoning_chain", []).append(
        f"Answer: recommendation={recommendation} | citations={len(citations)}"
    )
    logger.info(f"[Answer] recommendation={recommendation} | {len(citations)} citations")
    return state


def fallback_node(state: AgentState) -> AgentState:
    """LangGraph node: Structured fallback when retrieval is insufficient."""
    chunks    = state.get("retrieved_chunks", [])
    doc_types = state.get("retrieval_sources", ["policies"])
    confidence= state.get("reflection_result", {}).get("confidence", 0.0)
    gaps      = state.get("reflection_result", {}).get("gaps", [])

    best_match = ""
    if chunks:
        top = chunks[0]
        best_match = (
            f"Closest match: {top.get('source_name', 'Unknown')} "
            f"— {top.get('section_header', '')} "
            f"(similarity {top.get('score', 0):.2f})"
        )

    state["final_answer"] = (
        f"⚠️ **Insufficient Information Retrieved**\n\n"
        f"I could not find sufficient policy documentation to answer your query "
        f"with confidence (confidence score: {confidence:.0%}).\n\n"
        f"**Query:** {state['query']}\n"
        f"**Documents searched:** {', '.join(doc_types)}\n"
        f"{best_match}\n\n"
        f"**Information gaps identified:**\n"
        + ("\n".join(f"- {g}" for g in gaps) if gaps else "- General insufficiency")
        + "\n\n**Recommended next steps:**\n"
        f"1. Upload the relevant policy document using the Documents tab\n"
        f"2. Rephrase your query with more specific policy/procedure details\n"
        f"3. Contact your TPA representative directly for this query\n\n"
        f"*Case reference: {state.get('session_id', 'N/A')}*"
    )
    state["recommendation"] = "ESCALATE"
    state["citations"]      = []

    state.setdefault("reasoning_chain", []).append(
        f"Fallback triggered: confidence={confidence:.2f} "
        f"after {state.get('retry_count', 0)} retrieval attempt(s)"
    )
    logger.warning(f"[Fallback] Insufficient retrieval after {state.get('retry_count', 0)} attempts")
    return state
