"""
src/agents/reflection_agent.py
Stage 3: Reflection & Validation Node
  - Scores each retrieved chunk for relevance
  - Produces an overall confidence score
  - Identifies gaps
  - Recommends: 'answer' | 'expand_search' | 'fallback'
"""
from __future__ import annotations

import json
import re
from loguru import logger
from langchain_ollama import OllamaLLM

from src.config import OLLAMA_BASE_URL, LLM_MODEL, CONFIDENCE_THRESHOLD, FALLBACK_THRESHOLD
from src.agents.state import AgentState

_llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

_REFLECTION_PROMPT = """\
You are a senior medical insurance auditor reviewing retrieved policy evidence.
Evaluate whether the retrieved chunks are sufficient to answer the user's query.
Return ONLY valid JSON — no explanation, no markdown.

User query: {query}
Router intent: {intent}
Entities extracted: {entities}

Retrieved chunks ({n_chunks} total):
{chunks_formatted}

Return this exact JSON:
{{
  "is_sufficient": true,
  "confidence": 0.85,
  "chunk_scores": [
    {{"chunk_id": "0", "relevance": 0.9, "reason": "directly answers the query"}}
  ],
  "gaps": [],
  "recommendation": "answer",
  "validated_chunks": ["0", "1"]
}}

Confidence scoring guide:
- 0.9–1.0: Multiple highly relevant chunks, explicit answer found
- 0.7–0.9: Relevant chunks, minor context gap → recommendation: "answer"
- 0.5–0.7: Partial relevance, notable missing details → recommendation: "expand_search"
- 0.3–0.5: Weak relevance → recommendation: "expand_search"
- 0.0–0.3: No relevant chunks → recommendation: "fallback"

Return ONLY the JSON object.
"""


def _format_chunks(chunks: list[dict], max_chunks: int = 8) -> str:
    lines = []
    for i, c in enumerate(chunks[:max_chunks]):
        score = c.get("score", c.get("rrf_score", 0))
        lines.append(
            f"[{i}] ({score:.3f}) {c.get('source_name', 'Unknown')} "
            f"— {c.get('section_header', 'No Section')}\n"
            f"    {c.get('chunk_text', '')[:200]}"
        )
    return "\n".join(lines)


def _safe_parse(raw: str, n_chunks: int) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Reflection JSON parse failed — using heuristic confidence.")
    # Heuristic fallback: confidence proportional to number of chunks
    conf = min(0.5, n_chunks * 0.06)
    return {
        "is_sufficient":   conf >= CONFIDENCE_THRESHOLD,
        "confidence":      conf,
        "chunk_scores":    [],
        "gaps":            ["JSON parse failed — confidence estimated heuristically"],
        "recommendation":  "answer" if conf >= CONFIDENCE_THRESHOLD else (
                           "expand_search" if conf >= FALLBACK_THRESHOLD else "fallback"),
        "validated_chunks": [str(i) for i in range(min(n_chunks, 5))],
    }


def reflection_agent_node(state: AgentState) -> AgentState:
    """LangGraph node: Stage 3 — reflection and confidence scoring."""
    chunks  = state.get("retrieved_chunks", [])
    query   = state["query"]
    decision= state.get("router_decision", {})

    # Edge case: no chunks at all → instant fallback
    if not chunks:
        logger.warning("[Reflection] 0 chunks retrieved — instant fallback.")
        state["reflection_result"] = {
            "is_sufficient":   False,
            "confidence":      0.0,
            "chunk_scores":    [],
            "gaps":            ["No documents retrieved"],
            "recommendation":  "fallback",
            "validated_chunks":[],
        }
        state.setdefault("reasoning_chain", []).append(
            "Reflection: 0 chunks — instant fallback"
        )
        return state

    prompt = _REFLECTION_PROMPT.format(
        query           = query,
        intent          = decision.get("intent", "unknown"),
        entities        = json.dumps(decision.get("entities", {})),
        n_chunks        = len(chunks),
        chunks_formatted= _format_chunks(chunks),
    )

    raw    = _llm.invoke(prompt)
    result = _safe_parse(raw, len(chunks))

    state["reflection_result"] = result
    conf   = result.get("confidence", 0.0)
    rec    = result.get("recommendation", "fallback")

    state.setdefault("reasoning_chain", []).append(
        f"Reflection: confidence={conf:.2f} | recommendation={rec} | "
        f"gaps={result.get('gaps', [])}"
    )
    logger.info(f"[Reflection] confidence={conf:.2f} | recommendation={rec}")
    return state


def route_after_reflection(state: AgentState) -> str:
    """
    Conditional edge function used in the LangGraph graph.
    Returns the name of the next node to visit.
    """
    result      = state.get("reflection_result", {})
    confidence  = result.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    rec         = result.get("recommendation", "fallback")

    if rec == "answer" or confidence >= CONFIDENCE_THRESHOLD:
        return "synthesize"

    if rec == "expand_search" and retry_count < max_retries:
        # Expand search criteria for next attempt
        decision = state.get("router_decision", {})
        decision["expanded"]                = True
        decision["num_candidates_multiplier"] = NUM_CANDIDATES_MULT * 2
        decision["add_synonyms"]            = True
        # On second retry, search all doc_types
        if retry_count >= 1:
            decision["doc_types"] = ["policy", "claim", "regulation", "agreement"]
        state["router_decision"] = decision
        logger.info(f"[Reflection] Expanding search (retry {retry_count + 1})")
        return "parallel_retrieval"

    return "fallback"


# need config import for NUM_CANDIDATES_MULT
from src.config import NUM_CANDIDATES_MULT  # noqa: E402
