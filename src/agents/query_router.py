"""
src/agents/query_router.py
Stage 1: Query Router Node
  - Classifies intent
  - Extracts entities (policy_number, diagnosis, procedure, age, amount)
  - Detects ambiguity → sets needs_clarification
  - Detects follow-up queries → inherits entities from previous turn
"""
from __future__ import annotations

import json
import re
from loguru import logger
from langchain_ollama import OllamaLLM

from src.config import OLLAMA_BASE_URL, LLM_MODEL
from src.agents.state import AgentState

_llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

# ── Follow-up trigger words ───────────────────────────────────────────────────
_FOLLOWUP_TRIGGERS = re.compile(
    r"^\s*(what about|and if|also|how about|what if|regarding|for that|"
    r"in that case|same for|does it also|additionally|furthermore|"
    r"and for|but what|compared to|versus|vs\.?)\b",
    re.IGNORECASE,
)

# ── Prompt template ───────────────────────────────────────────────────────────
_ROUTER_PROMPT = """\
You are an expert medical insurance query classifier for an Indian hospital system.
Analyze the user query and return ONLY valid JSON — no explanation, no markdown.

Previous conversation summary: {context_summary}
Recent conversation (last 3 turns):
{recent_history}

Current query: {query}

Return this exact JSON:
{{
  "intent": "coverage_check" | "claims_precedent" | "regulatory_compliance" | "general",
  "doc_types": ["policy", "claim", "regulation", "agreement"],
  "entities": {{
    "policy_number": null,
    "diagnosis": null,
    "procedure": null,
    "patient_age": null,
    "amount": null
  }},
  "needs_clarification": false,
  "clarification_question": null,
  "is_followup": false,
  "reasoning": "one sentence"
}}

Rules:
- doc_types must only include collections needed for THIS query.
- Set needs_clarification=true ONLY if ALL of policy_number, diagnosis, procedure are null AND intent is coverage_check.
- Set is_followup=true if the query references previous conversation context.
- If is_followup=true and entities are missing, inherit them from the conversation.
- Return ONLY the JSON object, nothing else.
"""


def _format_recent_history(history: list[dict], n: int = 3) -> str:
    if not history:
        return "None"
    lines = []
    for turn in history[-n:]:
        lines.append(f"USER: {turn.get('query', '')}")
        lines.append(f"ASSISTANT: {turn.get('answer', '')[:200]}...")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from LLM response text."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in LLM output: {text[:200]!r}")


def _parse_router_output(raw: str, fallback_query: str) -> dict:
    """Parse LLM output with retries and safe defaults."""
    for attempt in range(2):
        try:
            return _extract_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            if attempt == 0:
                logger.warning(f"Router JSON parse failed, retrying: {exc}")
                # Retry with simpler prompt
                simple_prompt = (
                    f"Classify this insurance query. Return ONLY JSON.\n"
                    f"Query: {fallback_query}\n"
                    f'{{"intent":"coverage_check","doc_types":["policy"],'
                    f'"entities":{{"policy_number":null,"diagnosis":null,"procedure":null,'
                    f'"patient_age":null,"amount":null}},'
                    f'"needs_clarification":false,"clarification_question":null,'
                    f'"is_followup":false,"reasoning":"fallback classification"}}'
                )
                raw = _llm.invoke(simple_prompt)
            else:
                logger.error(f"Router JSON parse failed after 2 attempts: {exc}")
                # Return safe default
                return {
                    "intent": "general",
                    "doc_types": ["policy", "claim"],
                    "entities": {
                        "policy_number": None, "diagnosis": None,
                        "procedure": None, "patient_age": None, "amount": None,
                    },
                    "needs_clarification": False,
                    "clarification_question": None,
                    "is_followup": False,
                    "reasoning": "Fallback: JSON parse failed",
                }


def _inherit_entities(entities: dict, history: list[dict]) -> dict:
    """
    If is_followup and entities are missing, try to inherit from last turn.
    Last turn's router_decision is stored in the answer field as structured data.
    """
    # Simple heuristic: if current entities are mostly null,
    # try to extract from the last assistant response text.
    if not history:
        return entities
    # Re-use entities as-is (the LLM should handle inheritance via context_summary)
    return entities


def query_router_node(state: AgentState) -> AgentState:
    """LangGraph node: Stage 1 — query routing."""
    query   = state["query"]
    history = state.get("conversation_history", [])
    ctx_sum = state.get("context_summary", "")

    # Quick follow-up detection heuristic (before LLM call)
    is_obvious_followup = bool(_FOLLOWUP_TRIGGERS.match(query))

    logger.info(f"[Router] query={query[:80]!r} followup_heuristic={is_obvious_followup}")

    prompt = _ROUTER_PROMPT.format(
        context_summary=ctx_sum or "None",
        recent_history=_format_recent_history(history),
        query=query,
    )

    raw = _llm.invoke(prompt)
    decision = _parse_router_output(raw, query)

    # Override is_followup if heuristic detected it
    if is_obvious_followup:
        decision["is_followup"] = True

    # Inherit entities from previous turn if follow-up
    if decision.get("is_followup") and history:
        decision["entities"] = _inherit_entities(decision.get("entities", {}), history)

    # Ensure required fields exist
    decision.setdefault("expanded", False)
    decision.setdefault("num_candidates_multiplier", 1)
    decision.setdefault("add_synonyms", False)

    state["router_decision"]        = decision
    state["needs_clarification"]    = decision.get("needs_clarification", False)
    state["clarification_question"] = decision.get("clarification_question")

    state.setdefault("reasoning_chain", []).append(
        f"Router: intent={decision.get('intent')} | "
        f"doc_types={decision.get('doc_types')} | "
        f"is_followup={decision.get('is_followup')} | "
        f"{decision.get('reasoning', '')}"
    )

    logger.info(f"[Router] intent={decision.get('intent')} | "
                f"doc_types={decision.get('doc_types')} | "
                f"needs_clarification={decision.get('needs_clarification')}")
    return state
