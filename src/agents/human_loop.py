"""
src/agents/human_loop.py
Clarification and escalation nodes for the LangGraph workflow.
"""
from __future__ import annotations

from loguru import logger
from src.agents.state import AgentState


def ask_clarification_node(state: AgentState) -> AgentState:
    """
    LangGraph node: triggered when router sets needs_clarification=True.
    Builds a helpful clarification question from the identified gaps.
    Returns immediately without calling the LLM (fast path).
    """
    gaps = state.get("reflection_result", {}).get("gaps", [])

    if gaps:
        gap_text = "\n".join(f"- {g}" for g in gaps[:3])
        question = (
            f"To answer your query accurately, could you please provide:\n{gap_text}"
        )
    else:
        # Use the router's pre-generated clarification question if available
        question = state.get("clarification_question") or (
            "To help me find the right policy information, could you clarify:\n"
            "- Which **policy number** applies? (e.g., POL-2024-IND-001)\n"
            "- What specific **procedure or diagnosis** is involved?\n"
            "- What is the **patient's age**?"
        )

    state["clarification_question"] = question
    state["final_answer"]           = question
    state["recommendation"]         = "NEEDS_MORE_INFO"
    state["citations"]              = []

    state.setdefault("reasoning_chain", []).append(
        "Clarification requested: query was ambiguous or missing key entities"
    )
    logger.info(f"[Clarification] Question: {question[:100]!r}")
    return state
