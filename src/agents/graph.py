"""
src/agents/graph.py
LangGraph StateGraph assembly — wires all 6 nodes with conditional edges.

Graph topology:
  query_router
      ↓ (conditional)
  ┌── ask_clarification → END
  └── parallel_retrieval
          ↓
      reflect
          ↓ (conditional)
      ┌── synthesize → END
      ├── parallel_retrieval (retry loop)
      └── fallback → END
"""
from __future__ import annotations

import time
from datetime import datetime

from langgraph.graph import StateGraph, END
from loguru import logger

from src.agents.state import AgentState
from src.agents.query_router import query_router_node
from src.agents.retrieval_agent import run_parallel_retrieval
from src.agents.reflection_agent import reflection_agent_node, route_after_reflection
from src.agents.answer_agent import answer_agent_node, fallback_node
from src.agents.human_loop import ask_clarification_node


def _route_after_router(state: AgentState) -> str:
    """Conditional edge: after routing → clarify or retrieve."""
    if state.get("needs_clarification"):
        return "ask_clarification"
    return "parallel_retrieval"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph insurance analysis workflow."""
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("query_router",       query_router_node)
    graph.add_node("parallel_retrieval", run_parallel_retrieval)
    graph.add_node("reflect",            reflection_agent_node)
    graph.add_node("synthesize",         answer_agent_node)
    graph.add_node("fallback",           fallback_node)
    graph.add_node("ask_clarification",  ask_clarification_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("query_router")

    # ── Edge: router → clarify or retrieve ───────────────────────────────────
    graph.add_conditional_edges(
        "query_router",
        _route_after_router,
        {
            "ask_clarification":  "ask_clarification",
            "parallel_retrieval": "parallel_retrieval",
        },
    )

    # ── Edge: retrieval always feeds reflection ───────────────────────────────
    graph.add_edge("parallel_retrieval", "reflect")

    # ── Edge: reflection → synthesize / retry / fallback ─────────────────────
    graph.add_conditional_edges(
        "reflect",
        route_after_reflection,
        {
            "synthesize":         "synthesize",
            "parallel_retrieval": "parallel_retrieval",   # retry loop
            "fallback":           "fallback",
        },
    )

    # ── Terminal edges ────────────────────────────────────────────────────────
    graph.add_edge("synthesize",        END)
    graph.add_edge("fallback",          END)
    graph.add_edge("ask_clarification", END)

    return graph.compile()


# Compile once at module load time
INSURANCE_GRAPH = build_graph()
logger.info("LangGraph workflow compiled successfully.")


def run_agent(
    query: str,
    session_id: str,
    user_role: str = "insurance_staff",
    conversation_history: list[dict] | None = None,
    context_summary: str = "",
) -> AgentState:
    """
    Public entry point — initialise AgentState and invoke the compiled graph.
    Called by src/api/query.py process_query().
    """
    t_start = time.time()

    initial_state: AgentState = {
        # Input
        "query":      query,
        "session_id": session_id,
        "user_role":  user_role,

        # Conversation memory
        "conversation_history": conversation_history or [],
        "context_summary":      context_summary,

        # Clarification
        "needs_clarification":  False,
        "clarification_question": None,

        # Stages (initialised empty)
        "router_decision":   {},
        "retrieved_chunks":  [],
        "retrieval_sources": [],
        "retrieval_strategy":"parallel",

        "reflection_result": {},
        "retry_count":       0,
        "max_retries":       3,

        "final_answer":    "",
        "citations":       [],
        "reasoning_chain": [],
        "recommendation":  "",

        # Audit
        "response_time_ms": 0,
        "created_at":       datetime.utcnow(),
    }

    logger.info(f"[Graph] Invoking for session={session_id!r} query={query[:80]!r}")
    final_state: AgentState = INSURANCE_GRAPH.invoke(initial_state)

    elapsed_ms = int((time.time() - t_start) * 1000)
    final_state["response_time_ms"] = elapsed_ms
    logger.info(f"[Graph] Completed in {elapsed_ms}ms — "
                f"recommendation={final_state.get('recommendation')}")
    return final_state
