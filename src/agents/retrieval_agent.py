"""
src/agents/retrieval_agent.py
Stage 2: Parallel Multi-Agent Retrieval
  - 4 specialist sub-agents (Policy, Claims, Regulation, Agreement)
  - asyncio.gather for parallel execution
  - Reciprocal Rank Fusion (RRF) merge
  - Fallback keyword search when vector search returns nothing
"""
from __future__ import annotations

import asyncio
from loguru import logger

from src.agents.state import AgentState
from src.database.vector_store import vector_search, keyword_search
from src.ingestion.embedder import embed_query
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.config import DEFAULT_TOP_K, CLAIMS_TOP_K, NUM_CANDIDATES_MULT

# ── Synonym expansion map for retry ──────────────────────────────────────────
_SYNONYMS: dict[str, list[str]] = {
    "knee replacement": ["knee arthroplasty", "TKR", "total knee replacement"],
    "bypass":           ["CABG", "coronary artery bypass", "open heart surgery"],
    "dialysis":         ["hemodialysis", "renal replacement therapy", "HD"],
    "cataract":         ["phacoemulsification", "IOL implant", "lens extraction"],
    "appendix":         ["appendicitis", "appendectomy", "appendiceal"],
    "cancer":           ["malignancy", "tumour", "chemotherapy", "oncology"],
    "diabetes":         ["diabetic", "DM", "hyperglycemia", "insulin"],
    "cardiac":          ["heart", "cardiology", "myocardial", "coronary"],
    "stroke":           ["cerebral infarction", "TIA", "CVA", "brain attack"],
}


def _expand_query(query: str, entities: dict) -> str:
    """Append medical synonyms to the query for broader retrieval."""
    extras: list[str] = []
    combined = (query + " " + " ".join(str(v) for v in entities.values() if v)).lower()
    for term, synonyms in _SYNONYMS.items():
        if term in combined:
            extras.extend(synonyms)
    if extras:
        return query + " " + " ".join(extras[:4])
    return query


# ── Individual sub-agent coroutines ───────────────────────────────────────────

async def _policy_agent(
    query_emb: list[float],
    entities: dict,
    top_k: int,
    num_candidates_mult: int,
    expanded: bool,
) -> list[dict]:
    """Searches document_chunks filtered to doc_type=policy."""
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: vector_search(
            query_emb, ["policy"],
            top_k=top_k,
            num_candidates_mult=num_candidates_mult,
            expanded=expanded,
        ),
    )
    logger.debug(f"  [Policy Agent] returned {len(results)} chunks")
    return results


async def _claims_agent(
    query_emb: list[float],
    entities: dict,
    top_k: int,
    num_candidates_mult: int,
    expanded: bool,
) -> list[dict]:
    """Searches document_chunks filtered to doc_type=claim."""
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: vector_search(
            query_emb, ["claim"],
            top_k=CLAIMS_TOP_K,
            num_candidates_mult=num_candidates_mult,
            expanded=expanded,
        ),
    )
    logger.debug(f"  [Claims Agent] returned {len(results)} chunks")
    return results


async def _regulation_agent(
    query_emb: list[float],
    entities: dict,
    top_k: int,
    num_candidates_mult: int,
    expanded: bool,
) -> list[dict]:
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: vector_search(
            query_emb, ["regulation"],
            top_k=top_k,
            num_candidates_mult=num_candidates_mult,
            expanded=expanded,
        ),
    )
    logger.debug(f"  [Regulation Agent] returned {len(results)} chunks")
    return results


async def _agreement_agent(
    query_emb: list[float],
    entities: dict,
    top_k: int,
    num_candidates_mult: int,
    expanded: bool,
) -> list[dict]:
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: vector_search(
            query_emb, ["agreement"],
            top_k=top_k,
            num_candidates_mult=num_candidates_mult,
            expanded=expanded,
        ),
    )
    logger.debug(f"  [Agreement Agent] returned {len(results)} chunks")
    return results


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def _run_parallel(state: AgentState) -> list[dict]:
    decision = state["router_decision"]
    doc_types            = decision.get("doc_types", ["policy"])
    entities             = decision.get("entities", {})
    expanded             = decision.get("expanded", False)
    num_candidates_mult  = decision.get("num_candidates_multiplier", NUM_CANDIDATES_MULT)
    add_synonyms         = decision.get("add_synonyms", False)

    query = state["query"]
    if add_synonyms:
        query = _expand_query(query, entities)
        logger.info(f"  [Retrieval] Expanded query: {query[:120]}")

    # Embed the (potentially expanded) query once
    loop = asyncio.get_event_loop()
    query_emb = await loop.run_in_executor(None, lambda: embed_query(query))

    # Build coroutine tasks for selected doc_types
    tasks: list = []
    task_names: list[str] = []
    kwargs = dict(
        query_emb=query_emb,
        entities=entities,
        top_k=DEFAULT_TOP_K,
        num_candidates_mult=num_candidates_mult,
        expanded=expanded,
    )
    if "policy"     in doc_types:
        tasks.append(_policy_agent(**kwargs));     task_names.append("policy")
    if "claim"      in doc_types:
        tasks.append(_claims_agent(**kwargs));     task_names.append("claim")
    if "regulation" in doc_types:
        tasks.append(_regulation_agent(**kwargs)); task_names.append("regulation")
    if "agreement"  in doc_types:
        tasks.append(_agreement_agent(**kwargs));  task_names.append("agreement")

    if not tasks:
        logger.warning("No retrieval tasks — defaulting to policy search.")
        tasks = [_policy_agent(**kwargs)]
        task_names = ["policy"]

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out failed sub-agents gracefully
    valid: list[list[dict]] = []
    for name, result in zip(task_names, raw_results):
        if isinstance(result, Exception):
            logger.warning(f"  [{name.title()} Agent] failed: {result}")
        elif result:
            valid.append(result)

    if not valid:
        # Fallback: keyword search
        logger.warning("  All vector sub-agents returned empty — trying keyword fallback.")
        q_words = [w for w in state["query"].split() if len(w) > 4][:5]
        fallback = keyword_search(q_words, doc_types or ["policy"])
        return fallback

    return reciprocal_rank_fusion(*valid)


def run_parallel_retrieval(state: AgentState) -> AgentState:
    """Synchronous LangGraph node wrapper around the async orchestrator."""
    retry_count = state.get("retry_count", 0)
    logger.info(f"[Retrieval] attempt={retry_count + 1} / {state.get('max_retries', 3)}")

    try:
        # Use existing loop if available (Streamlit compatibility)
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        merged = loop.run_until_complete(_run_parallel(state))
    except RuntimeError:
        # If no event loop exists, create one
        merged = asyncio.run(_run_parallel(state))

    state["retrieved_chunks"]   = merged
    state["retrieval_sources"]  = state["router_decision"].get("doc_types", [])
    state["retrieval_strategy"] = "parallel"
    state["retry_count"]        = retry_count + 1

    state.setdefault("reasoning_chain", []).append(
        f"Retrieval (attempt {retry_count + 1}): "
        f"{len(merged)} chunks from {state['retrieval_sources']}"
    )
    logger.info(f"[Retrieval] merged {len(merged)} chunks via RRF")
    return state
