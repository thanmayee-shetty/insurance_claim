"""
src/retrieval/hybrid.py
Reciprocal Rank Fusion (RRF) merge of multiple retrieval result sets.
Formula: score(doc) = Σ 1 / (k + rank + 1) across all result sets.
"""
from __future__ import annotations
from src.config import RRF_K


def reciprocal_rank_fusion(
    *result_sets: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge multiple ranked result lists into a single ranked list using RRF.
    Each result dict must have an '_id' field (MongoDB ObjectId or string).

    Args:
        *result_sets: Variable number of ranked result lists.
        k: RRF constant (default 60).

    Returns:
        Merged list sorted by descending fused RRF score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for result_set in result_sets:
        for rank, doc in enumerate(result_set):
            key = str(doc["_id"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc   # keep most recent copy of the full doc

    ranked_keys = sorted(scores, key=lambda k_: scores[k_], reverse=True)
    merged = []
    for key in ranked_keys:
        doc = doc_map[key]
        doc["rrf_score"] = round(scores[key], 6)
        merged.append(doc)

    return merged
