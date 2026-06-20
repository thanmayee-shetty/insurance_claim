"""
src/database/vector_store.py
CRUD operations for document_metadata, document_chunks, and historical_claims.
Handles checksum deduplication so documents are never re-ingested.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from bson import ObjectId
from loguru import logger
from pymongo.errors import DuplicateKeyError

from src.database.connection import get_db
from src.config import VECTOR_INDEX_NAME, DEFAULT_TOP_K, NUM_CANDIDATES_MULT


# ══════════════════════════════════════════════════════════════════════════════
# Document Metadata
# ══════════════════════════════════════════════════════════════════════════════

def upsert_document_metadata(doc: dict) -> Optional[str]:
    """
    Insert document metadata. Skip if the same checksum already exists.
    Returns the ObjectId string of the inserted/existing document, or None on error.
    """
    db = get_db()
    existing = db.document_metadata.find_one({"checksum": doc["checksum"]})
    if existing:
        logger.info(f"  Duplicate detected for '{doc['filename']}' — skipping.")
        return str(existing["_id"])

    payload = {
        "filename":       doc["filename"],
        "doc_type":       doc["doc_type"],
        "source_name":    doc.get("source_name", doc["filename"]),
        "effective_date": doc.get("effective_date"),
        "policy_number":  doc.get("policy_number"),
        "insurer_name":   doc.get("insurer_name"),
        "hospital_name":  doc.get("hospital_name"),
        "checksum":       doc["checksum"],
        "total_chunks":   0,
        "ingested_at":    datetime.utcnow(),
    }
    result = db.document_metadata.insert_one(payload)
    return str(result.inserted_id)


def update_chunk_count(doc_id: str, count: int) -> None:
    get_db().document_metadata.update_one(
        {"_id": ObjectId(doc_id)},
        {"$set": {"total_chunks": count}},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Document Chunks
# ══════════════════════════════════════════════════════════════════════════════

def insert_chunks(doc_id: str, chunks: list[dict]) -> int:
    """
    Bulk-insert chunk dicts (each must have chunk_text + embedding).
    Returns number of chunks inserted.
    """
    if not chunks:
        return 0
    db = get_db()
    docs = [
        {
            "document_id":    ObjectId(doc_id),
            "chunk_index":    c["chunk_index"],
            "chunk_text":     c["chunk_text"],
            "embedding":      c["embedding"],       # list[float], 768 dims
            "section_header": c.get("section_header"),
            "token_count":    c.get("token_count"),
            "page_number":    c.get("page_number"),
            "created_at":     datetime.utcnow(),
        }
        for c in chunks
    ]
    result = db.document_chunks.insert_many(docs)
    return len(result.inserted_ids)


def delete_chunks_for_document(doc_id: str) -> int:
    """Delete all chunks belonging to a document (for re-indexing)."""
    result = get_db().document_chunks.delete_many(
        {"document_id": ObjectId(doc_id)}
    )
    return result.deleted_count


# ══════════════════════════════════════════════════════════════════════════════
# Historical Claims
# ══════════════════════════════════════════════════════════════════════════════

def upsert_claim(claim: dict) -> Optional[str]:
    """
    Insert a historical claim. Skips duplicates based on claim_id.
    Returns ObjectId string or None.
    """
    db = get_db()
    existing = db.historical_claims.find_one({"claim_id": claim["claim_id"]})
    if existing:
        return str(existing["_id"])
    try:
        result = db.historical_claims.insert_one(claim)
        return str(result.inserted_id)
    except DuplicateKeyError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# $vectorSearch
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_fallback(
    query_embedding: list[float],
    doc_types: list[str],
    top_k: int,
) -> list[dict]:
    """
    Pure-Python cosine similarity fallback used when $vectorSearch
    (Atlas-only) is not available on local MongoDB Community.
    Loads all embeddings into memory and ranks them with numpy.
    """
    import numpy as np

    db = get_db()

    # Load all chunks + their metadata in one aggregation pass
    pipeline = [
        {
            "$lookup": {
                "from":         "document_metadata",
                "localField":   "document_id",
                "foreignField": "_id",
                "as":           "metadata",
            }
        },
        {"$unwind": "$metadata"},
        {"$match": {"metadata.doc_type": {"$in": doc_types}}},
        {
            "$project": {
                "chunk_text":     1,
                "section_header": 1,
                "token_count":    1,
                "chunk_index":    1,
                "embedding":      1,
                "doc_type":       "$metadata.doc_type",
                "source_name":    "$metadata.source_name",
                "policy_number":  "$metadata.policy_number",
                "insurer_name":   "$metadata.insurer_name",
            }
        },
    ]

    docs = list(db.document_chunks.aggregate(pipeline))
    if not docs:
        return []

    # Build embedding matrix
    q   = np.array(query_embedding, dtype=np.float32)
    q  /= (np.linalg.norm(q) + 1e-10)

    embs = np.array([d["embedding"] for d in docs], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    embs /= norms

    scores = (embs @ q).tolist()

    # Rank and return top_k
    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True,
    )[:top_k]

    results = []
    for score, doc in ranked:
        doc_out = {k: v for k, v in doc.items() if k != "embedding"}
        doc_out["score"] = float(score)
        results.append(doc_out)
    return results


def vector_search(
    query_embedding: list[float],
    doc_types: list[str],
    top_k: int = DEFAULT_TOP_K,
    num_candidates_mult: int = NUM_CANDIDATES_MULT,
    expanded: bool = False,
) -> list[dict]:
    """
    Run a $vectorSearch aggregation pipeline.
    Falls back to cosine similarity in Python if Atlas Search is unavailable
    (standard MongoDB Community deployment).
    Returns up to top_k chunks enriched with document metadata.
    """
    db = get_db()
    num_candidates = top_k * num_candidates_mult * (2 if expanded else 1)

    pipeline = [
        {
            "$vectorSearch": {
                "index":         VECTOR_INDEX_NAME,
                "path":          "embedding",
                "queryVector":   query_embedding,
                "numCandidates": num_candidates,
                "limit":         top_k * 2,
            }
        },
        {
            "$lookup": {
                "from":         "document_metadata",
                "localField":   "document_id",
                "foreignField": "_id",
                "as":           "metadata",
            }
        },
        {"$unwind": "$metadata"},
        {"$match": {"metadata.doc_type": {"$in": doc_types}}},
        {"$limit": top_k},
        {
            "$project": {
                "chunk_text":     1,
                "section_header": 1,
                "token_count":    1,
                "chunk_index":    1,
                "doc_type":       "$metadata.doc_type",
                "source_name":    "$metadata.source_name",
                "policy_number":  "$metadata.policy_number",
                "insurer_name":   "$metadata.insurer_name",
                "score":          {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        return list(db.document_chunks.aggregate(pipeline))
    except Exception as exc:
        err = str(exc)
        if "SearchNotEnabled" in err or "search" in err.lower():
            logger.warning(
                "Atlas $vectorSearch not available — falling back to "
                "Python cosine similarity (local MongoDB Community mode)."
            )
            return _cosine_fallback(query_embedding, doc_types, top_k)
        logger.error(f"$vectorSearch failed: {exc}")
        return []



def keyword_search(
    keywords: list[str],
    doc_types: list[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Simple MongoDB text-style search using $regex on chunk_text.
    Used as fallback when vector search returns poor results.
    """
    db = get_db()
    if not keywords:
        return []

    pattern = "|".join(re.escape(k) for k in keywords)
    pipeline = [
        {
            "$lookup": {
                "from": "document_metadata",
                "localField": "document_id",
                "foreignField": "_id",
                "as": "metadata",
            }
        },
        {"$unwind": "$metadata"},
        {
            "$match": {
                "metadata.doc_type": {"$in": doc_types},
                "chunk_text": {"$regex": pattern, "$options": "i"},
            }
        },
        {"$limit": top_k},
        {
            "$project": {
                "chunk_text":     1,
                "section_header": 1,
                "doc_type":       "$metadata.doc_type",
                "source_name":    "$metadata.source_name",
                "policy_number":  "$metadata.policy_number",
                "score":          {"$literal": 0.5},   # fixed mid-range score
            }
        },
    ]
    try:
        return list(db.document_chunks.aggregate(pipeline))
    except Exception as exc:
        logger.error(f"keyword_search failed: {exc}")
        return []


# ── need re import for keyword_search ────────────────────────────────────────
import re
