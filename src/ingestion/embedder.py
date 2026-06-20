"""
src/ingestion/embedder.py
Wraps OllamaEmbeddings (nomic-embed-text) with:
  - Batched embedding for efficiency
  - Retry on transient errors
  - Single-query embed helper for retrieval
"""
from __future__ import annotations

import time
from typing import Optional
from loguru import logger

from langchain_ollama import OllamaEmbeddings

from src.config import OLLAMA_BASE_URL, EMBED_MODEL, EMBED_DIMENSION

# Singleton embedder (one per process)
_embedder: Optional[OllamaEmbeddings] = None


def _get_embedder() -> OllamaEmbeddings:
    global _embedder
    if _embedder is None:
        logger.info(f"Initialising embedder: {EMBED_MODEL} @ {OLLAMA_BASE_URL}")
        _embedder = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
    return _embedder


def embed_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """
    Embed a list of texts in batches.
    Returns list of 768-dim float vectors, one per input text.
    """
    if not texts:
        return []

    embedder = _get_embedder()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        for attempt in range(3):
            try:
                batch_embs = embedder.embed_documents(batch)
                # Validate dimensions
                for emb in batch_embs:
                    if len(emb) != EMBED_DIMENSION:
                        raise ValueError(
                            f"Unexpected embedding dimension: {len(emb)} "
                            f"(expected {EMBED_DIMENSION})"
                        )
                all_embeddings.extend(batch_embs)
                logger.debug(f"  Embedded batch {i // batch_size + 1}: "
                             f"{len(batch)} texts")
                break
            except Exception as exc:
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.warning(f"  Embedding failed (attempt {attempt + 1}/3): "
                                   f"{exc}. Retrying in {wait}s …")
                    time.sleep(wait)
                else:
                    logger.error(f"  Embedding failed after 3 attempts: {exc}")
                    raise

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string for retrieval.
    Uses embed_query (not embed_documents) for search-optimised embedding.
    """
    embedder = _get_embedder()
    for attempt in range(3):
        try:
            emb = embedder.embed_query(query)
            if len(emb) != EMBED_DIMENSION:
                raise ValueError(f"Bad embedding dim: {len(emb)}")
            return emb
        except Exception as exc:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Query embedding failed: {exc}")
                raise
