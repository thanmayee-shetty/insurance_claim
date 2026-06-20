"""
src/ingestion/chunker.py
Splits document text into overlapping chunks appropriate for each doc_type.
Returns a list of chunk dicts ready for embedding + MongoDB insert.
"""
from __future__ import annotations

import re
from typing import Optional
from loguru import logger

from src.config import (
    CHUNK_SIZE_POLICY, CHUNK_OVERLAP_POLICY,
    CHUNK_SIZE_REG, CHUNK_OVERLAP_REG,
    CHUNK_SIZE_CLAIMS,
)

# ── Section-aware split markers ───────────────────────────────────────────────
# These are the separator patterns used before character-level chunking.
_SECTION_SPLITS = re.compile(
    r"(?=^(?:SECTION|REGULATION|CIRCULAR|GUIDELINE)\s+\d+)",
    re.IGNORECASE | re.MULTILINE,
)

# ── Per-doc-type chunking config ──────────────────────────────────────────────
_CONFIG = {
    "policy":    {"size": CHUNK_SIZE_POLICY, "overlap": CHUNK_OVERLAP_POLICY},
    "agreement": {"size": CHUNK_SIZE_POLICY, "overlap": CHUNK_OVERLAP_POLICY},
    "regulation":{"size": CHUNK_SIZE_REG,    "overlap": CHUNK_OVERLAP_REG},
    "claim":     {"size": CHUNK_SIZE_CLAIMS,  "overlap": 0},   # one claim = one chunk
}


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character-level windows."""
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _detect_section(text: str, headers: list[str]) -> Optional[str]:
    """Return the most relevant section header for a given chunk of text."""
    text_up = text.upper()
    for header in headers:
        if header.upper()[:20] in text_up:
            return header
    return None


def chunk_document(doc: dict) -> list[dict]:
    """
    Chunk a loaded document dict (from loader.py).
    Returns list of chunk dicts with text, chunk_index, section_header, token_count.
    """
    doc_type = doc.get("doc_type", "policy")
    cfg = _CONFIG.get(doc_type, _CONFIG["policy"])
    full_text = doc.get("text", "")
    headers = doc.get("section_headers", [])

    if not full_text.strip():
        logger.warning(f"Empty document: {doc.get('filename')}")
        return []

    # For claims: one chunk per document (already short)
    if doc_type == "claim":
        chunk_text = full_text.strip()
        return [{
            "chunk_index":    0,
            "chunk_text":     chunk_text,
            "section_header": "Claim Record",
            "token_count":    len(chunk_text.split()),
            "page_number":    None,
        }]

    # For policies, agreements, regulations: split on section boundaries first
    sections = _SECTION_SPLITS.split(full_text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks: list[dict] = []
    idx = 0
    for section_text in sections:
        section_header = _detect_section(section_text[:200], headers)
        windows = _sliding_window(section_text, cfg["size"], cfg["overlap"])
        for window in windows:
            chunks.append({
                "chunk_index":    idx,
                "chunk_text":     window,
                "section_header": section_header,
                "token_count":    len(window.split()),
                "page_number":    None,
            })
            idx += 1

    logger.debug(f"  {doc.get('filename')}: {len(chunks)} chunks "
                 f"(doc_type={doc_type}, size={cfg['size']}, overlap={cfg['overlap']})")
    return chunks
