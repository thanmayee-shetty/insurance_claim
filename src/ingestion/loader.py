"""
src/ingestion/loader.py
Loads .txt and .pdf files from disk and extracts:
  - raw text
  - detected section headers (regex)
  - basic metadata (filename, doc_type, page count)
"""
from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import Optional

from loguru import logger

# Try PyMuPDF for PDF support; fall back gracefully
try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False
    logger.warning("PyMuPDF not installed — PDF loading will be skipped.")


# ── Section header detection ──────────────────────────────────────────────────
# Matches patterns like:
#   SECTION 4 — EXCLUSIONS
#   REGULATION 2 — CASHLESS CLAIMS
#   CIRCULAR 3 — GRIEVANCE REDRESSAL
#   2.3  Pre-existing Disease (PED)
_SECTION_RE = re.compile(
    r"^(?:SECTION|REGULATION|CIRCULAR|GUIDELINE)\s+\d+[\s\u2014\-]+(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBERED_HEADER_RE = re.compile(
    r"^(\d+\.\d+)\s{2,}([A-Z][A-Za-z\s\(\)\/\-]{5,60})$",
    re.MULTILINE,
)

# ── Doc-type inference from directory name ────────────────────────────────────
_DIR_TO_DOCTYPE = {
    "policies":           "policy",
    "provider_agreements": "agreement",
    "historical_claims":  "claim",
    "regulations":        "regulation",
}


def _infer_doc_type(filepath: Path) -> str:
    for part in filepath.parts:
        if part in _DIR_TO_DOCTYPE:
            return _DIR_TO_DOCTYPE[part]
    return "policy"   # safe default


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_section_headers(text: str) -> list[str]:
    """Return a list of all detected section headers in the text."""
    headers: list[str] = []
    for m in _SECTION_RE.finditer(text):
        headers.append(m.group(1).strip())
    for m in _NUMBERED_HEADER_RE.finditer(text):
        headers.append(f"{m.group(1)} {m.group(2).strip()}")
    return headers


def load_txt(filepath: Path) -> Optional[dict]:
    """Load a .txt file and return a document dict."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
        return {
            "filename":   filepath.name,
            "filepath":   str(filepath),
            "doc_type":   _infer_doc_type(filepath),
            "source_name": filepath.stem.replace("_", " ").title(),
            "text":       text,
            "page_texts": [text],   # single page for text files
            "page_count": 1,
            "checksum":   _sha256(text),
            "section_headers": _extract_section_headers(text),
        }
    except Exception as exc:
        logger.error(f"Failed to load {filepath}: {exc}")
        return None


def load_pdf(filepath: Path) -> Optional[dict]:
    """Load a .pdf file, extract per-page text, return document dict."""
    if not _HAS_FITZ:
        logger.warning(f"Skipping PDF {filepath} — PyMuPDF not available.")
        return None
    try:
        doc = fitz.open(str(filepath))
        page_texts = [page.get_text() for page in doc]
        full_text  = "\n".join(page_texts)
        doc.close()
        return {
            "filename":   filepath.name,
            "filepath":   str(filepath),
            "doc_type":   _infer_doc_type(filepath),
            "source_name": filepath.stem.replace("_", " ").title(),
            "text":       full_text,
            "page_texts": page_texts,
            "page_count": len(page_texts),
            "checksum":   _sha256(full_text),
            "section_headers": _extract_section_headers(full_text),
        }
    except Exception as exc:
        logger.error(f"Failed to load PDF {filepath}: {exc}")
        return None


def load_document(filepath: Path) -> Optional[dict]:
    """Dispatch to the appropriate loader based on file extension."""
    ext = filepath.suffix.lower()
    if ext == ".txt":
        return load_txt(filepath)
    elif ext == ".pdf":
        return load_pdf(filepath)
    else:
        logger.warning(f"Unsupported file type: {filepath}")
        return None


def discover_documents(root_dir: Path) -> list[Path]:
    """Recursively find all .txt and .pdf files under root_dir."""
    files = sorted(root_dir.rglob("*.txt")) + sorted(root_dir.rglob("*.pdf"))
    logger.info(f"Discovered {len(files)} document(s) in {root_dir}")
    return files
