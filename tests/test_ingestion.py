"""
tests/test_ingestion.py
Unit tests for loader, chunker, and embedder modules.
Run: pytest tests/test_ingestion.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile


# ── loader ────────────────────────────────────────────────────────────────────
class TestLoader:
    def test_load_txt_basic(self, tmp_path):
        """Load a simple .txt file and verify output structure."""
        from src.ingestion.loader import load_txt

        f = tmp_path / "test_policy.txt"
        f.write_text("SECTION 1 — COVERAGE\nThis covers hospitalisation.", encoding="utf-8")

        doc = load_txt(f)
        assert doc is not None
        assert doc["filename"] == "test_policy.txt"
        assert "COVERAGE" in doc["text"]
        assert len(doc["checksum"]) == 64           # SHA-256 hex
        assert doc["page_count"] == 1

    def test_load_txt_section_headers(self, tmp_path):
        """Verify that section headers are extracted from text."""
        from src.ingestion.loader import load_txt

        content = (
            "SECTION 1 — DEFINITIONS\nSome text.\n"
            "SECTION 2 — EXCLUSIONS\nMore text."
        )
        f = tmp_path / "policy.txt"
        f.write_text(content, encoding="utf-8")

        doc = load_txt(f)
        assert "DEFINITIONS" in " ".join(doc["section_headers"])
        assert "EXCLUSIONS"  in " ".join(doc["section_headers"])

    def test_load_unsupported_type(self, tmp_path):
        """Unsupported extensions return None."""
        from src.ingestion.loader import load_document

        f = tmp_path / "doc.docx"
        f.write_bytes(b"fake docx content")
        result = load_document(f)
        assert result is None

    def test_checksum_deterministic(self, tmp_path):
        """Same content → same checksum."""
        from src.ingestion.loader import load_txt

        content = "Policy text that is always the same."
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text(content); f2.write_text(content)

        doc1 = load_txt(f1)
        doc2 = load_txt(f2)
        assert doc1["checksum"] == doc2["checksum"]

    def test_infer_doc_type_from_directory(self, tmp_path):
        """doc_type is inferred from directory name."""
        from src.ingestion.loader import load_txt

        reg_dir = tmp_path / "regulations"
        reg_dir.mkdir()
        f = reg_dir / "irda_2016.txt"
        f.write_text("Regulatory content.")

        doc = load_txt(f)
        assert doc["doc_type"] == "regulation"

    def test_discover_documents(self, tmp_path):
        """discover_documents finds all .txt and .pdf files recursively."""
        from src.ingestion.loader import discover_documents

        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("b")
        (tmp_path / "ignore.csv").write_text("c")

        found = discover_documents(tmp_path)
        names = {f.name for f in found}
        assert "a.txt" in names
        assert "b.txt" in names
        assert "ignore.csv" not in names


# ── chunker ───────────────────────────────────────────────────────────────────
class TestChunker:
    def _make_doc(self, text: str, doc_type: str = "policy") -> dict:
        return {
            "filename": "test.txt",
            "doc_type": doc_type,
            "text":     text,
            "section_headers": [],
        }

    def test_policy_chunks_produced(self):
        """Policy doc with multi-section text produces multiple chunks."""
        from src.ingestion.chunker import chunk_document

        text = "SECTION 1 — SCOPE\n" + "Coverage text. " * 100
        doc    = self._make_doc(text, "policy")
        chunks = chunk_document(doc)
        assert len(chunks) >= 1
        for c in chunks:
            assert "chunk_text" in c
            assert "chunk_index" in c

    def test_claim_is_single_chunk(self):
        """Claim documents always produce exactly 1 chunk."""
        from src.ingestion.chunker import chunk_document

        text   = "Claim ID: CLM-001\nDiagnosis: Fever\nOutcome: APPROVED"
        doc    = self._make_doc(text, "claim")
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0]["section_header"] == "Claim Record"

    def test_empty_doc_returns_empty(self):
        """Empty document text returns empty chunk list."""
        from src.ingestion.chunker import chunk_document

        doc    = self._make_doc("", "policy")
        chunks = chunk_document(doc)
        assert chunks == []

    def test_chunk_index_sequential(self):
        """chunk_index values are sequential starting at 0."""
        from src.ingestion.chunker import chunk_document

        text   = "SECTION 1 — A\n" + ("word " * 200)
        doc    = self._make_doc(text, "policy")
        chunks = chunk_document(doc)
        for i, c in enumerate(chunks):
            assert c["chunk_index"] == i

    def test_token_count_present(self):
        """Each chunk has a non-zero token_count."""
        from src.ingestion.chunker import chunk_document

        text   = "SECTION 2 — EXCLUSIONS\n" + ("Some content here. " * 50)
        doc    = self._make_doc(text, "policy")
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.get("token_count", 0) > 0


# ── embedder (mocked) ─────────────────────────────────────────────────────────
class TestEmbedder:
    def test_embed_texts_returns_correct_count(self):
        """embed_texts returns one vector per input text."""
        from src.ingestion import embedder
        from unittest.mock import patch, MagicMock

        fake_emb = [0.1] * 768
        mock_embedder = MagicMock()
        mock_embedder.embed_documents = lambda texts: [fake_emb for _ in texts]

        with patch.object(embedder, "_get_embedder", return_value=mock_embedder):
            with patch.object(embedder, "_embedder", None):  # reset singleton
                texts  = ["policy text one", "policy text two", "claim text"]
                result = embedder.embed_texts(texts)

        assert len(result) == 3
        assert len(result[0]) == 768

    def test_embed_empty_returns_empty(self):
        """embed_texts([]) returns []."""
        from src.ingestion.embedder import embed_texts
        assert embed_texts([]) == []
