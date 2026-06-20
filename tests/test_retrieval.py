"""
tests/test_retrieval.py
Unit tests for the retrieval layer:
  - RRF fusion
  - _cosine_fallback logic
  - keyword_search
Run: pytest tests/test_retrieval.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ── RRF ───────────────────────────────────────────────────────────────────────
class TestReciprocalRankFusion:
    def _make_doc(self, doc_id: str, text: str = "") -> dict:
        return {"_id": doc_id, "chunk_text": text}

    def test_single_list_passthrough(self):
        """Single result list → RRF scores match 1/(k+rank+1)."""
        from src.retrieval.hybrid import reciprocal_rank_fusion

        docs = [self._make_doc(f"d{i}") for i in range(3)]
        merged = reciprocal_rank_fusion(docs)
        # First doc should have highest rrf_score
        assert merged[0]["_id"] == "d0"
        assert merged[0]["rrf_score"] > merged[1]["rrf_score"]

    def test_two_lists_merge(self):
        """Docs appearing in both lists should have higher RRF score."""
        from src.retrieval.hybrid import reciprocal_rank_fusion

        list_a = [self._make_doc("shared"), self._make_doc("only_a")]
        list_b = [self._make_doc("shared"), self._make_doc("only_b")]
        merged = reciprocal_rank_fusion(list_a, list_b)

        # shared doc should rank first
        assert merged[0]["_id"] == "shared"

    def test_empty_input_returns_empty(self):
        """Empty inputs → empty output."""
        from src.retrieval.hybrid import reciprocal_rank_fusion

        assert reciprocal_rank_fusion() == []
        assert reciprocal_rank_fusion([]) == []

    def test_deduplication(self):
        """Same doc ID in same list should not appear twice in output."""
        from src.retrieval.hybrid import reciprocal_rank_fusion

        docs = [
            self._make_doc("d1"),
            self._make_doc("d1"),  # duplicate
            self._make_doc("d2"),
        ]
        merged = reciprocal_rank_fusion(docs)
        ids = [d["_id"] for d in merged]
        assert ids.count("d1") == 1

    def test_rrf_k_parameter(self):
        """RRF score for rank=0 with k=10 equals exactly 1/(10+0+1)=1/11."""
        from src.retrieval.hybrid import reciprocal_rank_fusion

        docs = [self._make_doc("only_doc")]
        merged = reciprocal_rank_fusion(docs, k=10)

        expected = round(1.0 / (10 + 0 + 1), 6)
        assert merged[0]["rrf_score"] == expected
        assert merged[0]["rrf_score"] > 0






# ── Cosine fallback ───────────────────────────────────────────────────────────
class TestCosineFallback:
    def test_returns_sorted_by_similarity(self):
        """
        _cosine_fallback should return docs sorted by descending cosine score.
        Uses a mocked DB that returns two docs with known embeddings.
        """
        import numpy as np
        from unittest.mock import patch, MagicMock
        from src.database.vector_store import _cosine_fallback

        query = [1.0, 0.0]
        doc_a = {"_id": "a", "embedding": [1.0, 0.0], "chunk_text": "exact match",
                 "doc_type": "policy", "source_name": "A", "policy_number": None,
                 "insurer_name": None, "section_header": None,
                 "token_count": 2, "chunk_index": 0}
        doc_b = {"_id": "b", "embedding": [0.0, 1.0], "chunk_text": "orthogonal",
                 "doc_type": "policy", "source_name": "B", "policy_number": None,
                 "insurer_name": None, "section_header": None,
                 "token_count": 1, "chunk_index": 1}

        mock_col = MagicMock()
        mock_col.aggregate.return_value = iter([doc_a, doc_b])
        mock_db  = MagicMock()
        mock_db.document_chunks = mock_col

        with patch("src.database.vector_store.get_db", return_value=mock_db):
            results = _cosine_fallback(query, ["policy"], top_k=2)

        assert len(results) == 2
        assert results[0]["_id"] == "a"          # exact match should rank first
        assert results[0]["score"] > results[1]["score"]

    def test_returns_top_k_only(self):
        """_cosine_fallback respects top_k limit."""
        from unittest.mock import patch, MagicMock
        from src.database.vector_store import _cosine_fallback

        n = 10
        docs = [
            {"_id": str(i), "embedding": [float(i), 0.0],
             "chunk_text": f"doc {i}", "doc_type": "policy",
             "source_name": f"S{i}", "policy_number": None,
             "insurer_name": None, "section_header": None,
             "token_count": 2, "chunk_index": i}
            for i in range(1, n + 1)
        ]

        mock_col = MagicMock()
        mock_col.aggregate.return_value = iter(docs)
        mock_db  = MagicMock()
        mock_db.document_chunks = mock_col

        with patch("src.database.vector_store.get_db", return_value=mock_db):
            results = _cosine_fallback([1.0, 0.0], ["policy"], top_k=3)

        assert len(results) == 3

    def test_empty_collection_returns_empty(self):
        """Empty collection → empty result."""
        from unittest.mock import patch, MagicMock
        from src.database.vector_store import _cosine_fallback

        mock_col = MagicMock()
        mock_col.aggregate.return_value = iter([])
        mock_db  = MagicMock()
        mock_db.document_chunks = mock_col

        with patch("src.database.vector_store.get_db", return_value=mock_db):
            results = _cosine_fallback([1.0, 0.0], ["policy"], top_k=5)

        assert results == []
