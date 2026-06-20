"""
scripts/verify_setup.py
End-to-end smoke test — run after setup_mongo.py and ingest_documents.py.
Checks:
  1. MongoDB reachable + collections exist
  2. Vector search index exists
  3. Ollama reachable + both models loaded
  4. A sample embedding can be generated
  5. A $vectorSearch pipeline can execute
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from loguru import logger
from src.database.connection import get_db, check_health
from src.config import (
    OLLAMA_BASE_URL, LLM_MODEL, EMBED_MODEL, VECTOR_INDEX_NAME
)

PASS = "✓"
FAIL = "✗"


def check_mongodb() -> bool:
    logger.info("\n[1/5] MongoDB connectivity …")
    health = check_health()
    if health["status"] != "ok":
        logger.error(f"  {FAIL} MongoDB unreachable: {health.get('message')}")
        return False
    logger.success(f"  {PASS} Connected — DB: {health['database']}")
    db = get_db()
    for coll in ["document_metadata", "document_chunks",
                 "historical_claims", "audit_logs", "conversation_history"]:
        count = db[coll].count_documents({})
        logger.info(f"       {coll:35s} {count:>5} documents")
    return True


def check_vector_index() -> bool:
    logger.info("\n[2/5] $vectorSearch index …")
    db = get_db()
    try:
        indexes = list(db.document_chunks.list_search_indexes())
        names = [idx.get("name") for idx in indexes]
        if VECTOR_INDEX_NAME in names:
            logger.success(f"  {PASS} Index '{VECTOR_INDEX_NAME}' found.")
            return True
        else:
            logger.warning(f"  {FAIL} Index '{VECTOR_INDEX_NAME}' NOT found. "
                           f"Found: {names}")
            return False
    except Exception as exc:
        logger.warning(f"  Could not list search indexes: {exc}")
        logger.warning("  This is expected on MongoDB < 7.0 Community. "
                       "Vector search may not work.")
        return False


def check_ollama() -> bool:
    logger.info("\n[3/5] Ollama service …")
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        loaded = {m["name"] for m in resp.json().get("models", [])}
        ok = True
        for model in [LLM_MODEL, EMBED_MODEL]:
            # Allow partial match (e.g. "llama3.2:3b" matches "llama3.2:3b")
            found = any(model in name for name in loaded)
            status = PASS if found else FAIL
            logger.log("SUCCESS" if found else "WARNING",
                       f"  {status} Model '{model}' {'found' if found else 'NOT found — run: ollama pull ' + model}")
            if not found:
                ok = False
        return ok
    except Exception as exc:
        logger.error(f"  {FAIL} Ollama not reachable: {exc}")
        logger.error("  Start Ollama: ollama serve")
        return False


def check_embedding() -> bool:
    logger.info("\n[4/5] Sample embedding generation …")
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": "knee replacement surgery"},
            timeout=30,
        )
        resp.raise_for_status()
        emb = resp.json().get("embedding", [])
        if len(emb) == 768:
            logger.success(f"  {PASS} Embedding generated — dimension: {len(emb)}")
            return True
        else:
            logger.warning(f"  {FAIL} Unexpected embedding dimension: {len(emb)}")
            return False
    except Exception as exc:
        logger.error(f"  {FAIL} Embedding failed: {exc}")
        return False


def check_vector_search() -> bool:
    logger.info("\n[5/5] $vectorSearch pipeline …")
    db = get_db()
    chunk_count = db.document_chunks.count_documents({})
    if chunk_count == 0:
        logger.warning(f"  ⚠  No chunks in document_chunks. "
                       f"Run ingest_documents.py first.")
        return False
    try:
        # Generate a test embedding
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": "health insurance policy coverage"},
            timeout=30,
        )
        resp.raise_for_status()
        q_emb = resp.json()["embedding"]

        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": q_emb,
                    "numCandidates": 20,
                    "limit": 3,
                }
            },
            {"$project": {"chunk_text": 1, "_id": 0,
                          "score": {"$meta": "vectorSearchScore"}}},
        ]
        results = list(db.document_chunks.aggregate(pipeline))
        if results:
            logger.success(f"  {PASS} $vectorSearch returned {len(results)} result(s).")
            for r in results:
                logger.info(f"       score={r.get('score', 0):.3f} | "
                            f"{r.get('chunk_text', '')[:80]}…")
            return True
        else:
            logger.warning("  ⚠  $vectorSearch returned 0 results. "
                           "Check that the vector index is ready.")
            return False
    except Exception as exc:
        logger.error(f"  {FAIL} $vectorSearch failed: {exc}")
        return False


def main() -> None:
    logger.info("=" * 60)
    logger.info("  Insurance RAG — Setup Verification")
    logger.info("=" * 60)

    results = {
        "MongoDB":        check_mongodb(),
        "Vector Index":   check_vector_index(),
        "Ollama":         check_ollama(),
        "Embedding":      check_embedding(),
        "Vector Search":  check_vector_search(),
    }

    logger.info("\n" + "=" * 60)
    logger.info("  SUMMARY")
    logger.info("=" * 60)
    all_ok = True
    for name, passed in results.items():
        icon = PASS if passed else FAIL
        logger.log("SUCCESS" if passed else "WARNING", f"  {icon} {name}")
        if not passed:
            all_ok = False

    if all_ok:
        logger.success("\n✓ All checks passed — system is ready!")
        logger.info("  Launch the UI: streamlit run ui/app.py")
    else:
        logger.warning("\n⚠  Some checks failed. Fix the issues above, then re-run.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
