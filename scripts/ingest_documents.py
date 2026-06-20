"""
scripts/ingest_documents.py
Full ingestion pipeline:
  1. Discover all .txt/.pdf files in data/synthetic/
  2. Load each document (extract text, headers, checksum)
  3. Chunk each document
  4. Generate embeddings in batches
  5. Insert document_metadata + document_chunks into MongoDB
  6. Print summary

Run after:  python scripts/setup_mongo.py
            python scripts/generate_synthetic_data.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import DATA_SYNTHETIC_DIR
from src.ingestion.loader import discover_documents, load_document
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import embed_texts
from src.database.vector_store import (
    upsert_document_metadata,
    insert_chunks,
    update_chunk_count,
)
from src.database.connection import check_health


def ingest_all(root_dir: Path) -> dict:
    stats = {
        "docs_found":    0,
        "docs_ingested": 0,
        "docs_skipped":  0,
        "total_chunks":  0,
        "errors":        0,
        "elapsed_s":     0.0,
    }
    t0 = time.time()

    files = discover_documents(root_dir)
    stats["docs_found"] = len(files)
    logger.info(f"Found {len(files)} document(s) to ingest.")

    for filepath in files:
        logger.info(f"\n→ {filepath.name}")

        # 1. Load
        doc = load_document(filepath)
        if doc is None:
            stats["errors"] += 1
            continue

        # 2. Metadata insert (skip if duplicate)
        doc_id = upsert_document_metadata(doc)
        if doc_id is None:
            stats["errors"] += 1
            continue

        # Check if already had chunks (skip case)
        from src.database.connection import get_db
        from bson import ObjectId
        existing_meta = get_db().document_metadata.find_one({"_id": ObjectId(doc_id)})
        if existing_meta and existing_meta.get("total_chunks", 0) > 0:
            logger.info(f"  Already ingested — skipping chunks.")
            stats["docs_skipped"] += 1
            continue

        # 3. Chunk
        chunks = chunk_document(doc)
        if not chunks:
            logger.warning(f"  No chunks produced — skipping.")
            stats["errors"] += 1
            continue

        # 4. Embed
        logger.info(f"  Embedding {len(chunks)} chunks …")
        try:
            embeddings = embed_texts([c["chunk_text"] for c in chunks])
        except Exception as exc:
            logger.error(f"  Embedding failed: {exc}")
            stats["errors"] += 1
            continue

        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb

        # 5. Insert chunks
        inserted = insert_chunks(doc_id, chunks)
        update_chunk_count(doc_id, inserted)

        stats["docs_ingested"] += 1
        stats["total_chunks"] += inserted
        logger.success(f"  Inserted {inserted} chunks for '{doc['filename']}'")

    stats["elapsed_s"] = round(time.time() - t0, 1)
    return stats


def main() -> None:
    logger.info("=" * 60)
    logger.info("  Insurance RAG — Document Ingestion Pipeline")
    logger.info("=" * 60)

    health = check_health()
    if health["status"] != "ok":
        logger.error(f"MongoDB not reachable: {health.get('message')}")
        sys.exit(1)

    if not DATA_SYNTHETIC_DIR.exists():
        logger.error(f"Synthetic data directory not found: {DATA_SYNTHETIC_DIR}")
        logger.error("Run: python scripts/generate_synthetic_data.py first.")
        sys.exit(1)

    stats = ingest_all(DATA_SYNTHETIC_DIR)

    logger.info("\n" + "=" * 60)
    logger.info("  INGESTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Documents found:    {stats['docs_found']}")
    logger.info(f"  Documents ingested: {stats['docs_ingested']}")
    logger.info(f"  Documents skipped:  {stats['docs_skipped']} (duplicates)")
    logger.info(f"  Errors:             {stats['errors']}")
    logger.info(f"  Total chunks:       {stats['total_chunks']}")
    logger.info(f"  Time elapsed:       {stats['elapsed_s']}s")

    if stats["errors"] == 0:
        logger.success("\n✓ Ingestion complete. Run scripts/verify_setup.py to validate.")
    else:
        logger.warning(f"\n⚠ Ingestion completed with {stats['errors']} error(s).")


if __name__ == "__main__":
    main()
