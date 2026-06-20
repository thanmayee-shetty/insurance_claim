"""
scripts/setup_mongo.py
Run once to:
  1. Connect to MongoDB
  2. Create all collections
  3. Create the $vectorSearch index on document_chunks
  4. Create all regular indexes
  5. Print a confirmation summary
"""
import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from loguru import logger
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import OperationFailure

from src.database.connection import get_db, check_health
from src.config import VECTOR_INDEX_NAME, EMBED_DIMENSION


COLLECTIONS = [
    "document_metadata",
    "document_chunks",
    "historical_claims",
    "audit_logs",
    "conversation_history",
]


def create_collections(db) -> None:
    """Create collections that don't exist yet (MongoDB creates lazily on insert,
    but explicit creation lets us verify connectivity early)."""
    existing = set(db.list_collection_names())
    for name in COLLECTIONS:
        if name not in existing:
            db.create_collection(name)
            logger.info(f"  Created collection: {name}")
        else:
            logger.info(f"  Collection already exists: {name}")


def create_vector_search_index(db) -> None:
    """
    Create the Atlas-format Search Index used by $vectorSearch.
    Works on MongoDB 7.0+ Community Edition running locally.
    NOTE: This command is async on Atlas but synchronous on local mongod.
    """
    logger.info("Creating $vectorSearch index on document_chunks …")
    try:
        db.command({
            "createSearchIndexes": "document_chunks",
            "indexes": [{
                "name": VECTOR_INDEX_NAME,
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": EMBED_DIMENSION,
                                "similarity": "cosine",
                            },
                            "document_id":    {"type": "objectId"},
                            "section_header": {"type": "string"},
                        },
                    }
                },
            }],
        })
        logger.success(f"  Vector search index '{VECTOR_INDEX_NAME}' created.")
    except OperationFailure as exc:
        if "already exists" in str(exc).lower() or "IndexAlreadyExists" in str(exc):
            logger.info(f"  Vector search index '{VECTOR_INDEX_NAME}' already exists — skipping.")
        else:
            logger.warning(f"  $vectorSearch index creation failed: {exc}")
            logger.warning("  Continuing — $vectorSearch will not work until the index is created.")
            logger.warning("  On MongoDB Community < 7.0, use a manual Atlas Search index instead.")


def create_regular_indexes(db) -> None:
    """Create all standard (non-vector) indexes."""
    logger.info("Creating regular indexes …")

    # document_chunks
    db.document_chunks.create_index(
        [("document_id", ASCENDING), ("chunk_index", ASCENDING)], name="idx_chunks_doc_chunk"
    )

    # document_metadata
    db.document_metadata.create_index(
        [("doc_type", ASCENDING), ("policy_number", ASCENDING)], name="idx_meta_type_policy"
    )
    db.document_metadata.create_index("checksum", unique=True, name="idx_meta_checksum")

    # historical_claims
    db.historical_claims.create_index(
        [("outcome", ASCENDING), ("diagnosis_code", ASCENDING)], name="idx_claims_outcome_dx"
    )
    db.historical_claims.create_index("claim_id", unique=True, name="idx_claims_id")

    # audit_logs
    db.audit_logs.create_index(
        [("session_id", ASCENDING), ("created_at", DESCENDING)], name="idx_audit_session"
    )

    # conversation_history
    db.conversation_history.create_index("session_id", unique=True, name="idx_conv_session")
    db.conversation_history.create_index(
        [("last_updated", DESCENDING)], name="idx_conv_updated"
    )

    logger.success("  All regular indexes created.")


def verify(db) -> None:
    """Insert and delete a test document to confirm write access."""
    test_doc = {"_test": True, "ts": time.time()}
    result = db.document_metadata.insert_one(test_doc)
    db.document_metadata.delete_one({"_id": result.inserted_id})
    logger.success("  Write/delete test passed.")


def main() -> None:
    logger.info("=" * 60)
    logger.info("  Insurance RAG — MongoDB Setup")
    logger.info("=" * 60)

    health = check_health()
    if health["status"] != "ok":
        logger.error(f"MongoDB not reachable: {health['message']}")
        sys.exit(1)
    logger.success(f"Connected to MongoDB — database: {health['database']}")

    db = get_db()

    create_collections(db)
    create_vector_search_index(db)
    create_regular_indexes(db)
    verify(db)

    # Final summary
    collections = db.list_collection_names()
    logger.info("\nCollections in database:")
    for c in sorted(collections):
        count = db[c].count_documents({})
        logger.info(f"  {c:30s} ({count} documents)")

    logger.success("\n✓ MongoDB setup complete. Run generate_synthetic_data.py next.")


if __name__ == "__main__":
    main()
