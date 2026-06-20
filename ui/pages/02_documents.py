"""
ui/pages/02_documents.py
Page 2 — Document Management
  - Upload PDF/TXT files
  - Trigger ingestion pipeline
  - Show ingestion status table
  - Re-index button per document
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from datetime import datetime
from src.database.connection import get_db
from src.config import DATA_SYNTHETIC_DIR

st.set_page_config(page_title="InsureAI — Documents", page_icon="📂", layout="wide")

st.title("📂 Document Management")
st.caption("Upload new policy documents and monitor ingestion status.")

# ── Upload section ────────────────────────────────────────────────────────────
st.markdown("### Upload New Document")
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader(
        "Choose a PDF or TXT file (max 50 MB)",
        type=["pdf", "txt"],
        accept_multiple_files=False,
    )
with col2:
    doc_type = st.selectbox(
        "Document Type",
        ["policy", "agreement", "regulation", "claim"],
        index=0,
    )
    source_name = st.text_input("Source Name (optional)", placeholder="e.g. Star Health 2024")

if uploaded and st.button("⬆️ Upload & Ingest", type="primary"):
    # Save file to raw directory
    save_dir = DATA_SYNTHETIC_DIR / doc_type + "s"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / uploaded.name
    save_path.write_bytes(uploaded.getbuffer())

    with st.spinner(f"Ingesting {uploaded.name}…"):
        try:
            from src.ingestion.loader import load_document
            from src.ingestion.chunker import chunk_document
            from src.ingestion.embedder import embed_texts
            from src.database.vector_store import (
                upsert_document_metadata, insert_chunks, update_chunk_count
            )

            doc = load_document(save_path)
            if doc is None:
                st.error("Failed to load document.")
            else:
                if source_name:
                    doc["source_name"] = source_name
                doc["doc_type"] = doc_type
                doc_id = upsert_document_metadata(doc)
                chunks = chunk_document(doc)
                embeddings = embed_texts([c["chunk_text"] for c in chunks])
                for chunk, emb in zip(chunks, embeddings):
                    chunk["embedding"] = emb
                inserted = insert_chunks(doc_id, chunks)
                update_chunk_count(doc_id, inserted)
                st.success(f"✅ Ingested {uploaded.name} — {inserted} chunks created.")
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")

st.markdown("---")

# ── Ingestion status table ────────────────────────────────────────────────────
st.markdown("### Ingested Documents")
try:
    db   = get_db()
    docs = list(db.document_metadata.find({}, {
        "filename": 1, "doc_type": 1, "source_name": 1,
        "total_chunks": 1, "ingested_at": 1, "_id": 1,
    }).sort("ingested_at", -1).limit(100))

    if not docs:
        st.info("No documents ingested yet. Upload files above or run `python scripts/ingest_documents.py`.")
    else:
        for d in docs:
            d["_id"]       = str(d["_id"])
            d["ingested_at"] = d.get("ingested_at", datetime.utcnow()).strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            docs,
            column_order=["filename", "doc_type", "source_name", "total_chunks", "ingested_at"],
            use_container_width=True,
        )
        st.caption(f"Total: {len(docs)} document(s)")
except Exception as exc:
    st.error(f"Could not load document list: {exc}")
