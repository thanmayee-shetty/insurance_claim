"""
ui/components/upload_form.py
Reusable document upload form widget used in the Documents page.
Returns the uploaded file path and metadata on successful ingestion,
or None if no file was uploaded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st


def render_upload_form(save_root: Path) -> Optional[dict]:
    """
    Render an upload form and save the file to save_root/<doc_type>/.
    Returns a dict {filepath, doc_type, source_name} on upload, else None.
    """
    st.markdown("#### ⬆️ Upload a New Document")

    with st.form("upload_form", clear_on_submit=True):
        uploaded = st.file_uploader(
            "Choose a PDF or TXT file (max 50 MB)",
            type=["pdf", "txt"],
        )
        col1, col2 = st.columns(2)
        with col1:
            doc_type = st.selectbox(
                "Document Type",
                ["policy", "agreement", "regulation", "claim"],
            )
        with col2:
            source_name = st.text_input(
                "Source / Insurer Name",
                placeholder="e.g. Star Health 2024",
            )
        submitted = st.form_submit_button("Upload & Ingest", type="primary")

    if submitted and uploaded:
        # Resolve save directory (policies/, agreements/, etc.)
        folder_map = {
            "policy":    "policies",
            "agreement": "provider_agreements",
            "regulation":"regulations",
            "claim":     "historical_claims",
        }
        save_dir = save_root / folder_map.get(doc_type, doc_type + "s")
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / uploaded.name
        save_path.write_bytes(uploaded.getbuffer())

        st.success(f"✅ File saved: `{save_path.name}`")
        return {
            "filepath":    save_path,
            "doc_type":    doc_type,
            "source_name": source_name or uploaded.name,
        }

    if submitted and not uploaded:
        st.warning("Please select a file before clicking Upload.")

    return None
