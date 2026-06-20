"""
ui/components/source_panel.py
Sidebar or inline panel showing document source details
for the most recently retrieved results.
"""
from __future__ import annotations

import streamlit as st


def render_source_panel(citations: list[dict]) -> None:
    """
    Render a full source panel (used in sidebar or wide-mode column).
    Shows each citation with score, section, and snippet.
    """
    if not citations:
        st.info("No sources retrieved for this query.")
        return

    st.markdown(f"### 📚 {len(citations)} Source(s) Retrieved")
    for i, c in enumerate(citations, 1):
        with st.container():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**{i}. {c.get('doc_name', 'Unknown')}**")
                st.caption(f"Section: *{c.get('section', '—')}*")
            with col_b:
                score = c.get("similarity_score", 0)
                color = "green" if score >= 0.75 else "orange" if score >= 0.5 else "red"
                st.markdown(f":{color}[{score:.2f}]")
            snippet = c.get("snippet", "")
            if snippet:
                st.text_area(
                    label="",
                    value=snippet[:400],
                    height=80,
                    disabled=True,
                    key=f"src_{i}_{c.get('chunk_id', i)}",
                )
            st.divider()
