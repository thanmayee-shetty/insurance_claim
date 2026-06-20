"""
ui/components/chat_display.py
Reusable helpers for rendering chat messages, confidence meters,
and recommendation badges consistently across pages.
"""
from __future__ import annotations

import streamlit as st


# Recommendation → (color, emoji)
_REC_STYLE: dict[str, tuple[str, str]] = {
    "APPROVE":         ("green",  "✅"),
    "REJECT":          ("red",    "❌"),
    "ESCALATE":        ("orange", "⚠️"),
    "NEEDS_MORE_INFO": ("blue",   "❓"),
}


def render_recommendation_badge(recommendation: str) -> None:
    """Display a coloured recommendation badge."""
    color, icon = _REC_STYLE.get(recommendation, ("grey", "•"))
    st.markdown(
        f"**Recommendation:** :{color}[{icon} {recommendation}]"
    )


def render_confidence_bar(confidence: float) -> None:
    """Display a confidence score as a progress bar with colour coding."""
    if confidence <= 0:
        return
    color = "green" if confidence >= 0.7 else "orange" if confidence >= 0.4 else "red"
    st.markdown(f"**Confidence:** :{color}[{confidence:.0%}]")
    st.progress(confidence)


def render_reasoning_chain(steps: list[str]) -> None:
    """Display reasoning chain in an expander."""
    if not steps:
        return
    with st.expander("🔍 View Reasoning Chain"):
        for i, step in enumerate(steps, 1):
            st.markdown(f"**Step {i}:** {step}")


def render_citations(citations: list[dict]) -> None:
    """Display citation cards in an expander."""
    if not citations:
        return
    with st.expander(f"📎 {len(citations)} Source Document(s)"):
        for c in citations:
            st.markdown(
                f"**📄 {c.get('doc_name', 'Unknown')}**  \n"
                f"*Section:* {c.get('section', '—')}  \n"
                f"*Similarity:* `{c.get('similarity_score', 0):.3f}`"
            )
            snippet = c.get("snippet", "")
            if snippet:
                st.caption(f"> {snippet[:300]}")
            st.divider()
