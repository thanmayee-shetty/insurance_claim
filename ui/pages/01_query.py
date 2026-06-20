"""
ui/pages/01_query.py
Page 1 — Query Interface
  - Chat-style UI with conversation history
  - Citation expander cards
  - Confidence meter
  - Reasoning chain toggle
  - Clarification prompts
  - Clear conversation button
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from src.api.models import QueryRequest
from src.api.query import process_query
from src.memory.session import generate_session_id
from src.memory.conversation import ConversationManager

st.set_page_config(
    page_title="InsureAI — Query",
    page_icon="🔍",
    layout="wide",
)

# ── Session bootstrap ─────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id    = generate_session_id()
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ── Helper: render response metadata ─────────────────────────────────────────
def _render_response_meta(meta: dict, show_reasoning: bool) -> None:
    """Render confidence meter, citation cards, and reasoning chain."""
    confidence     = meta.get("confidence", 0.0)
    recommendation = meta.get("recommendation", "")
    citations      = meta.get("citations", [])
    reasoning      = meta.get("reasoning_chain", [])

    # Confidence + recommendation row
    if confidence > 0:
        color = "green" if confidence >= 0.7 else "orange" if confidence >= 0.4 else "red"
        rec_color = {
            "APPROVE": "green", "REJECT": "red",
            "ESCALATE": "orange", "NEEDS_MORE_INFO": "blue",
        }.get(recommendation, "grey")
        st.markdown(
            f"**Confidence:** :{color}[{confidence:.0%}]  |  "
            f"**Recommendation:** :{rec_color}[{recommendation}]"
        )

    # Citation cards
    if citations:
        with st.expander(f"📎 {len(citations)} Source Document(s)"):
            for c in citations:
                st.markdown(
                    f"**📄 {c['doc_name']}** — *{c['section']}*  \n"
                    f"Similarity: `{c['similarity_score']:.3f}`  \n"
                    f"> {c['snippet']}"
                )
                st.divider()

    # Reasoning chain
    if show_reasoning and reasoning:
        with st.expander("🔍 Reasoning Chain"):
            for step in reasoning:
                st.markdown(f"- {step}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Session Settings")
    st.caption(f"🔑 Session ID: `{st.session_state.session_id}`")

    show_reasoning = st.toggle("🔍 Show Reasoning Chain", value=False)

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        try:
            ConversationManager().clear_conversation_history(
                st.session_state.session_id
            )
        except Exception:
            pass
        st.session_state.chat_messages = []
        st.session_state.session_id    = generate_session_id()
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    examples = [
        "Does the Star Health policy cover knee replacement for a 62-year-old?",
        "Show me claims where cardiac bypass was rejected",
        "What does IRDA say about pre-authorisation timelines?",
        "What is the co-payment for non-network hospitals?",
        "Is dialysis covered for a diabetic patient on a 2-year-old policy?",
    ]
    for ex in examples:
        if st.button(ex[:55] + "…" if len(ex) > 55 else ex, use_container_width=True):
            st.session_state["_prefill_query"] = ex

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🔍 Insurance Policy Query")
st.caption(
    "Ask questions about coverage, claims precedents, IRDA regulations, or TPA agreements."
)

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg and msg["meta"]:
            _render_response_meta(msg["meta"], show_reasoning)

# ── Query input ───────────────────────────────────────────────────────────────
prefill    = st.session_state.pop("_prefill_query", None)
user_input = st.chat_input(
    "Describe the patient case or ask a policy question…",
    key="chat_input",
) or prefill

if user_input:
    # Display user message
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process query
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analysing policy documents… (this may take 30–90 seconds)"):
            try:
                response = process_query(QueryRequest(
                    query             = user_input,
                    session_id        = st.session_state.session_id,
                    include_reasoning = show_reasoning,
                ))
            except Exception as exc:
                st.error(f"An error occurred: {exc}")
                st.stop()

        # ── Clarification path ────────────────────────────────────────────────
        if response.status == "needs_clarification":
            cq = response.clarification_question or "Please provide more details."
            st.warning(f"❓ **Clarification needed**\n\n{cq}")
            st.session_state.chat_messages.append({
                "role":    "assistant",
                "content": f"❓ **Clarification needed**\n\n{cq}",
                "meta":    {},
            })

        # ── Success / Fallback path ───────────────────────────────────────────
        else:
            if response.status == "fallback":
                st.warning(response.answer)
            else:
                st.markdown(response.answer)

            meta = {
                "confidence":      response.confidence,
                "recommendation":  response.recommendation,
                "citations":       [c.model_dump() for c in response.citations],
                "reasoning_chain": response.reasoning_chain,
            }
            _render_response_meta(meta, show_reasoning)

            st.session_state.chat_messages.append({
                "role":    "assistant",
                "content": response.answer,
                "meta":    meta,
            })
