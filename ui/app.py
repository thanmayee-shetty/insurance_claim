"""
ui/app.py
Main Streamlit entrypoint — configures page, runs session expiry check,
and shows the home/navigation landing page.
Launch: streamlit run ui/app.py
"""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.memory.session import session_expiry_check

st.set_page_config(
    page_title  = "InsureAI — Policy Analysis",
    page_icon   = "🏥",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# Run session cleanup once per app start
if "expiry_checked" not in st.session_state:
    try:
        deleted = session_expiry_check()
        if deleted:
            st.toast(f"🗑️ Cleaned up {deleted} expired session(s).", icon="🗑️")
    except Exception:
        pass   # MongoDB not ready yet — silently skip
    st.session_state.expiry_checked = True

# ── Landing page ──────────────────────────────────────────────────────────────
st.title("🏥 InsureAI — Medical Insurance Policy Analyser")
st.markdown("""
Welcome to the **Agentic RAG system** for hospital insurance analysis.

### Features
| Page | Description |
|---|---|
| 🔍 **Query** | Ask questions about coverage, claims, and regulations |
| 📂 **Documents** | Upload and manage policy documents |
| 📋 **Audit Log** | View all past queries and compliance records |

### Quick Start
1. Open the **Query** page from the sidebar
2. Type your patient case or policy question
3. The AI will search through ingested policy documents and return a structured analysis

### System Status
""")

# Quick status check
col1, col2, col3 = st.columns(3)
try:
    from src.database.connection import check_health
    health = check_health()
    col1.metric("MongoDB", "✅ Connected" if health["status"] == "ok" else "❌ Error")
    db_count = sum(1 for _ in health.get("collections", []))
    col2.metric("Collections", db_count)
except Exception:
    col1.metric("MongoDB", "❌ Not Connected")
    col2.metric("Collections", "—")

try:
    import requests as _req
    from src.config import OLLAMA_BASE_URL, LLM_MODEL
    r = _req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
    models = [m["name"] for m in r.json().get("models", [])]
    llm_ok = any(LLM_MODEL in m for m in models)
    col3.metric("Ollama LLM", "✅ Ready" if llm_ok else "⚠️ Model not found")
except Exception:
    col3.metric("Ollama LLM", "❌ Not Connected")

st.markdown("---")
st.info("👈 Use the sidebar to navigate to the **Query** page to get started.")
