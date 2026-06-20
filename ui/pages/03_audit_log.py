"""
ui/pages/03_audit_log.py
Page 3 — Compliance Audit Log
  - Filterable table of all queries
  - Expandable detail view per entry
  - Export to CSV
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from src.database.connection import get_db

st.set_page_config(page_title="InsureAI — Audit Log", page_icon="📋", layout="wide")

st.title("📋 Compliance Audit Log")
st.caption("All queries, retrieved sources, and AI decisions are logged here for compliance review.")

# ── Filters ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    date_filter = st.selectbox(
        "Time Period",
        ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"],
        index=1,
    )
with col2:
    intent_filter = st.multiselect(
        "Intent",
        ["coverage_check", "claims_precedent", "regulatory_compliance", "general"],
        default=[],
    )
with col3:
    min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)

# Build date cutoff
cutoff_map = {
    "Last 24 hours": timedelta(hours=24),
    "Last 7 days":   timedelta(days=7),
    "Last 30 days":  timedelta(days=30),
    "All time":      None,
}
cutoff_delta = cutoff_map[date_filter]

# ── Fetch logs ────────────────────────────────────────────────────────────────
try:
    db    = get_db()
    query = {}
    if cutoff_delta:
        query["created_at"] = {"$gte": datetime.utcnow() - cutoff_delta}
    if intent_filter:
        query["query_intent"] = {"$in": intent_filter}
    if min_confidence > 0:
        query["confidence_score"] = {"$gte": min_confidence}

    logs = list(db.audit_logs.find(query).sort("created_at", -1).limit(200))

    if not logs:
        st.info("No audit log entries match the selected filters.")
    else:
        # Build DataFrame
        rows = []
        for log in logs:
            rows.append({
                "Timestamp":   log.get("created_at", datetime.utcnow()).strftime("%Y-%m-%d %H:%M"),
                "Query":       log.get("query_text", "")[:80],
                "Intent":      log.get("query_intent", "—"),
                "Confidence":  f"{log.get('confidence_score', 0):.0%}" if log.get("confidence_score") else "—",
                "Response ms": log.get("response_time_ms", "—"),
                "Session":     log.get("session_id", "")[-8:],
                "_id":         str(log.get("_id", "")),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df.drop(columns=["_id"]), use_container_width=True)
        st.caption(f"Showing {len(logs)} entries")

        # Export button
        csv = df.drop(columns=["_id"]).to_csv(index=False)
        st.download_button(
            label="⬇️ Export to CSV",
            data=csv,
            file_name=f"audit_log_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        # ── Detail expander ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔎 Detailed View")
        selected_id = st.selectbox(
            "Select an entry to view full details",
            options=[r["_id"] for r in rows],
            format_func=lambda x: next(
                (f"{r['Timestamp']} — {r['Query']}" for r in rows if r["_id"] == x), x
            ),
        )
        if selected_id:
            from bson import ObjectId
            detail = db.audit_logs.find_one({"_id": ObjectId(selected_id)})
            if detail:
                st.markdown(f"**Query:** {detail.get('query_text', '')}")
                st.markdown(f"**Intent:** {detail.get('query_intent', '—')} | "
                            f"**Confidence:** {detail.get('confidence_score', 0):.0%} | "
                            f"**Response time:** {detail.get('response_time_ms', '—')}ms")
                if detail.get("answer_text"):
                    with st.expander("📝 Answer"):
                        st.markdown(detail["answer_text"])
                if detail.get("reasoning_chain"):
                    with st.expander("🔍 Reasoning Chain"):
                        for step in detail["reasoning_chain"]:
                            st.markdown(f"- {step}")
                if detail.get("reflection_notes"):
                    with st.expander("🔎 Reflection Notes"):
                        st.text(detail["reflection_notes"])

except Exception as exc:
    st.error(f"Could not load audit log: {exc}")
    st.info("Make sure MongoDB is running and the system has processed at least one query.")
