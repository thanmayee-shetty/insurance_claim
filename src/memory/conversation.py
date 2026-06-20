"""
src/memory/conversation.py
ConversationManager — all conversation memory operations:
  - get_conversation_history()
  - get_langchain_memory()
  - update_conversation_history()
  - get_context_summary()
  - regenerate_context_summary()
  - clear_conversation_history()
"""
from __future__ import annotations

from datetime import datetime

from langchain.memory import ConversationBufferWindowMemory
from langchain_ollama import OllamaLLM
from loguru import logger

from src.database.connection import get_db
from src.config import (
    OLLAMA_BASE_URL, LLM_MODEL,
    SUMMARY_EVERY_N, MAX_MESSAGES_PER_SESSION, LANGCHAIN_MEMORY_K,
)


class ConversationManager:
    """Manages conversation history stored in MongoDB conversation_history collection."""

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_conversation_history(self, session_id: str) -> list[dict]:
        """
        Return list of {query, answer, timestamp} dicts for the session.
        Returns up to 20 most recent turns.
        """
        doc = get_db().conversation_history.find_one({"session_id": session_id})
        if not doc:
            return []

        messages = doc.get("messages", [])
        turns: list[dict] = []
        for i in range(0, len(messages) - 1, 2):
            if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                turns.append({
                    "query":     messages[i]["content"],
                    "answer":    messages[i + 1]["content"],
                    "timestamp": messages[i]["timestamp"],
                })
        return turns[-20:]  # last 20 turns

    def get_langchain_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """
        Build a LangChain ConversationBufferWindowMemory (k=LANGCHAIN_MEMORY_K)
        populated from the last k turns in MongoDB.
        """
        memory = ConversationBufferWindowMemory(
            k=LANGCHAIN_MEMORY_K,
            return_messages=True,
        )
        history = self.get_conversation_history(session_id)
        for turn in history[-LANGCHAIN_MEMORY_K:]:
            memory.save_context(
                {"input": turn["query"]},
                {"output": turn["answer"]},
            )
        return memory

    def get_context_summary(self, session_id: str) -> str | None:
        """Return stored context_summary string, or None if not set."""
        doc = get_db().conversation_history.find_one(
            {"session_id": session_id},
            {"context_summary": 1},
        )
        if doc:
            return doc.get("context_summary") or None
        return None

    # ── Write ─────────────────────────────────────────────────────────────────

    def update_conversation_history(
        self,
        session_id:         str,
        user_query:         str,
        assistant_response: str,
    ) -> None:
        """
        Append a user+assistant message pair.
        Triggers context summary regeneration every SUMMARY_EVERY_N user messages.
        Enforces MAX_MESSAGES_PER_SESSION cap (oldest messages dropped).
        """
        now = datetime.utcnow()
        new_msgs = [
            {"role": "user",      "content": user_query,          "timestamp": now},
            {"role": "assistant", "content": assistant_response,   "timestamp": now},
        ]

        db  = get_db()
        doc = db.conversation_history.find_one({"session_id": session_id})

        if doc is None:
            db.conversation_history.insert_one({
                "session_id":      session_id,
                "messages":        new_msgs,
                "context_summary": "",
                "last_updated":    now,
            })
        else:
            all_msgs = doc["messages"] + new_msgs
            # Enforce hard cap — drop oldest messages
            all_msgs = all_msgs[-MAX_MESSAGES_PER_SESSION:]
            db.conversation_history.update_one(
                {"session_id": session_id},
                {"$set": {"messages": all_msgs, "last_updated": now}},
            )

            # Auto-trigger summary every SUMMARY_EVERY_N user messages
            user_count = sum(1 for m in all_msgs if m["role"] == "user")
            if user_count > 0 and user_count % SUMMARY_EVERY_N == 0:
                try:
                    self.regenerate_context_summary(session_id)
                except Exception as exc:
                    logger.warning(f"Auto-summary failed for {session_id}: {exc}")

    def regenerate_context_summary(self, session_id: str) -> str:
        """
        Use llama3.2:3b to compress conversation history into ≤100 words.
        Stores the result back in MongoDB.
        Returns the new summary string.
        """
        doc = get_db().conversation_history.find_one({"session_id": session_id})
        if not doc or not doc.get("messages"):
            return ""

        # Build transcript from last 20 messages
        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}"
            for m in doc["messages"][-20:]
        )
        prompt = (
            "Summarize this insurance claims conversation focusing on: "
            "patient details, policy numbers mentioned, procedures discussed, "
            "and decisions reached. Keep it under 100 words.\n\n"
            f"{transcript}"
        )

        try:
            llm     = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
            summary = llm.invoke(prompt).strip()
        except Exception as exc:
            logger.error(f"LLM summary failed for {session_id}: {exc}")
            summary = "[Summary unavailable]"

        get_db().conversation_history.update_one(
            {"session_id": session_id},
            {"$set": {"context_summary": summary}},
        )
        logger.info(f"Context summary regenerated for {session_id}: {summary[:80]}…")
        return summary

    def clear_conversation_history(self, session_id: str) -> None:
        """Delete all conversation history for a session (user-triggered reset)."""
        get_db().conversation_history.delete_one({"session_id": session_id})
        logger.info(f"Cleared conversation history for session {session_id}")
