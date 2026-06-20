"""
tests/test_agents.py
Unit tests for agent nodes:
  - query_router (JSON parsing, follow-up detection, fallback)
  - reflection_agent (routing logic)
  - answer_agent (_extract_recommendation)
  - human_loop (ask_clarification_node)
  - api/models (state_to_response)
Run: pytest tests/test_agents.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock


def _base_state(**overrides) -> dict:
    """Helper: minimal AgentState-like dict."""
    state = {
        "query":                 "Is knee replacement covered?",
        "session_id":            "sess_abc123def456",
        "user_role":             "insurance_staff",
        "conversation_history":  [],
        "context_summary":       "",
        "needs_clarification":   False,
        "clarification_question":None,
        "router_decision":       {},
        "retrieved_chunks":      [],
        "retrieval_sources":     [],
        "retrieval_strategy":    "parallel",
        "reflection_result":     {},
        "retry_count":           0,
        "max_retries":           3,
        "final_answer":          "",
        "citations":             [],
        "reasoning_chain":       [],
        "recommendation":        "",
        "response_time_ms":      0,
        "created_at":            datetime.utcnow(),
    }
    state.update(overrides)
    return state


# ── Query Router ──────────────────────────────────────────────────────────────
class TestQueryRouter:
    def test_json_parse_valid(self):
        """_parse_router_output correctly parses well-formed JSON."""
        from src.agents.query_router import _parse_router_output

        raw = """{
          "intent": "coverage_check",
          "doc_types": ["policy"],
          "entities": {"policy_number": null, "diagnosis": "knee",
                       "procedure": "arthroplasty", "patient_age": 58, "amount": null},
          "needs_clarification": false,
          "clarification_question": null,
          "is_followup": false,
          "reasoning": "Coverage check for knee replacement"
        }"""
        result = _parse_router_output(raw, "knee replacement coverage")
        assert result["intent"] == "coverage_check"
        assert "policy" in result["doc_types"]
        assert result["needs_clarification"] is False

    def test_json_parse_with_markdown_fences(self):
        """_parse_router_output strips markdown code fences."""
        from src.agents.query_router import _parse_router_output

        raw = """```json
        {"intent":"general","doc_types":["policy"],"entities":{"policy_number":null,
        "diagnosis":null,"procedure":null,"patient_age":null,"amount":null},
        "needs_clarification":false,"clarification_question":null,
        "is_followup":false,"reasoning":"general"}
        ```"""
        result = _parse_router_output(raw, "test query")
        assert result["intent"] == "general"

    def test_json_parse_fallback_on_invalid(self):
        """_parse_router_output returns safe default when JSON is unparseable."""
        from src.agents.query_router import _parse_router_output

        with patch("src.agents.query_router._llm") as mock_llm:
            mock_llm.invoke.return_value = "still invalid json ..."
            result = _parse_router_output("this is not json", "any query")

        assert result["intent"] in ("general", "coverage_check")
        assert "doc_types" in result
        assert "entities" in result

    def test_followup_heuristic_detection(self):
        """Queries starting with 'what about' are detected as follow-ups."""
        from src.agents.query_router import _FOLLOWUP_TRIGGERS

        assert _FOLLOWUP_TRIGGERS.match("what about dialysis?")
        assert _FOLLOWUP_TRIGGERS.match("And if the patient is 70?")
        assert not _FOLLOWUP_TRIGGERS.match("Is knee replacement covered?")

    def test_router_node_sets_state_fields(self):
        """query_router_node populates router_decision and reasoning_chain."""
        from src.agents.query_router import query_router_node

        good_json = '{"intent":"coverage_check","doc_types":["policy"],' \
                    '"entities":{"policy_number":null,"diagnosis":"knee",' \
                    '"procedure":null,"patient_age":null,"amount":null},' \
                    '"needs_clarification":false,"clarification_question":null,' \
                    '"is_followup":false,"reasoning":"coverage check"}'

        with patch("src.agents.query_router._llm") as mock_llm:
            mock_llm.invoke.return_value = good_json
            state = query_router_node(_base_state())

        assert state["router_decision"]["intent"] == "coverage_check"
        assert len(state["reasoning_chain"]) == 1


# ── Reflection Agent ──────────────────────────────────────────────────────────
class TestReflectionAgent:
    def test_route_high_confidence_goes_to_synthesize(self):
        """confidence ≥ 0.7 → route to 'synthesize'."""
        from src.agents.reflection_agent import route_after_reflection

        state = _base_state(
            reflection_result={"confidence": 0.85, "recommendation": "answer"},
            retry_count=0,
        )
        assert route_after_reflection(state) == "synthesize"

    def test_route_low_confidence_triggers_retry(self):
        """confidence < 0.7 with retry budget → route to 'parallel_retrieval'."""
        from src.agents.reflection_agent import route_after_reflection

        state = _base_state(
            reflection_result={"confidence": 0.4, "recommendation": "expand_search"},
            retry_count=0,
            max_retries=3,
            router_decision={},
        )
        assert route_after_reflection(state) == "parallel_retrieval"

    def test_route_no_retry_budget_goes_fallback(self):
        """Exhausted retries → route to 'fallback'."""
        from src.agents.reflection_agent import route_after_reflection

        state = _base_state(
            reflection_result={"confidence": 0.2, "recommendation": "expand_search"},
            retry_count=3,
            max_retries=3,
            router_decision={},
        )
        assert route_after_reflection(state) == "fallback"

    def test_zero_chunks_instant_fallback(self):
        """0 retrieved chunks → reflection sets fallback immediately."""
        from src.agents.reflection_agent import reflection_agent_node

        with patch("src.agents.reflection_agent._llm"):
            state = reflection_agent_node(_base_state(retrieved_chunks=[]))

        assert state["reflection_result"]["recommendation"] == "fallback"
        assert state["reflection_result"]["confidence"] == 0.0


# ── Answer Agent ──────────────────────────────────────────────────────────────
class TestAnswerAgent:
    def test_extract_recommendation_approve(self):
        from src.agents.answer_agent import _extract_recommendation
        text = "## Recommendation\nAPPROVE — The claim is covered."
        assert _extract_recommendation(text) == "APPROVE"

    def test_extract_recommendation_reject(self):
        from src.agents.answer_agent import _extract_recommendation
        text = "The waiting period is not satisfied. REJECT this claim."
        assert _extract_recommendation(text) == "REJECT"

    def test_extract_recommendation_escalate_default(self):
        from src.agents.answer_agent import _extract_recommendation
        text = "This is ambiguous and requires further review."
        assert _extract_recommendation(text) == "ESCALATE"

    def test_fallback_node_sets_escalate(self):
        """fallback_node always sets recommendation=ESCALATE."""
        from src.agents.answer_agent import fallback_node

        state = fallback_node(_base_state(
            retrieved_chunks=[],
            reflection_result={"confidence": 0.1, "gaps": ["no data"]},
        ))
        assert state["recommendation"] == "ESCALATE"
        assert "⚠️" in state["final_answer"]

    def test_build_citations_uses_validated_chunks(self):
        """_build_citations returns only validated chunk IDs."""
        from src.agents.answer_agent import _build_citations

        chunks = [
            {"_id": "a1", "source_name": "Star Policy", "section_header": "Sec 3",
             "score": 0.9, "chunk_text": "text a"},
            {"_id": "b2", "source_name": "HDFC Policy", "section_header": "Sec 4",
             "score": 0.6, "chunk_text": "text b"},
        ]
        citations = _build_citations(chunks, validated_ids=["0"])   # index-based
        assert len(citations) == 1
        assert citations[0]["doc_name"] == "Star Policy"


# ── Human Loop ────────────────────────────────────────────────────────────────
class TestHumanLoop:
    def test_ask_clarification_sets_needs_more_info(self):
        """ask_clarification_node always sets recommendation=NEEDS_MORE_INFO."""
        from src.agents.human_loop import ask_clarification_node

        state = ask_clarification_node(_base_state(
            clarification_question="Please provide policy number."
        ))
        assert state["recommendation"] == "NEEDS_MORE_INFO"
        assert state["final_answer"] == "Please provide policy number."

    def test_ask_clarification_uses_gaps_when_no_question(self):
        """Without a pre-set question, it builds from reflection gaps."""
        from src.agents.human_loop import ask_clarification_node

        state = ask_clarification_node(_base_state(
            clarification_question=None,
            reflection_result={"gaps": ["Missing policy number", "Missing diagnosis"]},
        ))
        assert "Missing policy number" in state["final_answer"]


# ── API Models ────────────────────────────────────────────────────────────────
class TestApiModels:
    def test_state_to_response_success(self):
        """High confidence + non-empty answer → status=success."""
        from src.api.models import state_to_response

        state = _base_state(
            final_answer     = "## Summary\nThis is covered.\n\n## Recommendation\nAPPROVE",
            recommendation   = "APPROVE",
            reflection_result= {"confidence": 0.85},
            citations        = [],
            reasoning_chain  = ["Router: intent=coverage_check"],
            needs_clarification=False,
        )
        resp = state_to_response(state, include_reasoning=True)
        assert resp.status == "success"
        assert resp.recommendation == "APPROVE"
        assert resp.confidence == 0.85

    def test_state_to_response_clarification(self):
        """needs_clarification=True → status=needs_clarification."""
        from src.api.models import state_to_response

        state = _base_state(
            needs_clarification    = True,
            clarification_question = "Please provide the policy number.",
        )
        resp = state_to_response(state)
        assert resp.status == "needs_clarification"
        assert resp.clarification_question == "Please provide the policy number."

    def test_state_to_response_no_reasoning_when_disabled(self):
        """include_reasoning=False → reasoning_chain is empty."""
        from src.api.models import state_to_response

        state = _base_state(
            final_answer     = "APPROVE — covered.",
            recommendation   = "APPROVE",
            reflection_result= {"confidence": 0.8},
            reasoning_chain  = ["Step 1", "Step 2"],
        )
        resp = state_to_response(state, include_reasoning=False)
        assert resp.reasoning_chain == []
