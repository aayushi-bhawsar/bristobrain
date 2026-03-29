"""
agents/insight_agent.py
Answers natural language questions about the restaurant's performance
using RAG-retrieved POS context and Chain-of-Thought reasoning.
"""
from __future__ import annotations

import os
from datetime import date

import structlog

from data.rag_store import RAGStore
from data.schema import QueryRequest, QueryResponse
from utils.llm_client import get_llm_client

log = structlog.get_logger(__name__)

RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "Our Bistro")
CURRENCY = os.getenv("CURRENCY_SYMBOL", "₹")

SYSTEM_PROMPT = f"""You are BistroBrain — the AI Store Manager for {RESTAURANT_NAME}.
You have access to detailed Point-of-Sale (POS) data about the restaurant's performance.

Your role: Answer the owner's questions with analyst-level insight, expressed in plain English
that a busy restaurant owner can understand and act on immediately.

Guidelines:
- Always cite specific numbers from the provided data
- Identify root causes, not just symptoms ("food cost spiked because X, not just 'costs were high'")
- End every answer with 1-2 actionable follow-up suggestions
- If the data doesn't contain enough information, say so honestly
- Today's date is {date.today().isoformat()}
- Currency: {CURRENCY}

Reasoning style: Think step-by-step before answering (Chain-of-Thought), but only output
the final answer — not your reasoning process.
"""


class InsightAgent:
    """
    Handles natural language Q&A using:
    1. RAG retrieval from ChromaDB (relevant POS summaries)
    2. LLM reasoning (Claude/GPT-4o via Groq for speed)
    3. Structured QueryResponse output
    """

    def __init__(self) -> None:
        self._llm = get_llm_client()
        self._rag = RAGStore()

    def answer(self, request: QueryRequest) -> QueryResponse:
        """Process a natural language question and return an insight."""
        log.info("insight_query", question=request.question[:80])

        # Step 1: Retrieve relevant POS context
        context_docs = self._rag.query(request.question, n_results=6)
        context_text = "\n\n".join(context_docs) if context_docs else "No historical data available."

        # Step 2: Build the prompt with RAG context
        user_prompt = self._build_prompt(request.question, context_text, request.date_context)

        # Step 3: Call LLM (use Groq for fast response)
        try:
            raw_answer = self._llm.complete(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=800,
                temperature=0.2,
                fast=True,  # Route to Groq for sub-5s latency
            )
        except Exception as exc:
            log.error("insight_llm_failed", error=str(exc))
            raw_answer = (
                "I couldn't retrieve an answer right now. "
                "Please check your API key configuration."
            )

        # Step 4: Extract follow-up suggestions if present
        answer_text, follow_ups = self._parse_response(raw_answer)

        return QueryResponse(
            question=request.question,
            answer=answer_text,
            confidence=0.85 if context_docs else 0.40,
            sources=[f"POS data: {doc[:60]}..." for doc in context_docs[:3]],
            follow_up_suggestions=follow_ups,
        )

    def _build_prompt(
        self, question: str, context: str, date_context: date | None
    ) -> str:
        date_str = f" (focusing on {date_context})" if date_context else ""
        return f"""
Restaurant POS Data Context{date_str}:
─────────────────────────────
{context}
─────────────────────────────

Owner's Question: {question}

Answer the question based on the data above. Be specific with numbers.
End your response with a line starting "Follow-up:" listing 1-2 actionable next steps.
"""

    def _parse_response(self, raw: str) -> tuple[str, list[str]]:
        """Split main answer from follow-up suggestions."""
        follow_ups: list[str] = []
        lines = raw.strip().split("\n")
        answer_lines: list[str] = []

        in_followup = False
        for line in lines:
            if line.strip().lower().startswith("follow-up:") or line.strip().lower().startswith("follow up:"):
                in_followup = True
                rest = line.split(":", 1)[-1].strip()
                if rest:
                    follow_ups.append(rest)
            elif in_followup and line.strip().startswith(("- ", "• ", "1.", "2.", "3.")):
                follow_ups.append(line.strip().lstrip("-•123. ").strip())
            elif not in_followup:
                answer_lines.append(line)

        return "\n".join(answer_lines).strip(), follow_ups[:3]
