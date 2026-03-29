"""
api/routes/query.py
Natural language Q&A endpoint — "Why were my margins low last Tuesday?"
"""
from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, HTTPException

from agents.insight_agent import InsightAgent
from data.schema import QueryRequest, QueryResponse

log = structlog.get_logger(__name__)
router = APIRouter()

_insight_agent: InsightAgent | None = None


def _get_agent() -> InsightAgent:
    global _insight_agent
    if _insight_agent is None:
        _insight_agent = InsightAgent()
    return _insight_agent


@router.post("/query", response_model=QueryResponse)
async def query_insights(request: QueryRequest):
    """
    Ask a natural language question about your restaurant's performance.

    Examples:
    - "Why were my margins low last Tuesday?"
    - "What's my best-selling item this week?"
    - "Should I order more chicken this weekend?"
    - "How do my Thursday covers compare to Friday?"
    """
    start = time.perf_counter()

    try:
        agent = _get_agent()
        response = agent.answer(request)
    except Exception as exc:
        log.error("query_failed", question=request.question[:80], error=str(exc))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(exc)}")

    elapsed = time.perf_counter() - start
    log.info("query_answered", latency_s=round(elapsed, 2), question=request.question[:60])

    if elapsed > 5.0:
        log.warning("latency_target_missed", latency_s=round(elapsed, 2))

    return response


@router.get("/query/examples")
async def get_example_queries():
    """Return example questions an owner might ask."""
    return {
        "examples": [
            "Why were my margins low last Tuesday?",
            "What's driving my food costs up this week?",
            "Which items should I 86 from the menu?",
            "How do my weekend covers compare to weekdays?",
            "When should I reorder chicken stock?",
            "What's my most profitable item this month?",
            "Why did revenue drop on Wednesday?",
            "Should I run a happy hour this Friday?",
        ]
    }
