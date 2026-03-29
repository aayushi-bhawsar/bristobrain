"""
api/routes/action_cards.py
Endpoints for generating and retrieving daily action cards.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, timedelta

import structlog
from fastapi import APIRouter, HTTPException, Query

from agents import InventoryAgent, MarketingAgent, PricingAgent
from data.schema import ActionCard, DailySummary, DailyDigestResponse

log = structlog.get_logger(__name__)
router = APIRouter()

DB_PATH = os.getenv("SQLITE_DB_PATH", "bistrobrain.db")
RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "My Bistro")


def _load_recent_summaries(days: int = 7) -> list[DailySummary]:
    """Load the last N days of summaries from SQLite."""
    try:
        con = sqlite3.connect(DB_PATH)
        cutoff = str(date.today() - timedelta(days=days))
        rows = con.execute(
            "SELECT * FROM daily_summaries WHERE date >= ? ORDER BY date DESC",
            (cutoff,),
        ).fetchall()
        con.close()
    except Exception as exc:
        log.warning("sqlite_unavailable", error=str(exc))
        return []

    summaries = []
    for row in rows:
        summaries.append(
            DailySummary(
                date=date.fromisoformat(row[0]),
                total_revenue=row[1],
                total_food_cost=row[2],
                gross_profit=row[3],
                avg_margin=row[4],
                cover_count=row[5],
                top_items=json.loads(row[6]),
                low_margin_items=json.loads(row[7]),
            )
        )
    return summaries


@router.get("/digest", response_model=DailyDigestResponse)
async def get_daily_digest(
    days: int = Query(default=7, ge=1, le=30, description="Days of history to analyse"),
    send_whatsapp: bool = Query(default=False, description="Send digest to WhatsApp"),
):
    """
    Generate today's full daily digest:
    - Summary of recent performance
    - Inventory action cards
    - Marketing action cards
    - Pricing action cards

    Optionally sends the digest to the configured WhatsApp number.
    """
    summaries = _load_recent_summaries(days)
    if not summaries:
        raise HTTPException(
            status_code=404,
            detail="No POS data found. Run the ingestion pipeline first: "
                   "python -m data.ingestion --input your_data.csv",
        )

    today_summary = summaries[0]  # Most recent

    # Run all agents
    inventory_agent = InventoryAgent()
    marketing_agent = MarketingAgent()
    pricing_agent = PricingAgent()

    # In production, load real inventory from DB
    # Here we use an empty list as placeholder
    inventory_cards = inventory_agent.scan([])
    marketing_cards = marketing_agent.generate_cards(summaries)
    pricing_cards = pricing_agent.analyse(summaries)

    # Merge and sort by priority
    all_cards: list[ActionCard] = inventory_cards + marketing_cards + pricing_cards
    priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
    all_cards.sort(key=lambda c: priority_order.get(c.priority.value, 99))

    whatsapp_sent = False
    if send_whatsapp and all_cards:
        try:
            from utils.whatsapp import send_action_cards
            send_action_cards(all_cards[:5])  # Send top 5 cards
            whatsapp_sent = True
        except Exception as exc:
            log.warning("whatsapp_send_failed", error=str(exc))

    return DailyDigestResponse(
        date=date.today(),
        restaurant_name=RESTAURANT_NAME,
        summary=today_summary,
        action_cards=all_cards,
        whatsapp_sent=whatsapp_sent,
    )


@router.get("/cards/{card_date}", response_model=list[ActionCard])
async def get_cards_for_date(card_date: date):
    """Retrieve cached action cards for a specific date (if available)."""
    # In production this would load from a cards table in SQLite
    raise HTTPException(
        status_code=501,
        detail="Historical card retrieval coming in v1.1. Use /api/digest for today's cards.",
    )
