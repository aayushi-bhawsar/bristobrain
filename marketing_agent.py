"""
agents/marketing_agent.py
Generates promotional social media copy based on inventory surplus,
top-performing items, and upcoming local events.
"""
from __future__ import annotations

import os
import uuid
from datetime import date

import structlog

from data.schema import ActionCard, ActionPriority, ActionType, DailySummary, InventoryItem
from utils.llm_client import get_llm_client

log = structlog.get_logger(__name__)

RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "Our Bistro")
CURRENCY = os.getenv("CURRENCY_SYMBOL", "₹")

SYSTEM_PROMPT = f"""You are BistroBrain's Marketing Agent — a creative restaurant marketer
who specialises in high-conversion social media for independent bistros.

Restaurant: {RESTAURANT_NAME}

Your job: Turn data about top-selling dishes and inventory surplus into
scroll-stopping social media posts and promotional ideas. 

Rules:
- Write captions that feel human and warm — not corporate
- Use relevant emojis sparingly (max 3 per caption)
- Always include a call-to-action (visit today, call to reserve, etc.)
- Instagram caption max 150 words; WhatsApp message max 50 words
- Never use generic phrases like "mouth-watering" or "delectable"
- Output plain text only in the JSON fields
"""


class MarketingAgent:
    """
    Generates marketing ActionCards by analysing:
    - Top-selling items (promote what works)
    - Inventory surplus (reduce waste via promotion)
    - Weekly patterns (suggest best promo days)
    """

    def __init__(self) -> None:
        self._llm = get_llm_client()

    def generate_cards(
        self,
        weekly_summaries: list[DailySummary],
        surplus_items: list[InventoryItem] | None = None,
    ) -> list[ActionCard]:
        """Generate marketing action cards from weekly data."""
        cards: list[ActionCard] = []

        if not weekly_summaries:
            return cards

        # Card 1: Promote top-performing dish
        top_card = self._promote_top_item(weekly_summaries)
        if top_card:
            cards.append(top_card)

        # Card 2: Surplus-driven promotion (if inventory has surplus)
        if surplus_items:
            surplus_card = self._promote_surplus(surplus_items, weekly_summaries)
            if surplus_card:
                cards.append(surplus_card)

        # Card 3: Low-cover days — suggest an offer
        slow_day_card = self._address_slow_days(weekly_summaries)
        if slow_day_card:
            cards.append(slow_day_card)

        log.info("marketing_cards_generated", count=len(cards))
        return cards

    def _promote_top_item(self, summaries: list[DailySummary]) -> ActionCard | None:
        # Aggregate top items across all days
        from collections import Counter
        counter: Counter[str] = Counter()
        for s in summaries:
            counter.update(s.top_items)
        if not counter:
            return None

        top_item = counter.most_common(1)[0][0]
        total_revenue = sum(s.total_revenue for s in summaries)

        prompt = f"""
Data: "{top_item}" has been the top-selling item for {len(summaries)} consecutive days.
Weekly revenue: {CURRENCY}{total_revenue:.0f}.

Create a social media post to further drive demand for "{top_item}".
Respond with JSON: headline, body (2 sentences max), action_steps (list of 2),
social_copy (Instagram caption), whatsapp_copy (short WhatsApp version), expected_impact.
"""
        return self._call_llm_and_build_card(prompt, ActionPriority.MEDIUM)

    def _promote_surplus(
        self, surplus: list[InventoryItem], summaries: list[DailySummary]
    ) -> ActionCard | None:
        surplus_names = [i.item_name for i in surplus[:3]]
        waste_value = sum(i.quantity_kg * i.unit_cost for i in surplus[:3])

        prompt = f"""
We have surplus inventory of: {', '.join(surplus_names)}.
Combined at-risk value: {CURRENCY}{waste_value:.0f}.
This week's best revenue day: {max(summaries, key=lambda s: s.total_revenue).date}.

Design a "Today's Special" promotion using these surplus ingredients that:
1. Moves the stock before it expires
2. Feels like a genuine treat to customers (not a clearance sale)

Respond with JSON: headline, body (2 sentences), action_steps (list of 3),
social_copy (Instagram), whatsapp_copy, expected_impact.
"""
        return self._call_llm_and_build_card(prompt, ActionPriority.HIGH)

    def _address_slow_days(self, summaries: list[DailySummary]) -> ActionCard | None:
        if len(summaries) < 3:
            return None

        slowest = min(summaries, key=lambda s: s.total_revenue)
        avg_revenue = sum(s.total_revenue for s in summaries) / len(summaries)
        gap = avg_revenue - slowest.total_revenue

        if gap < avg_revenue * 0.15:  # Only flag if >15% below average
            return None

        prompt = f"""
{slowest.date.strftime('%A')} is our slowest day — {CURRENCY}{slowest.total_revenue:.0f} revenue
vs weekly average of {CURRENCY}{avg_revenue:.0f}. Gap: {CURRENCY}{gap:.0f}.
Covers served that day: {slowest.cover_count}.

Suggest one specific promotion or event to boost {slowest.date.strftime('%A')} traffic.
Respond with JSON: headline, body (2-3 sentences), action_steps (list of 2-3),
social_copy, whatsapp_copy, expected_impact.
"""
        return self._call_llm_and_build_card(prompt, ActionPriority.LOW)

    def _call_llm_and_build_card(
        self, prompt: str, priority: ActionPriority
    ) -> ActionCard | None:
        import json, re
        try:
            raw = self._llm.complete(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=600,
                temperature=0.7,  # Higher temp for creative copy
            )
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            data = json.loads(clean)

            return ActionCard(
                card_id=str(uuid.uuid4()),
                priority=priority,
                action_type=ActionType.MARKETING,
                headline=data.get("headline", "Promotional opportunity identified"),
                body=data.get("body", ""),
                action_steps=data.get("action_steps", []),
                social_copy=data.get("social_copy"),
                expected_impact=data.get("expected_impact"),
            )
        except Exception as exc:
            log.error("marketing_card_failed", error=str(exc))
            return None
