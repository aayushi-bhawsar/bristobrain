"""
agents/pricing_agent.py
Suggests dynamic pricing and combo deal opportunities based on
margin analysis, cover patterns, and weather/event context.
"""
from __future__ import annotations

import os
import uuid
from statistics import mean, stdev

import structlog

from data.schema import ActionCard, ActionPriority, ActionType, DailySummary
from utils.llm_client import get_llm_client

log = structlog.get_logger(__name__)
CURRENCY = os.getenv("CURRENCY_SYMBOL", "₹")
LOW_MARGIN = float(os.getenv("LOW_MARGIN_THRESHOLD", "0.28"))

SYSTEM_PROMPT = """You are BistroBrain's Pricing Strategist — a restaurant revenue expert
specialising in independent bistros and cafés in price-sensitive markets.

Your job: Identify pricing opportunities and inefficiencies from POS data.
Be practical — these owners can't do complex yield management, but they can:
- Add a weekend surcharge on high-demand items
- Bundle slow-moving items with popular ones
- Run a limited-time combo to boost slow covers

Always frame suggestions in terms of profit impact (e.g., "+₹400/day").
Output plain JSON only. No markdown.
"""


class PricingAgent:
    """
    Analyses daily summaries to surface:
    - Items with margins below threshold
    - Cover-count dips on specific days
    - Combo bundling opportunities
    """

    def __init__(self) -> None:
        self._llm = get_llm_client()

    def analyse(self, summaries: list[DailySummary]) -> list[ActionCard]:
        """Run pricing analysis and return ActionCards."""
        if len(summaries) < 3:
            log.warning("pricing_insufficient_data", days=len(summaries))
            return []

        cards: list[ActionCard] = []

        # Low margin alert
        low_margin_days = [s for s in summaries if s.avg_margin < LOW_MARGIN]
        if len(low_margin_days) >= 2:
            card = self._low_margin_card(low_margin_days, summaries)
            if card:
                cards.append(card)

        # Cover variability — spot dip days
        cover_counts = [s.cover_count for s in summaries if s.cover_count > 0]
        if len(cover_counts) >= 3:
            avg_covers = mean(cover_counts)
            sd = stdev(cover_counts) if len(cover_counts) > 1 else 0
            dip_days = [s for s in summaries if s.cover_count < avg_covers - sd]
            if dip_days:
                card = self._cover_dip_card(dip_days, avg_covers, summaries)
                if card:
                    cards.append(card)

        log.info("pricing_analysis_done", cards=len(cards))
        return cards

    def _low_margin_card(
        self, low_days: list[DailySummary], all_days: list[DailySummary]
    ) -> ActionCard | None:
        avg_margin = mean(s.avg_margin for s in low_days)
        worst_day = min(low_days, key=lambda s: s.avg_margin)
        low_items = list({item for s in low_days for item in s.low_margin_items})[:5]
        revenue_at_risk = sum(s.total_revenue * (LOW_MARGIN - s.avg_margin) for s in low_days)

        prompt = f"""
Margin Alert:
- {len(low_days)} days with avg margin {avg_margin:.1%} (target: {LOW_MARGIN:.0%})
- Worst day: {worst_day.date} at {worst_day.avg_margin:.1%} margin
- Low-margin items: {', '.join(low_items) or 'unknown'}
- Estimated revenue gap: {CURRENCY}{revenue_at_risk:.0f}

Suggest 2-3 pricing actions (price increase, combo, portion adjustment, or supplier negotiation)
that a single-outlet bistro owner can implement this week.

JSON: headline, body (2-3 sentences), action_steps (list of 3), expected_impact.
social_copy: null.
"""
        return self._llm_to_card(prompt, ActionPriority.HIGH)

    def _cover_dip_card(
        self,
        dip_days: list[DailySummary],
        avg_covers: float,
        all_days: list[DailySummary],
    ) -> ActionCard | None:
        from collections import Counter
        day_names = Counter(s.date.strftime("%A") for s in dip_days)
        worst_day_name = day_names.most_common(1)[0][0]
        avg_dip_revenue = mean(s.total_revenue for s in dip_days)
        avg_revenue = mean(s.total_revenue for s in all_days)

        prompt = f"""
Cover Dip Analysis:
- {worst_day_name} consistently underperforms: avg {mean(s.cover_count for s in dip_days):.0f} covers vs {avg_covers:.0f} weekly avg
- Revenue on low-cover days: {CURRENCY}{avg_dip_revenue:.0f} vs avg {CURRENCY}{avg_revenue:.0f}
- Revenue gap per low-cover day: {CURRENCY}{avg_revenue - avg_dip_revenue:.0f}

Suggest one targeted promotion or event specifically for {worst_day_name} to close this gap.

JSON: headline, body (2 sentences), action_steps (list of 2),
social_copy (a short {worst_day_name}-themed promo post), expected_impact.
"""
        return self._llm_to_card(prompt, ActionPriority.MEDIUM)

    def _llm_to_card(self, prompt: str, priority: ActionPriority) -> ActionCard | None:
        import json, re
        try:
            raw = self._llm.complete(SYSTEM_PROMPT, prompt, max_tokens=500, temperature=0.3)
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            data = json.loads(clean)
            return ActionCard(
                card_id=str(uuid.uuid4()),
                priority=priority,
                action_type=ActionType.PRICING,
                headline=data.get("headline", "Pricing opportunity detected"),
                body=data.get("body", ""),
                action_steps=data.get("action_steps", []),
                social_copy=data.get("social_copy"),
                expected_impact=data.get("expected_impact"),
            )
        except Exception as exc:
            log.error("pricing_card_failed", error=str(exc))
            return None
