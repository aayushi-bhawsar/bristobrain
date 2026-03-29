"""
agents/inventory_agent.py
Flags expiring inventory and low-stock items.
Produces ActionCards with specific, cost-aware recommendations.
"""
from __future__ import annotations

import os
import uuid
from datetime import date, timedelta

import structlog

from data.schema import (
    ActionCard,
    ActionPriority,
    ActionType,
    InventoryAlert,
    InventoryItem,
)
from utils.llm_client import get_llm_client

log = structlog.get_logger(__name__)

EXPIRY_ALERT_HOURS = int(os.getenv("EXPIRY_ALERT_HOURS", "48"))
CURRENCY = os.getenv("CURRENCY_SYMBOL", "₹")


SYSTEM_PROMPT = """You are BistroBrain's Inventory Agent — an expert restaurant manager
with 20 years of experience minimizing food waste and maximizing ingredient utilization.

Your job: Given an inventory alert, produce ONE specific, actionable recommendation
for the restaurant owner. Be direct, practical, and revenue-focused.

Rules:
- Always suggest a specific dish or promotion to use expiring stock
- Quantify the financial impact when possible (e.g., "recover ₹1,200 in potential waste")
- Keep the recommended action executable by a non-technical person
- If stock is critically low, suggest reorder quantities based on historical usage
- Output plain text only — no markdown headers, no bullet points in the body.
"""


class InventoryAgent:
    """
    Scans inventory items and generates ActionCards for:
    - Items expiring within EXPIRY_ALERT_HOURS
    - Items below reorder level
    """

    def __init__(self) -> None:
        self._llm = get_llm_client()

    def scan(self, inventory: list[InventoryItem]) -> list[ActionCard]:
        """Run a full inventory scan and return ActionCards."""
        alerts = self._detect_alerts(inventory)
        if not alerts:
            log.info("inventory_scan_clean", items=len(inventory))
            return []

        cards: list[ActionCard] = []
        for alert in alerts:
            card = self._generate_card(alert)
            if card:
                cards.append(card)

        log.info("inventory_scan_done", alerts=len(alerts), cards=len(cards))
        return cards

    def _detect_alerts(self, inventory: list[InventoryItem]) -> list[InventoryAlert]:
        """Identify expiring and low-stock items."""
        alerts: list[InventoryAlert] = []
        expiry_threshold = date.today() + timedelta(hours=EXPIRY_ALERT_HOURS)

        for item in inventory:
            # Expiry alert
            if item.expiry_date and item.expiry_date <= expiry_threshold:
                hours_left = item.days_until_expiry * 24 if item.days_until_expiry is not None else 0
                alerts.append(
                    InventoryAlert(
                        item=item,
                        alert_type="expiry",
                        urgency_hours=max(hours_left, 0),
                        suggested_action=f"Use {item.item_name} before expiry",
                    )
                )

            # Low stock alert
            elif item.is_low_stock:
                alerts.append(
                    InventoryAlert(
                        item=item,
                        alert_type="low_stock",
                        suggested_action=f"Reorder {item.item_name}",
                    )
                )

        # Sort by urgency: expiry first, then low stock
        return sorted(
            alerts,
            key=lambda a: (a.alert_type != "expiry", a.urgency_hours or 999),
        )

    def _generate_card(self, alert: InventoryAlert) -> ActionCard | None:
        """Use LLM to craft a specific, actionable recommendation."""
        item = alert.item
        waste_value = round(item.quantity_kg * item.unit_cost, 2)

        user_prompt = f"""
Inventory Alert:
- Item: {item.item_name}
- Category: {item.category.value}
- Quantity: {item.quantity_kg}kg
- Unit cost: {CURRENCY}{item.unit_cost}/kg
- Total at risk: {CURRENCY}{waste_value}
- Alert type: {alert.alert_type}
- Hours until expiry: {alert.urgency_hours or 'N/A'}
- Reorder level: {item.reorder_level_kg}kg

Write a SHORT, direct recommendation for the restaurant owner.
Include:
1. One headline (max 12 words)
2. A 2-3 sentence explanation
3. Exactly 2-3 concrete action steps
4. One ready-to-post social media caption (if it's an expiry alert)
5. The financial impact in {CURRENCY}

Format your response as JSON with keys:
headline, body, action_steps (list), social_copy (string or null), expected_impact
"""

        try:
            raw = self._llm.complete(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=512,
                temperature=0.4,
            )

            # Parse JSON response
            import json, re
            # Strip markdown fences if present
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            data = json.loads(clean)

            priority = (
                ActionPriority.URGENT
                if (alert.urgency_hours or 999) < 24
                else ActionPriority.HIGH
                if alert.alert_type == "expiry"
                else ActionPriority.MEDIUM
            )

            return ActionCard(
                card_id=str(uuid.uuid4()),
                priority=priority,
                action_type=ActionType.INVENTORY,
                headline=data.get("headline", alert.suggested_action),
                body=data.get("body", ""),
                action_steps=data.get("action_steps", []),
                social_copy=data.get("social_copy"),
                expected_impact=data.get("expected_impact"),
            )

        except Exception as exc:
            log.error("inventory_card_failed", item=item.item_name, error=str(exc))
            # Fallback: generate a basic card without LLM
            return ActionCard(
                card_id=str(uuid.uuid4()),
                priority=ActionPriority.HIGH,
                action_type=ActionType.INVENTORY,
                headline=f"{item.item_name} needs immediate attention",
                body=alert.suggested_action,
                action_steps=[alert.suggested_action],
                expected_impact=f"Prevent {CURRENCY}{waste_value} in waste",
            )
