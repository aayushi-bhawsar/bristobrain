"""
tests/test_agents.py
Unit tests for BistroBrain agents.
Uses mock LLM responses to keep tests fast and API-free.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from data.schema import (
    ActionPriority,
    ActionType,
    CategoryEnum,
    DailySummary,
    InventoryItem,
    POSTransaction,
    QueryRequest,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_inventory() -> list[InventoryItem]:
    return [
        InventoryItem(
            item_id="inv_001",
            item_name="Salmon",
            category=CategoryEnum.FOOD,
            quantity_kg=5.0,
            unit_cost=800,
            expiry_date=date.today() + timedelta(days=1),  # Expiring tomorrow
            reorder_level_kg=2.0,
        ),
        InventoryItem(
            item_id="inv_002",
            item_name="Chicken",
            category=CategoryEnum.FOOD,
            quantity_kg=1.5,  # Below reorder level of 2.0
            unit_cost=250,
            expiry_date=date.today() + timedelta(days=5),
            reorder_level_kg=2.0,
        ),
        InventoryItem(
            item_id="inv_003",
            item_name="Pasta",
            category=CategoryEnum.FOOD,
            quantity_kg=10.0,
            unit_cost=120,
            expiry_date=date.today() + timedelta(days=30),
            reorder_level_kg=2.0,
        ),
    ]


@pytest.fixture
def sample_summaries() -> list[DailySummary]:
    base_date = date.today() - timedelta(days=7)
    revenues = [9200, 11500, 8400, 9800, 14200, 16800, 7300]
    margins = [0.31, 0.29, 0.24, 0.30, 0.33, 0.35, 0.22]

    return [
        DailySummary(
            date=base_date + timedelta(days=i),
            total_revenue=revenues[i],
            total_food_cost=revenues[i] * (1 - margins[i]),
            gross_profit=revenues[i] * margins[i],
            avg_margin=margins[i],
            cover_count=int(revenues[i] / 220),
            top_items=["Pasta Arrabiata", "Grilled Salmon", "Cold Coffee"],
            low_margin_items=["Lamb Kebab Platter"] if margins[i] < 0.27 else [],
        )
        for i in range(7)
    ]


MOCK_LLM_INVENTORY = json.dumps({
    "headline": "5kg salmon expiring in 24h — run a Seafood Special",
    "body": "Your salmon stock expires tomorrow. At ₹800/kg, that's ₹4,000 at risk. A targeted lunch special can recover most of this.",
    "action_steps": [
        "Add 'Salmon Teriyaki Special' to tomorrow's lunch menu at ₹420",
        "Post the offer on Instagram tonight",
        "Brief kitchen staff on prep volume by 10am"
    ],
    "social_copy": "🐟 Fresh catch alert! Salmon Teriyaki Special — only tomorrow at lunch. Limited plates, big flavour.",
    "expected_impact": "Recover ₹3,200–₹4,000 in potential waste"
})

MOCK_LLM_MARKETING = json.dumps({
    "headline": "Pasta is your star — make it the hero this weekend",
    "body": "Pasta Arrabiata topped sales for 5 consecutive days. Featuring it in a weekend combo can boost covers by 12–15%.",
    "action_steps": [
        "Create a 'Pasta Night' combo (pasta + drink + dessert) at ₹480",
        "Post a reel on Instagram Friday evening"
    ],
    "social_copy": "🍝 Our Pasta Arrabiata keeps selling out — and we're not sorry. Grab yours this weekend!",
    "whatsapp_copy": "Weekend special: Pasta combo at ₹480 only! 🍝",
    "expected_impact": "+₹1,800 revenue over weekend"
})

MOCK_LLM_PRICING = json.dumps({
    "headline": "Tuesday margins 22% below target — fix pricing or portions",
    "body": "Tuesday averages 22% margin vs your 30% target. Two low-margin items (Lamb Kebab) are the main culprits.",
    "action_steps": [
        "Increase Lamb Kebab price by ₹40 (still competitive)",
        "Reduce portion size by 5% to hit target cost",
        "Consider removing from Tuesday menu if adjustment doesn't help"
    ],
    "social_copy": None,
    "expected_impact": "+₹620/week in recovered margin"
})


# ─────────────────────────────────────────────
# Inventory Agent Tests
# ─────────────────────────────────────────────

class TestInventoryAgent:
    def test_detects_expiry_alert(self, sample_inventory):
        from agents.inventory_agent import InventoryAgent
        agent = InventoryAgent()
        alerts = agent._detect_alerts(sample_inventory)
        expiry_alerts = [a for a in alerts if a.alert_type == "expiry"]
        assert len(expiry_alerts) == 1
        assert expiry_alerts[0].item.item_name == "Salmon"

    def test_detects_low_stock_alert(self, sample_inventory):
        from agents.inventory_agent import InventoryAgent
        agent = InventoryAgent()
        alerts = agent._detect_alerts(sample_inventory)
        low_stock_alerts = [a for a in alerts if a.alert_type == "low_stock"]
        assert len(low_stock_alerts) == 1
        assert low_stock_alerts[0].item.item_name == "Chicken"

    def test_no_alert_for_healthy_item(self, sample_inventory):
        from agents.inventory_agent import InventoryAgent
        agent = InventoryAgent()
        alerts = agent._detect_alerts(sample_inventory)
        pasta_alerts = [a for a in alerts if a.item.item_name == "Pasta"]
        assert len(pasta_alerts) == 0

    @patch("agents.inventory_agent.get_llm_client")
    def test_generates_action_card(self, mock_llm_factory, sample_inventory):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MOCK_LLM_INVENTORY
        mock_llm_factory.return_value = mock_llm

        from agents.inventory_agent import InventoryAgent
        agent = InventoryAgent()
        cards = agent.scan(sample_inventory)

        assert len(cards) >= 1
        assert cards[0].action_type == ActionType.INVENTORY
        assert cards[0].priority in (ActionPriority.URGENT, ActionPriority.HIGH)
        assert len(cards[0].action_steps) >= 1

    @patch("agents.inventory_agent.get_llm_client")
    def test_fallback_card_on_llm_failure(self, mock_llm_factory, sample_inventory):
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = Exception("API timeout")
        mock_llm_factory.return_value = mock_llm

        from agents.inventory_agent import InventoryAgent
        agent = InventoryAgent()
        cards = agent.scan(sample_inventory)

        # Should still produce fallback cards
        assert len(cards) >= 1
        assert cards[0].headline  # Non-empty headline


# ─────────────────────────────────────────────
# Marketing Agent Tests
# ─────────────────────────────────────────────

class TestMarketingAgent:
    @patch("agents.marketing_agent.get_llm_client")
    def test_generates_top_item_card(self, mock_llm_factory, sample_summaries):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MOCK_LLM_MARKETING
        mock_llm_factory.return_value = mock_llm

        from agents.marketing_agent import MarketingAgent
        agent = MarketingAgent()
        cards = agent.generate_cards(sample_summaries)

        assert len(cards) >= 1
        marketing_cards = [c for c in cards if c.action_type == ActionType.MARKETING]
        assert len(marketing_cards) >= 1

    def test_empty_summaries_returns_no_cards(self):
        from agents.marketing_agent import MarketingAgent
        agent = MarketingAgent()
        cards = agent.generate_cards([])
        assert cards == []


# ─────────────────────────────────────────────
# Pricing Agent Tests
# ─────────────────────────────────────────────

class TestPricingAgent:
    @patch("agents.pricing_agent.get_llm_client")
    def test_detects_low_margin_days(self, mock_llm_factory, sample_summaries):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MOCK_LLM_PRICING
        mock_llm_factory.return_value = mock_llm

        from agents.pricing_agent import PricingAgent
        agent = PricingAgent()
        cards = agent.analyse(sample_summaries)

        # sample_summaries has 2 days below 0.28 threshold (margins 0.24 and 0.22)
        assert len(cards) >= 1

    def test_insufficient_data_returns_empty(self):
        from agents.pricing_agent import PricingAgent
        agent = PricingAgent()
        cards = agent.analyse([])
        assert cards == []


# ─────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────

class TestPOSTransaction:
    def test_margin_calculation(self):
        txn = POSTransaction(
            transaction_id="test_001",
            timestamp="2026-03-01 12:00:00",
            item_name="Pasta",
            category=CategoryEnum.FOOD,
            quantity=2,
            unit_price=280,
            food_cost=62,
        )
        assert txn.revenue == 560
        gross = 560 - (2 * 62)
        assert txn.gross_profit == pytest.approx(gross)
        assert txn.margin == pytest.approx(gross / 560)

    def test_zero_revenue_margin(self):
        txn = POSTransaction(
            transaction_id="test_002",
            timestamp="2026-03-01 12:00:00",
            item_name="Free Item",
            category=CategoryEnum.MISC,
            quantity=1,
            unit_price=0,
            food_cost=50,
        )
        assert txn.margin == 0.0


class TestActionCard:
    def test_whatsapp_format(self, sample_summaries):
        from data.schema import ActionCard, ActionPriority, ActionType
        card = ActionCard(
            card_id=str(uuid.uuid4()),
            priority=ActionPriority.HIGH,
            action_type=ActionType.INVENTORY,
            headline="Salmon expiring tomorrow",
            body="5kg of salmon needs to be used by tomorrow.",
            action_steps=["Run a seafood special", "Post on Instagram"],
            social_copy="Fresh catch special 🐟",
            expected_impact="Recover ₹4,000",
        )
        msg = card.to_whatsapp_message()
        assert "Salmon expiring tomorrow" in msg
        assert "seafood special" in msg
        assert "🟠" in msg  # HIGH priority emoji
