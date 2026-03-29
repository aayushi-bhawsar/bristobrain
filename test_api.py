"""
tests/test_api.py
Integration tests for FastAPI endpoints.
Uses TestClient (no real server needed).
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def set_env(monkeypatch, tmp_path):
    """Set required env vars for tests."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
    monkeypatch.setenv("RESTAURANT_NAME", "Test Bistro")
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("APP_ENV", "test")

    # Seed the test DB with 7 days of data
    con = sqlite3.connect(db_path)
    con.execute(
        """CREATE TABLE daily_summaries (
            date TEXT PRIMARY KEY, total_revenue REAL, total_food_cost REAL,
            gross_profit REAL, avg_margin REAL, cover_count INTEGER,
            top_items TEXT, low_margin_items TEXT
        )"""
    )
    today = date.today()
    for i in range(7):
        d = today - timedelta(days=i)
        revenue = 10000 + i * 500
        margin = 0.30 - (i * 0.01)
        food_cost = revenue * (1 - margin)
        con.execute(
            "INSERT INTO daily_summaries VALUES (?,?,?,?,?,?,?,?)",
            (
                str(d), revenue, food_cost, revenue * margin,
                margin, 40 + i * 2,
                json.dumps(["Pasta Arrabiata", "Cold Coffee"]),
                json.dumps([]),
            ),
        )
    con.commit()
    con.close()
    return db_path


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


# ─────────────────────────────────────────────
# Health endpoints
# ─────────────────────────────────────────────

class TestHealthEndpoints:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "BistroBrain"
        assert data["status"] == "online"

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# ─────────────────────────────────────────────
# Action Cards endpoint
# ─────────────────────────────────────────────

class TestDigestEndpoint:
    @patch("api.routes.action_cards.InventoryAgent")
    @patch("api.routes.action_cards.MarketingAgent")
    @patch("api.routes.action_cards.PricingAgent")
    def test_digest_returns_200(
        self, mock_pricing, mock_marketing, mock_inventory, client
    ):
        # Mock all agents to return empty card lists
        for mock_cls in [mock_inventory, mock_marketing, mock_pricing]:
            instance = mock_cls.return_value
            instance.scan.return_value = []
            instance.generate_cards.return_value = []
            instance.analyse.return_value = []

        response = client.get("/api/digest")
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "action_cards" in data
        assert data["restaurant_name"] == "Test Bistro"


# ─────────────────────────────────────────────
# Q&A endpoint
# ─────────────────────────────────────────────

class TestQueryEndpoint:
    def test_example_queries_endpoint(self, client):
        response = client.get("/api/query/examples")
        assert response.status_code == 200
        data = response.json()
        assert "examples" in data
        assert len(data["examples"]) >= 5

    @patch("api.routes.query._get_agent")
    def test_query_returns_answer(self, mock_get_agent, client):
        from data.schema import QueryResponse

        mock_agent = MagicMock()
        mock_agent.answer.return_value = QueryResponse(
            question="Why were margins low last Tuesday?",
            answer="Food cost spiked due to a bulk chicken order on a low-revenue day.",
            confidence=0.85,
            sources=["POS data: Date: 2026-03-17..."],
            follow_up_suggestions=["Check if bulk orders align with high-cover days."],
        )
        mock_get_agent.return_value = mock_agent

        response = client.post(
            "/api/query",
            json={"question": "Why were margins low last Tuesday?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["confidence"] > 0

    def test_empty_question_rejected(self, client):
        response = client.post("/api/query", json={"question": "  "})
        assert response.status_code == 422  # Pydantic validation error

    def test_too_long_question_rejected(self, client):
        response = client.post("/api/query", json={"question": "x" * 501})
        assert response.status_code == 422


# ─────────────────────────────────────────────
# WhatsApp webhook
# ─────────────────────────────────────────────

class TestWhatsAppWebhook:
    def test_help_command_responds(self, client, monkeypatch):
        monkeypatch.setenv("RESTAURANT_OWNER_WHATSAPP", "")  # Allow any number
        response = client.post(
            "/api/webhook/whatsapp",
            data={"Body": "help", "From": "whatsapp:+91999999999", "To": "whatsapp:+14155238886"},
        )
        assert response.status_code == 200
        assert "BistroBrain" in response.text

    @patch("api.routes.webhook._get_agent")
    def test_question_routes_to_insight_agent(self, mock_get_agent, client, monkeypatch):
        monkeypatch.setenv("RESTAURANT_OWNER_WHATSAPP", "")
        from data.schema import QueryResponse

        mock_agent = MagicMock()
        mock_agent.answer.return_value = QueryResponse(
            question="Why were margins low?",
            answer="Bulk orders on slow days drove costs up.",
            confidence=0.80,
        )
        mock_get_agent.return_value = mock_agent

        response = client.post(
            "/api/webhook/whatsapp",
            data={
                "Body": "Why were margins low?",
                "From": "whatsapp:+91999999999",
                "To": "whatsapp:+14155238886",
            },
        )
        assert response.status_code == 200
        assert "Bulk orders" in response.text

    def test_unauthorized_number_blocked(self, client, monkeypatch):
        monkeypatch.setenv("RESTAURANT_OWNER_WHATSAPP", "whatsapp:+910000000000")
        response = client.post(
            "/api/webhook/whatsapp",
            data={
                "Body": "hello",
                "From": "whatsapp:+919999999999",  # Different number
                "To": "whatsapp:+14155238886",
            },
        )
        assert response.status_code == 200
        assert "Unauthorized" in response.text
