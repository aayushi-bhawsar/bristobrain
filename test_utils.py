"""
tests/test_utils.py
Tests for weather, events, and LLM client utilities.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────
# Weather utility tests
# ─────────────────────────────────────────────

class TestWeatherSnapshot:
    def _make_snapshot(self, condition="Clear", temp=25.0, rain_mm=0.0):
        from utils.weather import WeatherSnapshot
        return WeatherSnapshot(
            city="Mumbai",
            date=date.today(),
            condition=condition,
            temp_celsius=temp,
            humidity_pct=60,
            rain_mm=rain_mm,
        )

    def test_rainy_condition_is_rainy(self):
        snap = self._make_snapshot(condition="Rain")
        assert snap.is_rainy is True

    def test_heavy_rain_mm_is_rainy(self):
        snap = self._make_snapshot(condition="Clear", rain_mm=5.0)
        assert snap.is_rainy is True

    def test_clear_day_not_rainy(self):
        snap = self._make_snapshot(condition="Clear", rain_mm=0.0)
        assert snap.is_rainy is False

    def test_hot_day_flag(self):
        snap = self._make_snapshot(temp=38.0)
        assert snap.is_hot is True

    def test_normal_temp_not_hot(self):
        snap = self._make_snapshot(temp=28.0)
        assert snap.is_hot is False

    def test_rainy_footfall_modifier_below_one(self):
        snap = self._make_snapshot(condition="Rain")
        assert snap.footfall_modifier < 1.0

    def test_pleasant_day_footfall_modifier_above_one(self):
        snap = self._make_snapshot(condition="Clear", temp=25.0)
        assert snap.footfall_modifier >= 1.0

    def test_context_string_contains_key_info(self):
        snap = self._make_snapshot(condition="Rain", temp=22.0, rain_mm=3.5)
        ctx = snap.to_context_string()
        assert "Rain" in ctx
        assert "22.0" in ctx

    def test_no_api_key_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENWEATHER_API_KEY", "")
        from utils.weather import get_today_weather
        result = get_today_weather()
        assert result is None


# ─────────────────────────────────────────────
# Events utility tests
# ─────────────────────────────────────────────

class TestLocalEvent:
    def _make_event(self, category="festival", attendance=None, distance_km=None):
        from utils.events import LocalEvent
        return LocalEvent(
            title="Test Event",
            event_date=date.today() + timedelta(days=1),
            category=category,
            expected_attendance=attendance,
            distance_km=distance_km,
        )

    def test_holiday_boosts_footfall(self):
        event = self._make_event(category="holiday")
        assert event.footfall_modifier > 1.0

    def test_large_nearby_concert_boosts_footfall(self):
        event = self._make_event(category="concert", attendance=15000, distance_km=1.0)
        assert event.footfall_modifier >= 1.40

    def test_small_event_minimal_impact(self):
        event = self._make_event(category="other")
        assert 1.0 <= event.footfall_modifier <= 1.10

    def test_context_string_includes_title(self):
        event = self._make_event(category="festival", attendance=5000)
        ctx = event.to_context_string()
        assert "Test Event" in ctx
        assert "festival" in ctx

    def test_no_api_key_returns_empty_list(self, monkeypatch):
        monkeypatch.setenv("PREDICTHQ_API_TOKEN", "")
        from utils.events import get_upcoming_events
        events = get_upcoming_events(days=7)
        assert isinstance(events, list)

    def test_seed_file_loading(self, tmp_path):
        import json
        seed = [
            {"title": "Diwali Festival", "date": str(date.today() + timedelta(days=2)), "category": "festival"},
            {"title": "IPL Match", "date": str(date.today() + timedelta(days=4)), "category": "sports", "attendance": 30000},
        ]
        seed_file = tmp_path / "events.json"
        seed_file.write_text(json.dumps(seed))

        from utils.events import _load_seed_events
        events = _load_seed_events(str(seed_file))
        assert len(events) == 2
        assert events[0].title == "Diwali Festival"
        assert events[1].expected_attendance == 30000


# ─────────────────────────────────────────────
# LLM client tests
# ─────────────────────────────────────────────

class TestLLMClient:
    def test_raises_without_any_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("GROQ_API_KEY", "")

        from utils.llm_client import LLMClient
        client = LLMClient(provider="anthropic")

        with pytest.raises(RuntimeError, match="No LLM API key configured"):
            client.complete("system", "user")

    @patch("utils.llm_client.anthropic")
    def test_anthropic_complete_called(self, mock_anthropic_module, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
        monkeypatch.setenv("PRIMARY_LLM", "anthropic")

        # Mock the Anthropic client response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test answer")]
        mock_response.usage.output_tokens = 42
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client

        from utils.llm_client import LLMClient
        client = LLMClient(provider="anthropic")
        result = client.complete("system", "hello", fast=False)

        assert result == "Test answer"
        mock_client.messages.create.assert_called_once()

    def test_get_llm_client_returns_singleton(self):
        from utils.llm_client import get_llm_client
        a = get_llm_client()
        b = get_llm_client()
        assert a is b
