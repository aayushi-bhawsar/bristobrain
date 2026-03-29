"""
utils/events.py
Fetches local events (concerts, festivals, sports, holidays) near the
restaurant to predict footfall spikes and slow days.

Supports:
  - Google Calendar API (public holiday calendars)
  - Ticketmaster / PredictHQ API (concerts, sports, festivals)
  - Manual event seeding via .env or JSON file
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import httpx
import structlog

log = structlog.get_logger(__name__)

PREDICTHQ_TOKEN = os.getenv("PREDICTHQ_API_TOKEN", "")
RESTAURANT_CITY = os.getenv("RESTAURANT_CITY", "Mumbai")
RESTAURANT_TIMEZONE = os.getenv("RESTAURANT_TIMEZONE", "Asia/Kolkata")
EVENTS_SEED_FILE = os.getenv("EVENTS_SEED_FILE", "")   # Optional local JSON override


@dataclass
class LocalEvent:
    """A local event that may affect restaurant footfall."""
    title: str
    event_date: date
    category: str                             # "concert", "festival", "holiday", "sports", "other"
    expected_attendance: Optional[int] = None
    venue: Optional[str] = None
    distance_km: Optional[float] = None

    @property
    def footfall_modifier(self) -> float:
        """
        Estimate footfall impact on the restaurant.
        Nearby high-attendance events drive walk-ins; public holidays can cut or boost.
        """
        if self.category == "holiday":
            return 1.25       # Public holidays often boost dining out

        if self.expected_attendance and self.expected_attendance > 10_000:
            if self.distance_km and self.distance_km < 2.0:
                return 1.40   # Major nearby event = strong boost
            return 1.15       # Farther away, still some lift

        return 1.05           # Small event, marginal uplift

    def to_context_string(self) -> str:
        dist_str = f", {self.distance_km:.1f}km away" if self.distance_km else ""
        att_str = f", ~{self.expected_attendance:,} attendees" if self.expected_attendance else ""
        return (
            f"Local event on {self.event_date}: '{self.title}' ({self.category}{dist_str}{att_str}). "
            f"Expected footfall modifier: {self.footfall_modifier:.0%}."
        )


# ─────────────────────────────────────────────
# PredictHQ integration (recommended)
# ─────────────────────────────────────────────

def _fetch_predicthq(city: str, start: date, end: date) -> list[LocalEvent]:
    """Fetch events from PredictHQ API (free tier available)."""
    if not PREDICTHQ_TOKEN:
        return []

    try:
        resp = httpx.get(
            "https://api.predicthq.com/v1/events/",
            headers={"Authorization": f"Bearer {PREDICTHQ_TOKEN}"},
            params={
                "q": city,
                "active.gte": start.isoformat(),
                "active.lte": end.isoformat(),
                "category": "concerts,festivals,sports,public-holidays",
                "limit": 20,
                "sort": "rank",
            },
            timeout=8.0,
        )
        resp.raise_for_status()
        data = resp.json()

        events: list[LocalEvent] = []
        for item in data.get("results", []):
            start_str = item.get("start", "")[:10]
            try:
                event_date = date.fromisoformat(start_str)
            except ValueError:
                continue

            events.append(
                LocalEvent(
                    title=item.get("title", "Unknown event"),
                    event_date=event_date,
                    category=item.get("category", "other").replace("-", "_"),
                    expected_attendance=item.get("predicted_event_spend"),
                    venue=item.get("entities", [{}])[0].get("name") if item.get("entities") else None,
                )
            )

        log.info("predicthq_events_fetched", city=city, count=len(events))
        return events

    except Exception as exc:
        log.warning("predicthq_fetch_failed", error=str(exc))
        return []


# ─────────────────────────────────────────────
# Manual seed file fallback
# ─────────────────────────────────────────────

def _load_seed_events(seed_path: str) -> list[LocalEvent]:
    """
    Load manually curated events from a JSON file.
    Useful when no API key is available or for adding hyper-local events.

    JSON format:
    [
      {"title": "Ganesh Chaturthi", "date": "2026-08-25", "category": "festival"},
      {"title": "IPL Final", "date": "2026-05-26", "category": "sports", "attendance": 40000}
    ]
    """
    if not seed_path or not Path(seed_path).exists():
        return []

    try:
        with open(seed_path) as f:
            raw: list[dict] = json.load(f)

        events: list[LocalEvent] = []
        for item in raw:
            events.append(
                LocalEvent(
                    title=item["title"],
                    event_date=date.fromisoformat(item["date"]),
                    category=item.get("category", "other"),
                    expected_attendance=item.get("attendance"),
                    venue=item.get("venue"),
                    distance_km=item.get("distance_km"),
                )
            )

        log.info("seed_events_loaded", count=len(events), path=seed_path)
        return events

    except Exception as exc:
        log.warning("seed_events_failed", error=str(exc))
        return []


# ─────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────

def get_upcoming_events(
    days: int = 7,
    city: Optional[str] = None,
) -> list[LocalEvent]:
    """
    Return upcoming local events for the next N days.
    Merges PredictHQ results with any manually seeded events.
    De-duplicates by (title, date).
    """
    target_city = city or RESTAURANT_CITY
    start = date.today()
    end = start + timedelta(days=days)

    api_events = _fetch_predicthq(target_city, start, end)
    seed_events = _load_seed_events(EVENTS_SEED_FILE)

    # Merge and de-duplicate
    seen: set[tuple[str, str]] = set()
    merged: list[LocalEvent] = []

    for event in api_events + seed_events:
        key = (event.title.lower(), str(event.event_date))
        if key not in seen and start <= event.event_date <= end:
            seen.add(key)
            merged.append(event)

    merged.sort(key=lambda e: e.event_date)
    return merged


def events_to_context_string(events: list[LocalEvent]) -> str:
    """Convert a list of events to a compact string for LLM context."""
    if not events:
        return "No notable local events in the upcoming period."
    return "\n".join(e.to_context_string() for e in events)
