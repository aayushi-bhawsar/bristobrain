"""
utils/weather.py
Fetches current and forecast weather for the restaurant's city.
Used by agents to correlate sales with weather (e.g. rainy days → lower footfall).
"""
from __future__ import annotations

import os
from datetime import date, datetime
from typing import Optional

import httpx
import structlog

log = structlog.get_logger(__name__)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
RESTAURANT_CITY = os.getenv("RESTAURANT_CITY", "Mumbai")
BASE_URL = "https://api.openweathermap.org/data/2.5"


class WeatherSnapshot:
    """Lightweight weather data container."""

    def __init__(
        self,
        city: str,
        date: date,
        condition: str,           # e.g. "Rain", "Clear", "Clouds"
        temp_celsius: float,
        humidity_pct: int,
        rain_mm: float = 0.0,
    ) -> None:
        self.city = city
        self.date = date
        self.condition = condition
        self.temp_celsius = temp_celsius
        self.humidity_pct = humidity_pct
        self.rain_mm = rain_mm

    @property
    def is_rainy(self) -> bool:
        return self.condition.lower() in ("rain", "drizzle", "thunderstorm") or self.rain_mm > 2.0

    @property
    def is_hot(self) -> bool:
        return self.temp_celsius >= 35.0

    @property
    def footfall_modifier(self) -> float:
        """
        Heuristic modifier for expected footfall based on weather.
        Returns a multiplier: 1.0 = normal, <1.0 = reduced, >1.0 = boosted.
        """
        if self.is_rainy:
            return 0.75       # ~25% footfall reduction on rainy days
        if self.is_hot:
            return 0.85       # Heat reduces lunch covers
        if self.condition == "Clear" and 20 <= self.temp_celsius <= 30:
            return 1.10       # Pleasant weather boosts walk-ins
        return 1.0

    def to_context_string(self) -> str:
        return (
            f"Weather on {self.date}: {self.condition}, "
            f"{self.temp_celsius:.1f}°C, humidity {self.humidity_pct}%, "
            f"rain {self.rain_mm:.1f}mm. "
            f"Footfall modifier: {self.footfall_modifier:.0%}."
        )

    def __repr__(self) -> str:
        return f"<WeatherSnapshot {self.city} {self.date}: {self.condition} {self.temp_celsius:.0f}°C>"


def get_today_weather(city: Optional[str] = None) -> Optional[WeatherSnapshot]:
    """
    Fetch current weather for the restaurant's city.
    Returns None if API key is not configured or request fails.
    """
    if not OPENWEATHER_API_KEY:
        log.debug("weather_api_key_missing", hint="Set OPENWEATHER_API_KEY in .env")
        return None

    target_city = city or RESTAURANT_CITY

    try:
        resp = httpx.get(
            f"{BASE_URL}/weather",
            params={
                "q": target_city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric",
            },
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()

        return WeatherSnapshot(
            city=target_city,
            date=date.today(),
            condition=data["weather"][0]["main"],
            temp_celsius=data["main"]["temp"],
            humidity_pct=data["main"]["humidity"],
            rain_mm=data.get("rain", {}).get("1h", 0.0),
        )

    except httpx.HTTPStatusError as exc:
        log.warning("weather_http_error", status=exc.response.status_code)
    except Exception as exc:
        log.warning("weather_fetch_failed", error=str(exc))

    return None


def get_forecast(days: int = 5, city: Optional[str] = None) -> list[WeatherSnapshot]:
    """
    Fetch N-day weather forecast.
    Returns an empty list if API key is missing or request fails.
    """
    if not OPENWEATHER_API_KEY:
        return []

    target_city = city or RESTAURANT_CITY

    try:
        resp = httpx.get(
            f"{BASE_URL}/forecast",
            params={
                "q": target_city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric",
                "cnt": days * 8,       # 3-hour intervals × 8 = 1 day
            },
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()

        # Deduplicate to one snapshot per calendar day (noon reading preferred)
        seen_dates: set[str] = set()
        snapshots: list[WeatherSnapshot] = []

        for item in data.get("list", []):
            dt = datetime.fromtimestamp(item["dt"])
            day_str = dt.date().isoformat()

            if day_str in seen_dates:
                continue
            if dt.hour not in (11, 12, 13, 14):   # Prefer midday reading
                continue

            seen_dates.add(day_str)
            snapshots.append(
                WeatherSnapshot(
                    city=target_city,
                    date=dt.date(),
                    condition=item["weather"][0]["main"],
                    temp_celsius=item["main"]["temp"],
                    humidity_pct=item["main"]["humidity"],
                    rain_mm=item.get("rain", {}).get("3h", 0.0),
                )
            )

            if len(snapshots) >= days:
                break

        log.info("weather_forecast_fetched", city=target_city, days=len(snapshots))
        return snapshots

    except Exception as exc:
        log.warning("weather_forecast_failed", error=str(exc))
        return []
