"""
scripts/run_daily_digest.py
Cron-friendly script to generate and optionally send the daily digest.
Run this every morning via cron:
    0 8 * * * /path/to/venv/bin/python /path/to/bistrobrain/scripts/run_daily_digest.py --send
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agents import InventoryAgent, MarketingAgent, PricingAgent
from data.schema import ActionCard, ActionPriority, DailySummary
from data.ingestion import _load_recent_summaries_from_db

log = structlog.get_logger(__name__)
console = Console()

CURRENCY = os.getenv("CURRENCY_SYMBOL", "₹")
DB_PATH = os.getenv("SQLITE_DB_PATH", "bistrobrain.db")
RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "My Bistro")


def _load_summaries(days: int) -> list[DailySummary]:
    """Load recent summaries from SQLite."""
    import sqlite3, json as _json

    try:
        con = sqlite3.connect(DB_PATH)
        cutoff = str(date.today() - timedelta(days=days))
        rows = con.execute(
            "SELECT * FROM daily_summaries WHERE date >= ? ORDER BY date DESC",
            (cutoff,),
        ).fetchall()
        con.close()
    except Exception as exc:
        console.print(f"[red]Database error:[/red] {exc}")
        console.print(
            "Run ingestion first: [bold]python -m data.ingestion --input your_data.csv[/bold]"
        )
        sys.exit(1)

    if not rows:
        console.print("[yellow]No data found.[/yellow] Generate sample data first:")
        console.print("  python synthetic_data/generate_pos_data.py")
        console.print("  python -m data.ingestion --input synthetic_data/sample_data.csv")
        sys.exit(0)

    return [
        DailySummary(
            date=date.fromisoformat(row[0]),
            total_revenue=row[1],
            total_food_cost=row[2],
            gross_profit=row[3],
            avg_margin=row[4],
            cover_count=row[5],
            top_items=_json.loads(row[6]),
            low_margin_items=_json.loads(row[7]),
        )
        for row in rows
    ]


def _print_summary(summary: DailySummary) -> None:
    console.rule(f"[bold blue]📊 {RESTAURANT_NAME} — Daily Digest ({date.today()})[/bold blue]")
    console.print(f"  Revenue:      [green]{CURRENCY}{summary.total_revenue:,.0f}[/green]")
    console.print(f"  Food Cost:    [yellow]{CURRENCY}{summary.total_food_cost:,.0f}[/yellow]")
    console.print(f"  Gross Profit: [green]{CURRENCY}{summary.gross_profit:,.0f}[/green]")
    console.print(f"  Avg Margin:   {'[red]' if summary.avg_margin < 0.28 else '[green]'}{summary.avg_margin:.1%}[/]")
    console.print(f"  Covers:       {summary.cover_count}")
    if summary.top_items:
        console.print(f"  Top Items:    {', '.join(summary.top_items[:3])}")
    console.print()


def _print_card(card: ActionCard, index: int) -> None:
    priority_colors = {
        ActionPriority.URGENT: "red",
        ActionPriority.HIGH: "orange3",
        ActionPriority.MEDIUM: "yellow",
        ActionPriority.LOW: "green",
    }
    priority_emoji = {
        ActionPriority.URGENT: "🔴",
        ActionPriority.HIGH: "🟠",
        ActionPriority.MEDIUM: "🟡",
        ActionPriority.LOW: "🟢",
    }

    color = priority_colors[card.priority]
    emoji = priority_emoji[card.priority]

    title = Text()
    title.append(f"{emoji} [{card.priority.upper()}] ", style=f"bold {color}")
    title.append(card.headline, style="bold white")

    body_text = card.body
    if card.action_steps:
        body_text += "\n\nSteps:"
        for i, step in enumerate(card.action_steps, 1):
            body_text += f"\n  {i}. {step}"
    if card.social_copy:
        body_text += f"\n\n📱 Caption: {card.social_copy}"
    if card.expected_impact:
        body_text += f"\n\n💡 Impact: {card.expected_impact}"

    console.print(Panel(body_text, title=title, border_style=color, padding=(1, 2)))


def _send_whatsapp(cards: list[ActionCard]) -> bool:
    """Send top action cards via Twilio WhatsApp."""
    try:
        from twilio.rest import Client

        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_WHATSAPP_FROM")
        to_number = os.getenv("RESTAURANT_OWNER_WHATSAPP")

        if not all([account_sid, auth_token, from_number, to_number]):
            console.print("[yellow]WhatsApp not configured. Set Twilio env vars to enable.[/yellow]")
            return False

        client = Client(account_sid, auth_token)

        # Send header
        header = (
            f"🍽️ *{RESTAURANT_NAME} — Daily Digest*\n"
            f"📅 {date.today().strftime('%A, %d %b %Y')}\n"
            f"{'─' * 30}"
        )
        client.messages.create(body=header, from_=from_number, to=to_number)

        # Send top 5 cards
        for card in cards[:5]:
            client.messages.create(
                body=card.to_whatsapp_message(),
                from_=from_number,
                to=to_number,
            )

        console.print(f"[green]✅ Digest sent to WhatsApp ({to_number})[/green]")
        return True

    except ImportError:
        console.print("[yellow]twilio not installed. Run: pip install twilio[/yellow]")
        return False
    except Exception as exc:
        console.print(f"[red]WhatsApp send failed:[/red] {exc}")
        return False


def run_digest(days: int = 7, send: bool = False, output_json: str | None = None) -> None:
    console.print(f"\n[bold]BistroBrain Daily Digest[/bold] — {date.today()}\n")

    summaries = _load_summaries(days)
    today_summary = summaries[0]
    _print_summary(today_summary)

    # Run agents
    console.print("[dim]Running agents...[/dim]")
    inventory_agent = InventoryAgent()
    marketing_agent = MarketingAgent()
    pricing_agent = PricingAgent()

    inventory_cards = inventory_agent.scan([])  # Pass real inventory in production
    marketing_cards = marketing_agent.generate_cards(summaries)
    pricing_cards = pricing_agent.analyse(summaries)

    all_cards = inventory_cards + marketing_cards + pricing_cards
    priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
    all_cards.sort(key=lambda c: priority_order.get(c.priority.value, 99))

    if not all_cards:
        console.print("[green]✅ No action items today. All metrics look healthy![/green]")
        return

    console.print(f"\n[bold]Action Cards ({len(all_cards)} total):[/bold]\n")
    for i, card in enumerate(all_cards, 1):
        _print_card(card, i)

    if output_json:
        with open(output_json, "w") as f:
            json.dump([c.model_dump(mode="json") for c in all_cards], f, indent=2, default=str)
        console.print(f"\n[dim]Cards saved to {output_json}[/dim]")

    if send:
        _send_whatsapp(all_cards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BistroBrain Daily Digest Runner")
    parser.add_argument("--days", type=int, default=7, help="Days of history to analyse")
    parser.add_argument("--send", action="store_true", help="Send digest via WhatsApp")
    parser.add_argument("--output", type=str, default=None, help="Save cards as JSON file")
    args = parser.parse_args()

    run_digest(days=args.days, send=args.send, output_json=args.output)
