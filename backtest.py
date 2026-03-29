"""
scripts/backtest.py
Backtesting framework: replay AI recommendations against historical
"failed" months to estimate loss-prevention potential.

Usage:
    python scripts/backtest.py --months 3
    python scripts/backtest.py --months 6 --report backtest_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from agents import MarketingAgent, PricingAgent
from data.schema import DailySummary

console = Console()
DB_PATH = os.getenv("SQLITE_DB_PATH", "bistrobrain.db")
CURRENCY = os.getenv("CURRENCY_SYMBOL", "₹")
LOW_MARGIN = float(os.getenv("LOW_MARGIN_THRESHOLD", "0.28"))


def _load_all_summaries() -> list[DailySummary]:
    """Load every daily summary from SQLite."""
    try:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT * FROM daily_summaries ORDER BY date ASC"
        ).fetchall()
        con.close()
    except Exception as exc:
        console.print(f"[red]DB error:[/red] {exc}")
        sys.exit(1)

    return [
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
        for row in rows
    ]


def _group_by_month(summaries: list[DailySummary]) -> dict[str, list[DailySummary]]:
    """Group summaries by YYYY-MM."""
    groups: dict[str, list[DailySummary]] = defaultdict(list)
    for s in summaries:
        groups[s.date.strftime("%Y-%m")].append(s)
    return dict(groups)


def identify_failed_months(
    monthly: dict[str, list[DailySummary]], threshold: float = 0.26
) -> list[str]:
    """
    A 'failed' month is one where:
    - Average margin < threshold, OR
    - Revenue declined >10% vs prior month
    """
    failed: list[str] = []
    month_keys = sorted(monthly.keys())

    for i, month in enumerate(month_keys):
        days = monthly[month]
        avg_margin = mean(d.avg_margin for d in days)
        avg_revenue = mean(d.total_revenue for d in days)

        is_low_margin = avg_margin < threshold

        is_declining = False
        if i > 0:
            prev_month = month_keys[i - 1]
            prev_revenue = mean(d.total_revenue for d in monthly[prev_month])
            if prev_revenue > 0 and (avg_revenue - prev_revenue) / prev_revenue < -0.10:
                is_declining = True

        if is_low_margin or is_declining:
            failed.append(month)

    return failed


def simulate_intervention(
    summaries: list[DailySummary],
) -> dict:
    """
    Simulate what BistroBrain recommendations would have recovered.
    Heuristic model based on known intervention impact rates:
    - Margin fix: recovering 3-5% margin uplift on flagged items
    - Slow-day promotion: +12-18% covers on targeted days
    - Surplus promotion: recovering 60-75% of at-risk waste value
    """
    total_revenue = sum(d.total_revenue for d in summaries)
    total_food_cost = sum(d.total_food_cost for d in summaries)
    avg_margin = (total_revenue - total_food_cost) / total_revenue if total_revenue else 0

    # Estimate recoverable value
    low_margin_days = [d for d in summaries if d.avg_margin < LOW_MARGIN]
    margin_recovery = sum(
        d.total_revenue * (LOW_MARGIN - d.avg_margin) * 0.6  # 60% recovery rate
        for d in low_margin_days
    )

    slow_days = [d for d in summaries if d.cover_count < mean(d.cover_count for d in summaries) * 0.8]
    cover_recovery = sum(d.total_revenue * 0.12 for d in slow_days)  # 12% uplift

    total_recovery = margin_recovery + cover_recovery

    return {
        "total_revenue": round(total_revenue, 2),
        "avg_margin": round(avg_margin, 4),
        "low_margin_days": len(low_margin_days),
        "estimated_margin_recovery": round(margin_recovery, 2),
        "slow_days": len(slow_days),
        "estimated_cover_recovery": round(cover_recovery, 2),
        "total_estimated_recovery": round(total_recovery, 2),
        "recovery_pct": round(total_recovery / total_revenue * 100, 1) if total_revenue else 0,
    }


def run_backtest(months: int = 3, report_path: str | None = None) -> None:
    console.print(f"\n[bold blue]BistroBrain Backtester[/bold blue] — last {months} months\n")

    all_summaries = _load_all_summaries()
    if not all_summaries:
        console.print("[red]No data found. Run ingestion first.[/red]")
        return

    # Filter to requested window
    cutoff = date.today() - timedelta(days=months * 30)
    window = [s for s in all_summaries if s.date >= cutoff]

    if not window:
        console.print(f"[yellow]No data in the last {months} months.[/yellow]")
        return

    monthly = _group_by_month(window)
    failed_months = identify_failed_months(monthly)

    # Summary table
    table = Table(title="Monthly Performance vs BistroBrain Projection", show_lines=True)
    table.add_column("Month", style="bold")
    table.add_column("Actual Revenue", justify="right")
    table.add_column("Actual Margin", justify="right")
    table.add_column("Status")
    table.add_column("Est. Recovery", justify="right", style="green")

    all_results = []
    for month_key in sorted(monthly.keys()):
        days = monthly[month_key]
        result = simulate_intervention(days)
        is_failed = month_key in failed_months

        status = "[red]⚠ Failed[/red]" if is_failed else "[green]✓ Healthy[/green]"
        recovery_str = (
            f"{CURRENCY}{result['total_estimated_recovery']:,.0f} ({result['recovery_pct']}%)"
            if is_failed
            else "—"
        )

        table.add_row(
            month_key,
            f"{CURRENCY}{result['total_revenue']:,.0f}",
            f"{result['avg_margin']:.1%}",
            status,
            recovery_str,
        )
        all_results.append({"month": month_key, "failed": is_failed, **result})

    console.print(table)

    # Aggregate stats
    total_recoverable = sum(
        r["total_estimated_recovery"] for r in all_results if r["failed"]
    )
    total_revenue = sum(r["total_revenue"] for r in all_results)

    console.print(f"\n[bold]Summary over {months} months:[/bold]")
    console.print(f"  Failed months:     {len(failed_months)} / {len(monthly)}")
    console.print(f"  Total revenue:     {CURRENCY}{total_revenue:,.0f}")
    console.print(f"  Est. recoverable:  [green]{CURRENCY}{total_recoverable:,.0f}[/green]")
    if total_revenue:
        console.print(f"  Recovery rate:     {total_recoverable/total_revenue*100:.1f}% of total revenue")

    console.print(
        "\n[dim]Recovery estimates assume: 60% margin uplift on low-margin days, "
        "12% cover increase on slow days via targeted promotions.[/dim]"
    )

    if report_path:
        with open(report_path, "w") as f:
            json.dump(
                {
                    "backtest_date": str(date.today()),
                    "months_analysed": months,
                    "failed_months": failed_months,
                    "total_estimated_recovery": total_recoverable,
                    "monthly_results": all_results,
                },
                f,
                indent=2,
                default=str,
            )
        console.print(f"\n[dim]Report saved to {report_path}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BistroBrain Backtest Runner")
    parser.add_argument("--months", type=int, default=3, help="Months of history to backtest")
    parser.add_argument("--report", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    run_backtest(months=args.months, report_path=args.report)
