"""
data/ingestion.py
Normalizes raw POS CSV/JSON exports into BistroBrain's internal schema
and indexes them into ChromaDB for RAG retrieval.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd
import structlog

from data.schema import CategoryEnum, DailySummary, POSTransaction
from data.rag_store import RAGStore

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────
# Column name aliases for common POS systems
# (Square, Toast, Petpooja)
# ─────────────────────────────────────────────
COLUMN_ALIASES: dict[str, list[str]] = {
    "transaction_id": ["transaction_id", "order_id", "bill_no", "receipt_id"],
    "timestamp":      ["timestamp", "date_time", "order_time", "created_at"],
    "item_name":      ["item_name", "item", "product_name", "menu_item"],
    "category":       ["category", "item_category", "department", "type"],
    "quantity":       ["quantity", "qty", "amount"],
    "unit_price":     ["unit_price", "price", "selling_price", "rate"],
    "food_cost":      ["food_cost", "cost", "cogs", "item_cost"],
    "table_id":       ["table_id", "table_no", "table_number"],
    "server_id":      ["server_id", "waiter_id", "staff_id"],
    "payment_method": ["payment_method", "payment_type", "tender"],
}


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename POS-system-specific columns to BistroBrain canonical names."""
    rename_map: dict[str, str] = {}
    df_cols_lower = {c.lower(): c for c in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df_cols_lower:
                rename_map[df_cols_lower[alias]] = canonical
                break

    return df.rename(columns=rename_map)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load and normalize a POS CSV export."""
    df = pd.read_csv(path)
    df = _resolve_columns(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["category"] = df["category"].str.lower().str.strip()
    df["category"] = df["category"].map(
        lambda x: x if x in CategoryEnum.__members__.values() else "misc"
    )
    log.info("csv_loaded", path=str(path), rows=len(df))
    return df


def load_json(path: str | Path) -> pd.DataFrame:
    """Load and normalize a POS JSON export."""
    with open(path) as f:
        data = json.load(f)
    # Handle both list-of-records and {data: [...]} shapes
    records = data if isinstance(data, list) else data.get("data", data.get("orders", []))
    df = pd.json_normalize(records)
    return load_csv.__wrapped__(df) if hasattr(load_csv, "__wrapped__") else _resolve_columns(df)


def df_to_transactions(df: pd.DataFrame) -> list[POSTransaction]:
    """Convert a normalized DataFrame to POSTransaction objects."""
    transactions: list[POSTransaction] = []
    for _, row in df.iterrows():
        try:
            txn = POSTransaction(
                transaction_id=str(row.get("transaction_id", uuid.uuid4())),
                timestamp=row["timestamp"],
                item_name=str(row["item_name"]),
                category=CategoryEnum(row.get("category", "misc")),
                quantity=float(row["quantity"]),
                unit_price=float(row["unit_price"]),
                food_cost=float(row.get("food_cost", row["unit_price"] * 0.30)),
                table_id=row.get("table_id"),
                server_id=row.get("server_id"),
                payment_method=row.get("payment_method"),
            )
            transactions.append(txn)
        except Exception as exc:
            log.warning("row_skipped", error=str(exc), row=row.to_dict())
    return transactions


def compute_daily_summaries(transactions: list[POSTransaction]) -> list[DailySummary]:
    """Aggregate transactions into daily summaries."""
    from collections import defaultdict

    day_groups: dict[str, list[POSTransaction]] = defaultdict(list)
    for txn in transactions:
        day_key = txn.timestamp.date().isoformat()
        day_groups[day_key].append(txn)

    summaries: list[DailySummary] = []
    for day_str, day_txns in sorted(day_groups.items()):
        total_revenue = sum(t.revenue for t in day_txns)
        total_food_cost = sum(t.food_cost * t.quantity for t in day_txns)
        gross_profit = total_revenue - total_food_cost

        # Top items by revenue
        item_revenue: dict[str, float] = {}
        for t in day_txns:
            item_revenue[t.item_name] = item_revenue.get(t.item_name, 0) + t.revenue
        top_items = sorted(item_revenue, key=lambda k: item_revenue[k], reverse=True)[:5]

        # Low margin items
        low_margin = [t.item_name for t in day_txns if t.margin < 0.20]

        summaries.append(
            DailySummary(
                date=datetime.fromisoformat(day_str).date(),
                total_revenue=round(total_revenue, 2),
                total_food_cost=round(total_food_cost, 2),
                gross_profit=round(gross_profit, 2),
                avg_margin=round(gross_profit / total_revenue if total_revenue else 0, 4),
                cover_count=len({t.table_id for t in day_txns if t.table_id}),
                top_items=top_items,
                low_margin_items=list(set(low_margin)),
            )
        )
    return summaries


def persist_to_sqlite(summaries: list[DailySummary], db_path: str = "bistrobrain.db") -> None:
    """Persist daily summaries to SQLite for fast retrieval."""
    con = sqlite3.connect(db_path)
    records = [
        (
            str(s.date),
            s.total_revenue,
            s.total_food_cost,
            s.gross_profit,
            s.avg_margin,
            s.cover_count,
            json.dumps(s.top_items),
            json.dumps(s.low_margin_items),
        )
        for s in summaries
    ]
    con.execute(
        """CREATE TABLE IF NOT EXISTS daily_summaries (
            date TEXT PRIMARY KEY,
            total_revenue REAL,
            total_food_cost REAL,
            gross_profit REAL,
            avg_margin REAL,
            cover_count INTEGER,
            top_items TEXT,
            low_margin_items TEXT
        )"""
    )
    con.executemany(
        "INSERT OR REPLACE INTO daily_summaries VALUES (?,?,?,?,?,?,?,?)", records
    )
    con.commit()
    con.close()
    log.info("sqlite_persisted", rows=len(records), db=db_path)


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def ingest(input_path: str, db_path: str = "bistrobrain.db") -> list[DailySummary]:
    """Full ingestion pipeline: load → parse → persist → index."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    log.info("ingestion_started", file=str(path))

    if path.suffix.lower() == ".json":
        df = load_json(path)
    else:
        df = load_csv(path)

    transactions = df_to_transactions(df)
    summaries = compute_daily_summaries(transactions)
    persist_to_sqlite(summaries, db_path)

    # Index summaries into ChromaDB for RAG
    rag = RAGStore()
    rag.index_summaries(summaries)

    log.info("ingestion_complete", transactions=len(transactions), days=len(summaries))
    return summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BistroBrain POS Data Ingestion")
    parser.add_argument("--input", required=True, help="Path to POS CSV or JSON file")
    parser.add_argument("--db", default="bistrobrain.db", help="SQLite output path")
    args = parser.parse_args()

    summaries = ingest(args.input, args.db)
    print(f"✅ Ingested {len(summaries)} days of data successfully.")
