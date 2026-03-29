"""
tests/test_ingestion.py
Tests for the POS data ingestion pipeline.
"""
from __future__ import annotations

import csv
import io
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from data.ingestion import (
    _resolve_columns,
    compute_daily_summaries,
    df_to_transactions,
    load_csv,
)
from data.schema import CategoryEnum


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _write_sample_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


SAMPLE_ROWS = [
    {
        "transaction_id": "A001",
        "timestamp": "2026-03-01 12:30:00",
        "item_name": "Pasta Arrabiata",
        "category": "food",
        "quantity": "2",
        "unit_price": "280",
        "food_cost": "62",
        "table_id": "T01",
        "server_id": "server_01",
        "payment_method": "UPI",
    },
    {
        "transaction_id": "A002",
        "timestamp": "2026-03-01 13:00:00",
        "item_name": "Cold Coffee",
        "category": "beverage",
        "quantity": "1",
        "unit_price": "120",
        "food_cost": "22",
        "table_id": "T02",
        "server_id": "server_02",
        "payment_method": "Card",
    },
    {
        "transaction_id": "A003",
        "timestamp": "2026-03-02 19:30:00",
        "item_name": "Grilled Salmon",
        "category": "food",
        "quantity": "1",
        "unit_price": "480",
        "food_cost": "182",
        "table_id": "T05",
        "server_id": "server_01",
        "payment_method": "Cash",
    },
]


# ─────────────────────────────────────────────
# Column resolution
# ─────────────────────────────────────────────

class TestColumnResolution:
    def test_canonical_columns_unchanged(self):
        import pandas as pd
        df = pd.DataFrame(SAMPLE_ROWS)
        result = _resolve_columns(df)
        assert "transaction_id" in result.columns
        assert "item_name" in result.columns

    def test_alias_columns_renamed(self):
        import pandas as pd
        # Simulate Square-style column names
        rows = [
            {
                "order_id": "B001",
                "date_time": "2026-03-01 12:00:00",
                "product_name": "Pasta",
                "item_category": "food",
                "qty": "2",
                "selling_price": "280",
                "cogs": "62",
            }
        ]
        df = pd.DataFrame(rows)
        result = _resolve_columns(df)
        assert "transaction_id" in result.columns
        assert "item_name" in result.columns
        assert "unit_price" in result.columns
        assert "food_cost" in result.columns


# ─────────────────────────────────────────────
# CSV loading
# ─────────────────────────────────────────────

class TestCSVLoading:
    def test_load_valid_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        _write_sample_csv(str(csv_path), SAMPLE_ROWS)
        df = load_csv(str(csv_path))
        assert len(df) == 3

    def test_timestamp_parsed(self, tmp_path):
        import pandas as pd
        csv_path = tmp_path / "test.csv"
        _write_sample_csv(str(csv_path), SAMPLE_ROWS)
        df = load_csv(str(csv_path))
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_category_normalised(self, tmp_path):
        rows = [
            {**SAMPLE_ROWS[0], "category": "  FOOD  "},  # uppercase + whitespace
        ]
        csv_path = tmp_path / "test.csv"
        _write_sample_csv(str(csv_path), rows)
        df = load_csv(str(csv_path))
        assert df["category"].iloc[0] == "food"

    def test_unknown_category_becomes_misc(self, tmp_path):
        rows = [{**SAMPLE_ROWS[0], "category": "alcoholic_beverage"}]
        csv_path = tmp_path / "test.csv"
        _write_sample_csv(str(csv_path), rows)
        df = load_csv(str(csv_path))
        assert df["category"].iloc[0] == "misc"


# ─────────────────────────────────────────────
# Transaction parsing
# ─────────────────────────────────────────────

class TestTransactionParsing:
    def test_converts_rows_to_transactions(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        _write_sample_csv(str(csv_path), SAMPLE_ROWS)
        df = load_csv(str(csv_path))
        txns = df_to_transactions(df)
        assert len(txns) == 3

    def test_revenue_calculated_correctly(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        _write_sample_csv(str(csv_path), SAMPLE_ROWS[:1])  # Pasta row: 2 * 280 = 560
        df = load_csv(str(csv_path))
        txns = df_to_transactions(df)
        assert txns[0].revenue == pytest.approx(560.0)

    def test_bad_rows_skipped_gracefully(self, tmp_path):
        bad_rows = [
            {**SAMPLE_ROWS[0]},
            {**SAMPLE_ROWS[0], "unit_price": "not_a_number"},  # Bad row
        ]
        csv_path = tmp_path / "test.csv"
        _write_sample_csv(str(csv_path), bad_rows)
        df = load_csv(str(csv_path))
        txns = df_to_transactions(df)
        assert len(txns) == 1  # Bad row skipped


# ─────────────────────────────────────────────
# Daily summary aggregation
# ─────────────────────────────────────────────

class TestDailySummaries:
    def test_groups_by_date(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        _write_sample_csv(str(csv_path), SAMPLE_ROWS)
        df = load_csv(str(csv_path))
        txns = df_to_transactions(df)
        summaries = compute_daily_summaries(txns)
        # SAMPLE_ROWS spans 2 dates
        assert len(summaries) == 2

    def test_revenue_aggregated(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        _write_sample_csv(str(csv_path), SAMPLE_ROWS[:2])  # Both on 2026-03-01
        df = load_csv(str(csv_path))
        txns = df_to_transactions(df)
        summaries = compute_daily_summaries(txns)
        assert len(summaries) == 1
        # Revenue: (2*280) + (1*120) = 680
        assert summaries[0].total_revenue == pytest.approx(680.0)

    def test_margin_between_0_and_1(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        _write_sample_csv(str(csv_path), SAMPLE_ROWS)
        df = load_csv(str(csv_path))
        txns = df_to_transactions(df)
        summaries = compute_daily_summaries(txns)
        for s in summaries:
            assert 0.0 <= s.avg_margin <= 1.0
