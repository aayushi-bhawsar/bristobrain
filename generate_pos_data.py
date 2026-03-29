"""
synthetic_data/generate_pos_data.py
Generates a realistic 30-day synthetic POS dataset for BistroBrain demos.
Mimics data from Square, Toast, or Petpooja POS systems.
"""
from __future__ import annotations

import csv
import os
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────────
# Restaurant configuration
# ─────────────────────────────────────────────
RESTAURANT = {
    "name": "The Blue Plate Bistro",
    "city": "Mumbai",
    "avg_daily_covers": 45,
    "peak_days": ["Friday", "Saturday"],
    "slow_days": ["Tuesday", "Wednesday"],
}

MENU = [
    # (item_name, category, unit_price, food_cost_pct)
    ("Grilled Salmon", "food", 480, 0.38),
    ("Pasta Arrabiata", "food", 280, 0.22),
    ("Chicken Tikka Wrap", "food", 220, 0.30),
    ("Margherita Pizza", "food", 320, 0.26),
    ("Caesar Salad", "food", 190, 0.25),
    ("Mushroom Risotto", "food", 350, 0.32),
    ("Lamb Kebab Platter", "food", 520, 0.40),
    ("Fish & Chips", "food", 380, 0.35),
    ("Veg Burger", "food", 200, 0.24),
    ("Dal Makhani Bowl", "food", 180, 0.20),
    ("Cold Coffee", "beverage", 120, 0.18),
    ("Fresh Lime Soda", "beverage", 80, 0.15),
    ("Masala Chai", "beverage", 60, 0.12),
    ("Mango Lassi", "beverage", 110, 0.20),
    ("Sparkling Water", "beverage", 90, 0.10),
    ("Chocolate Lava Cake", "dessert", 220, 0.30),
    ("Gulab Jamun", "dessert", 120, 0.22),
    ("Cheesecake Slice", "dessert", 180, 0.28),
]

PAYMENT_METHODS = ["UPI", "Card", "Cash", "Wallet"]
SERVERS = ["server_01", "server_02", "server_03"]


def _covers_for_day(day_name: str) -> int:
    base = RESTAURANT["avg_daily_covers"]
    if day_name in RESTAURANT["peak_days"]:
        return int(base * random.uniform(1.3, 1.6))
    elif day_name in RESTAURANT["slow_days"]:
        return int(base * random.uniform(0.55, 0.75))
    return int(base * random.uniform(0.85, 1.15))


def generate_day(day_date: datetime) -> list[dict]:
    """Generate all transactions for a single day."""
    day_name = day_date.strftime("%A")
    covers = _covers_for_day(day_name)
    records: list[dict] = []

    for table_num in range(1, covers + 1):
        # Each table orders 2-4 items
        table_id = f"T{table_num:02d}"
        items_ordered = random.choices(MENU, k=random.randint(2, 4))

        # Random time within service hours (11am–10pm)
        hour = random.randint(11, 21)
        minute = random.randint(0, 59)
        order_time = day_date.replace(hour=hour, minute=minute, second=random.randint(0, 59))

        for item_name, category, unit_price, food_cost_pct in items_ordered:
            quantity = random.choices([1, 2], weights=[0.8, 0.2])[0]
            # Add slight price variation (±5%) to simulate real-world variance
            actual_price = round(unit_price * random.uniform(0.95, 1.05))
            food_cost = round(actual_price * food_cost_pct * random.uniform(0.9, 1.1), 2)

            records.append({
                "transaction_id": str(uuid.uuid4())[:8].upper(),
                "timestamp": order_time.strftime("%Y-%m-%d %H:%M:%S"),
                "item_name": item_name,
                "category": category,
                "quantity": quantity,
                "unit_price": actual_price,
                "food_cost": food_cost,
                "table_id": table_id,
                "server_id": random.choice(SERVERS),
                "payment_method": random.choice(PAYMENT_METHODS),
            })

    return records


def generate_dataset(days: int = 30, output_path: str | None = None) -> str:
    """Generate a full synthetic POS dataset and write to CSV."""
    if output_path is None:
        here = Path(__file__).parent
        output_path = str(here / "sample_data.csv")

    all_records: list[dict] = []
    start_date = datetime.now() - timedelta(days=days)

    for i in range(days):
        day = start_date + timedelta(days=i)
        day_records = generate_day(day)
        all_records.extend(day_records)

    if not all_records:
        raise RuntimeError("No records generated.")

    fieldnames = list(all_records[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"✅ Generated {len(all_records):,} transactions over {days} days → {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BistroBrain Synthetic POS Data Generator")
    parser.add_argument("--days", type=int, default=30, help="Days of data to generate")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    path = generate_dataset(days=args.days, output_path=args.output)
    print(f"Sample data available at: {path}")
    print("Next step: python -m data.ingestion --input", path)
