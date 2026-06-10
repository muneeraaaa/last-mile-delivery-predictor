"""
02_data_preview.py
──────────────────
Load and preview every table in the Olist dataset.

WHAT THIS SCRIPT DOES:
  1. Loads all 9 CSV files using our data_loader module
  2. Prints a column-level profile of every table
  3. Verifies that join keys align across tables
  4. Shows the order status breakdown (delivered, cancelled, etc.)
  5. Computes your baseline late delivery rate — the number the ML model must beat
  6. Shows the dataset date range

"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data_loader import load_all_tables, describe_table


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Load all tables
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  PHASE 2 — OLIST DATASET: LOAD & PREVIEW")
print("═" * 70)
print("\nLoading all 9 tables from data/raw/ ...\n")

dfs = load_all_tables(verbose=True)


orders               = dfs["orders"]
customers            = dfs["customers"]
order_items          = dfs["order_items"]
payments             = dfs["payments"]
reviews              = dfs["reviews"]
products             = dfs["products"]
sellers              = dfs["sellers"]
geolocation          = dfs["geolocation"]
category_translation = dfs["category_translation"]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Column-level profile of every table
# ═══════════════════════════════════════════════════════════════════════════════


print("\n\nColumn-level profile of every table...\n")

for name, df in dfs.items():
    describe_table(df, name)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Verify join keys
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 70)
print("  JOIN KEY VERIFICATION")
print("  (these counts tell you how well the tables align before merging)")
print("═" * 70 + "\n")

checks = [
    ("Unique order_id in  orders",     orders["order_id"].nunique()),
    ("Unique order_id in  items",      order_items["order_id"].nunique()),
    ("Unique order_id in  payments",   payments["order_id"].nunique()),
    ("Unique order_id in  reviews",    reviews["order_id"].nunique()),
    ("Unique customer_id in orders",   orders["customer_id"].nunique()),
    ("Unique customer_id in customers",customers["customer_id"].nunique()),
    ("Unique product_id in products",  products["product_id"].nunique()),
    ("Unique seller_id in sellers",    sellers["seller_id"].nunique()),
]

for label, count in checks:
    print(f"  {label:<40}  {count:>9,}")




# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Order status breakdown
# ═══════════════════════════════════════════════════════════════════════════════

print("═" * 70)
print("═" * 70 + "\n")

status_counts = orders["order_status"].value_counts()
total_orders  = len(orders)

for status, count in status_counts.items():
    pct = count / total_orders * 100
    bar = "█" * int(pct / 2)
    print(f"  {status:<20} {count:>8,}  ({pct:>5.1f}%)  {bar}")

delivered_count = status_counts.get("delivered", 0)



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Dataset date range
# ═══════════════════════════════════════════════════════════════════════════════

print("═" * 70)
print("  DATASET DATE RANGE")
print("═" * 70 + "\n")

earliest = orders["order_purchase_timestamp"].min()
latest   = orders["order_purchase_timestamp"].max()
span     = (latest - earliest).days

print(f"  Earliest order  :  {earliest}")
print(f"  Latest order    :  {latest}")
print(f"  Dataset spans   :  {span:,} days  (~{span // 365} years)")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Compute the baseline late delivery rate
# ═══════════════════════════════════════════════════════════════════════════════


print("═" * 70)
print("  BASELINE: ON-TIME DELIVERY RATE")
print("═" * 70 + "\n")

delivered = orders[orders["order_status"] == "delivered"].copy()

has_both_dates = (
    delivered["order_delivered_customer_date"].notna() &
    delivered["order_estimated_delivery_date"].notna()
)
delivered_clean = delivered[has_both_dates].copy()

late_mask  = (
    delivered_clean["order_delivered_customer_date"] >
    delivered_clean["order_estimated_delivery_date"]
)
late_count  = late_mask.sum()
total_clean = len(delivered_clean)
late_pct    = late_count / total_clean * 100
on_time_pct = 100 - late_pct

print(f"  Delivered orders with valid dates : {total_clean:>8,}")
print(f"  Late deliveries                   : {late_count:>8,}  ({late_pct:.1f}%)")
print(f"  On-time deliveries                : {total_clean - late_count:>8,}  ({on_time_pct:.1f}%)")


print("  Missing value summary (orders table):")
key_cols = [
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for col in key_cols:
    nulls = orders[col].isna().sum()
    pct   = nulls / len(orders) * 100
    print(f"    {col:<45} {nulls:>5,}  ({pct:.1f}% missing)")



print("═" * 70)
print("═" * 70 + "\n")
