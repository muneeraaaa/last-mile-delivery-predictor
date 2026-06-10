"""
03_eda.py
---------
Exploratory Data Analysis for the Last-Mile Delivery Risk Predictor.

This version combines BOTH:
1. Visual EDA:
   - interactive Plotly HTML charts
   - saved inside reports/figures/

2. CSV/report EDA:
   - table overview
   - missing values
   - baseline summary
   - late rates by state/category/payment/time
   - correlation reports
   - processed EDA dataset for Phase 4

"""

import sys
from pathlib import Path

# =============================================================================
# 0. PROJECT PATH SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# 1. OUTPUT FOLDERS
# =============================================================================

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. CONSTANTS
# =============================================================================

DAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

LATE_COLOR = "#e74c3c"
ONTIME_COLOR = "#2ecc71"
BAR_SCALE = "Reds"


# =============================================================================
# 3. ROBUST DATA LOADING
# =============================================================================
# We first try to use your src/data_loader.py.
# If it fails because of the category filename mismatch, this script falls back
# to loading the CSV files directly using the correct filenames.

TABLE_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
    "order_items": [
        "shipping_limit_date",
    ],
}


def local_load_all_tables(verbose=True):
   
    dfs = {}

    for table_name, filename in TABLE_FILES.items():
        path = DATA_RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"\nMissing file:\n  {path}\n\n"
                f"Check that the Olist CSV files are inside data/raw/."
            )

        parse_dates = DATE_COLUMNS.get(table_name, False)
        df = pd.read_csv(path, parse_dates=parse_dates)
        dfs[table_name] = df

        if verbose:
            print(
                f"  ✓ {table_name:<25} "
                f"{df.shape[0]:>9,} rows × {df.shape[1]:>2} cols"
            )

    return dfs


def load_tables_safely():
    
    try:
        from src.data_loader import load_all_tables

        print("\nTrying to load data using src/data_loader.py ...")
        dfs = load_all_tables(verbose=True)
        print("Loaded successfully using src/data_loader.py")
        return dfs

    except Exception as e:
        print("\nCould not load using src/data_loader.py.")
        print("Reason:")
        print(f"  {type(e).__name__}: {e}")
        print("\nFalling back to direct CSV loading from data/raw/ ...\n")

        dfs = local_load_all_tables(verbose=True)
        print("\nLoaded successfully using fallback loader.")
        return dfs


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def save_csv(df, filename):
    path = REPORTS_DIR / filename
    df.to_csv(path, index=False)
    print(f"  ✓ CSV saved: {path}")
    return path


def save_processed_csv(df, filename):
    path = DATA_PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    print(f"  ✓ Processed data saved: {path}")
    return path


def save_chart(fig, filename):
    
    path = FIGURES_DIR / filename
    fig.write_html(path)
    print(f"  ✓ Chart saved: {path}")
    return path


def most_common(series):
    mode_values = series.dropna().mode()

    if len(mode_values) == 0:
        return np.nan

    return mode_values.iloc[0]


def safe_describe(df):
    
    return df.describe(include="all").transpose().reset_index().rename(
        columns={"index": "column"}
    )


def late_rate_by_category(df, column_name, min_orders=100):
    
    temp = df.copy()
    temp[column_name] = temp[column_name].fillna("missing")

    result = (
        temp.groupby(column_name, dropna=False)
        .agg(
            orders=("order_id", "count"),
            late_orders=("is_late", "sum"),
            late_rate=("is_late", "mean"),
            avg_delay_days=("delivery_delay_days", "mean"),
            avg_delivery_days=("delivery_days_actual", "mean"),
        )
        .reset_index()
    )

    result["late_rate_pct"] = result["late_rate"] * 100
    result = result[result["orders"] >= min_orders]
    result = result.sort_values("late_rate_pct", ascending=False)

    return result


def make_quartile_stats(df, numeric_col, target_col="is_late"):
    
    temp = df.dropna(subset=[numeric_col]).copy()

    temp[f"{numeric_col}_quartile"] = pd.qcut(
        temp[numeric_col],
        q=4,
        duplicates="drop"
    )

    stats = (
        temp.groupby(f"{numeric_col}_quartile", observed=True)[target_col]
        .agg(late_rate="mean", order_count="count")
        .reset_index()
    )

    stats["late_rate_pct"] = stats["late_rate"] * 100
    stats[f"{numeric_col}_quartile"] = stats[f"{numeric_col}_quartile"].astype(str)

    return stats


# =============================================================================
# 5. LOAD DATA
# =============================================================================

print("\n" + "═" * 80)
print("  PHASE 3 — EXPLORATORY DATA ANALYSIS")
print("═" * 80)

dfs = load_tables_safely()

orders = dfs["orders"]
customers = dfs["customers"]
order_items = dfs["order_items"]
payments = dfs["payments"]
reviews = dfs["reviews"]
products = dfs["products"]
sellers = dfs["sellers"]
geolocation = dfs["geolocation"]
category_translation = dfs["category_translation"]


# =============================================================================
# 6. TABLE OVERVIEW CSV
# =============================================================================

print("\n" + "═" * 80)
print("  TABLE OVERVIEW")
print("═" * 80)

table_overview_rows = []

for table_name, df in dfs.items():
    table_overview_rows.append(
        {
            "table_name": table_name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }
    )

table_overview = pd.DataFrame(table_overview_rows)
print(table_overview.to_string(index=False))
save_csv(table_overview, "phase3_table_overview.csv")


# =============================================================================
# 7. SUMMARY STATISTICS CSVs
# =============================================================================

print("\n" + "═" * 80)
print("  SUMMARY STATISTICS")
print("═" * 80)

orders_summary = safe_describe(orders)
items_summary = safe_describe(order_items[["price", "freight_value"]])
products_summary = safe_describe(
    products[
        [
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]
    ]
)

save_csv(orders_summary, "phase3_summary_orders.csv")
save_csv(items_summary, "phase3_summary_order_items_price_freight.csv")
save_csv(products_summary, "phase3_summary_product_dimensions.csv")

print("\nOrders summary preview:")
print(orders_summary.head(10).to_string(index=False))


# =============================================================================
# 8. MISSING VALUES CSV
# =============================================================================

print("\n" + "═" * 80)
print("  MISSING VALUE ANALYSIS")
print("═" * 80)

missing_rows = []

for table_name, df in dfs.items():
    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_pct = missing_count / len(df) * 100

        missing_rows.append(
            {
                "table_name": table_name,
                "column": column,
                "dtype": str(df[column].dtype),
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 2),
                "unique_values": int(df[column].nunique(dropna=True)),
            }
        )

missing_values = pd.DataFrame(missing_rows)
missing_values = missing_values.sort_values(
    ["missing_pct", "missing_count"],
    ascending=False
)

print(missing_values.head(25).to_string(index=False))
save_csv(missing_values, "phase3_missing_values.csv")


# =============================================================================
# 9. BUILD EDA DATAFRAME
# =============================================================================

print("\n" + "═" * 80)
print("  BUILDING ENRICHED EDA DATAFRAME")
print("═" * 80)

# -------------------------------------------------------------------------
# A. Keep only delivered orders with the dates needed to label late/on-time.
# -------------------------------------------------------------------------

delivered = (
    orders
    .loc[orders["order_status"] == "delivered"]
    .dropna(
        subset=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    )
    .copy()
)

raw_delivered_count = int((orders["order_status"] == "delivered").sum())
dropped_count = raw_delivered_count - len(delivered)

print(f"Delivered orders before date filtering : {raw_delivered_count:,}")
print(f"Dropped due to missing required dates  : {dropped_count:,}")
print(f"Working delivered dataset             : {len(delivered):,}")


# -------------------------------------------------------------------------
# B. Delivery-time calculations.
# -------------------------------------------------------------------------
# delivery_days_actual:
#   actual delivery date - order purchase date
#
# delivery_days_estimated:
#   estimated/promised delivery date - order purchase date
#
# delivery_delay_days:
#   actual delivery date - estimated delivery date
#   positive = late
#   negative = early

delivered["delivery_days_actual"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400

delivered["delivery_days_estimated"] = (
    delivered["order_estimated_delivery_date"]
    - delivered["order_purchase_timestamp"]
).dt.total_seconds() / 86400

delivered["delivery_delay_days"] = (
    delivered["order_delivered_customer_date"]
    - delivered["order_estimated_delivery_date"]
).dt.total_seconds() / 86400


# -------------------------------------------------------------------------
# C. Target variable.
# -------------------------------------------------------------------------
# This is what the model will predict in Phase 5.
# 1 = late
# 0 = on time / early

delivered["is_late"] = (delivered["delivery_delay_days"] > 0).astype(int)


# -------------------------------------------------------------------------
# D. Time features.
# -------------------------------------------------------------------------

delivered["purchase_year"] = delivered["order_purchase_timestamp"].dt.year
delivered["purchase_month_num"] = delivered["order_purchase_timestamp"].dt.month
delivered["purchase_month"] = delivered["order_purchase_timestamp"].dt.month_name()
delivered["purchase_day_of_week"] = delivered["order_purchase_timestamp"].dt.day_name()
delivered["purchase_hour"] = delivered["order_purchase_timestamp"].dt.hour


# -------------------------------------------------------------------------
# E. Product volume.
# -------------------------------------------------------------------------

products = products.copy()

products["product_volume_cm3"] = (
    products["product_length_cm"]
    * products["product_height_cm"]
    * products["product_width_cm"]
)


# -------------------------------------------------------------------------
# F. Join item + product + seller data.
# -------------------------------------------------------------------------
# order_items has multiple rows per order.
# We must aggregate it into one row per order.

items_products = order_items.merge(
    products,
    on="product_id",
    how="left"
)

items_products = items_products.merge(
    category_translation,
    on="product_category_name",
    how="left"
)

items_products = items_products.merge(
    sellers[["seller_id", "seller_city", "seller_state"]],
    on="seller_id",
    how="left"
)

order_item_features = (
    items_products.groupby("order_id")
    .agg(
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        item_count=("order_item_id", "count"),
        product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        avg_product_weight_g=("product_weight_g", "mean"),
        avg_product_length_cm=("product_length_cm", "mean"),
        avg_product_height_cm=("product_height_cm", "mean"),
        avg_product_width_cm=("product_width_cm", "mean"),
        avg_product_volume_cm3=("product_volume_cm3", "mean"),
        main_product_category=("product_category_name_english", most_common),
        main_seller_state=("seller_state", most_common),
        main_seller_city=("seller_city", most_common),
    )
    .reset_index()
)


# -------------------------------------------------------------------------
# G. Payment features.
# -------------------------------------------------------------------------

payment_features = (
    payments.groupby("order_id")
    .agg(
        total_payment_value=("payment_value", "sum"),
        max_payment_installments=("payment_installments", "max"),
        payment_count=("payment_sequential", "count"),
        main_payment_type=("payment_type", most_common),
    )
    .reset_index()
)


# -------------------------------------------------------------------------
# H. Review features.
# -------------------------------------------------------------------------
# Review score happens after delivery.
# It is useful for EDA/business impact, but should NOT be used as a model input.

review_features = (
    reviews.groupby("order_id")
    .agg(
        avg_review_score=("review_score", "mean"),
        review_count=("review_id", "count"),
    )
    .reset_index()
)


# -------------------------------------------------------------------------
# I. Merge everything into delivered.
# -------------------------------------------------------------------------

delivered = delivered.merge(
    customers[["customer_id", "customer_city", "customer_state"]],
    on="customer_id",
    how="left"
)

delivered = delivered.merge(
    order_item_features,
    on="order_id",
    how="left"
)

delivered = delivered.merge(
    payment_features,
    on="order_id",
    how="left"
)

delivered = delivered.merge(
    review_features,
    on="order_id",
    how="left"
)

late_n = int(delivered["is_late"].sum())
total_n = len(delivered)
on_time_n = total_n - late_n
late_pct = late_n / total_n * 100
on_time_pct = 100 - late_pct

print(f"\nFinal EDA DataFrame: {total_n:,} rows × {delivered.shape[1]} columns")
print(f"Late deliveries    : {late_n:,} ({late_pct:.2f}%)")
print(f"On-time deliveries : {on_time_n:,} ({on_time_pct:.2f}%)")

save_processed_csv(delivered, "phase3_eda_orders.csv")


# =============================================================================
# 10. BASELINE SUMMARY CSV
# =============================================================================

print("\n" + "═" * 80)
print("  BASELINE SUMMARY")
print("═" * 80)

baseline_summary = pd.DataFrame(
    [
        {
            "delivered_orders_with_valid_dates": total_n,
            "late_orders": late_n,
            "on_time_orders": on_time_n,
            "late_rate_pct": round(late_pct, 2),
            "on_time_rate_pct": round(on_time_pct, 2),
            "baseline_accuracy_if_predict_all_on_time": round(on_time_pct, 2),
            "avg_delivery_days": round(delivered["delivery_days_actual"].mean(), 2),
            "median_delivery_days": round(delivered["delivery_days_actual"].median(), 2),
            "avg_delay_days": round(delivered["delivery_delay_days"].mean(), 2),
            "median_delay_days": round(delivered["delivery_delay_days"].median(), 2),
        }
    ]
)

print(baseline_summary.to_string(index=False))
save_csv(baseline_summary, "phase3_baseline_summary.csv")


# =============================================================================
# 11. LATE RATE TABLES BY CATEGORY
# =============================================================================

print("\n" + "═" * 80)
print("  LATE RATE TABLES")
print("═" * 80)

late_by_customer_state = late_rate_by_category(
    delivered,
    "customer_state",
    min_orders=100
)

late_by_seller_state = late_rate_by_category(
    delivered,
    "main_seller_state",
    min_orders=100
)

late_by_payment_type = late_rate_by_category(
    delivered,
    "main_payment_type",
    min_orders=100
)

late_by_product_category = late_rate_by_category(
    delivered,
    "main_product_category",
    min_orders=100
)

late_by_day = late_rate_by_category(
    delivered,
    "purchase_day_of_week",
    min_orders=100
)

late_by_month = late_rate_by_category(
    delivered,
    "purchase_month",
    min_orders=100
)

save_csv(late_by_customer_state, "phase3_late_by_customer_state.csv")
save_csv(late_by_seller_state, "phase3_late_by_seller_state.csv")
save_csv(late_by_payment_type, "phase3_late_by_payment_type.csv")
save_csv(late_by_product_category, "phase3_late_by_product_category.csv")
save_csv(late_by_day, "phase3_late_by_day_of_week.csv")
save_csv(late_by_month, "phase3_late_by_month.csv")

print("\nTop customer states by late rate:")
print(late_by_customer_state.head(10).to_string(index=False))

print("\nTop seller states by late rate:")
print(late_by_seller_state.head(10).to_string(index=False))

print("\nTop product categories by late rate:")
print(late_by_product_category.head(10).to_string(index=False))


# =============================================================================
# 12. PRICE/FREIGHT QUARTILE TABLES
# =============================================================================

print("\n" + "═" * 80)
print("  PRICE/FREIGHT QUARTILE ANALYSIS")
print("═" * 80)

price_quartile_stats = make_quartile_stats(delivered, "total_price")
freight_quartile_stats = make_quartile_stats(delivered, "total_freight")

save_csv(price_quartile_stats, "phase3_late_by_price_quartile.csv")
save_csv(freight_quartile_stats, "phase3_late_by_freight_quartile.csv")

print("\nPrice quartile stats:")
print(price_quartile_stats.to_string(index=False))

print("\nFreight quartile stats:")
print(freight_quartile_stats.to_string(index=False))


# =============================================================================
# 13. CORRELATIONS
# =============================================================================
# WARNING:
# Do NOT use delivery_delay_days as a model input.
# It is calculated using actual delivery date, which is future information.
# Using it would be data leakage.

print("\n" + "═" * 80)
print("  NUMERIC CORRELATIONS")
print("═" * 80)

candidate_numeric_features = [
    "is_late",
    "delivery_days_estimated",
    "purchase_month_num",
    "purchase_hour",
    "total_price",
    "total_freight",
    "item_count",
    "product_count",
    "seller_count",
    "avg_product_weight_g",
    "avg_product_length_cm",
    "avg_product_height_cm",
    "avg_product_width_cm",
    "avg_product_volume_cm3",
    "total_payment_value",
    "max_payment_installments",
    "payment_count",
]

numeric_features = [
    col for col in candidate_numeric_features
    if col in delivered.columns
]

correlation_data = delivered[numeric_features].copy()
corr_matrix = correlation_data.corr(numeric_only=True)

corr_matrix_output = (
    corr_matrix
    .reset_index()
    .rename(columns={"index": "feature"})
)

save_csv(corr_matrix_output, "phase3_correlation_matrix.csv")

corr_with_target = (
    corr_matrix["is_late"]
    .drop("is_late")
    .sort_values(key=lambda s: s.abs(), ascending=False)
)

corr_with_target_df = (
    corr_with_target
    .reset_index()
)

corr_with_target_df.columns = [
    "feature",
    "correlation_with_late_delivery"
]

save_csv(corr_with_target_df, "phase3_numeric_correlations_with_late_delivery.csv")

print("\nFeatures sorted by absolute correlation with is_late:")
print(corr_with_target_df.to_string(index=False))


# =============================================================================
# 14. CHART 1 — BASELINE ON-TIME VS LATE
# =============================================================================

print("\n" + "═" * 80)
print("  CREATING CHARTS")
print("═" * 80)

fig = go.Figure(
    data=[
        go.Bar(
            x=["On Time / Early", "Late"],
            y=[on_time_n, late_n],
            marker_color=[ONTIME_COLOR, LATE_COLOR],
            text=[f"{on_time_pct:.1f}%", f"{late_pct:.1f}%"],
            textposition="outside",
        )
    ]
)

fig.update_layout(
    title=(
        f"Delivery Outcome: {total_n:,} Delivered Orders "
        f"| Baseline Accuracy = {on_time_pct:.1f}%"
    ),
    xaxis_title="Delivery outcome",
    yaxis_title="Number of orders",
    plot_bgcolor="white",
    showlegend=False,
)

save_chart(fig, "01_baseline_on_time_vs_late.html")


# =============================================================================
# 15. CHART 2 — DELIVERY TIME DISTRIBUTION
# =============================================================================

delivery_plot_df = (
    delivered[
        delivered["delivery_days_actual"].between(1, 100)
    ]
    .copy()
)

delivery_plot_df["Outcome"] = delivery_plot_df["is_late"].map(
    {
        0: "On Time / Early",
        1: "Late",
    }
)

fig = px.histogram(
    delivery_plot_df,
    x="delivery_days_actual",
    color="Outcome",
    nbins=80,
    barmode="overlay",
    opacity=0.75,
    title="Distribution of Delivery Time",
    labels={
        "delivery_days_actual": "Delivery time in days",
        "count": "Orders",
    },
    color_discrete_map={
        "On Time / Early": ONTIME_COLOR,
        "Late": LATE_COLOR,
    },
)

fig.update_layout(
    plot_bgcolor="white",
    yaxis_title="Number of orders",
)

save_chart(fig, "02_delivery_time_distribution.html")


# =============================================================================
# 16. CHART 3 — DELAY DISTRIBUTION VIOLIN
# =============================================================================

delay_plot_df = (
    delivered[
        delivered["delivery_delay_days"].between(-60, 60)
    ]
    .copy()
)

delay_plot_df["Outcome"] = delay_plot_df["is_late"].map(
    {
        0: "On Time / Early",
        1: "Late",
    }
)

fig = px.violin(
    delay_plot_df,
    x="Outcome",
    y="delivery_delay_days",
    color="Outcome",
    box=True,
    points=False,
    title="Delivery Delay: Early vs Late",
    labels={
        "delivery_delay_days": "Delay days: negative = early, positive = late",
    },
    color_discrete_map={
        "On Time / Early": ONTIME_COLOR,
        "Late": LATE_COLOR,
    },
)

fig.add_hline(
    y=0,
    line_dash="dash",
    line_color="#666666",
    annotation_text="Estimated delivery date",
    annotation_position="top right",
)

fig.update_layout(
    plot_bgcolor="white",
    showlegend=False,
)

save_chart(fig, "03_delivery_delay_violin.html")


# =============================================================================
# 17. CHART 4 — LATE RATE BY DAY OF WEEK
# =============================================================================

day_stats_chart = (
    delivered
    .groupby("purchase_day_of_week")["is_late"]
    .agg(late_rate="mean", order_count="count")
    .reset_index()
)

day_stats_chart["late_rate_pct"] = day_stats_chart["late_rate"] * 100

day_stats_chart["purchase_day_of_week"] = pd.Categorical(
    day_stats_chart["purchase_day_of_week"],
    categories=DAY_ORDER,
    ordered=True,
)

day_stats_chart = day_stats_chart.sort_values("purchase_day_of_week")

fig = px.bar(
    day_stats_chart,
    x="purchase_day_of_week",
    y="late_rate_pct",
    text=day_stats_chart["late_rate_pct"].round(1),
    title="Late Delivery Rate by Day of Week",
    labels={
        "purchase_day_of_week": "Day of week",
        "late_rate_pct": "Late rate (%)",
    },
    color="late_rate_pct",
    color_continuous_scale=BAR_SCALE,
    hover_data=["order_count"],
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
)

fig.update_layout(
    plot_bgcolor="white",
    coloraxis_showscale=False,
)

save_chart(fig, "04_late_rate_by_day_of_week.html")


# =============================================================================
# 18. CHART 5 — LATE RATE BY MONTH
# =============================================================================

month_stats_chart = (
    delivered
    .groupby("purchase_month")["is_late"]
    .agg(late_rate="mean", order_count="count")
    .reset_index()
)

month_stats_chart["late_rate_pct"] = month_stats_chart["late_rate"] * 100

month_stats_chart["purchase_month"] = pd.Categorical(
    month_stats_chart["purchase_month"],
    categories=MONTH_ORDER,
    ordered=True,
)

month_stats_chart = month_stats_chart.sort_values("purchase_month")

fig = px.line(
    month_stats_chart,
    x="purchase_month",
    y="late_rate_pct",
    markers=True,
    text=month_stats_chart["late_rate_pct"].round(1),
    title="Late Delivery Rate by Month",
    labels={
        "purchase_month": "Month",
        "late_rate_pct": "Late rate (%)",
    },
    hover_data=["order_count"],
)

fig.update_traces(
    line_color=LATE_COLOR,
    marker_size=8,
    texttemplate="%{text:.1f}%",
    textposition="top center",
)

fig.update_layout(
    plot_bgcolor="white",
)

save_chart(fig, "05_late_rate_by_month.html")


# =============================================================================
# 19. CHART 6 — LATE RATE BY CUSTOMER STATE
# =============================================================================

customer_state_chart = late_by_customer_state.sort_values(
    "late_rate_pct",
    ascending=True
).tail(15)

fig = px.bar(
    customer_state_chart,
    x="late_rate_pct",
    y="customer_state",
    orientation="h",
    title="Top Customer States by Late Delivery Rate",
    labels={
        "late_rate_pct": "Late rate (%)",
        "customer_state": "Customer state",
    },
    color="late_rate_pct",
    color_continuous_scale=BAR_SCALE,
    hover_data=["orders", "late_orders", "avg_delay_days"],
)

fig.update_layout(
    plot_bgcolor="white",
    coloraxis_showscale=False,
)

save_chart(fig, "06_late_rate_by_customer_state.html")


# =============================================================================
# 20. CHART 7 — LATE RATE BY SELLER STATE
# =============================================================================

seller_state_chart = late_by_seller_state.sort_values(
    "late_rate_pct",
    ascending=True
).tail(15)

fig = px.bar(
    seller_state_chart,
    x="late_rate_pct",
    y="main_seller_state",
    orientation="h",
    title="Top Seller States by Late Delivery Rate",
    labels={
        "late_rate_pct": "Late rate (%)",
        "main_seller_state": "Seller state",
    },
    color="late_rate_pct",
    color_continuous_scale=BAR_SCALE,
    hover_data=["orders", "late_orders", "avg_delay_days"],
)

fig.update_layout(
    plot_bgcolor="white",
    coloraxis_showscale=False,
)

save_chart(fig, "07_late_rate_by_seller_state.html")


# =============================================================================
# 21. CHART 8 — LATE RATE BY PRODUCT CATEGORY
# =============================================================================

category_chart = late_by_product_category.sort_values(
    "late_rate_pct",
    ascending=True
).tail(15)

fig = px.bar(
    category_chart,
    x="late_rate_pct",
    y="main_product_category",
    orientation="h",
    title="Top Product Categories by Late Delivery Rate",
    labels={
        "late_rate_pct": "Late rate (%)",
        "main_product_category": "Product category",
    },
    color="late_rate_pct",
    color_continuous_scale=BAR_SCALE,
    hover_data=["orders", "late_orders", "avg_delay_days"],
)

fig.update_layout(
    plot_bgcolor="white",
    coloraxis_showscale=False,
    height=700,
)

save_chart(fig, "08_late_rate_by_product_category.html")


# =============================================================================
# 22. CHART 9 — PRICE AND FREIGHT QUARTILES
# =============================================================================

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Late Rate by Price Quartile", "Late Rate by Freight Quartile"),
)

fig.add_trace(
    go.Bar(
        x=price_quartile_stats["total_price_quartile"],
        y=price_quartile_stats["late_rate_pct"].round(1),
        text=price_quartile_stats["late_rate_pct"].round(1),
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_color=LATE_COLOR,
        showlegend=False,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Bar(
        x=freight_quartile_stats["total_freight_quartile"],
        y=freight_quartile_stats["late_rate_pct"].round(1),
        text=freight_quartile_stats["late_rate_pct"].round(1),
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_color="#e67e22",
        showlegend=False,
    ),
    row=1,
    col=2,
)

fig.update_layout(
    title="Late Rate by Price and Freight Quartiles",
    plot_bgcolor="white",
)

fig.update_yaxes(title_text="Late rate (%)")

save_chart(fig, "09_late_rate_by_price_and_freight_quartiles.html")


# =============================================================================
# 23. CHART 10 — CORRELATION MATRIX
# =============================================================================

fig = px.imshow(
    corr_matrix,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    text_auto=".2f",
    aspect="auto",
    title="Correlation Matrix — Numeric Features vs Late Delivery",
)

save_chart(fig, "10_correlation_matrix.html")


# =============================================================================
# 24. CHART 11 — REVIEW SCORE: LATE VS ON-TIME
# =============================================================================


review_plot_data = (
    delivered.dropna(subset=["avg_review_score"])
    .groupby("is_late")
    .agg(
        avg_review_score=("avg_review_score", "mean"),
        orders=("order_id", "count"),
    )
    .reset_index()
)

review_plot_data["delivery_status"] = review_plot_data["is_late"].map(
    {
        0: "On Time / Early",
        1: "Late",
    }
)

fig = px.bar(
    review_plot_data,
    x="delivery_status",
    y="avg_review_score",
    text=review_plot_data["avg_review_score"].round(2),
    title="Average Review Score: Late vs On-Time Deliveries",
    labels={
        "delivery_status": "Delivery status",
        "avg_review_score": "Average review score",
    },
    color="delivery_status",
    color_discrete_map={
        "On Time / Early": ONTIME_COLOR,
        "Late": LATE_COLOR,
    },
    hover_data=["orders"],
)

fig.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
)

fig.update_layout(
    plot_bgcolor="white",
    showlegend=False,
)

save_chart(fig, "11_review_score_late_vs_ontime.html")


# =============================================================================
# 25. KEY FINDINGS TEXT FILE
# =============================================================================

print("\n" + "═" * 80)
print("  KEY FINDINGS")
print("═" * 80)

top_corr_feature = corr_with_target_df.iloc[0]["feature"]
top_corr_value = corr_with_target_df.iloc[0]["correlation_with_late_delivery"]

worst_customer_state = (
    late_by_customer_state.iloc[0]["customer_state"]
    if len(late_by_customer_state) > 0
    else "N/A"
)

worst_seller_state = (
    late_by_seller_state.iloc[0]["main_seller_state"]
    if len(late_by_seller_state) > 0
    else "N/A"
)

worst_category = (
    late_by_product_category.iloc[0]["main_product_category"]
    if len(late_by_product_category) > 0
    else "N/A"
)

key_findings = f"""
PHASE 3 EDA KEY FINDINGS
========================

DATASET
-------
Delivered orders with valid dates: {total_n:,}
Late orders: {late_n:,} ({late_pct:.2f}%)
On-time / early orders: {on_time_n:,} ({on_time_pct:.2f}%)

BASELINE
--------
baseline accuracy is:
{on_time_pct:.2f}%


IMPORTANT BUSINESS INSIGHTS
---------------------------
1. Worst customer state by late rate:
   {worst_customer_state}

2. Worst seller state by late rate:
   {worst_seller_state}

3. Worst product category by late rate:
   {worst_category}

4. Strongest numeric correlation with late delivery:
   {top_corr_feature} with correlation {top_corr_value:.4f}

DATA LEAKAGE WARNING
--------------------
Do NOT use these as model input features:
- delivery_delay_days
- order_delivered_customer_date
- avg_review_score

Reason:
They are only known after delivery happens.

PHASE 4 FEATURE ENGINEERING CANDIDATES
--------------------------------------
Numeric:
- delivery_days_estimated
- total_price
- total_freight
- item_count
- product_count
- seller_count
- avg_product_weight_g
- avg_product_volume_cm3
- total_payment_value
- max_payment_installments
- payment_count
- purchase_hour
- purchase_month_num

Categorical:
- purchase_day_of_week
- purchase_month
- customer_state
- main_seller_state
- main_payment_type
- main_product_category
"""

key_findings_path = REPORTS_DIR / "phase3_key_findings.txt"

with open(key_findings_path, "w", encoding="utf-8") as f:
    f.write(key_findings)

print(key_findings)
print(f"  ✓ Key findings saved: {key_findings_path}")


# =============================================================================
# 26. FINAL OUTPUT SUMMARY
# =============================================================================

print("\n" + "═" * 80)
print("  PHASE 3 COMPLETE")
print("═" * 80)

print(f"""
Created outputs:

CSV reports saved in:
  {REPORTS_DIR}

Important CSV files:
  - phase3_table_overview.csv
  - phase3_missing_values.csv
  - phase3_baseline_summary.csv
  - phase3_late_by_customer_state.csv
  - phase3_late_by_seller_state.csv
  - phase3_late_by_payment_type.csv
  - phase3_late_by_product_category.csv
  - phase3_numeric_correlations_with_late_delivery.csv
  - phase3_correlation_matrix.csv
  - phase3_key_findings.txt

Processed dataset saved in:
  {DATA_PROCESSED_DIR / "phase3_eda_orders.csv"}

Interactive charts saved in:
  {FIGURES_DIR}

Charts:
  01_baseline_on_time_vs_late.html
  02_delivery_time_distribution.html
  03_delivery_delay_violin.html
  04_late_rate_by_day_of_week.html
  05_late_rate_by_month.html
  06_late_rate_by_customer_state.html
  07_late_rate_by_seller_state.html
  08_late_rate_by_product_category.html
  09_late_rate_by_price_and_freight_quartiles.html
  10_correlation_matrix.html
  11_review_score_late_vs_ontime.html

""")