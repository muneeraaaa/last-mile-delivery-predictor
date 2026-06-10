import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"

TABLE_FILES: dict[str, str] = {
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

DATE_COLUMNS: dict[str, list[str]] = {
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
}


def load_table(name: str) -> pd.DataFrame:
    if name not in TABLE_FILES:
        raise ValueError(
            f"Unknown table '{name}'.\n"
            f"Valid names: {list(TABLE_FILES)}"
        )

    filepath = RAW_DATA_PATH / TABLE_FILES[name]

    if not filepath.exists():
        raise FileNotFoundError(
            f"\n  File not found: {filepath}"
            f"\n\n  Have you downloaded the Kaggle dataset into data/raw/ ?"
            f"\n  Run this in your terminal (with venv active):"
            f"\n    kaggle datasets download -d olistbr/brazilian-ecommerce"
            f"\n  Then unzip into data/raw/"
        )

    parse_dates = DATE_COLUMNS.get(name, False)
    df = pd.read_csv(filepath, parse_dates=parse_dates)
    return df


def load_all_tables(verbose: bool = True) -> dict[str, pd.DataFrame]:
    dataframes: dict[str, pd.DataFrame] = {}

    for name in TABLE_FILES:
        df = load_table(name)
        dataframes[name] = df
        if verbose:
            print(f"  ✓ {name:<25}  {df.shape[0]:>9,} rows  ×  {df.shape[1]:>2} cols")

    if verbose:
        total = sum(d.shape[0] for d in dataframes.values())
        print(f"\n  Total rows across all 9 tables: {total:,}")

    return dataframes


def describe_table(df: pd.DataFrame, name: str = "") -> None:
    header = f"  TABLE: {name.upper()}  " if name else "  TABLE  "
    width = 70

    print(f"\n{'═' * width}")
    print(f"{header:^{width}}")
    print(f"{'═' * width}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    print(f"  {'Column':<42} {'Dtype':<14} {'Nulls':>6}  Sample values")
    print(f"  {'─' * 66}")

    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isna().sum())
        sample = df[col].dropna().unique()[:3].tolist()
        sample_str = str(sample)[:36]
        print(f"  {col:<42} {dtype:<14} {null_count:>6}  {sample_str}")
