from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


LEAKY_COLUMNS: list[str] = [
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "delivery_days_actual",
    "delivery_delay_days",
    "avg_review_score",
    "review_count",
]

ID_COLUMNS: list[str] = [
    "order_id",
    "customer_id",
]

RAW_DATE_COLUMNS: list[str] = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
]

EDA_ONLY_COLUMNS: list[str] = [
    "order_status",
    "purchase_year",
    "purchase_month",
    "customer_city",
    "main_seller_city",
]

CATEGORICAL_COLUMNS: list[str] = [
    "purchase_day_of_week",
    "customer_state",
    "main_seller_state",
    "main_payment_type",
    "main_product_category",
]

TARGET_COLUMN: str = "is_late"

FINAL_FEATURE_COLUMNS: list[str] = [
    "delivery_days_estimated",
    "purchase_hour",
    "purchase_month_num",
    "is_weekend",
    "total_price",
    "total_freight",
    "freight_ratio",
    "price_per_item",
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
    "cross_state",
    "purchase_day_of_week_encoded",
    "customer_state_encoded",
    "main_seller_state_encoded",
    "main_payment_type_encoded",
    "main_product_category_encoded",
    "approval_time_hours",
]


def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["order_purchase_timestamp", "order_approved_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "purchase_day_of_week" in df.columns:
        df["is_weekend"] = df["purchase_day_of_week"].isin(
            ["Saturday", "Sunday"]
        ).astype(int)
    else:
        df["is_weekend"] = 0

    if "customer_state" in df.columns and "main_seller_state" in df.columns:
        df["cross_state"] = (
            df["customer_state"].notna()
            & df["main_seller_state"].notna()
            & (df["customer_state"] != df["main_seller_state"])
        ).astype(int)
    else:
        df["cross_state"] = 0

    if "total_price" in df.columns and "item_count" in df.columns:
        df["price_per_item"] = df["total_price"] / df["item_count"].clip(lower=1)
    else:
        df["price_per_item"] = np.nan

    if "total_freight" in df.columns and "total_price" in df.columns:
        df["freight_ratio"] = (
            df["total_freight"] / (df["total_price"] + 1)
        ).clip(upper=10.0)
    else:
        df["freight_ratio"] = np.nan

    if (
        "order_purchase_timestamp" in df.columns
        and "order_approved_at" in df.columns
    ):
        approval_delta = df["order_approved_at"] - df["order_purchase_timestamp"]
        df["approval_time_hours"] = (
            approval_delta.dt.total_seconds()
            .div(3600)
            .clip(lower=0.0, upper=720.0)
        )
    else:
        df["approval_time_hours"] = np.nan

    return df


def impute_missing(
    df: pd.DataFrame,
    numeric_fill_values: dict | None = None,
    categorical_fill_values: dict | None = None,
    training: bool = True,
) -> tuple[pd.DataFrame, dict, dict]:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if training:
        numeric_fill_values = {
            col: float(df[col].median())
            for col in numeric_cols
            if df[col].isna().any()
        }
        categorical_fill_values = {
            col: str(df[col].mode().iloc[0])
            for col in categorical_cols
            if df[col].isna().any() and len(df[col].dropna()) > 0
        }

    for col, value in (numeric_fill_values or {}).items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    for col, value in (categorical_fill_values or {}).items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    return df, numeric_fill_values or {}, categorical_fill_values or {}


def encode_categoricals(
    df: pd.DataFrame,
    columns: list[str],
    encoders: dict | None = None,
    training: bool = True,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    if training:
        encoders = {}

    for col in columns:
        if col not in df.columns:
            continue

        df[col] = df[col].fillna("missing").astype(str)
        encoded_col = f"{col}_encoded"

        if training:
            le = LabelEncoder()
            df[encoded_col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            known_classes = set(le.classes_)
            df[col] = df[col].apply(
                lambda x: x if x in known_classes else "missing"
            )
            if "missing" not in known_classes:
                le.classes_ = np.append(le.classes_, "missing")
            df[encoded_col] = le.transform(df[col])

    return df, encoders or {}
