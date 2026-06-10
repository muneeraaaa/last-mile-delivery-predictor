"""
04_feature_engineering.py
--------------------------
Feature Engineering

Transform the EDA DataFrame into a clean, fully numeric,model-ready feature matrix.

Loads  : data/processed/phase3_eda_orders.csv

Saves  : data/processed/phase4_feature_matrix.csv
All features (X) + target (is_late). Input for training.

models/preprocessor.pkl
Fitted encoders + imputation fill values.
Used by Streamlit to transform new orders.

reports/phase4_feature_report.csv
Per-feature stats and correlation with target.


"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.features import (
    CATEGORICAL_COLUMNS,
    EDA_ONLY_COLUMNS,
    FINAL_FEATURE_COLUMNS,
    ID_COLUMNS,
    LEAKY_COLUMNS,
    RAW_DATE_COLUMNS,
    TARGET_COLUMN,
    build_derived_features,
    encode_categoricals,
    impute_missing,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD PHASE 3 DATA
# ═══════════════════════════════════════════════════════════════════════════════
# We load from the CSV saved at the end of Phase 3.
# This makes phases independent: you can re-run Phase 4 any time without
# re-running the full EDA.

print("\n" + "═" * 68)
print("  PHASE 4 — FEATURE ENGINEERING")
print("═" * 68)

phase3_path = PROCESSED_DIR / "phase3_eda_orders.csv"

if not phase3_path.exists():
    raise FileNotFoundError(
        f"\nFile not found: {phase3_path}"
        f"\nRun notebooks/03_eda.py first to generate the Phase 3 output."
    )

print(f"\nLoading: {phase3_path.name} ...")
df = pd.read_csv(phase3_path)

print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Target (is_late) distribution:")
print(f"    Late     : {int(df[TARGET_COLUMN].sum()):,}  "
      f"({df[TARGET_COLUMN].mean()*100:.2f}%)")
print(f"    On time  : {int((df[TARGET_COLUMN] == 0).sum()):,}  "
      f"({(1-df[TARGET_COLUMN].mean())*100:.2f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD DERIVED FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
# New columns added here:
#   is_weekend            (1 if order placed Sat or Sun)
#   cross_state           (1 if seller and customer are in different states)
#   price_per_item        (total_price / item_count)
#   freight_ratio         (total_freight / (total_price + 1), capped at 10)
#   approval_time_hours   (hours from purchase to payment approval)
#
# WHY DERIVED FEATURES MATTER:
# A tree model splitting on raw "customer_state" and raw "main_seller_state"
# separately might never learn "same vs different state" without seeing that
# exact combination. cross_state pre-computes this relationship explicitly.
# We're doing the domain-knowledge work up front so the model doesn't have to.

print("\n\n" + "═" * 68)
print("  STEP 2 — Building derived features")
print("═" * 68)

df = build_derived_features(df)

new_features = [
    "is_weekend",
    "cross_state",
    "price_per_item",
    "freight_ratio",
    "approval_time_hours",
]

for feat in new_features:
    if feat in df.columns:
        n_null = df[feat].isna().sum()
        mean   = df[feat].mean()
        corr   = df[feat].corr(df[TARGET_COLUMN])
        print(f"  {feat:<28} mean={mean:>7.3f}   nulls={n_null:>5,}   "
              f"corr_with_is_late={corr:+.4f}")
    else:
        print(f"  {feat:<28} NOT CREATED (check column availability)")

print(f"\n  cross_state rate : "
      f"{df['cross_state'].mean()*100:.1f}% of orders cross state lines")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — IMPUTE MISSING VALUES
# ═══════════════════════════════════════════════════════════════════════════════
# WHAT IS IMPUTATION?
# Replacing null (NaN) values with a sensible estimate so the model
# receives a complete, null-free input.
#
# Imputation strategy used here:
#   Numeric   → median  (robust to the extreme outliers common in logistics data)
#   Categorical → mode  (the most common value = neutral "typical" assumption)
#
# CRITICAL RULE: compute fill values from this DataFrame (training data).
# Save those values in preprocessor.pkl. When Phase 6 encounters a new order
# with a missing product weight, it fills with the TRAINING median —
# not the median of the new order batch. Using new-data statistics would be
# a form of data leakage.

print("\n\n" + "═" * 68)
print("  STEP 3 — Imputing missing values")
print("═" * 68)

pre_null = df.isnull().sum()
pre_null = pre_null[pre_null > 0].sort_values(ascending=False)

if len(pre_null) > 0:
    print(f"\n  {len(pre_null)} columns have missing values before imputation:\n")
    for col, cnt in pre_null.items():
        pct = cnt / len(df) * 100
        bar = "░" * max(1, int(pct / 2))
        print(f"  {col:<45} {cnt:>6,}  ({pct:>5.1f}%)  {bar}")
else:
    print("\n  No missing values found before imputation.")

df, numeric_fills, categorical_fills = impute_missing(df, training=True)

post_null = df.isnull().sum().sum()
if post_null == 0:
    print(f"\n  ✓ All nulls imputed.")
    print(f"  {len(numeric_fills)} numeric columns filled with median.")
    print(f"  {len(categorical_fills)} categorical columns filled with mode.")
else:
    print(f"\n  WARNING: {post_null} nulls remain after imputation.")
    remaining = df.isnull().sum()
    print(remaining[remaining > 0])

print("\n  Top 5 numeric fill values (training medians):")
for col, val in list(numeric_fills.items())[:5]:
    print(f"    {col:<40} → {val:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — ENCODE CATEGORICAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
# WHAT LABEL ENCODING DOES:
# Converts each unique string to an integer.
# LabelEncoder sorts values alphabetically and assigns 0, 1, 2, etc.
#
# A new column {col}_encoded is created alongside the original.

print("\n\n" + "═" * 68)
print("  STEP 4 — Encoding categorical features")
print("═" * 68)

df, label_encoders = encode_categoricals(
    df, CATEGORICAL_COLUMNS, training=True
)

print(f"\n  Encoded {len(label_encoders)} categorical columns:\n")
for col, le in label_encoders.items():
    encoded_col = f"{col}_encoded"
    n_classes   = len(le.classes_)
    sample      = list(le.classes_[:3])
    print(f"  {col:<35} → {encoded_col}")
    print(f"    {n_classes} classes. Examples: {sample} → "
          f"{list(le.transform([c for c in sample]))}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — DROP NON-FEATURE COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════
# Four categories of columns are removed:
#
# LEAKY (post-delivery info):
#   delivery_delay_days, delivery_days_actual, order_delivered_customer_date,
#   avg_review_score, review_count
#   → Only known after the package arrives. Can't use at prediction time.
#
# IDENTIFIERS (unique per row):
#   order_id, customer_id
#   → A model that memorises order IDs fails on any new order.
#
# RAW DATETIMES (already decomposed into numeric features):
#   order_purchase_timestamp, order_approved_at, order_estimated_delivery_date
#   → extracted hour, month, approval_time_hours from these.
#     The raw strings add no new information and aren't numeric.


print("\n\n" + "═" * 68)
print("  STEP 5 — Dropping non-feature columns")
print("═" * 68)

all_drop_cols = list(dict.fromkeys(
    LEAKY_COLUMNS +
    ID_COLUMNS +
    RAW_DATE_COLUMNS +
    EDA_ONLY_COLUMNS +
    CATEGORICAL_COLUMNS          
))

present_drop_cols = [c for c in all_drop_cols if c in df.columns]
absent_drop_cols  = [c for c in all_drop_cols if c not in df.columns]

df_clean = df.drop(columns=present_drop_cols)

print(f"\n  Dropped  : {len(present_drop_cols)} columns")
print(f"  Not found (already absent): {len(absent_drop_cols)}")
print(f"  Remaining: {df_clean.shape[1]} columns")

if absent_drop_cols:
    print(f"\n  Columns listed for drop but not present (OK to ignore):")
    for c in absent_drop_cols:
        print(f"    {c}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SELECT FINAL FEATURES AND TARGET
# ═══════════════════════════════════════════════════════════════════════════════
# FINAL_FEATURE_COLUMNS is the authoritative list defined in src/features.py.


print("\n\n" + "═" * 68)
print("  STEP 6 — Selecting final feature matrix")
print("═" * 68)

available_features = [
    c for c in FINAL_FEATURE_COLUMNS if c in df_clean.columns
]
missing_features = [
    c for c in FINAL_FEATURE_COLUMNS if c not in df_clean.columns
]

if missing_features:
    print(f"\n  WARNING: {len(missing_features)} expected features not in DataFrame:")
    for f in missing_features:
        print(f"    - {f}")
    print(f"\n  Proceeding with {len(available_features)} available features.")
else:
    print(f"\n  All {len(available_features)} expected features are present.")

X = df_clean[available_features]
y = df_clean[TARGET_COLUMN]

print(f"\n  X (feature matrix) : {X.shape[0]:,} rows × {X.shape[1]} columns")
print(f"  y (target)         : {y.shape[0]:,} values")
print(f"  Late rate in y     : {y.mean()*100:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — FINAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
# The model will break if X contains nulls or non-numeric values.
# This check catches any issues before they cause cryptic XGBoost errors.

print("\n\n" + "═" * 68)
print("  STEP 7 — Validation checks")
print("═" * 68)

null_check = X.isnull().sum()
null_cols  = null_check[null_check > 0]
if len(null_cols) > 0:
    print(f"\n  ✗ {len(null_cols)} columns still have nulls:")
    print(null_cols)
    print("\n  Fix: check impute_missing() is being called before this step.")
else:
    print(f"\n  ✓ No nulls in feature matrix X.")

non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"\n  ✗ {len(non_numeric)} non-numeric columns in X:")
    print(non_numeric)
    print("\n  Fix: check encode_categoricals() and EDA_ONLY_COLUMNS drop.")
else:
    print(f"  ✓ All {len(available_features)} features are numeric.")

if y.isnull().sum() > 0:
    print(f"\n  ✗ Target y has {y.isnull().sum()} nulls — check filtering logic.")
else:
    print(f"  ✓ Target y is clean (no nulls).")

late_rate = y.mean()
print(f"\n  Class balance check:")
print(f"    Late (1)    : {int(y.sum()):>8,}  ({late_rate*100:.2f}%)")
print(f"    On-time (0) : {int((y==0).sum()):>8,}  ({(1-late_rate)*100:.2f}%)")
if late_rate < 0.05:
    print(f"  NOTE: Only {late_rate*100:.1f}% late orders. "
          f"Phase 5 will use scale_pos_weight to compensate.")
elif late_rate < 0.15:
    print(f"  NOTE: Moderate imbalance ({late_rate*100:.1f}% late). "
          f"XGBoost handles this well with scale_pos_weight.")
else:
    print(f"  NOTE: Balanced enough for standard training.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 68)
print("  STEP 8 — Saving outputs")
print("═" * 68)

# ── Feature matrix CSV ─────────────────────────────────────────────────────────
# X columns + target column combined into one file.
# Phase 5 loads this, splits into X and y, then trains XGBoost.
feature_matrix = X.copy()
feature_matrix[TARGET_COLUMN] = y.values

matrix_path = PROCESSED_DIR / "phase4_feature_matrix.csv"
feature_matrix.to_csv(matrix_path, index=False)
print(f"\n  ✓ Feature matrix  → {matrix_path}")
print(f"    {feature_matrix.shape[0]:,} rows × {feature_matrix.shape[1]} columns "
      f"({len(available_features)} features + target)")

# ── Preprocessor pkl ───────────────────────────────────────────────────────────
# Packages all the fitted transformation artefacts into one object.
#
# WHY SAVE THE PREPROCESSOR?
# When the Phase 6 Streamlit dashboard receives a new order from a user,
# it must apply the exact same transformations that were applied to training data:
# Without the preprocessor, prediction at inference time would be impossible.
preprocessor = {
    "label_encoders":            label_encoders,
    "numeric_fill_values":       numeric_fills,
    "categorical_fill_values":   categorical_fills,
    "feature_columns":           available_features,
    "target_column":             TARGET_COLUMN,
}
preprocessor_path = MODELS_DIR / "preprocessor.pkl"
joblib.dump(preprocessor, preprocessor_path)
print(f"\n  ✓ Preprocessor    → {preprocessor_path}")
print(f"    {len(label_encoders)} encoders, "
      f"{len(numeric_fills)} numeric fills, "
      f"{len(categorical_fills)} categorical fills")

feature_report_rows = []
for col in available_features:
    series = X[col]
    feature_report_rows.append({
        "feature":              col,
        "dtype":                str(series.dtype),
        "null_count":           int(series.isnull().sum()),
        "mean":                 round(float(series.mean()), 4),
        "median":               round(float(series.median()), 4),
        "std":                  round(float(series.std()), 4),
        "min":                  round(float(series.min()), 4),
        "max":                  round(float(series.max()), 4),
        "corr_with_is_late":    round(float(series.corr(y)), 4),
    })

feature_report = (
    pd.DataFrame(feature_report_rows)
    .sort_values("corr_with_is_late", key=lambda s: s.abs(), ascending=False)
    .reset_index(drop=True)
)

report_path = REPORTS_DIR / "phase4_feature_report.csv"
feature_report.to_csv(report_path, index=False)
print(f"\n  ✓ Feature report  → {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT — FEATURE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 68)
print("  FEATURE REPORT (sorted by |correlation with is_late|)")
print("═" * 68)
print(f"\n  {'#':<4} {'Feature':<38} {'Dtype':<9} {'Corr with target':>16}  Visual")
print(f"  {'─' * 64}")

for rank, row in feature_report.iterrows():
    corr_val = row["corr_with_is_late"]
    bar      = "█" * max(1, int(abs(corr_val) * 30))
    sign     = "+" if corr_val >= 0 else "−"
    print(f"  {rank+1:<4} {row['feature']:<38} {row['dtype']:<9} "
          f"  {sign}{abs(corr_val):.4f}         {bar}")

print(f"""

  PHASE 4 COMPLETE
  ──────────────────────────────────────────────────────────────────
  Feature matrix  : {len(available_features)} features, {len(X):,} orders
  Target          : is_late  ({late_rate*100:.2f}% late)
  Baseline        : {(1-late_rate)*100:.2f}% accuracy (predict all on-time)

  Files created:
    data/processed/phase4_feature_matrix.csv   ← input for Phase 5
    models/preprocessor.pkl                    ← input for Phase 6 dashboard
    reports/phase4_feature_report.csv          ← feature summary

""")
