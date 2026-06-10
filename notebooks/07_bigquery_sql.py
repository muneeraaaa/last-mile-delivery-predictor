"""
07_bigquery_sql.py
------------------
Upload processed delivery datasets to BigQuery and run SQL KPI queries.

Inputs:
    data/processed/phase3_eda_orders.csv
    data/processed/phase4_feature_matrix.csv

Creates BigQuery dataset

Creates BigQuery tables:
    phase3_eda_orders
    phase4_feature_matrix

Outputs:
    reports/phase7_bigquery_kpis.csv
    reports/phase7_bigquery_state_late_rates.csv
    reports/phase7_bigquery_feature_risk.csv
    reports/phase7_bigquery_monthly_late_rate.csv

"""

import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PHASE3_PATH = PROCESSED_DIR / "phase3_eda_orders.csv"
PHASE4_PATH = PROCESSED_DIR / "phase4_feature_matrix.csv"




import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

if not PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT is not set.")

DATASET_ID = "aramex_delivery_predictor"

PHASE3_TABLE = "phase3_eda_orders"
PHASE4_TABLE = "phase4_feature_matrix"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace(".", "_", regex=False)
        .str.lower()
    )

    return df


def reduce_object_column_risk(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].replace("nan", None)

    return df


def upload_dataframe_to_bigquery(
    client: bigquery.Client,
    df: pd.DataFrame,
    table_name: str,
) -> str:
    
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )

    print("\n" + "-" * 80)
    print(f"Uploading table: {table_id}")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")

    load_job = client.load_table_from_dataframe(
        df,
        table_id,
        job_config=job_config,
    )

    load_job.result()

    table = client.get_table(table_id)

    print(
        f"Upload complete: {table.num_rows:,} rows, "
        f"{len(table.schema)} columns"
    )

    return table_id


def run_query_to_csv(
    client: bigquery.Client,
    query: str,
    output_filename: str,
) -> pd.DataFrame:
    
    print("\n" + "-" * 80)
    print(f"Running SQL query → {output_filename}")

    result_df = client.query(query).to_dataframe()

    output_path = REPORTS_DIR / output_filename
    result_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(result_df.head(10).to_string(index=False))

    return result_df


print("\n" + "═" * 80)
print("  BIGQUERY & SQL — CLOUD ANALYTICS LAYER")
print("═" * 80)

print(f"\nGoogle Cloud project: {PROJECT_ID}")
print(f"BigQuery dataset    : {DATASET_ID}")

if not PHASE3_PATH.exists():
    raise FileNotFoundError(
        f"\nMissing file:\n  {PHASE3_PATH}\n\n"
        "Run your EDA script first to generate phase3_eda_orders.csv."
    )

if not PHASE4_PATH.exists():
    raise FileNotFoundError(
        f"\nMissing file:\n  {PHASE4_PATH}\n\n"
        "Run your feature engineering script first to generate phase4_feature_matrix.csv."
    )


client = bigquery.Client(project=PROJECT_ID)


dataset_id_full = f"{PROJECT_ID}.{DATASET_ID}"
dataset = bigquery.Dataset(dataset_id_full)
dataset.location = "US"

print("\nCreating BigQuery dataset if it does not exist...")

dataset = client.create_dataset(
    dataset,
    exists_ok=True,
)

print(f"Dataset ready: {dataset.full_dataset_id}")


print("\nLoading local processed CSV files...")

phase3_df = pd.read_csv(PHASE3_PATH)
phase4_df = pd.read_csv(PHASE4_PATH)

phase3_df = clean_column_names(phase3_df)
phase4_df = clean_column_names(phase4_df)

phase3_df = reduce_object_column_risk(phase3_df)
phase4_df = reduce_object_column_risk(phase4_df)

for col in [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
    "order_delivered_customer_date",
]:
    if col in phase3_df.columns:
        phase3_df[col] = pd.to_datetime(phase3_df[col], errors="coerce")


phase3_table_id = upload_dataframe_to_bigquery(
    client=client,
    df=phase3_df,
    table_name=PHASE3_TABLE,
)

phase4_table_id = upload_dataframe_to_bigquery(
    client=client,
    df=phase4_df,
    table_name=PHASE4_TABLE,
)


# OVERALL KPIs

query_kpis = f"""
SELECT
  COUNT(*) AS total_orders,
  SUM(is_late) AS late_orders,
  COUNT(*) - SUM(is_late) AS on_time_orders,
  ROUND(AVG(is_late) * 100, 2) AS late_rate_pct,
  ROUND(AVG(delivery_days_estimated), 2) AS avg_estimated_delivery_days,
  ROUND(AVG(total_price), 2) AS avg_order_value,
  ROUND(AVG(total_freight), 2) AS avg_freight_value,
  ROUND(AVG(cross_state) * 100, 2) AS cross_state_rate_pct
FROM `{phase4_table_id}`
"""

run_query_to_csv(
    client=client,
    query=query_kpis,
    output_filename="phase7_bigquery_kpis.csv",
)


#  STATE-LEVEL LATE RATES

query_state_late_rates = f"""
SELECT
  customer_state,
  COUNT(*) AS total_orders,
  SUM(is_late) AS late_orders,
  ROUND(AVG(is_late) * 100, 2) AS late_rate_pct,
  ROUND(AVG(delivery_days_actual), 2) AS avg_actual_delivery_days,
  ROUND(AVG(delivery_delay_days), 2) AS avg_delay_days
FROM `{phase3_table_id}`
WHERE customer_state IS NOT NULL
GROUP BY customer_state
HAVING total_orders >= 100
ORDER BY late_rate_pct DESC
LIMIT 20
"""

run_query_to_csv(
    client=client,
    query=query_state_late_rates,
    output_filename="phase7_bigquery_state_late_rates.csv",
)


#  ROUTE / SELLER RISK SEGMENTS

query_feature_risk = f"""
SELECT
  CASE
    WHEN cross_state = 1 THEN 'Cross-state'
    ELSE 'Same-state'
  END AS route_type,

  CASE
    WHEN seller_count >= 2 THEN 'Multi-seller'
    ELSE 'Single-seller'
  END AS seller_complexity,

  COUNT(*) AS total_orders,
  SUM(is_late) AS late_orders,
  ROUND(AVG(is_late) * 100, 2) AS late_rate_pct,
  ROUND(AVG(total_freight), 2) AS avg_freight,
  ROUND(AVG(delivery_days_estimated), 2) AS avg_estimated_days
FROM `{phase4_table_id}`
GROUP BY route_type, seller_complexity
ORDER BY late_rate_pct DESC
"""

run_query_to_csv(
    client=client,
    query=query_feature_risk,
    output_filename="phase7_bigquery_feature_risk.csv",
)


#  QUERY - MONTHLY LATE RATE

query_monthly = f"""
SELECT
  purchase_month_num,
  COUNT(*) AS total_orders,
  SUM(is_late) AS late_orders,
  ROUND(AVG(is_late) * 100, 2) AS late_rate_pct,
  ROUND(AVG(total_freight), 2) AS avg_freight_value
FROM `{phase4_table_id}`
GROUP BY purchase_month_num
ORDER BY purchase_month_num
"""

run_query_to_csv(
    client=client,
    query=query_monthly,
    output_filename="phase7_bigquery_monthly_late_rate.csv",
)

#  QUERY 1 - BASELINE DELIVERY SUMMARY
# Purpose: KPI row for booth/demo slides:
# total orders, on-time rate, average delivery time, average delay.

query_baseline_summary = f"""
SELECT
  COUNT(*) AS total_delivered_orders,
  SUM(is_late) AS late_orders,
  COUNT(*) - SUM(is_late) AS on_time_orders,
  ROUND((1 - AVG(is_late)) * 100, 2) AS on_time_rate_pct,
  ROUND(AVG(is_late) * 100, 2) AS late_rate_pct,
  ROUND(AVG(delivery_days_actual), 2) AS avg_actual_delivery_days,
  ROUND(AVG(delivery_delay_days), 2) AS avg_delay_days
FROM `{phase3_table_id}`
"""

run_query_to_csv(
    client=client,
    query=query_baseline_summary,
    output_filename="phase7_query1_baseline_summary.csv",
)


#  QUERY 2 - SELLER STATE RANKING
# Purpose: Identify seller states with higher late-delivery rates.

query_seller_state_ranking = f"""
SELECT
  main_seller_state AS seller_state,
  COUNT(*) AS total_orders,
  SUM(is_late) AS late_orders,
  ROUND(AVG(is_late) * 100, 2) AS late_rate_pct,
  ROUND(AVG(delivery_days_actual), 2) AS avg_actual_delivery_days,
  ROUND(AVG(delivery_delay_days), 2) AS avg_delay_days
FROM `{phase3_table_id}`
WHERE main_seller_state IS NOT NULL
GROUP BY main_seller_state
HAVING COUNT(*) >= 50
ORDER BY late_rate_pct DESC
LIMIT 20
"""

run_query_to_csv(
    client=client,
    query=query_seller_state_ranking,
    output_filename="phase7_query2_seller_state_ranking.csv",
)


# QUERY 4 - MONTHLY TRENDS WITH 3-MONTH ROLLING AVERAGE
# Purpose: Show monthly late-rate trend with a rolling average.


query_monthly_rolling_average = f"""
WITH monthly AS (
  SELECT
    purchase_month_num,
    COUNT(*) AS total_orders,
    SUM(is_late) AS late_orders,
    ROUND(AVG(is_late) * 100, 2) AS late_rate_pct,
    ROUND(AVG(total_freight), 2) AS avg_freight_value
  FROM `{phase4_table_id}`
  GROUP BY purchase_month_num
)

SELECT
  purchase_month_num,
  total_orders,
  late_orders,
  late_rate_pct,
  avg_freight_value,
  ROUND(
    AVG(late_rate_pct) OVER (
      ORDER BY purchase_month_num
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ),
    2
  ) AS rolling_3_month_late_rate_pct
FROM monthly
ORDER BY purchase_month_num
"""

run_query_to_csv(
    client=client,
    query=query_monthly_rolling_average,
    output_filename="phase7_query4_monthly_rolling_average.csv",
)


#  QUERY 5 - CROSS-STATE VS SAME-STATE
# Purpose:
# Compare delivery risk for same-state vs cross-state shipments.
#
# CASE WHEN turns 0/1 into readable business labels.

query_cross_state_vs_same_state = f"""
SELECT
  CASE
    WHEN cross_state = 1 THEN 'Cross-state shipment'
    ELSE 'Same-state shipment'
  END AS shipment_type,
  COUNT(*) AS total_orders,
  SUM(is_late) AS late_orders,
  ROUND(AVG(is_late) * 100, 2) AS late_rate_pct,
  ROUND(AVG(total_freight), 2) AS avg_freight_value,
  ROUND(AVG(delivery_days_estimated), 2) AS avg_estimated_delivery_days
FROM `{phase4_table_id}`
GROUP BY shipment_type
ORDER BY late_rate_pct DESC
"""

run_query_to_csv(
    client=client,
    query=query_cross_state_vs_same_state,
    output_filename="phase7_query5_cross_state_vs_same_state.csv",
)


#  QUERY 8 -WORST ROUTE PAIRS
# Purpose:
# Find seller-state → customer-state route pairs with the worst late rates.
#
# HAVING COUNT(*) >= 50 removes noisy tiny routes.

query_worst_route_pairs = f"""
SELECT
  main_seller_state AS seller_state,
  customer_state,
  CONCAT(main_seller_state, ' → ', customer_state) AS route_pair,
  COUNT(*) AS total_orders,
  SUM(is_late) AS late_orders,
  ROUND(AVG(is_late) * 100, 2) AS late_rate_pct,
  ROUND(AVG(delivery_days_actual), 2) AS avg_actual_delivery_days,
  ROUND(AVG(delivery_delay_days), 2) AS avg_delay_days
FROM `{phase3_table_id}`
WHERE main_seller_state IS NOT NULL
  AND customer_state IS NOT NULL
GROUP BY main_seller_state, customer_state
HAVING COUNT(*) >= 50
ORDER BY late_rate_pct DESC
LIMIT 10
"""

run_query_to_csv(
    client=client,
    query=query_worst_route_pairs,
    output_filename="phase7_query8_worst_route_pairs.csv",
)


print("\n" + "═" * 80)
print("  BIGQUERY SQL WORKFLOW COMPLETE")
print("═" * 80)

print(f"""
BigQuery dataset created/used:
  {PROJECT_ID}.{DATASET_ID}

BigQuery tables:
  {phase3_table_id}
  {phase4_table_id}

Local SQL report outputs:
  {REPORTS_DIR / "phase7_bigquery_kpis.csv"}
  {REPORTS_DIR / "phase7_bigquery_state_late_rates.csv"}
  {REPORTS_DIR / "phase7_bigquery_feature_risk.csv"}
  {REPORTS_DIR / "phase7_bigquery_monthly_late_rate.csv"}
  {REPORTS_DIR / "phase7_query1_baseline_summary.csv"}
  {REPORTS_DIR / "phase7_query2_seller_state_ranking.csv"}
  {REPORTS_DIR / "phase7_query4_monthly_rolling_average.csv"}
  {REPORTS_DIR / "phase7_query5_cross_state_vs_same_state.csv"}
  {REPORTS_DIR / "phase7_query8_worst_route_pairs.csv"}

""")
