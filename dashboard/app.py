"""
app.py
------
Phase 6: Streamlit Dashboard — Last-Mile Delivery Risk Predictor

A live web app that:
  1. Displays model KPIs and EDA charts from the real Olist dataset
  2. Shows a Brazil map of late-delivery rates by state
  3. Lets users input any order profile and receive a live prediction

Run from the project root:
    streamlit run dashboard/app.py

Prerequisites:
  - Phase 3 complete  → data/processed/phase3_eda_orders.csv
  - Phase 5 complete  → models/xgboost_model.pkl
                        models/preprocessor.pkl
                        models/model_metrics.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


LATE_COLOR   = "#e74c3c"
ONTIME_COLOR = "#27ae60"
ACCENT       = "#2980b9"

# Approximate centre coordinates for each Brazilian state
BRAZIL_COORDS = {
    "AC": (-9.02, -70.81),  "AL": (-9.57, -36.78),  "AP": ( 1.41, -51.77),
    "AM": (-3.42, -65.86),  "BA": (-12.58,-41.70),   "CE": (-5.50, -39.32),
    "DF": (-15.78,-47.93),  "ES": (-19.18,-40.31),   "GO": (-15.83,-49.84),
    "MA": (-4.96, -45.27),  "MT": (-12.68,-56.92),   "MS": (-20.77,-54.79),
    "MG": (-18.51,-44.56),  "PA": (-3.42, -52.76),   "PB": (-7.24, -36.78),
    "PR": (-24.89,-51.90),  "PE": (-8.81, -36.95),   "PI": (-7.72, -42.73),
    "RJ": (-22.91,-43.17),  "RN": (-5.81, -36.21),   "RS": (-30.03,-51.22),
    "RO": (-11.51,-63.58),  "RR": ( 2.74, -62.08),   "SC": (-27.24,-50.22),
    "SP": (-23.55,-46.63),  "SE": (-10.57,-37.39),   "TO": (-10.18,-48.30),
}
BR_STATES    = sorted(BRAZIL_COORDS.keys())
PAYMENT_TYPES = ["credit_card", "boleto", "debit_card", "voucher"]
DAYS_OF_WEEK  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MONTH_NAMES   = ["January","February","March","April","May","June",
                 "July","August","September","October","November","December"]
MONTH_NUMS    = {m: i+1 for i, m in enumerate(MONTH_NAMES)}
MONTH_ORDER   = MONTH_NAMES   # used for chart sorting


st.set_page_config(
    page_title="Delivery Risk Control Panel",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.85rem; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 0.82rem; }
    .verdict-box {
        border-radius: 10px; padding: 1.1rem 1.4rem;
        text-align: center;  margin: 0.6rem 0 0.8rem;
    }
    .high-risk { background:#fdecea; border:2px solid #e74c3c; }
    .low-risk  { background:#eafaf1; border:2px solid #27ae60; }
    .verdict-label { font-size:1.3rem; font-weight:700; margin-bottom:0.2rem; }
    .verdict-prob  { font-size:2rem;   font-weight:800; }
    .verdict-sub   { font-size:0.82rem; color:#666; margin-top:0.3rem; }
    .factor-card {
    background:#fff8f0;
    border-left:4px solid #e67e22;
    padding:0.55rem 0.9rem;
    border-radius:4px;
    margin:0.35rem 0;
    font-size:0.88rem;
    color:#1f2937 !important;
}

.factor-card * {
    color:#1f2937 !important;
}
</style>
""", unsafe_allow_html=True)



@st.cache_data(show_spinner="Loading EDA data …")
def load_eda() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "data" / "processed" / "phase3_eda_orders.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource(show_spinner="Loading model …")
def load_model_artifacts():
    """
    Returns (model, preprocessor) or (None, None).
    @st.cache_resource keeps the live objects in RAM across reruns —
    crucial for a 100 MB XGBoost model that would otherwise reload every
    time a slider moves.
    """
    mp = PROJECT_ROOT / "models" / "xgboost_model.pkl"
    pp = PROJECT_ROOT / "models" / "preprocessor.pkl"
    model       = joblib.load(mp) if mp.exists() else None
    preprocessor = joblib.load(pp) if pp.exists() else None
    return model, preprocessor


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    path = PROJECT_ROOT / "models" / "model_metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def predict_order(inputs: dict, preprocessor: dict, model) -> float:
    """
    Apply the Phase 4 pipeline to one order dict and return P(late).

    TRAINING-SERVING SKEW WARNING:
    Every transformation here must exactly mirror src/features.py.
    The column ORDER must also match — XGBoost assigns meaning to column
    positions at training time. Passing columns in a different order
    silently produces wrong predictions.
    """
    le    = preprocessor["label_encoders"]
    fills = preprocessor["numeric_fill_values"]
    fcols = preprocessor["feature_columns"]

    fr = min(inputs["total_freight"] / (inputs["total_price"] + 1), 10.0)
    row = {
        "delivery_days_estimated":  inputs["delivery_days_estimated"],
        "purchase_hour":            inputs["purchase_hour"],
        "purchase_month_num":       inputs["purchase_month_num"],
        "is_weekend":   int(inputs["purchase_day_of_week"] in ["Saturday","Sunday"]),
        "total_price":              inputs["total_price"],
        "total_freight":            inputs["total_freight"],
        "freight_ratio":            fr,
        "price_per_item":           inputs["total_price"] / max(inputs["item_count"], 1),
        "item_count":               inputs["item_count"],
        "product_count":            1,
        "seller_count":             1,
        "avg_product_weight_g":     inputs["avg_product_weight_g"],
        "avg_product_length_cm":    fills.get("avg_product_length_cm",   30.0),
        "avg_product_height_cm":    fills.get("avg_product_height_cm",   15.0),
        "avg_product_width_cm":     fills.get("avg_product_width_cm",    20.0),
        "avg_product_volume_cm3":   fills.get("avg_product_volume_cm3", 9000.0),
        "total_payment_value":      inputs["total_price"] + inputs["total_freight"],
        "max_payment_installments": inputs["installments"],
        "payment_count":            1,
        "cross_state":  int(inputs["customer_state"] != inputs["main_seller_state"]),
        "approval_time_hours":      fills.get("approval_time_hours", 2.0),
    }

    for col in ["purchase_day_of_week","customer_state","main_seller_state",
                "main_payment_type","main_product_category"]:
        if col not in le:
            continue
        encoder = le[col]
        val = str(inputs.get(col, "missing"))
        if val not in set(encoder.classes_):
            val = "missing"
            if "missing" not in set(encoder.classes_):
                encoder.classes_ = np.append(encoder.classes_, "missing")
        row[f"{col}_encoded"] = int(encoder.transform([val])[0])

    df_row = pd.DataFrame([row])
    for col in fcols:
        if col not in df_row.columns:
            df_row[col] = fills.get(col, 0.0)

    return float(model.predict_proba(df_row[fcols].astype(float))[0, 1])


def risk_factors(inputs: dict, preprocessor: dict) -> list[str]:
    
    fills   = preprocessor["numeric_fill_values"]
    factors = []

    if inputs["delivery_days_estimated"] > 18:
        factors.append(
            f"Long promised window: {inputs['delivery_days_estimated']} days "
            f"(sellers with long estimates frequently still miss them)"
        )
    if inputs["customer_state"] != inputs["main_seller_state"]:
        factors.append(
            f"Cross-state shipment: {inputs['main_seller_state']} → "
            f"{inputs['customer_state']} — more carrier handoffs, longer route"
        )
    fr = inputs["total_freight"] / (inputs["total_price"] + 1)
    if fr > 0.28:
        factors.append(
            f"High freight ratio ({fr:.2f}) — heavy or bulky item, "
            f"or long-distance delivery"
        )
    if inputs["purchase_day_of_week"] in ["Friday", "Saturday", "Sunday"]:
        factors.append(
            f"Order placed on {inputs['purchase_day_of_week']} — "
            f"weekend warehouse shift reduces on-time pick-up likelihood"
        )
    median_w = fills.get("avg_product_weight_g", 2_000)
    if inputs["avg_product_weight_g"] > median_w * 2.5:
        factors.append(
            f"Heavy product ({inputs['avg_product_weight_g']/1000:.1f} kg) — "
            f"special handling required"
        )
    if inputs["installments"] > 6:
        factors.append(
            f"High-value purchase: {inputs['installments']} installments "
            f"(may require additional payment approval steps)"
        )
    return factors


def gauge_chart(prob: float, threshold: float) -> go.Figure:
    """Plotly gauge showing P(late) as a 0–100 % arc."""
    colour = LATE_COLOR if prob >= threshold else ONTIME_COLOR
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 52, "color": colour}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar":  {"color": colour, "thickness": 0.22},
            "steps": [
                {"range": [0,  30], "color": "#d5f5e3"},
                {"range": [30, 55], "color": "#fef9e7"},
                {"range": [55,100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "#333333", "width": 3},
                "thickness": 0.82,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=16, r=16, t=24, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Delivery Risk Control Panel")
    st.markdown(
        "XGBoost classifier trained on **100 k+ real Brazilian e-commerce orders** "
        "to predict last-mile delivery failures *before* they happen."
    )
    st.divider()

    st.markdown("**Tech stack**")
    st.markdown("""
| Layer | Tool |
|-------|------|
| Data | Olist / Kaggle |
| Model | XGBoost |
| Features | 26 engineered |
| Dashboard | Streamlit |
| Cloud | BigQuery |
""")
    st.divider()

    with st.expander("How it works"):
        st.markdown("""
- Order details are converted into 26 model features.
- Categorical fields are encoded using the saved preprocessor.
- XGBoost returns a late-delivery probability.
- High-risk orders can be flagged for operational review.
- Logistics teams can monitor, reroute, or notify customers proactively.
        """)

    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# LOAD EVERYTHING
# ══════════════════════════════════════════════════════════════════════════════

df_eda               = load_eda()
model, preprocessor  = load_model_artifacts()
metrics              = load_metrics()
model_ready          = model is not None and preprocessor is not None

if not model_ready:
    st.warning(
        "Model files not found ; run `python notebooks/05_model.py`, "
        "then refresh this page."
    )

if df_eda is None:
    st.info(
        " EDA dataset not found — run `python notebooks/03_eda.py` "
        "to enable the Overview and Geography tabs."
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER + KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════

st.title("Last-Mile Delivery Risk Predictor")
st.markdown("*XGBoost · Olist e-commerce dataset · 100 k+ real orders*")
st.divider()

if metrics:
    opt  = metrics.get("at_optimal_threshold", {})
    n_tr = metrics.get("n_train", 0)
    late_rate_tr = metrics.get("late_rate_train", 0.07)
    thr  = metrics.get("optimal_threshold", 0.5)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Training orders",
            f"{n_tr:,}",
            help="Delivered orders used to train the XGBoost model",
        )
    with k2:
        baseline = round((1 - late_rate_tr) * 100, 1)
        st.metric(
            "On-time baseline",
            f"{baseline} %",
            help="Accuracy if we predict every order on-time",
        )
    with k3:
        auc = opt.get("auc_roc", 0)
        st.metric(
            "AUC-ROC",
            f"{auc:.4f}",
            delta=f"+{auc - 0.5:.4f} vs random",
            help="Area Under ROC Curve. 1.0 = perfect, 0.5 = random guessing",
        )
    with k4:
        recall = opt.get("recall", 0)
        st.metric(
            "Late orders caught",
            f"{recall*100:.1f} %",
            help=f"Recall at threshold {thr:.2f}: fraction of late orders the model flags",
        )
else:
    st.info("Run Phase 5 to see model metrics here.")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["  Overview", "  Geography", "  Live Predictor"])


# ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────────
with tab1:
    if df_eda is None:
        st.info("Run `python notebooks/03_eda.py` to populate this tab.")
    else:
        st.subheader("Delivery time distribution")
        hist_df = (
            df_eda[df_eda["delivery_days_actual"].between(1, 80)]
            .copy()
            .assign(Outcome=lambda d: d["is_late"].map({0: "On Time", 1: "Late"}))
        )
        fig_hist = px.histogram(
            hist_df,
            x="delivery_days_actual",
            color="Outcome",
            nbins=70,
            barmode="overlay",
            opacity=0.72,
            color_discrete_map={"On Time": ONTIME_COLOR, "Late": LATE_COLOR},
            labels={"delivery_days_actual": "Days (order → delivery)", "count": "Orders"},
        )
        fig_hist.update_layout(
            plot_bgcolor="white",
            legend_title_text="",
            yaxis=dict(gridcolor="#f0f0f0"),
            height=320,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.subheader("Late rate by month")
            if "purchase_month" in df_eda.columns:
                m_stats = (
                    df_eda.groupby("purchase_month")["is_late"]
                    .agg(late_rate="mean", orders="count")
                    .reset_index()
                    .assign(
                        pct=lambda d: d["late_rate"] * 100,
                        purchase_month=lambda d: pd.Categorical(
                            d["purchase_month"], categories=MONTH_ORDER, ordered=True
                        ),
                    )
                    .sort_values("purchase_month")
                )
                fig_m = px.line(
                    m_stats, x="purchase_month", y="pct",
                    markers=True,
                    labels={"purchase_month": "", "pct": "Late rate (%)"},
                    hover_data={"orders": True},
                )
                fig_m.update_traces(line_color=LATE_COLOR, marker_size=8)
                fig_m.update_layout(
                    plot_bgcolor="white",
                    yaxis=dict(gridcolor="#f0f0f0"),
                    xaxis=dict(tickangle=40),
                    height=310,
                )
                st.plotly_chart(fig_m, use_container_width=True)

        with right:
            st.subheader("Highest-risk product categories")
            cat_col = "main_product_category" if "main_product_category" in df_eda.columns else None
            if cat_col:
                cat_stats = (
                    df_eda.dropna(subset=[cat_col])
                    .groupby(cat_col)["is_late"]
                    .agg(late_rate="mean", orders="count")
                    .reset_index()
                    .assign(pct=lambda d: d["late_rate"] * 100)
                    .query("orders >= 100")
                    .sort_values("pct", ascending=True)
                    .tail(12)
                )
                fig_c = px.bar(
                    cat_stats, x="pct", y=cat_col,
                    orientation="h",
                    color="pct",
                    color_continuous_scale="Reds",
                    labels={"pct": "Late rate (%)", cat_col: ""},
                    hover_data={"orders": True},
                )
                fig_c.update_layout(
                    plot_bgcolor="white",
                    coloraxis_showscale=False,
                    height=310,
                )
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.info("Product category column not found in EDA dataset.")


# ── TAB 2: GEOGRAPHY ──────────────────────────────────────────────────────────
with tab2:
    if df_eda is None:
        st.info("Run `python notebooks/03_eda.py` to populate this tab.")
    else:
        st.subheader("Late delivery rate across Brazilian states")

        state_col = next(
            (c for c in ["customer_state", "main_seller_state"] if c in df_eda.columns),
            None,
        )

        if state_col is None:
            st.info("State column not found in EDA dataset.")
        else:
            state_stats = (
                df_eda.groupby(state_col)["is_late"]
                .agg(late_rate="mean", orders="count")
                .reset_index()
                .rename(columns={state_col: "state"})
                .assign(
                    late_rate_pct=lambda d: d["late_rate"] * 100,
                    lat=lambda d: d["state"].map(
                        lambda s: BRAZIL_COORDS.get(s, (None, None))[0]
                    ),
                    lon=lambda d: d["state"].map(
                        lambda s: BRAZIL_COORDS.get(s, (None, None))[1]
                    ),
                )
                .dropna(subset=["lat", "lon"])
                .query("orders >= 30")
            )

            map_col, tbl_col = st.columns([3, 2], gap="medium")

            with map_col:
                fig_map = px.scatter_geo(
                    state_stats,
                    lat="lat", lon="lon",
                    size="orders",
                    color="late_rate_pct",
                    hover_name="state",
                    hover_data={
                        "orders": True,
                        "late_rate_pct": ":.1f",
                        "lat": False, "lon": False,
                    },
                    color_continuous_scale="Reds",
                    size_max=48,
                    labels={"late_rate_pct": "Late %"},
                )
                fig_map.update_geos(
                    scope="south america",
                    showland=True,       landcolor="#f5f5f0",
                    showocean=True,      oceancolor="#d6eaf8",
                    showcountries=True,  countrycolor="#cccccc",
                    showcoastlines=True, coastlinecolor="#cccccc",
                    center={"lat": -15, "lon": -52},
                    lataxis={"range": [-35, 6]},
                    lonaxis={"range": [-75, -28]},
                )
                fig_map.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=500,
                    coloraxis_colorbar=dict(title="Late %"),
                )
                st.plotly_chart(fig_map, use_container_width=True)
                st.caption(
                    "Bubble size = order volume. Colour intensity = late rate. "
                    "Hover any bubble for exact figures."
                )

            with tbl_col:
                st.markdown("**State ranking — highest late rate first**")
                tbl_display = (
                    state_stats
                    .sort_values("late_rate_pct", ascending=False)
                    .rename(columns={
                        "state":         "State",
                        "orders":        "Orders",
                        "late_rate_pct": "Late %",
                    })
                    [["State", "Orders", "Late %"]]
                    .assign(**{"Late %": lambda d: d["Late %"].round(1)})
                    .reset_index(drop=True)
                )
                st.dataframe(tbl_display, height=490, use_container_width=True)


# ── TAB 3: LIVE PREDICTOR ─────────────────────────────────────────────────────
with tab3:
    if not model_ready:
        st.warning(
            "Model not ready. Run `python notebooks/05_model.py` then refresh."
        )
        st.stop()

    threshold = metrics.get("optimal_threshold", 0.5) if metrics else 0.5

    if df_eda is not None and "main_product_category" in df_eda.columns:
        all_categories = sorted(df_eda["main_product_category"].dropna().unique())
    else:
        all_categories = [
            "bed_bath_table","computers_accessories","cool_stuff",
            "furniture_decor","garden_tools","health_beauty","housewares",
            "perfumery","pet_shop","sports_leisure","telephony","toys",
            "watches_gifts",
        ]

    st.markdown(
        "Adjust the order profile on the left. The model recalculates "
        "**instantly** on every change — no submit button needed."
    )
    st.markdown("---")

    form_col, result_col = st.columns([5, 6], gap="large")

    with form_col:
        st.markdown("#### Order profile")

        with st.expander("Delivery & items", expanded=True):
            delivery_days = st.slider(
                "Estimated delivery window (days)", 1, 60, 14,
                help="How many days the seller promised between order and delivery",
            )
            total_price = st.slider("Order value (R$)", 10.0, 2_000.0, 150.0, 10.0)
            total_freight = st.slider("Freight cost (R$)", 5.0, 500.0, 25.0, 1.0)
            item_count = st.slider("Number of items in order", 1, 20, 1)
            product_weight = st.slider(
                "Product weight (g)", 100, 30_000, 1_500, 100,
                help="Average weight per item",
            )

        with st.expander(" Geography", expanded=True):
            seller_state = st.selectbox(
                "Seller state", BR_STATES, index=BR_STATES.index("SP"),
                help="Brazilian state of the seller's warehouse",
            )
            customer_state = st.selectbox(
                "Customer state", BR_STATES, index=BR_STATES.index("RJ"),
            )
            if seller_state != customer_state:
                st.info(
                    f" Cross-state shipment detected "
                    f"({seller_state} → {customer_state}) — risk signal"
                )

        with st.expander("Product & payment", expanded=False):
            product_category = st.selectbox("Product category", all_categories)
            payment_type = st.selectbox("Payment method", PAYMENT_TYPES)
            installments = st.slider("Payment installments", 1, 24, 1)

        with st.expander("Order timing", expanded=False):
            day_of_week = st.selectbox("Day of week order was placed", DAYS_OF_WEEK)
            month_name  = st.selectbox("Month order was placed", MONTH_NAMES, index=5)
            hour        = st.slider("Hour of day (0 = midnight)", 0, 23, 14)

    inputs = {
        "delivery_days_estimated": delivery_days,
        "total_price":             total_price,
        "total_freight":           total_freight,
        "item_count":              item_count,
        "avg_product_weight_g":    float(product_weight),
        "main_seller_state":       seller_state,
        "customer_state":          customer_state,
        "main_product_category":   product_category,
        "main_payment_type":       payment_type,
        "installments":            installments,
        "purchase_day_of_week":    day_of_week,
        "purchase_month_num":      MONTH_NUMS[month_name],
        "purchase_hour":           hour,
    }

    prob     = predict_order(inputs, preprocessor, model)
    is_late  = prob >= threshold

    with result_col:
        st.markdown("#### Prediction")

        st.plotly_chart(gauge_chart(prob, threshold), use_container_width=True)

        if is_late:
            css_cls     = "verdict-box high-risk"
            label_text  = "HIGH RISK — Likely late"
            label_color = LATE_COLOR
        else:
            css_cls     = "verdict-box low-risk"
            label_text  = "LOW RISK — Likely on time"
            label_color = ONTIME_COLOR

        st.markdown(
            f'<div class="{css_cls}">'
            f'<div class="verdict-label" style="color:{label_color}">{label_text}</div>'
            f'<div class="verdict-prob"  style="color:{label_color}">'
            f'{prob*100:.1f}% probability of late delivery</div>'
            f'<div class="verdict-sub">'
            f'Decision threshold: {threshold:.2f} '
            f'(flag if P ≥ {threshold*100:.0f}%)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Recommended operational action
        st.markdown("#### Recommended action")
        if prob >= 0.72:
            st.error(
                "**Immediate escalation** — reroute to express carrier, "
                "notify customer proactively, alert seller warehouse now."
            )
        elif is_late:
            st.warning(
                "**Flag for monitoring** — add to priority watch list, "
                "send seller on-time reminder, prepare customer comms."
            )
        elif prob >= 0.25:
            st.info(
                "**Watch list** — track carrier updates; escalate if "
                "pickup delay exceeds 12 hours."
            )
        else:
            st.success(
                "**Standard handling** — no immediate action required. "
                "Low-risk order profile."
            )

        # Risk factor breakdown
        factors = risk_factors(inputs, preprocessor)
        if factors:
            st.markdown("#### Risk factors driving this prediction")
            for f in factors:
                st.markdown(
                    f'<div class="factor-card">⚠ {f}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "*No significant risk factors detected for this order profile.*"
            )

        # Collapsible model context — useful at the booth
        with st.expander("Model performance summary"):
            if metrics:
                opt_m = metrics.get("at_optimal_threshold", {})
                d050  = metrics.get("at_threshold_0.50", {})
                st.markdown(f"""
| Metric | Threshold {threshold:.2f} (optimal) | Threshold 0.50 |
|--------|--------------------------------------|----------------|
| AUC-ROC | `{opt_m.get('auc_roc', 0):.4f}` | `{d050.get('auc_roc', 0):.4f}` |
| Recall  | `{opt_m.get('recall', 0):.4f}` | `{d050.get('recall', 0):.4f}` |
| Precision | `{opt_m.get('precision', 0):.4f}` | `{d050.get('precision', 0):.4f}` |
| F1 | `{opt_m.get('f1', 0):.4f}` | `{d050.get('f1', 0):.4f}` |
| Late orders caught (TP) | `{opt_m.get('tp', 0):,}` | `{d050.get('tp', 0):,}` |
| False alarms (FP) | `{opt_m.get('fp', 0):,}` | `{d050.get('fp', 0):,}` |
| Training orders | `{metrics.get('n_train', 0):,}` | — |
| Features | `{metrics.get('n_features', 0)}` | — |
                """)
            else:
                st.info("Run Phase 5 to populate model metrics.")

        with st.expander("Order summary (what the model sees)"):
            fr = min(total_freight / (total_price + 1), 10.0)
            summary = pd.DataFrame([{
                "delivery_days_estimated": delivery_days,
                "total_price (R$)":        total_price,
                "total_freight (R$)":      total_freight,
                "freight_ratio":           round(fr, 3),
                "price_per_item (R$)":     round(total_price / max(item_count, 1), 2),
                "avg_product_weight_g":    product_weight,
                "is_weekend":              int(day_of_week in ["Saturday","Sunday"]),
                "cross_state":             int(seller_state != customer_state),
                "seller_state":            seller_state,
                "customer_state":          customer_state,
                "product_category":        product_category,
                "payment_type":            payment_type,
                "installments":            installments,
                "purchase_hour":           hour,
                "purchase_month_num":      MONTH_NUMS[month_name],
                "purchase_day_of_week":    day_of_week,
            }]).T.rename(columns={0: "Value"})
            st.dataframe(summary, use_container_width=True)
