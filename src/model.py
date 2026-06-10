
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_scale_pos_weight(y: pd.Series) -> float:
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        raise ValueError("No positive (late) examples in target y.")
    return float(n_neg / n_pos)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "auc_roc":   float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy":  float((y_pred == y_test).mean()),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "y_pred": y_pred,
        "y_prob":  y_prob,
        "classification_report": classification_report(y_test, y_pred),
    }


def find_optimal_threshold(y_test: pd.Series, y_prob: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    f1s = (
        2 * precisions[:-1] * recalls[:-1] /
        (precisions[:-1] + recalls[:-1] + 1e-9)
    )
    return float(thresholds[int(f1s.argmax())])


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
    save_path: str | None = None,
) -> go.Figure:
    raw_importance = model.get_booster().get_score(importance_type="gain")


    importance = pd.Series(
        {feat: raw_importance.get(feat, 0.0) for feat in feature_names}
    ).sort_values(ascending=True).tail(top_n)

    colors = ["#3498db"] * len(importance)
    colors[-1] = "#e74c3c"

    fig = go.Figure(go.Bar(
        x=importance.values,
        y=importance.index,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:,.1f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top {min(top_n, len(importance))} features — XGBoost gain importance",
        xaxis_title="Gain (avg. loss reduction per split)",
        yaxis_title="",
        plot_bgcolor="white",
        height=max(420, len(importance) * 28),
        xaxis=dict(gridcolor="#f0f0f0"),
        margin=dict(l=230),
    )
    if save_path:
        fig.write_html(save_path)
    return fig


def plot_confusion_matrix_chart(
    metrics: dict,
    save_path: str | None = None,
) -> go.Figure:
    cm = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ])
    labels = [["TN\nCorrect ✓", "FP\nFalse alarm ⚠"],
              ["FN\nMissed! ✗",  "TP\nCaught! ✓"]]

    annotations = []
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            text_color = "white" if count > cm.max() * 0.4 else "#333333"
            annotations.append(dict(
                x=j, y=i,
                text=f"<b>{count:,}</b><br><span style='font-size:11px'>"
                     f"{labels[i][j].replace(chr(10), '<br>')}</span>",
                showarrow=False,
                font=dict(size=14, color=text_color),
            ))

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted: On-Time", "Predicted: Late"],
        y=["Actual: On-Time", "Actual: Late"],
        colorscale=[[0, "#f0f6fc"], [0.5, "#7fb5d8"], [1, "#1a5276"]],
        showscale=False,
    ))
    fig.update_layout(
        annotations=annotations,
        title=(f"Confusion matrix  |  threshold={metrics['threshold']:.2f}  "
               f"|  Recall={metrics['recall']:.3f}  "
               f"Precision={metrics['precision']:.3f}  "
               f"F1={metrics['f1']:.3f}"),
        xaxis=dict(side="top"),
        plot_bgcolor="white",
        height=360,
    )
    if save_path:
        fig.write_html(save_path)
    return fig


def plot_roc_curve_chart(
    y_test: pd.Series,
    y_prob: np.ndarray,
    save_path: str | None = None,
) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val     = roc_auc_score(y_test, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"XGBoost  AUC = {auc_val:.4f}",
        line=dict(color="#e74c3c", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random guess  AUC = 0.500",
        line=dict(color="#aaaaaa", width=1.2, dash="dash"),
    ))
    fig.update_layout(
        title=f"ROC curve  |  AUC-ROC = {auc_val:.4f}",
        xaxis_title="False positive rate (on-time orders wrongly flagged)",
        yaxis_title="True positive rate = Recall (late orders correctly caught)",
        plot_bgcolor="white",
        legend=dict(x=0.52, y=0.08),
        xaxis=dict(gridcolor="#f0f0f0", range=[0, 1]),
        yaxis=dict(gridcolor="#f0f0f0", range=[0, 1.02]),
    )
    if save_path:
        fig.write_html(save_path)
    return fig


def plot_threshold_analysis_chart(
    y_test: pd.Series,
    y_prob: np.ndarray,
    save_path: str | None = None,
) -> go.Figure:
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = (
        2 * precisions[:-1] * recalls[:-1] /
        (precisions[:-1] + recalls[:-1] + 1e-9)
    )
    best_idx = int(f1s.argmax())
    best_thr = float(thresholds[best_idx])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=precisions[:-1],
        mode="lines", name="Precision",
        line=dict(color="#F59E0B", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=recalls[:-1],
        mode="lines", name="Recall",
        line=dict(color="#e74c3c", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=f1s,
        mode="lines", name="F1",
        line=dict(color="#3498db", width=2.5),
    ))
    fig.add_vline(
        x=best_thr,
        line_dash="dash", line_color="#3498db",
        annotation_text=f"Best F1 = {best_thr:.2f}",
        annotation_position="top right",
    )
    fig.add_vline(
        x=0.5,
        line_dash="dot", line_color="#999999",
        annotation_text="Default = 0.50",
        annotation_position="bottom left",
    )
    fig.update_layout(
        title="Precision, Recall, and F1 vs classification threshold",
        xaxis_title="Threshold — predict Late if P(late) ≥ threshold",
        yaxis_title="Score",
        plot_bgcolor="white",
        yaxis=dict(range=[0, 1.05], gridcolor="#f0f0f0"),
        xaxis=dict(range=[0, 1],    gridcolor="#f0f0f0"),
    )
    if save_path:
        fig.write_html(save_path)
    return fig
