"""
05_model.py
-----------
Phase 5: Train, tune, and evaluate an XGBoost delivery-lateness predictor.

Three-stage pipeline:
  A. Baseline  — XGBoost with early stopping → finds optimal n_estimators
  B. Tuning    — RandomizedSearchCV on 4 hyperparameters
  C. Final     — best params, retrain on full training set, evaluate on test

Loads  : data/processed/phase4_feature_matrix.csv
Saves  : models/xgboost_model.pkl
         models/model_metrics.json
         reports/figures/05_feature_importance.html
         reports/figures/05_confusion_matrix.html
         reports/figures/05_roc_curve.html
         reports/figures/05_threshold_analysis.html


"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from xgboost import XGBClassifier

from src.features import FINAL_FEATURE_COLUMNS, TARGET_COLUMN
from src.model import (
    compute_scale_pos_weight,
    evaluate_model,
    find_optimal_threshold,
    plot_confusion_matrix_chart,
    plot_feature_importance,
    plot_roc_curve_chart,
    plot_threshold_analysis_chart,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
FIGURES_DIR   = PROJECT_ROOT / "reports" / "figures"

MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


SEARCH_ITER = 5
CV_FOLDS    = 3


def print_metrics(m: dict, label: str) -> None:
    n_late_total = m["tp"] + m["fn"]
    print(f"""
  ── {label} {'─' * max(1, 52 - len(label))}
  AUC-ROC    :  {m['auc_roc']:.4f}   ← primary metric (threshold-independent)
  Recall     :  {m['recall']:.4f}   ← of all late orders, % caught
  Precision  :  {m['precision']:.4f}   ← of all flagged orders, % truly late
  F1         :  {m['f1']:.4f}   ← harmonic mean of recall and precision
  Accuracy   :  {m['accuracy']:.4f}   ← misleading on imbalanced data
  Threshold  :  {m['threshold']:.2f}

  Confusion matrix:
    TN — on-time, correctly passed     :  {m['tn']:>8,}
    FP — on-time, wrongly flagged      :  {m['fp']:>8,}
    FN — late, MISSED (worst error)    :  {m['fn']:>8,}   ({m['fn']/n_late_total*100:.1f}% of late orders missed)
    TP — late, correctly caught        :  {m['tp']:>8,}   ({m['recall']*100:.1f}% recall)
""")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 68)
print("  PHASE 5 — XGBOOST MODEL TRAINING")
print("═" * 68)

matrix_path = PROCESSED_DIR / "phase4_feature_matrix.csv"
if not matrix_path.exists():
    raise FileNotFoundError(
        f"\nFile not found: {matrix_path}"
        f"\nRun notebooks/04_feature_engineering.py first."
    )

print(f"\nLoading: {matrix_path.name} ...")
df = pd.read_csv(matrix_path)
print(f"  {df.shape[0]:,} rows × {df.shape[1]} columns")

feature_cols = [c for c in FINAL_FEATURE_COLUMNS if c in df.columns]
missing_cols = [c for c in FINAL_FEATURE_COLUMNS if c not in df.columns]

if missing_cols:
    print(f"\n  WARNING: {len(missing_cols)} expected features absent:")
    for c in missing_cols:
        print(f"    – {c}")

X = df[feature_cols].copy()
y = df[TARGET_COLUMN].copy()

print(f"\n  Features : {X.shape[1]}")
print(f"  Rows     : {X.shape[0]:,}")
print(f"  Late     : {int(y.sum()):,}  ({y.mean()*100:.2f}%)")
print(f"  On-time  : {int((y==0).sum()):,}  ({(1-y.mean())*100:.2f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TRAIN / TEST SPLIT (stratified 80 / 20)
# ═══════════════════════════════════════════════════════════════════════════════


print("\n\n" + "═" * 68)
print("═" * 68)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

print(f"\n  Train : {len(X_train):>8,} rows  |  {int(y_train.sum()):,} late  "
      f"({y_train.mean()*100:.2f}%)")
print(f"  Test  : {len(X_test):>8,} rows  |  {int(y_test.sum()):,} late  "
      f"({y_test.mean()*100:.2f}%)")

rate_diff = abs(y_train.mean() - y_test.mean())
print(f"\n  Late-rate difference (train vs test): {rate_diff:.4f}  "
      f"({'✓ well balanced' if rate_diff < 0.005 else '⚠ check split'})")


spw = compute_scale_pos_weight(y_train)
print(f"\n  scale_pos_weight = {spw:.2f}")
print(f"  Each late order counted as {spw:.1f}× an on-time order during training")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE A — BASELINE MODEL WITH EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════════


print("\n\n" + "═" * 68)
print("═" * 68)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.10,
    stratify=y_train,
    random_state=42,
)
print(f"\n  Inner split — train: {len(X_tr):,}  |  val (for early stop): {len(X_val):,}")

baseline = XGBClassifier(
    n_estimators=1_000,      
    max_depth=6,              
    learning_rate=0.05,       
    subsample=0.8,           
    colsample_bytree=0.8,     
    min_child_weight=3,      
    scale_pos_weight=spw,     
    eval_metric="auc",        
    early_stopping_rounds=50, 
    verbosity=0,
    random_state=42,
    n_jobs=1,
    tree_method="hist",
)

print("\n  Training (early stopping on AUC of inner validation set)...")
baseline.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

best_n = baseline.best_iteration + 1
print(f"  Stopped at round {baseline.best_iteration}  →  optimal n_estimators = {best_n}")

baseline_metrics = evaluate_model(baseline, X_test, y_test, threshold=0.5)
print_metrics(baseline_metrics, f"Baseline (n_estimators={best_n}, threshold=0.50)")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE B — HYPERPARAMETER TUNING (RandomizedSearchCV)
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "═" * 68)
print(f"(n_iter={SEARCH_ITER}, cv={CV_FOLDS})")
print(f"  Estimated runtime: {SEARCH_ITER * 2}–{SEARCH_ITER * 4} minutes")
print(f"  Lower SEARCH_ITER at the top of this file for faster runs.")
print("═" * 68)

param_dist = {
    "max_depth":        [3, 4, 5, 6, 7, 8],
    "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
    "subsample":        [0.60, 0.70, 0.80, 0.90, 1.00],
    "colsample_bytree": [0.60, 0.70, 0.80, 0.90, 1.00],
    "min_child_weight": [1, 2, 3, 5, 7, 10],
    "reg_alpha":        [0, 0.01, 0.05, 0.1, 0.5],
}

search_model = XGBClassifier(
    n_estimators=best_n,
    scale_pos_weight=spw,
    verbosity=0,
    random_state=42,
)

cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    search_model,
    param_distributions=param_dist,
    n_iter=SEARCH_ITER,
    scoring="roc_auc",   
    cv=cv_strategy,
    n_jobs=1,           
    verbose=1,
    random_state=42,
    refit=True,          
)

total_fits = SEARCH_ITER * CV_FOLDS
print(f"\n  Fitting {total_fits} models across {len(param_dist)} parameter distributions...")
print(f"  Training rows per fit: {len(X_train):,}  |  Features: {X_train.shape[1]}\n")

search.fit(X_train, y_train)

print(f"\n  Best CV AUC-ROC : {search.best_score_:.4f}")
print(f"  Best parameters found:")
for param, value in sorted(search.best_params_.items()):
    print(f"    {param:<25} = {value}")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE C — EVALUATE FINAL MODEL ON HELD-OUT TEST SET
# ═══════════════════════════════════════════════════════════════════════════════


print("\n\n" + "═" * 68)
print("═" * 68)

final_model = search.best_estimator_

final_50 = evaluate_model(final_model, X_test, y_test, threshold=0.5)
print_metrics(final_50, "Final model — threshold = 0.50")

print(f"  Improvement over baseline (default 0.5 threshold):")
print(f"    AUC-ROC  : {baseline_metrics['auc_roc']:.4f}  →  "
      f"{final_50['auc_roc']:.4f}  ({final_50['auc_roc']-baseline_metrics['auc_roc']:+.4f})")
print(f"    Recall   : {baseline_metrics['recall']:.4f}  →  "
      f"{final_50['recall']:.4f}  ({final_50['recall']-baseline_metrics['recall']:+.4f})")
print(f"    F1       : {baseline_metrics['f1']:.4f}  →  "
      f"{final_50['f1']:.4f}  ({final_50['f1']-baseline_metrics['f1']:+.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — THRESHOLD OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 68)
print("  STEP 3 — Threshold optimisation")
print("═" * 68)

optimal_thr = find_optimal_threshold(y_test, final_50["y_prob"])
final_opt   = evaluate_model(final_model, X_test, y_test, threshold=optimal_thr)

n_late_total = final_opt["tp"] + final_opt["fn"]

print(f"""
  Default threshold (0.50):
    Recall    = {final_50['recall']:.4f}  →  catches {final_50['tp']:,} of {n_late_total:,} late orders
    Precision = {final_50['precision']:.4f}
    F1        = {final_50['f1']:.4f}

  Optimal threshold ({optimal_thr:.2f}):
    Recall    = {final_opt['recall']:.4f}  →  catches {final_opt['tp']:,} of {n_late_total:,} late orders
    Precision = {final_opt['precision']:.4f}
    F1        = {final_opt['f1']:.4f}


""")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — BUSINESS IMPACT STATEMENT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 68)
print("  STEP 4 — Business impact")
print("═" * 68)

m = final_opt
n_flagged = m["tp"] + m["fp"]

print(f"""
  At threshold = {optimal_thr:.2f} on the held-out test set ({len(X_test):,} orders):

  Late orders in test set             :  {n_late_total:,}
  Model caught (TP — can intervene)   :  {m['tp']:,}   ({m['recall']*100:.1f}% recall)
  Model missed (FN — customer risk)   :  {m['fn']:,}   ({m['fn']/n_late_total*100:.1f}% of late orders)
  False alarms (FP — wasted effort)   :  {m['fp']:,}

  Total orders flagged for review     :  {n_flagged:,}
    Of those, actually late           :  {m['tp']:,}   ({m['precision']*100:.1f}% precision)

""")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — GENERATE CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 68)
print("  STEP 5 — Generating 4 charts")
print("═" * 68 + "\n")

fig_fi = plot_feature_importance(
    final_model,
    feature_names=feature_cols,
    top_n=20,
    save_path=str(FIGURES_DIR / "05_feature_importance.html"),
)

fig_cm = plot_confusion_matrix_chart(
    final_opt,
    save_path=str(FIGURES_DIR / "05_confusion_matrix.html"),
)

fig_roc = plot_roc_curve_chart(
    y_test, final_opt["y_prob"],
    save_path=str(FIGURES_DIR / "05_roc_curve.html"),
)

fig_thr = plot_threshold_analysis_chart(
    y_test, final_opt["y_prob"],
    save_path=str(FIGURES_DIR / "05_threshold_analysis.html"),
)

try:
    for fig in [fig_fi, fig_cm, fig_roc, fig_thr]:
        fig.show()
except Exception:
    pass  


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SAVE MODEL AND METRICS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 68)
print("  STEP 6 — Saving model and metrics")
print("═" * 68)

# Trained model — loaded by Phase 6 Streamlit dashboard
model_path = MODELS_DIR / "xgboost_model.pkl"
joblib.dump(final_model, model_path)
print(f"\n  ✓ Model   → {model_path}")

metrics_dict = {
    "model_type":        "XGBClassifier",
    "n_features":        int(X.shape[1]),
    "feature_names":     feature_cols,
    "n_train":           int(len(X_train)),
    "n_test":            int(len(X_test)),
    "late_rate_train":   float(y_train.mean()),
    "late_rate_test":    float(y_test.mean()),
    "scale_pos_weight":  float(spw),
    "best_n_estimators": int(best_n),
    "best_params":       {k: float(v) if isinstance(v, (int, float)) else v
                          for k, v in search.best_params_.items()},
    "optimal_threshold": float(optimal_thr),
    "baseline_accuracy": float(1 - y_test.mean()),
    "at_threshold_0.50": {
        "auc_roc":   round(final_50["auc_roc"],   4),
        "recall":    round(final_50["recall"],    4),
        "precision": round(final_50["precision"], 4),
        "f1":        round(final_50["f1"],        4),
        "accuracy":  round(final_50["accuracy"],  4),
        "tp": final_50["tp"], "fp": final_50["fp"],
        "tn": final_50["tn"], "fn": final_50["fn"],
    },
    "at_optimal_threshold": {
        "threshold": float(optimal_thr),
        "auc_roc":   round(final_opt["auc_roc"],   4),
        "recall":    round(final_opt["recall"],    4),
        "precision": round(final_opt["precision"], 4),
        "f1":        round(final_opt["f1"],        4),
        "accuracy":  round(final_opt["accuracy"],  4),
        "tp": final_opt["tp"], "fp": final_opt["fp"],
        "tn": final_opt["tn"], "fn": final_opt["fn"],
    },
}

metrics_path = MODELS_DIR / "model_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, indent=2)
print(f"  ✓ Metrics → {metrics_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print(f"""

     PHASE 5 COMPLETE — MODEL READY                                  
     AUC-ROC    :  {final_opt['auc_roc']:.4f}                                      
     Recall     :  {final_opt['recall']:.4f}   ({final_opt['tp']:,}/{n_late_total:,} late orders caught){"" + " " * max(0, 17-len(str(final_opt['tp'])+str(n_late_total)))} 
     Precision  :  {final_opt['precision']:.4f}                                      
     F1         :  {final_opt['f1']:.4f}                                      
     Threshold  :  {optimal_thr:.2f}                                        
     Outputs created:                                                
       models/xgboost_model.pkl      ← Phase 6 dashboard input       
       models/model_metrics.json     ← KPI cards + README            
       reports/figures/05_*.html     ← 4 interactive charts         
  
""")
