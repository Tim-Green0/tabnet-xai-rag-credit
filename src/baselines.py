"""Day 2 — 베이스라인 모델 학습 + 평가.

대상 모델:
  1) Logistic Regression  (scaled 입력, class_weight='balanced')
  2) XGBoost              (unscaled 입력, scale_pos_weight)
  3) LightGBM             (unscaled 입력, class_weight='balanced')

산출물:
  - results/baseline_models/{logistic,xgboost,lightgbm}.pkl
  - results/baseline_metrics.csv      (모델 × split × 지표)
  - results/baseline_summary.json     (전체 요약 + 학습 시간 + threshold)
  - figures/08_roc_pr_curves.png      (ROC + PR, test 기준)
  - figures/09_threshold_sweep.png    (threshold별 F1 비교)
  - figures/10_feature_importance.png (Logistic |coef| / GBM gain top 20)

실행:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.baselines [--nrows N]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.metrics import compute_metrics, find_threshold_youden, metrics_table_row
from src.utils import FIGURES_DIR, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = RESULTS_DIR / "baseline_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "TARGET"


# ─────────────────────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────────────────────
def load_processed() -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """전처리된 parquet 6개를 모두 로드.

    반환:
      {
        'scaled':   (X_tr, y_tr, X_val, y_val, X_te, y_te),
        'unscaled': (X_tr, y_tr, X_val, y_val, X_te, y_te),
      }
    """
    out = {}
    for tag in ("scaled", "unscaled"):
        d = {}
        for split in ("train", "val", "test"):
            df = pd.read_parquet(PROCESSED_DIR / f"{split}_{tag}.parquet")
            d[f"X_{split}"] = df.drop(columns=[TARGET_COL])
            d[f"y_{split}"] = df[TARGET_COL].astype(int).values
        out[tag] = d
    return out


def maybe_subset(d: dict, nrows: int | None) -> dict:
    """dry-run용으로 train만 nrows로 잘라냄. val/test는 그대로."""
    if nrows is None:
        return d
    out = dict(d)
    out["X_train"] = d["X_train"].iloc[:nrows]
    out["y_train"] = d["y_train"][:nrows]
    return out


# ─────────────────────────────────────────────────────────────
# 모델별 학습 함수
# ─────────────────────────────────────────────────────────────
def train_logistic(d: dict) -> Tuple[LogisticRegression, np.ndarray, np.ndarray, float]:
    print("  > Logistic Regression 학습 중...")
    t0 = time.time()
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(d["X_train"], d["y_train"])
    elapsed = time.time() - t0
    val_score = clf.predict_proba(d["X_val"])[:, 1]
    test_score = clf.predict_proba(d["X_test"])[:, 1]
    return clf, val_score, test_score, elapsed


def train_xgboost(d: dict) -> Tuple[xgb.XGBClassifier, np.ndarray, np.ndarray, float]:
    print("  > XGBoost 학습 중...")
    t0 = time.time()
    pos_w = float((d["y_train"] == 0).sum() / max((d["y_train"] == 1).sum(), 1))
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=pos_w,
        random_state=SEED,
        eval_metric="auc",
        early_stopping_rounds=30,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(
        d["X_train"], d["y_train"],
        eval_set=[(d["X_val"], d["y_val"])],
        verbose=False,
    )
    elapsed = time.time() - t0
    val_score = clf.predict_proba(d["X_val"])[:, 1]
    test_score = clf.predict_proba(d["X_test"])[:, 1]
    return clf, val_score, test_score, elapsed


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """LightGBM이 거부하는 콤마, 콜론 등을 언더스코어로 치환."""
    import re
    new_cols = [re.sub(r'[",:{}\[\]\s]+', "_", c).strip("_") for c in df.columns]
    out = df.copy()
    out.columns = new_cols
    return out


def train_lightgbm(d: dict) -> Tuple[lgb.LGBMClassifier, np.ndarray, np.ndarray, float]:
    print("  > LightGBM 학습 중...")
    t0 = time.time()
    # LGBM은 'feature name에 특수문자 금지' 제약이 있어 별도 sanitize
    d = {**d,
         "X_train": _sanitize_columns(d["X_train"]),
         "X_val":   _sanitize_columns(d["X_val"]),
         "X_test":  _sanitize_columns(d["X_test"])}
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=-1,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        d["X_train"], d["y_train"],
        eval_set=[(d["X_val"], d["y_val"])],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    elapsed = time.time() - t0
    val_score = clf.predict_proba(d["X_val"])[:, 1]
    test_score = clf.predict_proba(d["X_test"])[:, 1]
    return clf, val_score, test_score, elapsed


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────
def plot_roc_pr(test_scores: Dict[str, np.ndarray], y_test: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for name, score in test_scores.items():
        fpr, tpr, _ = roc_curve(y_test, score)
        prec, rec, _ = precision_recall_curve(y_test, score)
        auroc = roc_auc_score(y_test, score)
        auprc = average_precision_score(y_test, score)
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC={auroc:.4f})")
        axes[1].plot(rec, prec, lw=2, label=f"{name} (AP={auprc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve (Test)")
    axes[0].legend(loc="lower right")

    base = float(y_test.mean())
    axes[1].axhline(base, color="k", linestyle="--", alpha=0.4, label=f"base rate={base:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (Test)")
    axes[1].legend(loc="upper right")

    plt.suptitle("베이스라인 비교 (Test set)")
    return savefig(fig, "08_roc_pr_curves")


def plot_threshold_sweep(test_scores: Dict[str, np.ndarray], y_test: np.ndarray,
                          chosen_thr: Dict[str, float]) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    thrs = np.linspace(0.02, 0.98, 100)
    for name, score in test_scores.items():
        f1s = [f1_score(y_test, (score >= t).astype(int), zero_division=0) for t in thrs]
        line, = ax.plot(thrs, f1s, lw=2, label=name)
        # validation에서 정한 threshold 위치 표시
        if name in chosen_thr:
            t = chosen_thr[name]
            f = f1_score(y_test, (score >= t).astype(int), zero_division=0)
            ax.scatter([t], [f], color=line.get_color(), s=80, zorder=5,
                       edgecolor="black", label=f"{name} val-thr={t:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 (Test)")
    ax.set_title("임계치 sweep — validation에서 선택한 threshold(점)을 test에 적용")
    ax.legend(loc="best", fontsize=9)
    return savefig(fig, "09_threshold_sweep")


def plot_feature_importance(
    log_clf: LogisticRegression,
    xgb_clf: xgb.XGBClassifier,
    lgb_clf: lgb.LGBMClassifier,
    feature_names_scaled: list,
    feature_names_unscaled: list,
    top: int = 20,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Logistic |coef|
    coef = np.abs(log_clf.coef_[0])
    idx = np.argsort(coef)[-top:]
    axes[0].barh([feature_names_scaled[i] for i in idx], coef[idx], color="#4C72B0")
    axes[0].set_title(f"Logistic |coef| Top {top}")
    axes[0].set_xlabel("|coefficient|")

    # XGBoost gain
    booster = xgb_clf.get_booster()
    score_dict = booster.get_score(importance_type="gain")
    # XGB는 'f0','f1',... 또는 컬럼명. sklearn API라 컬럼명일 가능성.
    xgb_imp = pd.Series(score_dict).reindex(feature_names_unscaled).fillna(0)
    idx = np.argsort(xgb_imp.values)[-top:]
    axes[1].barh([xgb_imp.index[i] for i in idx], xgb_imp.values[idx], color="#DD8452")
    axes[1].set_title(f"XGBoost Gain Top {top}")
    axes[1].set_xlabel("gain")

    # LightGBM gain — LGBM 내부 sanitized 컬럼명 사용
    lgb_imp = pd.Series(
        lgb_clf.booster_.feature_importance(importance_type="gain"),
        index=lgb_clf.booster_.feature_name(),
    )
    idx = np.argsort(lgb_imp.values)[-top:]
    axes[2].barh([lgb_imp.index[i] for i in idx], lgb_imp.values[idx], color="#55A868")
    axes[2].set_title(f"LightGBM Gain Top {top}")
    axes[2].set_xlabel("gain")

    plt.suptitle("모델별 변수 중요도 (Top 20) — Day 4 SHAP과 비교 예정")
    return savefig(fig, "10_feature_importance")


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main(nrows: int | None = None) -> None:
    set_seed(SEED)
    print(f"[1/5] 전처리된 데이터 로딩 (nrows={nrows or 'all'})")
    data_all = load_processed()
    sc = maybe_subset(data_all["scaled"], nrows)
    un = maybe_subset(data_all["unscaled"], nrows)
    print(f"     scaled  : train={sc['X_train'].shape}, val={sc['X_val'].shape}, test={sc['X_test'].shape}")
    print(f"     unscaled: train={un['X_train'].shape}, val={un['X_val'].shape}, test={un['X_test'].shape}")
    feat_sc = sc["X_train"].columns.tolist()
    feat_un = un["X_train"].columns.tolist()

    rows = []
    val_scores: Dict[str, np.ndarray] = {}
    test_scores: Dict[str, np.ndarray] = {}
    chosen_thr: Dict[str, float] = {}

    # ──────── Logistic ────────
    print("\n[2/5] Logistic Regression (scaled)")
    log_clf, vs, ts, t = train_logistic(sc)
    joblib.dump(log_clf, MODELS_DIR / "logistic.pkl")
    val_scores["Logistic"] = vs; test_scores["Logistic"] = ts
    thr, _ = find_threshold_youden(sc["y_val"], vs)
    chosen_thr["Logistic"] = thr
    rows.append(metrics_table_row("Logistic", "val", t, compute_metrics(sc["y_val"], vs, thr)))
    rows.append(metrics_table_row("Logistic", "test", t, compute_metrics(sc["y_test"], ts, thr)))
    print(f"     elapsed={t:.1f}s, val AUROC={rows[-2]['auroc']:.4f}, test AUROC={rows[-1]['auroc']:.4f}")

    # ──────── XGBoost ────────
    print("\n[3/5] XGBoost (unscaled)")
    xgb_clf, vs, ts, t = train_xgboost(un)
    joblib.dump(xgb_clf, MODELS_DIR / "xgboost.pkl")
    val_scores["XGBoost"] = vs; test_scores["XGBoost"] = ts
    thr, _ = find_threshold_youden(un["y_val"], vs)
    chosen_thr["XGBoost"] = thr
    rows.append(metrics_table_row("XGBoost", "val", t, compute_metrics(un["y_val"], vs, thr)))
    rows.append(metrics_table_row("XGBoost", "test", t, compute_metrics(un["y_test"], ts, thr)))
    print(f"     elapsed={t:.1f}s, val AUROC={rows[-2]['auroc']:.4f}, test AUROC={rows[-1]['auroc']:.4f}")
    print(f"     best_iteration={xgb_clf.best_iteration if hasattr(xgb_clf, 'best_iteration') else 'n/a'}")

    # ──────── LightGBM ────────
    print("\n[4/5] LightGBM (unscaled)")
    lgb_clf, vs, ts, t = train_lightgbm(un)
    joblib.dump(lgb_clf, MODELS_DIR / "lightgbm.pkl")
    val_scores["LightGBM"] = vs; test_scores["LightGBM"] = ts
    thr, _ = find_threshold_youden(un["y_val"], vs)
    chosen_thr["LightGBM"] = thr
    rows.append(metrics_table_row("LightGBM", "val", t, compute_metrics(un["y_val"], vs, thr)))
    rows.append(metrics_table_row("LightGBM", "test", t, compute_metrics(un["y_test"], ts, thr)))
    print(f"     elapsed={t:.1f}s, val AUROC={rows[-2]['auroc']:.4f}, test AUROC={rows[-1]['auroc']:.4f}")
    print(f"     best_iteration={lgb_clf.best_iteration_ if hasattr(lgb_clf, 'best_iteration_') else 'n/a'}")

    # ──────── 결과 저장 ────────
    print("\n[5/5] 결과 저장 + 시각화")
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)

    # 보기 좋은 형태로 한 번 더 print
    pivot = metrics_df.pivot_table(
        index="model", columns="split",
        values=["auroc", "auprc", "ks", "f1", "threshold", "time_sec"],
    )
    print("\n[메트릭 요약]")
    print(pivot.round(4).to_string())

    fig_roc = plot_roc_pr(test_scores, sc["y_test"])
    fig_thr = plot_threshold_sweep(test_scores, sc["y_test"], chosen_thr)
    fig_imp = plot_feature_importance(log_clf, xgb_clf, lgb_clf, feat_sc, feat_un)

    summary = {
        "config": {
            "seed": SEED,
            "models": {
                "Logistic": {
                    "input": "scaled", "class_weight": "balanced",
                    "solver": "lbfgs", "max_iter": 2000,
                },
                "XGBoost": {
                    "input": "unscaled", "n_estimators": 500, "max_depth": 6,
                    "learning_rate": 0.05, "scale_pos_weight": "auto",
                    "early_stopping_rounds": 30, "tree_method": "hist",
                },
                "LightGBM": {
                    "input": "unscaled", "n_estimators": 500, "num_leaves": 63,
                    "learning_rate": 0.05, "class_weight": "balanced",
                    "early_stopping": 30,
                },
            },
            "threshold_strategy": "Youden's J on validation, applied to test",
        },
        "results": rows,
        "chosen_thresholds": chosen_thr,
        "best_iterations": {
            "XGBoost": int(getattr(xgb_clf, "best_iteration", -1)) if hasattr(xgb_clf, "best_iteration") else None,
            "LightGBM": int(getattr(lgb_clf, "best_iteration_", -1)) if hasattr(lgb_clf, "best_iteration_") else None,
        },
        "files": {
            "metrics_csv": str(RESULTS_DIR / "baseline_metrics.csv"),
            "roc_pr": str(fig_roc),
            "threshold_sweep": str(fig_thr),
            "feature_importance": str(fig_imp),
            "models": {
                "Logistic": str(MODELS_DIR / "logistic.pkl"),
                "XGBoost": str(MODELS_DIR / "xgboost.pkl"),
                "LightGBM": str(MODELS_DIR / "lightgbm.pkl"),
            },
        },
    }
    with open(RESULTS_DIR / "baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n[OK] Day 2 베이스라인 완료")
    print(f"     - 모델 가중치: {MODELS_DIR}")
    print(f"     - 메트릭 CSV : results/baseline_metrics.csv")
    print(f"     - 요약 JSON  : results/baseline_summary.json")
    print(f"     - figures    : 08, 09, 10")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None,
                    help="dry-run: train만 일부 행으로 (val/test는 그대로)")
    args = ap.parse_args()
    main(nrows=args.nrows)
