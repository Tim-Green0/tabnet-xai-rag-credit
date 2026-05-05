"""5-fold Stratified CV 재평가.

목적: 단발 학습이 아닌 5-fold mean ± std로 안정성 확인.
방식:
  - train + val parquet 합쳐 246K rows → 5-fold
  - test parquet 61.5K는 hold-out (안 건드림)
  - 매 fold: 4/5 학습, 1/5 validation → val에서 threshold(Youden) 결정 → test에서 평가
  - 5번 반복 후 test 메트릭의 mean ± std 보고

산출물:
  - results/cv_metrics.csv      (모델 × fold × split × 지표 raw)
  - results/cv_summary.csv      (모델 × split, mean ± std)
  - figures/13_cv_comparison.png

실행: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.cv_eval [--skip-tabnet]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path
from typing import Callable, Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.metrics import compute_metrics, find_threshold_youden, metrics_table_row
from src.utils import FIGURES_DIR, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

PROCESSED_DIR = Path("data/processed")
TARGET_COL = "TARGET"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────────
def load_for_cv(scaled: bool):
    tag = "scaled" if scaled else "unscaled"
    train = pd.read_parquet(PROCESSED_DIR / f"train_{tag}.parquet")
    val = pd.read_parquet(PROCESSED_DIR / f"val_{tag}.parquet")
    test = pd.read_parquet(PROCESSED_DIR / f"test_{tag}.parquet")
    df_all = pd.concat([train, val], ignore_index=True)
    y_all = df_all[TARGET_COL].astype(int).values
    X_all = df_all.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL].astype(int).values
    X_test = test.drop(columns=[TARGET_COL])
    return X_all, y_all, X_test, y_test


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """LightGBM이 거부하는 콤마/콜론 등을 _로 치환."""
    new_cols = [re.sub(r'[",:{}\[\]\s]+', "_", c).strip("_") for c in df.columns]
    out = df.copy()
    out.columns = new_cols
    return out


# ─────────────────────────────────────────────────────────────
# 모델별 fold 학습 (baselines.py의 패턴 재사용, 압축)
# ─────────────────────────────────────────────────────────────
def fit_logistic(X_tr, y_tr, X_val, y_val, X_test):
    t0 = time.time()
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                              solver="lbfgs", n_jobs=-1, random_state=SEED)
    clf.fit(X_tr, y_tr)
    el = time.time() - t0
    return clf, clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1], el


def fit_xgboost(X_tr, y_tr, X_val, y_val, X_test):
    t0 = time.time()
    pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    clf = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             scale_pos_weight=pos_w, random_state=SEED,
                             eval_metric="auc", early_stopping_rounds=30,
                             n_jobs=-1, tree_method="hist")
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    el = time.time() - t0
    return clf, clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1], el


def fit_lightgbm(X_tr, y_tr, X_val, y_val, X_test):
    t0 = time.time()
    Xt, Xv, Xs = _sanitize(X_tr), _sanitize(X_val), _sanitize(X_test)
    clf = lgb.LGBMClassifier(n_estimators=500, max_depth=-1, num_leaves=63,
                              learning_rate=0.05, subsample=0.9,
                              colsample_bytree=0.9, class_weight="balanced",
                              random_state=SEED, n_jobs=-1, verbose=-1)
    clf.fit(Xt, y_tr, eval_set=[(Xv, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(0)])
    el = time.time() - t0
    return clf, clf.predict_proba(Xv)[:, 1], clf.predict_proba(Xs)[:, 1], el


# tabnet_train.py에서 가져온 best params (Day 3 Optuna 결과)
TABNET_BEST_PARAMS = {
    "n_d": 16, "n_a": 16, "n_steps": 3, "gamma": 1.1559945203362028,
    "lambda_sparse": 1.3066739238053272e-05, "lr": 0.03674059202635224,
    "mask_type": "entmax",
}


def fit_tabnet(X_tr, y_tr, X_val, y_val, X_test):
    t0 = time.time()
    clf = TabNetClassifier(
        n_d=TABNET_BEST_PARAMS["n_d"], n_a=TABNET_BEST_PARAMS["n_a"],
        n_steps=TABNET_BEST_PARAMS["n_steps"],
        gamma=TABNET_BEST_PARAMS["gamma"],
        lambda_sparse=TABNET_BEST_PARAMS["lambda_sparse"],
        n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=TABNET_BEST_PARAMS["lr"], weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.7),
        mask_type=TABNET_BEST_PARAMS["mask_type"],
        seed=SEED, verbose=0, device_name=DEVICE,
    )
    clf.fit(X_tr.values.astype(np.float32), y_tr,
            eval_set=[(X_val.values.astype(np.float32), y_val)],
            eval_name=["val"], eval_metric=["auc"],
            max_epochs=60, patience=12, batch_size=1024, virtual_batch_size=128,
            num_workers=0, drop_last=False, weights=1)
    el = time.time() - t0
    val_s = clf.predict_proba(X_val.values.astype(np.float32))[:, 1]
    test_s = clf.predict_proba(X_test.values.astype(np.float32))[:, 1]
    return clf, val_s, test_s, el


# ─────────────────────────────────────────────────────────────
# CV 루프
# ─────────────────────────────────────────────────────────────
def cv_evaluate(
    model_name: str,
    X_all: pd.DataFrame, y_all: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    fit_fn: Callable,
    n_splits: int = 5,
    seed: int = SEED,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    test_pred_per_fold = []
    print(f"\n=== CV: {model_name} ===")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all)):
        X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all.iloc[va_idx], y_all[va_idx]
        clf, val_s, test_s, t = fit_fn(X_tr, y_tr, X_va, y_va, X_test)
        thr, _ = find_threshold_youden(y_va, val_s)
        m_val = compute_metrics(y_va, val_s, thr)
        m_test = compute_metrics(y_test, test_s, thr)
        rows.append(metrics_table_row(model_name, f"fold{fold}_val", t, m_val))
        rows.append(metrics_table_row(model_name, f"fold{fold}_test", t, m_test))
        test_pred_per_fold.append(test_s)
        print(f"  fold{fold}: time={t:.1f}s, val_AUROC={m_val['auroc']:.4f}, test_AUROC={m_test['auroc']:.4f}, thr={thr:.3f}")
        # GPU cleanup
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows), np.array(test_pred_per_fold)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """fold별 → mean ± std."""
    rows = []
    for (model, split_kind), grp in df.assign(
            split_kind=df["split"].str.replace(r"fold\d+_", "", regex=True)
        ).groupby(["model", "split_kind"]):
        for col in ["auroc", "auprc", "ks", "f1", "precision", "recall", "accuracy", "time_sec"]:
            if col in grp.columns:
                rows.append({"model": model, "split": split_kind, "metric": col,
                              "mean": float(grp[col].mean()),
                              "std": float(grp[col].std()),
                              "n": int(len(grp))})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────
def plot_cv_comparison(summary: pd.DataFrame, out_name: str = "13_cv_comparison") -> Path:
    """test 메트릭 mean ± std 막대그래프."""
    test_summary = summary[summary["split"] == "test"]
    metrics = ["auroc", "auprc", "ks", "f1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    for ax, m in zip(axes, metrics):
        sub = test_summary[test_summary["metric"] == m].sort_values("mean", ascending=False)
        if len(sub) == 0:
            continue
        x = np.arange(len(sub))
        bars = ax.bar(x, sub["mean"], yerr=sub["std"], capsize=6,
                       color=["#DD8452", "#4C72B0", "#55A868", "#8172B2"][:len(sub)])
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"].tolist(), rotation=15, ha="right")
        ax.set_ylabel(m.upper())
        ax.set_title(f"{m.upper()} (test, mean ± std, 5-fold)")
        for i, (mn, sd) in enumerate(zip(sub["mean"].values, sub["std"].values)):
            ax.text(i, mn + sd + 0.005, f"{mn:.4f}\n±{sd:.4f}",
                    ha="center", va="bottom", fontsize=8)
    plt.suptitle("5-fold Stratified CV — model comparison (test set)")
    return savefig(fig, out_name)


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main(skip_tabnet: bool = False, only_tabnet: bool = False) -> None:
    set_seed(SEED)
    print(f"[device] {DEVICE}")

    # scaled 입력 (Logistic, TabNet) / unscaled 입력 (XGB, LGBM)
    X_sc, y_sc, X_test_sc, y_test_sc = load_for_cv(scaled=True)
    X_un, y_un, X_test_un, y_test_un = load_for_cv(scaled=False)
    print(f"trainval={X_sc.shape}, test={X_test_sc.shape}")
    assert (y_sc == y_un).all() and (y_test_sc == y_test_un).all()

    all_rows = []

    if only_tabnet:
        # 기존 baseline cv_metrics.csv 로딩 (있으면)
        prev_path = RESULTS_DIR / "cv_metrics.csv"
        if prev_path.exists():
            prev = pd.read_csv(prev_path)
            all_rows.append(prev)
            print(f"[append] 기존 cv_metrics.csv 로딩 ({len(prev)} rows)")
    else:
        # ── 1) Logistic (scaled) ──
        rows, _ = cv_evaluate("Logistic", X_sc, y_sc, X_test_sc, y_test_sc,
                                fit_logistic, n_splits=5)
        all_rows.append(rows)

        # ── 2) XGBoost (unscaled) ──
        rows, _ = cv_evaluate("XGBoost", X_un, y_un, X_test_un, y_test_un,
                                fit_xgboost, n_splits=5)
        all_rows.append(rows)

        # ── 3) LightGBM (unscaled) ──
        rows, _ = cv_evaluate("LightGBM", X_un, y_un, X_test_un, y_test_un,
                                fit_lightgbm, n_splits=5)
        all_rows.append(rows)

    # ── 4) TabNet (scaled, best params) ──
    if not skip_tabnet:
        rows, _ = cv_evaluate("TabNet", X_sc, y_sc, X_test_sc, y_test_sc,
                                fit_tabnet, n_splits=5)
        all_rows.append(rows)
    else:
        print("[skip] TabNet 5-fold skipped (--skip-tabnet)")

    raw = pd.concat(all_rows, ignore_index=True)
    raw.to_csv(RESULTS_DIR / "cv_metrics.csv", index=False)

    summary = aggregate(raw)
    summary.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)

    # 보기 좋게 출력
    print("\n[CV summary — test set]")
    test_only = summary[summary["split"] == "test"].pivot_table(
        index="model", columns="metric", values=["mean", "std"]
    )
    print(test_only.round(4).to_string())

    plot_cv_comparison(summary)
    print("\n[OK] 5-fold CV 완료")
    print("     - results/cv_metrics.csv (raw)")
    print("     - results/cv_summary.csv (mean/std)")
    print("     - figures/13_cv_comparison.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tabnet", action="store_true",
                    help="TabNet 5-fold 건너뛰기 (디버그용)")
    ap.add_argument("--only-tabnet", action="store_true",
                    help="기존 baseline cv_metrics.csv에 TabNet만 append")
    args = ap.parse_args()
    main(skip_tabnet=args.skip_tabnet, only_tabnet=args.only_tabnet)
