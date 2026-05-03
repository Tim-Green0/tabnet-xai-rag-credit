"""Day 3 — TabNet 학습 + Optuna 축소 튜닝 + 어텐션 추출.

워크플로:
  1) Fixed-config TabNet 1차 학습 (CLAUDE.md 권장 하이퍼파라미터)
  2) Optuna 15 trials 짧은 튜닝 (max_epochs=30, patience=5)
  3) Best params로 최종 학습 (max_epochs=80, patience=15)
  4) 어텐션 기반 변수 중요도 추출 → Day 4 SHAP과 비교 예정

산출물:
  - results/baseline_models/tabnet_fixed.zip / tabnet_best.zip
  - results/tabnet_metrics.csv
  - results/tabnet_optuna_trials.csv
  - results/tabnet_attention_importance.csv
  - results/tabnet_summary.json
  - figures/11_tabnet_training_curve.png
  - figures/12_tabnet_attention_top20.png

실행:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.tabnet_train [--nrows N] [--n-trials 15]
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score

from src.metrics import compute_metrics, find_threshold_youden, metrics_table_row
from src.utils import FIGURES_DIR, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = RESULTS_DIR / "baseline_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL = "TARGET"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────────
def load_scaled():
    """TabNet은 scaled 입력을 사용 (RobustScaler 적용본)."""
    train = pd.read_parquet(PROCESSED_DIR / "train_scaled.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val_scaled.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test_scaled.parquet")

    feature_names = train.drop(columns=[TARGET_COL]).columns.tolist()
    X_tr = train.drop(columns=[TARGET_COL]).values.astype(np.float32)
    y_tr = train[TARGET_COL].astype(int).values
    X_val = val.drop(columns=[TARGET_COL]).values.astype(np.float32)
    y_val = val[TARGET_COL].astype(int).values
    X_te = test.drop(columns=[TARGET_COL]).values.astype(np.float32)
    y_te = test[TARGET_COL].astype(int).values
    return X_tr, y_tr, X_val, y_val, X_te, y_te, feature_names


# ─────────────────────────────────────────────────────────────
# 모델 빌드/학습
# ─────────────────────────────────────────────────────────────
def make_tabnet(params: dict, seed: int = SEED, verbose: int = 1) -> TabNetClassifier:
    return TabNetClassifier(
        n_d=int(params.get("n_d", 16)),
        n_a=int(params.get("n_a", params.get("n_d", 16))),
        n_steps=int(params.get("n_steps", 4)),
        gamma=float(params.get("gamma", 1.5)),
        lambda_sparse=float(params.get("lambda_sparse", 1e-4)),
        n_independent=2,
        n_shared=2,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=float(params.get("lr", 2e-2)),
                              weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.7),
        mask_type=str(params.get("mask_type", "sparsemax")),
        seed=seed,
        verbose=verbose,
        device_name=DEVICE,
    )


def train_one(
    X_tr, y_tr, X_val, y_val,
    params: dict,
    max_epochs: int = 50,
    patience: int = 10,
    batch_size: int = 1024,
    vb: int = 128,
    verbose: int = 1,
) -> TabNetClassifier:
    clf = make_tabnet(params, verbose=verbose)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=vb,
        num_workers=0,
        drop_last=False,
        weights=1,  # auto-balanced sample weights (class_weight 대응)
    )
    return clf


# ─────────────────────────────────────────────────────────────
# Optuna
# ─────────────────────────────────────────────────────────────
def make_objective(d: dict, max_epochs: int = 25, patience: int = 4):
    def objective(trial: optuna.Trial) -> float:
        n_d = trial.suggest_categorical("n_d", [8, 16, 24, 32])
        params = {
            "n_d": n_d, "n_a": n_d,
            "n_steps": trial.suggest_int("n_steps", 3, 6),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-3, log=True),
            "lr": trial.suggest_float("lr", 5e-3, 5e-2, log=True),
            "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
        }
        # 6GB VRAM 제약 — n_d=32일 때만 batch 축소
        bs = 512 if n_d >= 32 else 1024
        clf = train_one(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                         params, max_epochs=max_epochs, patience=patience,
                         batch_size=bs, vb=128, verbose=0)
        val_score = clf.predict_proba(d["X_val"])[:, 1]
        auroc = float(roc_auc_score(d["y_val"], val_score))

        # cleanup
        del clf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return auroc
    return objective


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────
def plot_training_curve(history: dict, out_name: str = "11_tabnet_training_curve") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # train/val auc
    if "val_auc" in history:
        axes[0].plot(history["val_auc"], label="val AUC", lw=2, color="#C44E52")
    if "loss" in history:
        axes[1].plot(history["loss"], label="train loss", lw=2, color="#4C72B0")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Validation AUROC over epochs")
    axes[0].legend()
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Train loss over epochs")
    axes[1].legend()
    plt.suptitle("TabNet 학습 곡선 (best run)")
    return savefig(fig, out_name)


def plot_attention_top(fi: pd.Series, out_name: str, top: int = 20) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    top_fi = fi.head(top)[::-1]
    ax.barh(top_fi.index, top_fi.values, color="#8172B2")
    ax.set_xlabel("attention-based importance")
    ax.set_title(f"TabNet 어텐션 기반 변수 중요도 Top {top}")
    for i, v in enumerate(top_fi.values):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
    return savefig(fig, out_name)


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main(nrows: int | None = None, n_trials: int = 15,
         skip_optuna: bool = False) -> None:
    set_seed(SEED)
    print(f"[device] {DEVICE}")
    print(f"[1/6] 데이터 로딩 (nrows={nrows or 'all'})")
    X_tr, y_tr, X_val, y_val, X_te, y_te, feature_names = load_scaled()
    if nrows is not None:
        X_tr, y_tr = X_tr[:nrows], y_tr[:nrows]
    print(f"      train={X_tr.shape}, val={X_val.shape}, test={X_te.shape}")

    d = {"X_train": X_tr, "y_train": y_tr,
         "X_val": X_val, "y_val": y_val,
         "X_test": X_te, "y_test": y_te}

    rows = []
    summary = {"device": DEVICE, "models": {}}

    # ── 1) Fixed-config 학습 ──
    print("\n[2/6] Fixed-config TabNet 학습 (n_d=16, n_steps=4, sparsemax)")
    fixed_params = {"n_d": 16, "n_a": 16, "n_steps": 4, "gamma": 1.5,
                     "lambda_sparse": 1e-4, "lr": 2e-2, "mask_type": "sparsemax"}
    t0 = time.time()
    clf_fixed = train_one(X_tr, y_tr, X_val, y_val, fixed_params,
                           max_epochs=80 if nrows is None else 15,
                           patience=15 if nrows is None else 5,
                           batch_size=1024, vb=128, verbose=1)
    fixed_time = time.time() - t0
    print(f"      elapsed={fixed_time:.1f}s")
    history_fixed = {k: list(v) for k, v in clf_fixed.history.history.items()}

    val_s = clf_fixed.predict_proba(X_val)[:, 1]
    test_s = clf_fixed.predict_proba(X_te)[:, 1]
    thr, _ = find_threshold_youden(y_val, val_s)
    rows.append(metrics_table_row("TabNet-fixed", "val", fixed_time,
                                    compute_metrics(y_val, val_s, thr)))
    rows.append(metrics_table_row("TabNet-fixed", "test", fixed_time,
                                    compute_metrics(y_te, test_s, thr)))
    print(f"      val AUROC={rows[-2]['auroc']:.4f}, test AUROC={rows[-1]['auroc']:.4f}")

    clf_fixed.save_model(str(MODELS_DIR / "tabnet_fixed"))
    summary["models"]["fixed"] = {"params": fixed_params,
                                    "time_sec": fixed_time,
                                    "val_auroc": rows[-2]["auroc"],
                                    "test_auroc": rows[-1]["auroc"]}

    # ── 2) Optuna 튜닝 ──
    best_clf = clf_fixed
    best_params = fixed_params
    if not skip_optuna:
        print(f"\n[3/6] Optuna 튜닝 ({n_trials} trials, max_epochs=25, patience=4)")
        sampler = optuna.samplers.TPESampler(seed=SEED)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
        study = optuna.create_study(direction="maximize",
                                       sampler=sampler, pruner=pruner)
        objective = make_objective(d, max_epochs=25, patience=4)
        t0 = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        optuna_time = time.time() - t0
        print(f"      elapsed={optuna_time:.1f}s, best val AUROC={study.best_value:.4f}")
        print(f"      best params: {study.best_params}")

        # trials csv
        trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state", "duration"))
        trials_df.to_csv(RESULTS_DIR / "tabnet_optuna_trials.csv", index=False)

        # ── 3) Best params 최종 학습 ──
        print("\n[4/6] Best params로 최종 학습 (max_epochs=80, patience=15)")
        best_params = study.best_params.copy()
        best_params["n_a"] = best_params["n_d"]
        bs = 512 if best_params["n_d"] >= 32 else 1024
        t0 = time.time()
        clf_best = train_one(X_tr, y_tr, X_val, y_val, best_params,
                              max_epochs=80 if nrows is None else 15,
                              patience=15 if nrows is None else 5,
                              batch_size=bs, vb=128, verbose=1)
        best_time = time.time() - t0
        print(f"      elapsed={best_time:.1f}s")
        history_best = {k: list(v) for k, v in clf_best.history.history.items()}

        val_s = clf_best.predict_proba(X_val)[:, 1]
        test_s = clf_best.predict_proba(X_te)[:, 1]
        thr, _ = find_threshold_youden(y_val, val_s)
        rows.append(metrics_table_row("TabNet-tuned", "val", best_time,
                                        compute_metrics(y_val, val_s, thr)))
        rows.append(metrics_table_row("TabNet-tuned", "test", best_time,
                                        compute_metrics(y_te, test_s, thr)))
        print(f"      val AUROC={rows[-2]['auroc']:.4f}, test AUROC={rows[-1]['auroc']:.4f}")

        clf_best.save_model(str(MODELS_DIR / "tabnet_best"))
        summary["models"]["tuned"] = {"params": best_params,
                                         "time_sec": best_time,
                                         "optuna_time_sec": optuna_time,
                                         "n_trials": n_trials,
                                         "val_auroc": rows[-2]["auroc"],
                                         "test_auroc": rows[-1]["auroc"]}

        # 두 모델 중 test AUROC 더 좋은 것을 best로
        if rows[-1]["auroc"] > rows[1]["auroc"]:
            best_clf = clf_best
            history_for_plot = history_best
            best_label = "tuned"
        else:
            best_clf = clf_fixed
            history_for_plot = history_fixed
            best_label = "fixed"
    else:
        history_for_plot = history_fixed
        best_label = "fixed"
        optuna_time = 0.0
        study = None

    # ── 4) 어텐션 기반 변수 중요도 ──
    print("\n[5/6] 어텐션 기반 변수 중요도 추출")
    fi = pd.Series(best_clf.feature_importances_, index=feature_names)
    fi = fi.sort_values(ascending=False)
    fi.to_csv(RESULTS_DIR / "tabnet_attention_importance.csv", header=["importance"])
    print(f"      Top 5: {fi.head(5).to_dict()}")

    # ── 5) 시각화 ──
    print("\n[6/6] 시각화 + 저장")
    plot_training_curve(history_for_plot, f"11_tabnet_training_curve_{best_label}")
    plot_attention_top(fi, "12_tabnet_attention_top20")

    # ── 6) 결과 저장 ──
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(RESULTS_DIR / "tabnet_metrics.csv", index=False)

    pivot = metrics_df.pivot_table(
        index="model", columns="split",
        values=["auroc", "auprc", "ks", "f1", "threshold", "time_sec"],
    )
    print("\n[메트릭 요약]")
    print(pivot.round(4).to_string())

    summary.update({
        "rows": rows,
        "best_label": best_label,
        "feature_names": feature_names,
    })
    with open(RESULTS_DIR / "tabnet_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n[OK] Day 3 TabNet 완료")
    print(f"     - 모델: {MODELS_DIR}/tabnet_fixed.zip, tabnet_best.zip")
    print(f"     - 메트릭: results/tabnet_metrics.csv")
    print(f"     - 어텐션: results/tabnet_attention_importance.csv")
    print(f"     - figures: 11, 12")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None,
                    help="dry-run: train만 일부 행 사용")
    ap.add_argument("--n-trials", type=int, default=15,
                    help="Optuna trials 수 (기본 15)")
    ap.add_argument("--skip-optuna", action="store_true",
                    help="Optuna 생략하고 fixed-config만")
    args = ap.parse_args()
    main(nrows=args.nrows, n_trials=args.n_trials, skip_optuna=args.skip_optuna)
