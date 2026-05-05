"""Step 3-B-5: aux feature 추가 후 5-fold CV 재평가.

Step 1의 cv_metrics (baseline)와 비교하여 보조 테이블 임팩트를 정량화.
- 입력: data/processed/{train,val,test}_unscaled_aux.parquet
- 모델: XGBoost (메인 비교), LightGBM (보조)
  Logistic/TabNet은 715 feature → 학습 시간 부담 큼. 미팅 메시지엔 XGBoost가 핵심이라 OK.

산출:
    results/cv_metrics_aux.csv   — fold별 raw
    results/cv_summary_aux.csv   — mean ± std
    results/cv_aux_vs_baseline.csv — Step 1 baseline 대비 차이
    figures/24_cv_aux_comparison.png

실행:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.cv_eval_aux
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.cv_eval import (
    aggregate,
    cv_evaluate,
    fit_lightgbm,
    fit_xgboost,
)
from src.utils import FIGURES_DIR, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

PROCESSED_DIR = Path("data/processed")
TARGET_COL = "TARGET"


def load_for_cv_aux(scaled: bool):
    tag = "scaled" if scaled else "unscaled"
    train = pd.read_parquet(PROCESSED_DIR / f"train_{tag}_aux.parquet")
    val = pd.read_parquet(PROCESSED_DIR / f"val_{tag}_aux.parquet")
    test = pd.read_parquet(PROCESSED_DIR / f"test_{tag}_aux.parquet")
    df_all = pd.concat([train, val], ignore_index=True)
    y_all = df_all[TARGET_COL].astype(int).values
    X_all = df_all.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL].astype(int).values
    X_test = test.drop(columns=[TARGET_COL])
    return X_all, y_all, X_test, y_test


def compare_with_baseline(aux_summary: pd.DataFrame) -> pd.DataFrame:
    """Step 1의 cv_summary.csv 와 aux 결과를 같은 행에 나란히 정리."""
    base_path = RESULTS_DIR / "cv_summary.csv"
    if not base_path.exists():
        print(f"     [warn] baseline {base_path} 없음 - 비교 스킵")
        return pd.DataFrame()
    base = pd.read_csv(base_path)
    base = base[base["split"] == "test"][["model", "metric", "mean", "std"]].rename(
        columns={"mean": "mean_baseline", "std": "std_baseline"})
    aux_t = aux_summary[aux_summary["split"] == "test"][["model", "metric", "mean", "std"]].rename(
        columns={"mean": "mean_aux", "std": "std_aux"})
    df = aux_t.merge(base, on=["model", "metric"], how="left")
    df["delta"] = df["mean_aux"] - df["mean_baseline"]
    df["delta_pct"] = df["delta"] / df["mean_baseline"] * 100
    return df


def plot_aux_comparison(summary: pd.DataFrame, baseline_df: pd.DataFrame,
                         out_name: str = "27_cv_aux_comparison") -> Path:
    """baseline vs aux 비교 (test AUROC)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics = ["auroc", "auprc", "ks", "f1"]
    for ax, m in zip(axes, metrics):
        rows = []
        for _, r in baseline_df[baseline_df["metric"] == m].iterrows():
            rows.append({"model": r["model"], "set": "Baseline (Step 1)",
                          "mean": r["mean_baseline"], "std": r["std_baseline"]})
            rows.append({"model": r["model"], "set": "+ Aux features (Step 3-B)",
                          "mean": r["mean_aux"], "std": r["std_aux"]})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        sns.barplot(data=df, x="model", y="mean", hue="set", ax=ax,
                     errorbar=None, palette=["#A0A0A0", "#DD8452"])
        # error bars
        for i, (mod, st_) in enumerate(zip(df["model"], df["set"])):
            sub = df[(df["model"] == mod) & (df["set"] == st_)].iloc[0]
            x = i // 2 + (-0.2 if "Baseline" in st_ else 0.2)
            ax.errorbar(x, sub["mean"], yerr=sub["std"], fmt="none",
                          ecolor="black", capsize=4)
        ax.set_title(f"{m.upper()} (test, 5-fold mean ± std)")
        ax.set_ylabel(m.upper())
        ax.set_xlabel("")
        if m != "auroc":
            ax.legend_.remove()
    plt.suptitle("Step 3-B: Auxiliary tables impact (5-fold CV, test set)")
    return savefig(fig, out_name)


def main(only_xgb: bool = False) -> None:
    set_seed(SEED)
    print("[1/3] aux 데이터 로드 (unscaled)")
    X_un, y_un, X_test_un, y_test_un = load_for_cv_aux(scaled=False)
    print(f"     trainval={X_un.shape}, test={X_test_un.shape}")

    all_rows = []

    print("\n[2/3] CV 실행")
    rows_xgb, _ = cv_evaluate("XGBoost", X_un, y_un, X_test_un, y_test_un,
                                fit_xgboost, n_splits=5)
    all_rows.append(rows_xgb)

    if not only_xgb:
        rows_lgb, _ = cv_evaluate("LightGBM", X_un, y_un, X_test_un, y_test_un,
                                    fit_lightgbm, n_splits=5)
        all_rows.append(rows_lgb)

    raw = pd.concat(all_rows, ignore_index=True)
    raw.to_csv(RESULTS_DIR / "cv_metrics_aux.csv", index=False)
    print(f"\n[3/3] 집계 + 비교")
    summary = aggregate(raw)
    summary.to_csv(RESULTS_DIR / "cv_summary_aux.csv", index=False)
    print(summary[summary["split"] == "test"].to_string(index=False))

    cmp = compare_with_baseline(summary)
    if not cmp.empty:
        cmp.to_csv(RESULTS_DIR / "cv_aux_vs_baseline.csv", index=False)
        print("\n[delta] aux - baseline (test set):")
        print(cmp[["model", "metric", "mean_baseline", "mean_aux", "delta", "delta_pct"]
               ].to_string(index=False))
        plot_aux_comparison(summary, cmp)
        print(f"     figure saved.")

    print("\n[OK] aux CV 완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-xgb", action="store_true", help="LightGBM 스킵 (시간 절약)")
    args = ap.parse_args()
    main(only_xgb=args.only_xgb)
