"""Day 1 EDA 스크립트.

application_train.csv 전체를 로드하여 다음 산출물을 생성:
  - figures/01_target_distribution.png
  - figures/02_missing_top30.png
  - figures/03_protected_attrs.png
  - figures/04_numeric_distributions.png
  - figures/05_correlation_top.png
  - figures/06_target_vs_features.png
  - figures/07_categorical_cardinality.png
  - results/eda_summary.json
  - results/eda_missing_table.csv
  - results/eda_target_corr.csv

실행: .venv/Scripts/python.exe -m src.eda
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_application_train, basic_info
from src.utils import FIGURES_DIR, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
set_seed(SEED)
sns.set_theme(style="whitegrid", context="notebook")

# 본 EDA에서 핵심으로 살피는 수치형 변수 (Home Credit 도메인 표준)
CORE_NUMERIC = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "REGION_POPULATION_RELATIVE",
]


def main() -> None:
    print("[1/8] 데이터 로딩...")
    df = load_application_train()
    info = basic_info(df)
    print(f"  shape = {df.shape}, memory = {info['memory_mb']:.1f} MB")

    summary: dict = {
        "shape": list(df.shape),
        "memory_mb": info["memory_mb"],
        "dtypes": {str(k): int(v) for k, v in info["dtypes"].items()},
        "target": {
            "n_pos": info["n_target_pos"],
            "n_neg": int(df.shape[0] - info["n_target_pos"]),
            "pos_rate": info["target_rate"],
        },
    }

    # ─────────────────────────────────────────
    # 1. TARGET 분포
    # ─────────────────────────────────────────
    print("[2/8] TARGET 분포...")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    counts = df["TARGET"].value_counts().sort_index()
    ax[0].bar(["0 (non-default)", "1 (default)"], counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, f"{v:,}\n({v/len(df)*100:.2f}%)", ha="center", va="bottom")
    ax[0].set_title("TARGET class counts")
    ax[0].set_ylabel("count")

    ax[1].pie(counts.values, labels=["non-default", "default"], autopct="%1.2f%%",
              colors=["#4C72B0", "#C44E52"], startangle=90)
    ax[1].set_title(f"Class ratio (total N={len(df):,})")
    plt.suptitle("Home Credit Default Risk — TARGET distribution (imbalanced)")
    savefig(fig, "01_target_distribution")

    # ─────────────────────────────────────────
    # 2. 결측률 분석
    # ─────────────────────────────────────────
    print("[3/8] 결측률 분석...")
    miss = df.isna().sum().sort_values(ascending=False)
    miss_pct = (miss / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"n_missing": miss, "pct_missing": miss_pct})
    miss_df = miss_df[miss_df["n_missing"] > 0]
    miss_df.to_csv(RESULTS_DIR / "eda_missing_table.csv")

    summary["missing"] = {
        "n_columns_with_missing": int(len(miss_df)),
        "n_columns_no_missing": int(df.shape[1] - len(miss_df)),
        "max_missing_pct": float(miss_df["pct_missing"].max()) if len(miss_df) else 0.0,
        "median_missing_pct": float(miss_df["pct_missing"].median()) if len(miss_df) else 0.0,
        "n_cols_above_50pct": int((miss_df["pct_missing"] > 50).sum()),
    }

    top30 = miss_df.head(30)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(top30.index[::-1], top30["pct_missing"][::-1].values, color="#C44E52")
    ax.set_xlabel("Missing rate (%)")
    ax.set_title("Top 30 columns by missing rate")
    for i, v in enumerate(top30["pct_missing"][::-1].values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    savefig(fig, "02_missing_top30")

    # ─────────────────────────────────────────
    # 3. 보호 속성 (성별, 연령) 분포
    # ─────────────────────────────────────────
    print("[4/8] 보호 속성 분포...")
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # 3-1 성별 분포
    g_counts = df["CODE_GENDER"].value_counts()
    ax[0, 0].bar(g_counts.index, g_counts.values, color=["#4C72B0", "#DD8452", "#55A868"][: len(g_counts)])
    ax[0, 0].set_title("CODE_GENDER distribution")
    for i, (k, v) in enumerate(g_counts.items()):
        ax[0, 0].text(i, v, f"{v:,}", ha="center", va="bottom")

    # 3-2 성별 × TARGET
    g_tgt = df.groupby("CODE_GENDER")["TARGET"].mean().sort_values()
    ax[0, 1].bar(g_tgt.index, g_tgt.values, color="#C44E52")
    ax[0, 1].axhline(df["TARGET"].mean(), color="black", linestyle="--", label=f"overall {df['TARGET'].mean():.3f}")
    ax[0, 1].set_title("Default rate by gender (P(TARGET=1))")
    ax[0, 1].set_ylabel("default rate")
    ax[0, 1].legend()
    for i, (k, v) in enumerate(g_tgt.items()):
        ax[0, 1].text(i, v, f"{v:.3f}", ha="center", va="bottom")

    # 3-3 연령 분포 (DAYS_BIRTH 음수)
    age = (-df["DAYS_BIRTH"] / 365.25).clip(0, 100)
    ax[1, 0].hist(age, bins=40, color="#4C72B0", edgecolor="white")
    ax[1, 0].set_title(f"Age distribution (mean={age.mean():.1f}, median={age.median():.1f})")
    ax[1, 0].set_xlabel("Age (years)")
    ax[1, 0].set_ylabel("count")

    # 3-4 연령 구간 × TARGET
    age_bins = pd.cut(age, bins=[0, 25, 35, 45, 55, 65, 100],
                      labels=["~25", "25-35", "35-45", "45-55", "55-65", "65+"])
    age_tgt = df.assign(_age_bin=age_bins).groupby("_age_bin", observed=True)["TARGET"].mean()
    ax[1, 1].bar(age_tgt.index.astype(str), age_tgt.values, color="#C44E52")
    ax[1, 1].axhline(df["TARGET"].mean(), color="black", linestyle="--", label=f"overall {df['TARGET'].mean():.3f}")
    ax[1, 1].set_title("Default rate by age bin")
    ax[1, 1].set_ylabel("default rate")
    ax[1, 1].legend()
    for i, v in enumerate(age_tgt.values):
        ax[1, 1].text(i, v, f"{v:.3f}", ha="center", va="bottom")

    plt.suptitle("Protected attributes: gender and age")
    savefig(fig, "03_protected_attrs")

    summary["protected_attrs"] = {
        "gender": {
            "counts": g_counts.to_dict(),
            "default_rate": g_tgt.to_dict(),
        },
        "age": {
            "mean": float(age.mean()),
            "median": float(age.median()),
            "default_rate_by_bin": {str(k): float(v) for k, v in age_tgt.items()},
        },
    }

    # ─────────────────────────────────────────
    # 4. 핵심 수치형 변수 분포
    # ─────────────────────────────────────────
    print("[5/8] 핵심 수치형 변수 분포...")
    cols = [c for c in CORE_NUMERIC if c in df.columns]
    n = len(cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows))
    axes = axes.flatten()
    for i, c in enumerate(cols):
        ax = axes[i]
        s = df[c].dropna()
        # 이상치 제거하지 않고 1~99 quantile로 시각 범위만 제한
        lo, hi = s.quantile([0.01, 0.99])
        ax.hist(s.clip(lo, hi), bins=40, color="#4C72B0", edgecolor="white", alpha=0.7)
        ax.set_title(f"{c}\n(n={s.size:,}, na={(df[c].isna().sum()):,})", fontsize=9)
        ax.set_yscale("log")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Core numeric distributions (1-99% clipped, y log scale)")
    savefig(fig, "04_numeric_distributions")

    # ─────────────────────────────────────────
    # 5. 상관 매트릭스 (TARGET 포함)
    # ─────────────────────────────────────────
    print("[6/8] 상관 매트릭스 ...")
    num_df = df.select_dtypes(include=[np.number])
    # 결측 너무 많은 컬럼 제외 (50%+) — 시각 용도
    keep = num_df.columns[num_df.isna().mean() <= 0.5]
    corr = num_df[keep].corr()["TARGET"].abs().sort_values(ascending=False)
    top = corr.iloc[1:21].index.tolist()  # TARGET 자기 자신 제외 + top 20
    cm = num_df[["TARGET"] + top].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.7}, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Top 20 |correlation| vs TARGET")
    savefig(fig, "05_correlation_top")

    target_corr = num_df[keep].corr()["TARGET"].drop("TARGET").sort_values()
    target_corr.to_csv(RESULTS_DIR / "eda_target_corr.csv", header=["pearson"])
    summary["target_corr_top10_neg"] = target_corr.head(10).round(4).to_dict()
    summary["target_corr_top10_pos"] = target_corr.tail(10).round(4).to_dict()

    # ─────────────────────────────────────────
    # 6. TARGET별 핵심 변수 분포 비교 (KDE 또는 박스플롯)
    # ─────────────────────────────────────────
    print("[7/8] TARGET별 핵심 변수 비교...")
    # 가장 상관 강한 6개를 골라 KDE
    top6 = target_corr.abs().sort_values(ascending=False).head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, c in enumerate(top6):
        ax = axes[i]
        for tgt, color, label in [(0, "#4C72B0", "non-default (0)"), (1, "#C44E52", "default (1)")]:
            s = df.loc[df["TARGET"] == tgt, c].dropna()
            if len(s) == 0:
                continue
            lo, hi = s.quantile([0.01, 0.99])
            sns.kdeplot(s.clip(lo, hi), ax=ax, label=label, color=color, fill=True, alpha=0.3)
        ax.set_title(f"{c}  (|ρ|={target_corr[c]:.3f})", fontsize=9)
        ax.legend(fontsize=8)
    plt.suptitle("Distribution by TARGET: top 6 by |correlation|")
    savefig(fig, "06_target_vs_features")

    # ─────────────────────────────────────────
    # 7. 범주형 cardinality
    # ─────────────────────────────────────────
    print("[8/8] 범주형 cardinality ...")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_card = pd.Series({c: df[c].nunique(dropna=True) for c in cat_cols}).sort_values()
    summary["categorical"] = {
        "n_columns": len(cat_cols),
        "cardinality": cat_card.to_dict(),
        "high_card_cols": cat_card[cat_card > 10].index.tolist(),
    }

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(cat_card))))
    ax.barh(cat_card.index, cat_card.values, color="#55A868")
    for i, v in enumerate(cat_card.values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=8)
    ax.set_xlabel("unique values (cardinality)")
    ax.set_title(f"Categorical cardinality (total {len(cat_cols)} columns)")
    savefig(fig, "07_categorical_cardinality")

    # ─────────────────────────────────────────
    # 도메인 의심 변수: DAYS_EMPLOYED 365243 sentinel
    # ─────────────────────────────────────────
    de_outlier = (df["DAYS_EMPLOYED"] == 365243).sum()
    summary["data_quality_flags"] = {
        "DAYS_EMPLOYED_sentinel_365243_count": int(de_outlier),
        "DAYS_EMPLOYED_sentinel_pct": float(de_outlier / len(df) * 100),
        "note": "365243 = 약 1000년, 사실상 '무직' 의미의 sentinel 값. 전처리에서 NaN 변환 권장.",
    }

    # ─────────────────────────────────────────
    # JSON 저장
    # ─────────────────────────────────────────
    with open(RESULTS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("\n[OK] 산출물 저장됨:")
    print("  figures/  -> 7개 png")
    print("  results/eda_summary.json")
    print("  results/eda_missing_table.csv")
    print("  results/eda_target_corr.csv")


if __name__ == "__main__":
    main()
