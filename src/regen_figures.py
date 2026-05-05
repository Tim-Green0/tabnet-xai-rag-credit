"""기존 csv/json에서 모든 figure를 영어로 재생성.

학습/LLM 호출은 다시 안 함. 결과 파일에서 plot만 다시 그림.

실행:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.regen_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import RESULTS_DIR

sns.set_theme(style="whitegrid", context="notebook")

# ─────────────────────────────────────────────────────────────
# 13_cv_comparison
# ─────────────────────────────────────────────────────────────
def regen_13():
    from src.cv_eval import plot_cv_comparison
    summary = pd.read_csv(RESULTS_DIR / "cv_summary.csv")
    plot_cv_comparison(summary)
    print("  [13] cv_comparison")


# ─────────────────────────────────────────────────────────────
# 18_fairness_metrics + 19_fairness_mitigation
# ─────────────────────────────────────────────────────────────
def regen_18_19():
    from src.fairness import plot_fairness_metrics, plot_mitigation
    base_df = pd.read_csv(RESULTS_DIR / "fairness_metrics.csv")
    rows = []
    for _, r in base_df.iterrows():
        rows.append({
            "model": r["model"],
            "protected_attr": r["protected_attr"],
            "demographic_parity_diff": r["demographic_parity_diff"],
            "equal_opportunity_diff": r["equal_opportunity_diff"],
            "equalized_odds_diff": r["equalized_odds_diff"],
            "disparate_impact_ratio": r["disparate_impact_ratio"],
        })
    plot_fairness_metrics(rows)
    print("  [18] fairness_metrics")

    mit_df = pd.read_csv(RESULTS_DIR / "fairness_mitigation.csv")
    plot_mitigation(mit_df)
    print("  [19] fairness_mitigation")


# ─────────────────────────────────────────────────────────────
# 21_llm_comparison (compare_llms.main 재호출)
# ─────────────────────────────────────────────────────────────
def regen_21():
    from src.compare_llms import main as cmp_main
    cmp_main()
    print("  [21] llm_comparison")


# ─────────────────────────────────────────────────────────────
# 23_baseline_vs_xairag (csv에서 직접 plot)
# ─────────────────────────────────────────────────────────────
def regen_23():
    from src.utils import savefig
    cmp_df = pd.read_csv(RESULTS_DIR / "baseline_comparison.csv")

    fig, ax = plt.subplots(figsize=(11, 5))
    providers = cmp_df["provider"].tolist()
    x = np.arange(len(providers))
    width = 0.35
    ax.bar(x - width/2, cmp_df["xai_rag_halluc_strict_mean"], width,
            yerr=cmp_df["xai_rag_halluc_strict_std"], capsize=4,
            label="XAI-RAG (SHAP context)", color="#55A868")
    ax.bar(x + width/2, cmp_df["baseline_halluc_strict_mean"], width,
            yerr=cmp_df["baseline_halluc_strict_std"], capsize=4,
            label="baseline (no SHAP)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([p.title() for p in providers])
    ax.set_ylabel("Hallucination Rate (strict)")
    ax.set_title("XAI-RAG vs baseline (no SHAP) — Hallucination Rate (11 samples)")
    ax.legend()
    for i, (a_, b_) in enumerate(zip(
            cmp_df["xai_rag_halluc_strict_mean"],
            cmp_df["baseline_halluc_strict_mean"])):
        ax.text(i - width/2, a_ + 0.01, f"{a_:.3f}", ha="center")
        ax.text(i + width/2, b_ + 0.01, f"{b_:.3f}", ha="center")
    savefig(fig, "23_baseline_vs_xairag")
    print("  [23] baseline_vs_xairag")


# ─────────────────────────────────────────────────────────────
# eda 03/06/07 — 데이터 로딩 후 eda main 재호출
# ─────────────────────────────────────────────────────────────
def regen_eda():
    """eda.py main 재실행. 데이터 로딩 1회 + 빠름."""
    import warnings
    warnings.filterwarnings("ignore")
    from src.eda import main as eda_main
    eda_main()
    print("  [01-07] eda figures")


# ─────────────────────────────────────────────────────────────
# 08/09/10 baselines plot only — 모델 가중치 로드 후 plot 함수만
# ─────────────────────────────────────────────────────────────
def regen_08_09_10():
    import joblib
    from src.baselines import (
        plot_roc_pr, plot_threshold_sweep, plot_feature_importance,
        load_processed,
    )
    from src.metrics import find_threshold_youden

    data = load_processed()
    sc = data["scaled"]
    un = data["unscaled"]
    log_clf = joblib.load(RESULTS_DIR / "baseline_models" / "logistic.pkl")
    xgb_clf = joblib.load(RESULTS_DIR / "baseline_models" / "xgboost.pkl")
    lgb_clf = joblib.load(RESULTS_DIR / "baseline_models" / "lightgbm.pkl")

    test_scores = {}
    chosen_thr = {}
    # Logistic
    vs = log_clf.predict_proba(sc["X_val"])[:, 1]
    ts = log_clf.predict_proba(sc["X_test"])[:, 1]
    chosen_thr["Logistic"], _ = find_threshold_youden(sc["y_val"], vs)
    test_scores["Logistic"] = ts
    # XGB
    vs = xgb_clf.predict_proba(un["X_val"])[:, 1]
    ts = xgb_clf.predict_proba(un["X_test"])[:, 1]
    chosen_thr["XGBoost"], _ = find_threshold_youden(un["y_val"], vs)
    test_scores["XGBoost"] = ts
    # LGBM (sanitize 적용)
    from src.baselines import _sanitize_columns
    vs = lgb_clf.predict_proba(_sanitize_columns(un["X_val"]))[:, 1]
    ts = lgb_clf.predict_proba(_sanitize_columns(un["X_test"]))[:, 1]
    chosen_thr["LightGBM"], _ = find_threshold_youden(un["y_val"], vs)
    test_scores["LightGBM"] = ts

    plot_roc_pr(test_scores, sc["y_test"])
    plot_threshold_sweep(test_scores, sc["y_test"], chosen_thr)
    plot_feature_importance(log_clf, xgb_clf, lgb_clf,
                              sc["X_train"].columns.tolist(),
                              un["X_train"].columns.tolist())
    print("  [08-10] baseline figures")


# ─────────────────────────────────────────────────────────────
# 11/12 tabnet
# ─────────────────────────────────────────────────────────────
def regen_11_12():
    from src.tabnet_train import plot_attention_top
    fi = pd.read_csv(RESULTS_DIR / "tabnet_attention_importance.csv",
                       index_col=0)["importance"]
    fi = fi.sort_values(ascending=False)
    plot_attention_top(fi, "12_tabnet_attention_top20")
    print("  [12] tabnet_attention_top20")
    # 11번은 history가 raw로 없어 skip
    print("  [11] training_curve — skipped (history not persisted)")


# ─────────────────────────────────────────────────────────────
# 14/15/16/17 SHAP figures
# ─────────────────────────────────────────────────────────────
def regen_14_15_16_17():
    from src.shap_analysis import (
        plot_shap_global, plot_attention_vs_shap_scatter,
        plot_local_waterfall, attention_vs_shap,
    )
    imp_xgb = pd.read_csv(RESULTS_DIR / "shap_global_xgboost.csv",
                            index_col=0)["mean_abs_shap"]
    imp_log = pd.read_csv(RESULTS_DIR / "shap_global_logistic.csv",
                            index_col=0)["mean_abs_shap"]
    imp_tab = pd.read_csv(RESULTS_DIR / "shap_global_tabnet.csv",
                            index_col=0)["mean_abs_shap"]
    plot_shap_global(imp_xgb, "XGBoost", out_name="14_shap_global_xgboost")
    plot_shap_global(imp_log, "Logistic", out_name="14b_shap_global_logistic")
    plot_shap_global(imp_tab, "TabNet", out_name="15_shap_global_tabnet")
    print("  [14, 14b, 15] shap_global figures")

    att = pd.read_csv(RESULTS_DIR / "tabnet_attention_importance.csv",
                        index_col=0)["importance"]
    stats = json.loads((RESULTS_DIR / "attention_vs_shap.json").read_text(encoding="utf-8"))
    plot_attention_vs_shap_scatter(att, imp_tab, stats)
    print("  [16] attention_vs_shap_scatter")

    # 17 local waterfall 은 SHAP value 객체가 메모리에 없어 재생성 어려움 — skip
    print("  [17] local_waterfall — skipped (raw SHAP arrays not persisted)")


# ─────────────────────────────────────────────────────────────
# 22 demo walkthrough
# ─────────────────────────────────────────────────────────────
def regen_22():
    from src.demo import plot_walkthrough
    walk = json.loads((RESULTS_DIR / "demo_walkthrough.json").read_text(encoding="utf-8"))
    plot_walkthrough(walk)
    print("  [22] demo_walkthrough")


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    print("[regen figures]")
    regen_eda()
    regen_08_09_10()
    regen_11_12()
    regen_13()
    regen_14_15_16_17()
    regen_18_19()
    regen_21()
    regen_22()
    regen_23()
    print("\n[OK] all figures regenerated")


if __name__ == "__main__":
    main()
