"""Step 3-C-2-f: Cross-judge G-Eval 비교 분석.

Claude judge와 Gemini judge가 같은 4 그룹(SHAP-only/Fusion × Anthropic/Gemini target)을
평가한 결과를 머지해서 cross-judge 일관성을 정량화.

산출:
    results/cross_judge_comparison.csv  — Claude vs Gemini judge 양옆 비교
    figures/32_cross_judge_geval.png    — 4 메트릭 × 2 target × 2 judge 막대그래프

사용:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.cross_judge_analysis
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import FIGURES_DIR, RESULTS_DIR, savefig

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")


def load_judge_summary(path: Path, judge: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["judge"] = judge
    return df


def main() -> None:
    claude = load_judge_summary(
        RESULTS_DIR / "fusion_vs_shaponly_claude_judge.csv", "claude")
    gemini = load_judge_summary(
        RESULTS_DIR / "fusion_vs_shaponly_gemini_judge.csv", "gemini")

    # G-Eval 메트릭만 추출 (룰은 judge 무관해서 둘이 동일)
    geval_metrics = [
        "geval_factual_accuracy", "geval_completeness",
        "geval_sensitive_leak", "geval_style",
    ]
    cl = claude[claude["metric"].isin(geval_metrics)].copy()
    ge = gemini[gemini["metric"].isin(geval_metrics)].copy()

    # long format → wide format으로 fusion vs shaponly 비교
    def _wide(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (provider, metric), grp in df.groupby(["provider", "metric"]):
            shap_row = grp[grp["mode"] == "shaponly"]
            fus_row = grp[grp["mode"] == "fusion"]
            if shap_row.empty or fus_row.empty:
                continue
            shap_mean = float(shap_row.iloc[0]["mean"])
            fus_mean = float(fus_row.iloc[0]["mean"])
            rows.append({
                "target_llm": provider, "metric": metric,
                "shaponly": shap_mean, "fusion": fus_mean,
                "delta": fus_mean - shap_mean,
            })
        return pd.DataFrame(rows)

    cl_w = _wide(cl)
    ge_w = _wide(ge)

    # 비교 표
    cmp_rows = []
    for _, r in cl_w.iterrows():
        match = ge_w[(ge_w["target_llm"] == r["target_llm"]) &
                       (ge_w["metric"] == r["metric"])]
        if match.empty:
            continue
        m = match.iloc[0]
        cmp_rows.append({
            "target_llm": r["target_llm"],
            "metric": r["metric"],
            "shaponly_claude_judge": r["shaponly"],
            "fusion_claude_judge": r["fusion"],
            "delta_claude_judge": r["delta"],
            "shaponly_gemini_judge": m["shaponly"],
            "fusion_gemini_judge": m["fusion"],
            "delta_gemini_judge": m["delta"],
            "judge_agreement_delta_diff": abs(r["delta"] - m["delta"]),
        })
    cmp_df = pd.DataFrame(cmp_rows).sort_values(["target_llm", "metric"])
    out_csv = RESULTS_DIR / "cross_judge_comparison.csv"
    cmp_df.to_csv(out_csv, index=False)
    print(f"[save] {out_csv}")
    print(cmp_df.to_string(index=False))

    # ─────────────────────────────────────────────
    # Figure: 4 메트릭 × 2 target × 2 judge × {shaponly,fusion}
    # ─────────────────────────────────────────────
    metrics_label = {
        "geval_factual_accuracy": "Factual Accuracy (1-5)",
        "geval_completeness":     "Completeness (1-5)",
        "geval_sensitive_leak":   "Sensitive Leak (1-5)",
        "geval_style":            "Style (1-5)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for ax, (met, label) in zip(axes, metrics_label.items()):
        sub = cmp_df[cmp_df["metric"] == met]
        if sub.empty:
            ax.set_visible(False)
            continue
        # row 만들기
        rows = []
        for _, r in sub.iterrows():
            for j in ["claude", "gemini"]:
                for mode in ["shaponly", "fusion"]:
                    rows.append({
                        "target_llm": r["target_llm"],
                        "judge_mode": f"{j}/{mode}",
                        "judge": j,
                        "mode": mode,
                        "score": r[f"{mode}_{j}_judge"],
                    })
        plot_df = pd.DataFrame(rows)
        # x = target_llm, hue = judge_mode (4가지)
        order_hue = ["claude/shaponly", "claude/fusion",
                      "gemini/shaponly", "gemini/fusion"]
        palette = {
            "claude/shaponly": "#A0A0A0",
            "claude/fusion":   "#DD8452",
            "gemini/shaponly": "#7AABCC",
            "gemini/fusion":   "#55A868",
        }
        sns.barplot(data=plot_df, x="target_llm", y="score", hue="judge_mode",
                     hue_order=order_hue, palette=palette, ax=ax,
                     errorbar=None)
        ax.set_title(label)
        ax.set_ylabel(label.split(" (")[0])
        ax.set_xlabel("Target LLM")
        if met != "geval_factual_accuracy":
            if ax.legend_:
                ax.legend_.remove()
        else:
            ax.legend(title="Judge / Mode", fontsize=8, title_fontsize=9,
                       loc="lower right")
        ax.set_ylim(0, 5.5)

    plt.suptitle("Cross-Judge G-Eval: Claude vs Gemini judge × SHAP-only vs Fusion (n=30)")
    out_fig = savefig(fig, "32_cross_judge_geval")
    print(f"[save] {out_fig}")

    # 추가 요약 print
    print("\n[delta] judge별 fusion 효과 비교 (Δ = fusion - shaponly):")
    for _, r in cmp_df.iterrows():
        print(f"  [{r['target_llm']:9s}] {r['metric']:30s} "
              f"Claude judge: {r['delta_claude_judge']:+.3f}, "
              f"Gemini judge: {r['delta_gemini_judge']:+.3f}, "
              f"|차이|={r['judge_agreement_delta_diff']:.3f}")


if __name__ == "__main__":
    main()
