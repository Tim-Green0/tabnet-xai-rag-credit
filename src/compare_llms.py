"""Day 7 — 두 LLM (Gemini vs Claude) 평가 결과 비교 및 시각화.

산출물:
  - results/llm_comparison.csv         : 차원별 두 LLM 비교
  - figures/21_llm_comparison.png      : 막대그래프 비교
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import FIGURES_DIR, RESULTS_DIR, savefig

sns.set_theme(style="whitegrid", context="notebook")


def load_summary(suffix: str) -> dict:
    return json.loads(
        (RESULTS_DIR / f"explanation_eval_summary_{suffix}.json").read_text(encoding="utf-8")
    )


def load_explanations(suffix: str) -> list:
    """LLM별 elapsed/token 통계."""
    if suffix == "gemini":
        d = RESULTS_DIR / "explanations"
    else:
        d = RESULTS_DIR / f"explanations_{suffix}"
    rows = []
    for fp in sorted(d.glob("*.json")):
        if fp.name == "_index.json":
            continue
        ex = json.loads(fp.read_text(encoding="utf-8"))
        rows.append({
            "sample_id": fp.stem,
            "elapsed_sec": ex.get("elapsed_sec"),
            "total_tokens": ex.get("usage_metadata", {}).get("total_token_count"),
            "input_tokens": ex.get("usage_metadata", {}).get(
                "input_tokens", ex.get("usage_metadata", {}).get("prompt_token_count")
            ),
            "output_tokens": ex.get("usage_metadata", {}).get(
                "output_tokens", ex.get("usage_metadata", {}).get("candidates_token_count")
            ),
        })
    return rows


def main() -> None:
    g = load_summary("gemini")
    a = load_summary("anthropic")

    # 비교할 차원
    rule_dims = ["feat_match_rate", "val_match_rate", "sign_match_rate",
                  "full_match_rate", "halluc_rate_strict", "halluc_rate_broad"]
    geval_dims = ["geval_factual_accuracy", "geval_completeness",
                   "geval_sensitive_leak", "geval_style"]

    rows = []
    for d in rule_dims + geval_dims:
        gv = g.get(d, {})
        av = a.get(d, {})
        rows.append({
            "metric": d,
            "category": "rule_based" if d in rule_dims else "geval",
            "gemini_mean": gv.get("mean"),
            "gemini_std": gv.get("std"),
            "gemini_n": gv.get("n"),
            "anthropic_mean": av.get("mean"),
            "anthropic_std": av.get("std"),
            "anthropic_n": av.get("n"),
        })
    cmp_df = pd.DataFrame(rows)

    # 효율성 (시간/토큰)
    g_exp = pd.DataFrame(load_explanations("gemini"))
    a_exp = pd.DataFrame(load_explanations("anthropic"))
    eff_rows = []
    for col in ["elapsed_sec", "total_tokens"]:
        eff_rows.append({
            "metric": col,
            "category": "efficiency",
            "gemini_mean": float(g_exp[col].mean()),
            "gemini_std": float(g_exp[col].std()),
            "gemini_n": int(len(g_exp)),
            "anthropic_mean": float(a_exp[col].mean()),
            "anthropic_std": float(a_exp[col].std()),
            "anthropic_n": int(len(a_exp)),
        })
    cmp_df = pd.concat([cmp_df, pd.DataFrame(eff_rows)], ignore_index=True)
    cmp_df.to_csv(RESULTS_DIR / "llm_comparison.csv", index=False)

    print("[LLM 비교 표]")
    print(cmp_df.round(3).to_string(index=False))

    # ─── 시각화 ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (1) Rule-based 충실성/환각
    rule_show = cmp_df[cmp_df["category"] == "rule_based"]
    x = np.arange(len(rule_show))
    width = 0.35
    axes[0].bar(x - width/2, rule_show["gemini_mean"], width,
                 yerr=rule_show["gemini_std"], capsize=4,
                 color="#4C72B0", label="Gemini 2.5 Flash")
    axes[0].bar(x + width/2, rule_show["anthropic_mean"], width,
                 yerr=rule_show["anthropic_std"], capsize=4,
                 color="#DD8452", label="Claude Sonnet 4.5")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rule_show["metric"].str.replace("_rate", "").str.replace("halluc_", "halluc-"),
                              rotation=30, ha="right", fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("rate")
    axes[0].set_title("룰 기반 충실성 / 환각률\n(Faithfulness + Hallucination)")
    axes[0].legend(loc="best")

    # (2) G-Eval (Gemini-judge 기준)
    geval_show = cmp_df[cmp_df["category"] == "geval"].dropna(subset=["gemini_mean"])
    if len(geval_show):
        x2 = np.arange(len(geval_show))
        axes[1].bar(x2 - width/2, geval_show["gemini_mean"], width,
                     yerr=geval_show["gemini_std"], capsize=4,
                     color="#4C72B0", label="Gemini 2.5 Flash")
        if geval_show["anthropic_mean"].notna().any():
            axes[1].bar(x2 + width/2, geval_show["anthropic_mean"], width,
                         yerr=geval_show["anthropic_std"], capsize=4,
                         color="#DD8452", label="Claude Sonnet 4.5")
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(geval_show["metric"].str.replace("geval_", ""),
                                 rotation=20, ha="right", fontsize=9)
        axes[1].set_ylim(0, 5.5)
        axes[1].set_ylabel("score (1-5)")
        axes[1].set_title("G-Eval (Gemini self-judge)\n(Factual + Complete + Sensitive + Style)")
        axes[1].legend(loc="best")
        for i, v in enumerate(geval_show["gemini_mean"]):
            if pd.notna(v):
                axes[1].text(i - width/2, v + 0.1, f"{v:.2f}",
                              ha="center", fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "G-Eval 결과 없음", ha="center", va="center",
                      transform=axes[1].transAxes)
        axes[1].set_title("G-Eval (생성 실패)")

    # (3) Efficiency
    eff_show = cmp_df[cmp_df["category"] == "efficiency"]
    x3 = np.arange(len(eff_show))
    g_norm = eff_show.apply(lambda r: r["gemini_mean"], axis=1).values
    a_norm = eff_show.apply(lambda r: r["anthropic_mean"], axis=1).values

    # 두 metric 단위가 다르니 두 y축
    metric_pretty = ["elapsed (s)", "total tokens"]
    bx = axes[2].bar(x3 - width/2, g_norm, width, color="#4C72B0",
                       label="Gemini")
    cx = axes[2].bar(x3 + width/2, a_norm, width, color="#DD8452",
                       label="Claude")
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(metric_pretty)
    axes[2].set_ylabel("value")
    axes[2].set_title("효율성 (호출당)\nelapsed seconds / total tokens")
    axes[2].legend(loc="best")
    for bars in [bx, cx]:
        for b in bars:
            h = b.get_height()
            axes[2].text(b.get_x() + b.get_width()/2, h, f"{h:.0f}",
                          ha="center", va="bottom", fontsize=8)

    plt.suptitle("Gemini 2.5 Flash vs Claude Sonnet 4.5 — XAI-RAG 설명 품질 비교 (10 샘플)",
                  fontsize=12, y=1.02)
    plt.tight_layout()
    out = savefig(fig, "21_llm_comparison")

    print(f"\n[OK] LLM 비교 완료")
    print(f"     - results/llm_comparison.csv")
    print(f"     - {out}")


if __name__ == "__main__":
    main()
