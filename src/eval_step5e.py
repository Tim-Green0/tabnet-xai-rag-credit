"""Step 5-E 평가: fusion_friendly vs fusion 비교 (NLI + value_match + G-Eval).

대상:
  - Home Credit: results/explanations_friendly_home_{anthropic,gemini}_30/
  - German Credit: results/explanations_friendly_german_{anthropic,gemini}_30/

Premise는 fusion context와 동일 (friendly는 prompt만 변경, context 동일).
NLI/value_match/halluc는 src.german_eval 함수 재사용.
G-Eval은 cross_llm_geval.judge_one 재사용 (Anthropic judge).

산출:
  results/step5e_friendly_eval.csv     — friendly 30 × 2 dataset × 2 LLM = 120 rows
  results/step5e_comparison.csv        — fusion vs friendly mean ± std
  figures/42_friendly_vs_fusion.png
  results/step5e_summary.md

실행:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.eval_step5e \\
        [--skip-geval] [--geval-sleep 3]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.cross_llm_geval import judge_one as _judge_one
from src.german_eval import (
    context_to_premise, evaluate_nli_one, halluc_rate, load_nli_model,
    split_sentences, value_match_rate,
)
from src.llm_explainer import PROVIDER_DEFAULTS, make_client
from src.utils import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# all_features 로드 — dataset별
# ─────────────────────────────────────────────────────────────
def load_features_home() -> set:
    df = pd.read_parquet("data/processed/train_scaled.parquet")
    return set(df.columns) - {"TARGET"}


def load_features_german() -> set:
    df = pd.read_parquet("data/german_credit/processed/train_scaled.parquet")
    return set(df.columns) - {"TARGET"}


# ─────────────────────────────────────────────────────────────
# friendly 디렉토리 평가
# ─────────────────────────────────────────────────────────────
def evaluate_friendly_directory(directory: Path, dataset: str, provider: str,
                                  all_features: set,
                                  nli_tokenizer=None, nli_model=None,
                                  geval_client=None, geval_model: str = None,
                                  geval_sleep: float = 4.0,
                                  skip_nli: bool = False,
                                  skip_geval: bool = False) -> List[Dict]:
    rows = []
    files = sorted([p for p in directory.glob("*.json") if p.name != "_index.json"])
    print(f"  [friendly/{dataset}/{provider}] {len(files)}개 평가")

    for i, exp_path in enumerate(files, start=1):
        with open(exp_path, "r", encoding="utf-8") as f:
            exp = json.load(f)
        ctx = exp["context_sent"]
        text = exp["explanation"]

        row = {
            "sample_id": exp_path.stem,
            "mode": "fusion_friendly",
            "dataset": dataset,
            "provider": provider,
            "decision": exp.get("decision"),
            "true_label": exp.get("true_label"),
            "default_proba": ctx.get("default_probability"),
        }
        row.update(halluc_rate(text, ctx, all_features))
        row.update(value_match_rate(text, ctx))

        if not skip_nli and nli_tokenizer is not None:
            row.update(evaluate_nli_one(nli_tokenizer, nli_model, ctx, text))

        if not skip_geval and geval_client is not None:
            ge = None
            for attempt in range(3):
                try:
                    t0 = time.time()
                    ge = _judge_one(geval_client, geval_model, text, ctx)
                    row["geval_elapsed_sec"] = round(time.time() - t0, 2)
                    break
                except Exception as e:
                    print(f"    G-Eval ERROR (a={attempt+1}/3): {str(e)[:100]}")
                    if attempt < 2:
                        time.sleep(30 * (2 ** attempt))
            if ge is not None:
                p = ge["parsed"]
                if not p.get("parse_error"):
                    row.update({
                        "geval_factual_accuracy": p.get("factual_accuracy"),
                        "geval_completeness": p.get("completeness"),
                        "geval_sensitive_leak": p.get("sensitive_leak"),
                        "geval_style": p.get("style"),
                    })
            time.sleep(geval_sleep)

        rows.append(row)
        if i % 5 == 0:
            print(f"    [{i}/{len(files)}] running")
    return rows


METRIC_COLS = [
    "halluc_rate_strict", "val_match_rate",
    "entailment_rate", "contradiction_rate", "min_entailment",
    "geval_factual_accuracy", "geval_completeness",
    "geval_sensitive_leak", "geval_style",
]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (dataset, mode, provider), grp in df.groupby(["dataset", "mode", "provider"]):
        for col in METRIC_COLS:
            if col in grp.columns:
                vals = grp[col].dropna()
                if len(vals) > 0:
                    out.append({
                        "dataset": dataset, "mode": mode, "provider": provider,
                        "metric": col,
                        "mean": float(vals.mean()),
                        "std": float(vals.std()) if len(vals) > 1 else 0.0,
                        "n": int(len(vals)),
                    })
    return pd.DataFrame(out)


def merge_with_existing_fusion(friendly_summary: pd.DataFrame) -> pd.DataFrame:
    """기존 fusion 평가 결과와 합침."""
    parts = [friendly_summary]

    # German fusion (results/german_eval_summary.csv)
    g_path = RESULTS_DIR / "german_eval_summary.csv"
    if g_path.exists():
        gdf = pd.read_csv(g_path)
        gdf = gdf[gdf["mode"] == "fusion"].copy()
        gdf["dataset"] = "german"
        parts.append(gdf[["dataset", "mode", "provider", "metric", "mean", "std", "n"]])

    # Home fusion (results/generic_rag_summary.csv) — Step 5-B 4-way 통합
    h_path = RESULTS_DIR / "generic_rag_summary.csv"
    if h_path.exists():
        hdf = pd.read_csv(h_path)
        hdf = hdf[hdf["mode"] == "fusion"].copy()
        hdf["dataset"] = "home"
        # Home Credit eval은 nli_entailment_rate 명명 → 통일
        hdf["metric"] = hdf["metric"].replace({
            "nli_entailment_rate": "entailment_rate",
            "nli_contradiction_rate": "contradiction_rate",
        })
        parts.append(hdf[["dataset", "mode", "provider", "metric", "mean", "std", "n"]])

    out = pd.concat(parts, ignore_index=True)
    out = out[out["metric"].isin(METRIC_COLS)]
    return out


def plot_comparison(df: pd.DataFrame, name: str = "42_friendly_vs_fusion") -> Path:
    metrics = [
        ("entailment_rate", "NLI Entailment (↑)"),
        ("val_match_rate", "Value Match Rate (↑)"),
        ("geval_factual_accuracy", "G-Eval Factual (1-5, ↑)"),
        ("geval_completeness", "G-Eval Completeness (1-5, ↑)"),
        ("geval_sensitive_leak", "G-Eval Sensitive (1-5, ↑)"),
        ("geval_style", "G-Eval Style (1-5, ↑)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    palette = {"fusion": "#A0A0A0", "fusion_friendly": "#DD8452"}
    for ax, (m, label) in zip(axes.flat, metrics):
        sub = df[df["metric"] == m].copy()
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        # facet: x=dataset+provider, hue=mode
        sub["dataset_provider"] = sub["dataset"] + "/" + sub["provider"]
        sub = sub.sort_values("dataset_provider")
        sns.barplot(data=sub, x="dataset_provider", y="mean", hue="mode",
                     hue_order=["fusion", "fusion_friendly"], ax=ax,
                     errorbar=None, palette=palette)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=15)
        if ax.get_legend():
            ax.legend(loc="best", fontsize=8)
        # 값 라벨
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=3)
    plt.suptitle("Step 5-E — Customer-friendly fusion vs original fusion", fontsize=14, y=0.998)
    plt.tight_layout()
    return savefig(fig, name)


def write_summary_md(df: pd.DataFrame) -> Path:
    out = RESULTS_DIR / "step5e_summary.md"

    def get_mean(dataset, metric, mode, provider=None):
        sub = df[(df["dataset"] == dataset) & (df["metric"] == metric)
                 & (df["mode"] == mode)]
        if provider:
            sub = sub[sub["provider"] == provider]
        if len(sub) == 0:
            return None
        return float(sub["mean"].mean())

    def fmt(v):
        return "—" if v is None else f"{v:.3f}"

    def diff(a, b):
        if a is None or b is None:
            return "—"
        d = b - a
        sign = "+" if d > 0 else ""
        return f"{sign}{d:.3f}"

    lines = [
        "# Step 5-E 요약 — Customer-friendly fusion vs original fusion",
        "",
        "## 동기 (Step 5-C+5-D 일관 약점)",
        "- Step 5-C Customer persona clarity: fusion 2.67 (vs generic_rag 4.93)",
        "- Step 5-D G-Eval Factual: fusion 3.97/2.97 (vs generic_rag 5.0/5.0)",
        "→ fusion의 SHAP 부호 raw + agreement 라벨이 over-explain으로 평가됨.",
        "",
        "## 변경 사항",
        "- 같은 fusion context 그대로 사용",
        "- Prompt만 변경: SHAP 부호 자연어화, agreement 라벨 직관 표현, 정성 표현 추가, 친근한 톤",
        "- 새 mode: `fusion_friendly`",
        "",
        "## NLI Entailment (사실성 유지 여부, ↑)",
        "",
        "| Dataset | fusion | fusion_friendly | Δ |",
        "|---|---|---|---|",
    ]
    for ds in ["home", "german"]:
        f = get_mean(ds, "entailment_rate", "fusion")
        ff = get_mean(ds, "entailment_rate", "fusion_friendly")
        lines.append(f"| {ds} | {fmt(f)} | {fmt(ff)} | {diff(f, ff)} |")

    lines += [
        "",
        "## G-Eval Factual Accuracy (★ Step 5-D fusion 약점 차원)",
        "",
        "| Dataset | fusion | fusion_friendly | Δ |",
        "|---|---|---|---|",
    ]
    for ds in ["home", "german"]:
        f = get_mean(ds, "geval_factual_accuracy", "fusion")
        ff = get_mean(ds, "geval_factual_accuracy", "fusion_friendly")
        lines.append(f"| {ds} | {fmt(f)} | {fmt(ff)} | {diff(f, ff)} |")

    lines += [
        "",
        "## G-Eval Completeness (충실성)",
        "",
        "| Dataset | fusion | fusion_friendly | Δ |",
        "|---|---|---|---|",
    ]
    for ds in ["home", "german"]:
        f = get_mean(ds, "geval_completeness", "fusion")
        ff = get_mean(ds, "geval_completeness", "fusion_friendly")
        lines.append(f"| {ds} | {fmt(f)} | {fmt(ff)} | {diff(f, ff)} |")

    lines += [
        "",
        "## G-Eval Style (친근함)",
        "",
        "| Dataset | fusion | fusion_friendly | Δ |",
        "|---|---|---|---|",
    ]
    for ds in ["home", "german"]:
        f = get_mean(ds, "geval_style", "fusion")
        ff = get_mean(ds, "geval_style", "fusion_friendly")
        lines.append(f"| {ds} | {fmt(f)} | {fmt(ff)} | {diff(f, ff)} |")

    lines += [
        "",
        "## Value Match Rate (값 정확 인용)",
        "",
        "| Dataset | fusion | fusion_friendly | Δ |",
        "|---|---|---|---|",
    ]
    for ds in ["home", "german"]:
        f = get_mean(ds, "val_match_rate", "fusion")
        ff = get_mean(ds, "val_match_rate", "fusion_friendly")
        lines.append(f"| {ds} | {fmt(f)} | {fmt(ff)} | {diff(f, ff)} |")

    lines += [
        "",
        "## 결론",
        "(자동 결론 — 정확한 해석은 사람 검토 권장)",
        "",
        "- NLI 사실성이 유지/향상되면 fact-grounded 메시지 보존",
        "- G-Eval factual / style 향상이 보이면 over-explain 약점 개선",
        "- Δ가 음수인 메트릭은 trade-off로 honest reporting",
        "",
        "## 산출 파일",
        "",
        "- `results/explanations_friendly_{home,german}_{anthropic,gemini}_30/`",
        "- `results/step5e_friendly_eval.csv`",
        "- `results/step5e_comparison.csv`",
        "- `figures/42_friendly_vs_fusion.png`",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out}")
    return out


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main(skip_nli: bool = False, skip_geval: bool = False,
         geval_judge: str = "anthropic", geval_sleep: float = 4.0,
         n_samples: int = 30) -> None:
    set_seed(SEED)
    print(f"[device] {DEVICE}")

    nli_tok, nli_mod = (None, None)
    if not skip_nli:
        nli_tok, nli_mod = load_nli_model()

    geval_client, geval_model = None, None
    if not skip_geval:
        geval_client = make_client(geval_judge)
        geval_model = PROVIDER_DEFAULTS[geval_judge]["model"]
        print(f"[G-Eval] judge={geval_judge}/{geval_model}")

    feat_home = load_features_home()
    feat_german = load_features_german()
    print(f"[data] home features={len(feat_home)}, german features={len(feat_german)}")

    all_rows = []
    for dataset in ["home", "german"]:
        feat = feat_home if dataset == "home" else feat_german
        for provider in ["anthropic", "gemini"]:
            d = RESULTS_DIR / f"explanations_friendly_{dataset}_{provider}_{n_samples}"
            if not d.exists():
                print(f"[skip] {d} 없음")
                continue
            rows = evaluate_friendly_directory(
                d, dataset, provider, feat,
                nli_tokenizer=nli_tok, nli_model=nli_mod,
                geval_client=geval_client, geval_model=geval_model,
                geval_sleep=geval_sleep,
                skip_nli=skip_nli, skip_geval=skip_geval)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "step5e_friendly_eval.csv", index=False)
    print(f"[saved] results/step5e_friendly_eval.csv ({len(df)} rows)")

    summary = aggregate(df)
    summary["dataset"] = summary["dataset"].fillna("unknown")
    merged = merge_with_existing_fusion(summary)
    merged.to_csv(RESULTS_DIR / "step5e_comparison.csv", index=False)
    print(f"[saved] results/step5e_comparison.csv ({len(merged)} rows)")

    print("\n[summary - mode mean]")
    pv = merged.pivot_table(index=["dataset", "metric"], columns="mode",
                              values="mean").round(3)
    print(pv.to_string())

    plot_comparison(merged)
    write_summary_md(merged)
    print("\n[OK] Step 5-E 평가 완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nli", action="store_true")
    ap.add_argument("--skip-geval", action="store_true")
    ap.add_argument("--geval-judge", default="anthropic",
                    choices=["anthropic", "gemini"])
    ap.add_argument("--geval-sleep", type=float, default=4.0)
    ap.add_argument("--n-samples", type=int, default=30)
    args = ap.parse_args()
    main(skip_nli=args.skip_nli, skip_geval=args.skip_geval,
         geval_judge=args.geval_judge, geval_sleep=args.geval_sleep,
         n_samples=args.n_samples)
