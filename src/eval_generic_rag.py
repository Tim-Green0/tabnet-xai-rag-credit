"""Step 5-B-3: Generic RAG vs SHAP-RAG vs no-SHAP 3-way 평가.

Generic RAG (Step 5-B), SHAP-RAG / Fusion-RAG (Step 1/3-C-1), no-SHAP baseline (Step 1)
의 3가지 컨텍스트 모드를 동일 30 idx에 대해 비교.

평가:
    1. 룰 — Halluc strict (변수 토큰 환각), 변수명/값 매칭률
    2. NLI — entailment / contradiction (mDeBERTa-xnli)
    3. G-Eval — Claude judge factual / completeness / sensitive / style

산출:
    results/generic_rag_eval.csv     — 모든 mode × provider × idx의 메트릭
    results/generic_rag_summary.csv  — mode × provider mean ± std
    figures/35_generic_rag_3way.png  — 3-way 비교

사용:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.eval_generic_rag \
        [--skip-geval] [--skip-nli]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eval_explanation import (
    faithfulness_per_driver as _faithfulness_per_driver_orig,
    faithfulness_score, load_all_feature_names,
)
from src.eval_fusion import _english_prefixes
from src.cross_llm_geval import judge_one as _judge_one_cross
from src.llm_explainer import PROVIDER_DEFAULTS, make_client as _make_provider_client
from src.utils import FIGURES_DIR, RESULTS_DIR, savefig

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")


# ─────────────────────────────────────────────────────────────
# Generic RAG context → pseudo-drivers (faithfulness 룰용)
# ─────────────────────────────────────────────────────────────
def extract_drivers_generic_rag(ctx: Dict) -> List[Dict]:
    """customer_data를 driver list로 변환.

    Generic RAG는 SHAP 부호 없음 → sign 평가 X (None).
    feature_raw는 DOMAIN_GLOSSARY 역매핑.
    """
    from src.context_builder import DOMAIN_GLOSSARY
    inv_glossary = {v: k for k, v in DOMAIN_GLOSSARY.items()}

    out = []
    for kr, val in ctx.get("customer_data", {}).items():
        raw = inv_glossary.get(kr, kr)
        out.append({
            "feature": kr,
            "feature_raw": raw,
            "value": val,
            "value_raw": val,
            "shap": 0.0,  # 더미 (sign 평가 안 함)
            "group": "generic",
        })
    return out


# ─────────────────────────────────────────────────────────────
# Hallucination — generic RAG는 customer_data만 in_context
# ─────────────────────────────────────────────────────────────
def hallucination_rate_generic(text: str, ctx: Dict, all_features: set) -> Dict:
    drivers = extract_drivers_generic_rag(ctx)
    in_context_features = {d["feature_raw"] for d in drivers}
    # prefix 확장 (one-hot 등)
    in_context_prefixes = set()
    for f in in_context_features:
        for p in _english_prefixes(f):
            in_context_prefixes.add(p)
    dataset_prefixes = set()
    for f in all_features:
        for p in _english_prefixes(f):
            dataset_prefixes.add(p)

    raw_candidates = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text))
    common_excludes = {"SHAP", "REJECT", "APPROVE", "TARGET", "JSON", "API",
                        "AUC", "AUROC", "AUPRC", "OK", "AI", "LLM", "TABNET",
                        "DTI", "LTV", "DSR"}  # generic RAG에서 자주 인용되는 일반 약어
    raw_candidates -= common_excludes

    raw_in_ctx = (raw_candidates & in_context_features) | (raw_candidates & in_context_prefixes)
    raw_in_dataset = (raw_candidates & all_features) | (raw_candidates & dataset_prefixes)
    raw_outside = raw_candidates - raw_in_dataset
    raw_inside_dataset_outside_ctx = raw_in_dataset - raw_in_ctx

    return {
        "n_raw_candidates": len(raw_candidates),
        "n_raw_in_context": len(raw_in_ctx),
        "n_raw_in_dataset_only": len(raw_inside_dataset_outside_ctx),
        "n_raw_outside_dataset": len(raw_outside),
        "raw_outside_dataset_list": sorted(list(raw_outside)),
        "hallucination_rate_strict": (
            len(raw_outside) / max(len(raw_candidates), 1)),
        "hallucination_rate_broad": (
            (len(raw_outside) + len(raw_inside_dataset_outside_ctx))
            / max(len(raw_candidates), 1)),
    }


def faithfulness_generic(text: str, drivers: List[Dict]) -> List[Dict]:
    """generic RAG는 sign 평가 X — 모든 driver를 sign_in=None."""
    results = _faithfulness_per_driver_orig(text, drivers)
    for r in results:
        r["sign_in"] = None
        r["all_match"] = bool(r["feat_in"] and r["val_in"])
    return results


# ─────────────────────────────────────────────────────────────
# NLI premise builder for generic RAG
# ─────────────────────────────────────────────────────────────
def context_to_premise_generic(ctx: Dict, max_len: int = 1500) -> str:
    """customer_data + decision/probability를 자연어로."""
    decision = ctx.get("decision", "")
    prob = ctx.get("default_probability")
    decision_kr = "거절" if decision == "REJECT" else "승인" if decision == "APPROVE" else decision
    parts = [f"이 신청은 {decision_kr} 결정이며 부도 확률은 {prob:.1%}이다."]
    for kr, val in ctx.get("customer_data", {}).items():
        parts.append(f"{kr}이(가) {val}이다.")
    text = " ".join(parts)
    return text[:max_len] if len(text) > max_len else text


# ─────────────────────────────────────────────────────────────
# 통합 평가 (기존 SHAP-RAG / Fusion 결과는 cache)
# ─────────────────────────────────────────────────────────────
def evaluate_directory(directory: Path, mode: str, provider: str,
                        all_features: set, geval_client=None,
                        geval_model: str = None,
                        nli_tokenizer=None, nli_model=None,
                        geval_sleep: float = 4.0,
                        target_idx: List[int] = None) -> List[Dict]:
    """단일 디렉토리(mode/provider) 평가.

    mode: "generic_rag" | "shaponly" | "fusion" | "no_shap"
    """
    rows = []
    files = sorted([p for p in directory.glob("*.json")
                     if p.name != "_index.json"])
    if target_idx is not None:
        target_set = set(target_idx)
        files = [p for p in files
                  if int(p.stem.split("_")[0]) in target_set]
    print(f"  [{mode}/{provider}] {len(files)}개 평가")

    # mode별 driver/halluc/premise 함수 선택
    if mode == "generic_rag" or mode == "no_shap":
        # 둘 다 customer_data 기반 (baseline_no_shap은 SHAP 없는 raw, generic_rag는 raw + chunks)
        ext_fn = extract_drivers_generic_rag
        halluc_fn = hallucination_rate_generic
        faith_fn = faithfulness_generic
        prem_fn = context_to_premise_generic
    elif mode == "fusion":
        from src.eval_fusion import (
            extract_drivers_fusion, hallucination_rate_fusion,
            faithfulness_per_driver as fusion_faith_fn)
        from src.nli_eval import context_to_premise as fusion_prem_fn
        ext_fn = extract_drivers_fusion
        halluc_fn = hallucination_rate_fusion
        faith_fn = fusion_faith_fn
        prem_fn = fusion_prem_fn
    elif mode == "shaponly":
        from src.eval_explanation import (
            extract_drivers_from_context as shap_ext_fn,
            hallucination_rate as shap_halluc_fn,
            faithfulness_per_driver as shap_faith_fn)
        from src.nli_eval import context_to_premise as shap_prem_fn
        ext_fn = shap_ext_fn
        halluc_fn = shap_halluc_fn
        faith_fn = shap_faith_fn
        prem_fn = shap_prem_fn
    else:
        raise ValueError(f"unknown mode: {mode}")

    for i, exp_path in enumerate(files, start=1):
        with open(exp_path, "r", encoding="utf-8") as f:
            exp = json.load(f)
        ctx = exp["context_sent"]
        text = exp["explanation"]
        sample_id = exp_path.stem

        drivers = ext_fn(ctx)
        per_driver = faith_fn(text, drivers)
        f_scores = faithfulness_score(per_driver)
        h_scores = halluc_fn(text, ctx, all_features)

        row = {
            "sample_id": sample_id, "mode": mode, "provider": provider,
            "decision": exp.get("decision"),
            "true_label": exp.get("true_label"),
            "n_drivers": len(drivers),
            **f_scores,
            "halluc_rate_strict": h_scores["hallucination_rate_strict"],
            "halluc_rate_broad": h_scores["hallucination_rate_broad"],
            "n_raw_candidates": h_scores["n_raw_candidates"],
            "n_raw_outside_dataset": h_scores["n_raw_outside_dataset"],
            "raw_outside_list": ";".join(h_scores.get("raw_outside_dataset_list", [])),
        }

        # NLI
        if nli_tokenizer is not None and nli_model is not None:
            from src.nli_eval import nli_score, split_sentences
            premise = prem_fn(ctx)
            sentences = split_sentences(text)
            if sentences:
                scores = [nli_score(nli_tokenizer, nli_model, premise, s)
                           for s in sentences]
                df_s = pd.DataFrame(scores)
                row.update({
                    "nli_n_sentences": len(sentences),
                    "nli_entailment_rate": float(df_s["entailment"].mean()),
                    "nli_contradiction_rate": float(df_s["contradiction"].mean()),
                    "nli_neutral_rate": float(df_s["neutral"].mean()),
                    "nli_min_entailment": float(df_s["entailment"].min()),
                })

        # G-Eval
        if geval_client is not None:
            for attempt in range(4):
                try:
                    t0 = time.time()
                    ge = _judge_one_cross(geval_client, geval_model, text, ctx)
                    row["geval_elapsed_sec"] = round(time.time() - t0, 2)
                    p = ge["parsed"]
                    if not p.get("parse_error"):
                        row.update({
                            "geval_factual_accuracy": p.get("factual_accuracy"),
                            "geval_completeness": p.get("completeness"),
                            "geval_sensitive_leak": p.get("sensitive_leak"),
                            "geval_style": p.get("style"),
                        })
                    break
                except Exception as e:
                    err = str(e)
                    is_503 = "503" in err or "UNAVAILABLE" in err or "overloaded" in err.lower()
                    backoff = 30 * (2 ** attempt) if is_503 else 10
                    print(f"    G-Eval ERROR (attempt {attempt+1}/4): {err[:80]}")
                    if attempt < 3:
                        time.sleep(backoff)
                    else:
                        row["geval_error"] = err[:200]
            if i < len(files):
                time.sleep(geval_sleep)

        rows.append(row)
    return rows


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    metric_cols = [
        "halluc_rate_strict", "halluc_rate_broad",
        "feat_match_rate", "val_match_rate", "full_match_rate",
        "nli_entailment_rate", "nli_contradiction_rate",
        "geval_factual_accuracy", "geval_completeness",
        "geval_sensitive_leak", "geval_style",
    ]
    for (mode, provider), grp in df.groupby(["mode", "provider"]):
        for col in metric_cols:
            if col in grp.columns:
                vals = grp[col].dropna()
                if len(vals) > 0:
                    out.append({
                        "mode": mode, "provider": provider, "metric": col,
                        "mean": float(vals.mean()),
                        "std": float(vals.std()) if len(vals) > 1 else 0.0,
                        "n": int(len(vals)),
                    })
    return pd.DataFrame(out)


def plot_3way(summary: pd.DataFrame,
               out_name: str = "35_generic_rag_3way") -> Path:
    """4 핵심 메트릭 × 2 LLM × N modes 비교."""
    metrics = [
        ("halluc_rate_strict", "Halluc Rate (strict, ↓)"),
        ("nli_entailment_rate", "NLI Entailment (↑)"),
        ("nli_contradiction_rate", "NLI Contradiction (↓)"),
        ("geval_completeness", "G-Eval Completeness (↑)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    palette = {"shaponly": "#A0A0A0", "fusion": "#DD8452",
                "generic_rag": "#55A868", "no_shap": "#C44E52"}

    for ax, (m, label) in zip(axes, metrics):
        sub = summary[summary["metric"] == m]
        if len(sub) == 0:
            ax.set_visible(False); continue
        sns.barplot(data=sub, x="provider", y="mean", hue="mode",
                     ax=ax, errorbar=None,
                     hue_order=["no_shap", "generic_rag", "shaponly", "fusion"],
                     palette=palette)
        ax.set_title(label)
        ax.set_ylabel(label.split(" (")[0])
        ax.set_xlabel("LLM provider")
        if m != "halluc_rate_strict":
            if ax.legend_:
                ax.legend_.remove()
    plt.suptitle("3-way comparison — no_shap / generic_rag / shaponly / fusion (n=30 each)")
    return savefig(fig, out_name)


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main(skip_geval: bool = False, skip_nli: bool = False,
         judge: str = "anthropic", n_samples: int = 30) -> None:
    all_features = load_all_feature_names()

    # target idx (fusion 결과의 _index.json에서)
    fusion_idx_path = RESULTS_DIR / f"explanations_fusion_anthropic_{n_samples}" / "_index.json"
    with open(fusion_idx_path, "r", encoding="utf-8") as f:
        target_idx = json.load(f)["selected_idx"]
    print(f"[match] target_idx {len(target_idx)}개")

    # NLI 모델
    if not skip_nli:
        from src.nli_eval import load_nli_model
        tokenizer, nli_model = load_nli_model()
    else:
        tokenizer, nli_model = None, None

    # G-Eval client
    if skip_geval:
        client = None; judge_model = None
    else:
        client = _make_provider_client(judge)
        judge_model = PROVIDER_DEFAULTS[judge]["model"]
        print(f"[judge] {judge} / {judge_model}")

    # 평가 plan
    plan = [
        ("no_shap", "anthropic", RESULTS_DIR / "explanations_baseline_noshap_anthropic"),
        ("no_shap", "gemini", RESULTS_DIR / "explanations_baseline_noshap_gemini"),
        ("generic_rag", "anthropic", RESULTS_DIR / f"explanations_generic_rag_anthropic_{n_samples}"),
        ("generic_rag", "gemini", RESULTS_DIR / f"explanations_generic_rag_gemini_{n_samples}"),
        ("shaponly", "anthropic", RESULTS_DIR / "explanations_anthropic_100"),
        ("shaponly", "gemini", RESULTS_DIR / "explanations_gemini_100"),
        ("fusion", "anthropic", RESULTS_DIR / f"explanations_fusion_anthropic_{n_samples}"),
        ("fusion", "gemini", RESULTS_DIR / f"explanations_fusion_gemini_{n_samples}"),
    ]
    rows = []
    for mode, provider, directory in plan:
        if not directory.exists():
            print(f"[skip] {directory} 없음")
            continue
        rows.extend(evaluate_directory(
            directory, mode, provider, all_features,
            geval_client=client, geval_model=judge_model,
            nli_tokenizer=tokenizer, nli_model=nli_model,
            target_idx=target_idx,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "generic_rag_eval.csv", index=False)
    print(f"\n[save] generic_rag_eval.csv ({len(df)} rows)")

    summary = aggregate(df)
    summary.to_csv(RESULTS_DIR / "generic_rag_summary.csv", index=False)
    print(f"[save] generic_rag_summary.csv")

    # mode별 핵심 메트릭 print
    print("\n[summary] 핵심 메트릭 (mode × provider):")
    pivot = summary.pivot_table(index="metric", columns=["provider", "mode"],
                                  values="mean").round(3)
    print(pivot.to_string())

    plot_3way(summary)
    print(f"[save] figures/35_generic_rag_3way.png")
    print("\n[OK] 3-way 평가 완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-geval", action="store_true")
    ap.add_argument("--skip-nli", action="store_true")
    ap.add_argument("--judge", default="anthropic",
                    choices=["anthropic", "gemini"])
    ap.add_argument("--n-samples", type=int, default=30)
    args = ap.parse_args()
    main(skip_geval=args.skip_geval, skip_nli=args.skip_nli,
          judge=args.judge, n_samples=args.n_samples)
