"""UCI German Credit — 4-mode 평가 (Step 5-D Day 3).

평가 차원:
  1. NLI Entailment / Contradiction (mDeBERTa-multilingual-NLI)
  2. G-Eval (Claude judge) — factual / completeness / sensitive / style
  3. value_match_rate — 컨텍스트 값이 explanation에 인용되는 비율 (간단 룰)
  4. halluc_rate — 영문 토큰이 in-dataset 외에 등장하는 비율 (raw_outside)

대상: results/explanations_german_{mode}_{provider}_30/
  modes: no_shap, generic_rag, shaponly, fusion
  providers: anthropic, gemini

산출:
  results/german_eval.csv       — sample × mode × provider × metrics
  results/german_eval_summary.csv — mode × provider × metric mean ± std
  figures/40_german_4way.png    — 4-mode × 2 LLM 비교

실행:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.german_eval \\
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
import torch

from src.cross_llm_geval import judge_one as _judge_one
from src.eval_explanation import G_EVAL_RUBRIC  # noqa: F401
from src.llm_explainer import PROVIDER_DEFAULTS, make_client
from src.utils import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

GERMAN_DIR = PROJECT_ROOT / "data" / "german_credit"
PROCESSED_DIR = GERMAN_DIR / "processed"
NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# Premise builder (4-mode 모두 지원)
# ─────────────────────────────────────────────────────────────
def context_to_premise(ctx: Dict, max_len: int = 1500) -> str:
    decision = ctx.get("decision", "")
    prob = ctx.get("default_probability")
    decision_kr = "거절" if decision == "REJECT" else "승인"
    parts = [f"이 신청은 {decision_kr} 결정이며 부도 확률은 {prob:.1%}이다."]

    if "agreed_drivers" in ctx:
        # fusion
        for d in ctx.get("agreed_drivers", []):
            parts.append(_driver_sentence(d, "agreed"))
        for d in ctx.get("shap_only_drivers", []):
            parts.append(_driver_sentence(d, "shap_only"))
        for d in ctx.get("attention_only_drivers", []):
            parts.append(_driver_sentence(d, "attention_only"))
    elif "top_drivers_for_default" in ctx:
        # shaponly
        for d in ctx.get("top_drivers_for_default", []):
            d2 = dict(d); d2["sign_for_default"] = "+"
            parts.append(_driver_sentence(d2, None))
        for d in ctx.get("top_drivers_against_default", []):
            d2 = dict(d); d2["sign_for_default"] = "-"
            parts.append(_driver_sentence(d2, None))
    elif "customer_data" in ctx:
        # no_shap / generic_rag
        for k, v in ctx.get("customer_data", {}).items():
            parts.append(f"{k}: {v}.")
    text = " ".join(parts)
    if len(text) > max_len:
        text = text[:max_len]
    return text


def _driver_sentence(d: Dict, group: str) -> str:
    feat = d.get("feature", "")
    val = d.get("value", "")
    sign = d.get("sign_for_default")
    if sign == "+":
        direction = "부도 가능성을 높이는 요인"
    elif sign == "-":
        direction = "부도 가능성을 낮추는 요인"
    else:
        direction = "결정에 영향을 준 변수"
    if group == "agreed":
        prefix = "두 모델이 동의: "
    elif group == "shap_only":
        prefix = "SHAP 분석: "
    elif group == "attention_only":
        prefix = "TabNet 어텐션: "
    else:
        prefix = ""
    return f"{prefix}{feat}이(가) {val}로, {direction}이다."


# ─────────────────────────────────────────────────────────────
# 한국어 문장 분리 (nli_eval와 동일)
# ─────────────────────────────────────────────────────────────
def split_sentences(text: str) -> List[str]:
    sections = re.split(r"\n\n+", text)
    out = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        first_line = sec.split("\n", 1)[0].strip()
        if "개선 권고" in first_line or "면책 고지" in first_line:
            continue
        body = sec[len(first_line):].strip() if first_line.startswith("[") else sec
        # markdown 헤더(## …) 제거
        body = re.sub(r"^#+\s.*?$", "", body, flags=re.MULTILINE)
        lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
        for line in lines:
            line = re.sub(r"^[\-\d]+\.?\s*", "", line)
            line = re.sub(r"\*+", "", line)  # markdown bold 제거
            sents = re.split(r"(?<=[.!?])\s+", line)
            for s in sents:
                s = s.strip()
                if len(s) >= 6:
                    out.append(s)
    return out


# ─────────────────────────────────────────────────────────────
# NLI 모델
# ─────────────────────────────────────────────────────────────
def load_nli_model(model_name: str = NLI_MODEL_NAME):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f"[load] NLI model: {model_name} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def nli_score(tokenizer, model, premise: str, hypothesis: str) -> Dict[str, float]:
    inputs = tokenizer(premise, hypothesis, return_tensors="pt",
                        truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    id2label = model.config.id2label
    out = {}
    for i, p in enumerate(probs):
        lbl = id2label[i].lower()
        if "entail" in lbl:
            out["entailment"] = float(p)
        elif "contradict" in lbl:
            out["contradiction"] = float(p)
        else:
            out["neutral"] = float(p)
    return out


def evaluate_nli_one(tokenizer, model, ctx: Dict, text: str) -> Dict:
    premise = context_to_premise(ctx)
    sentences = split_sentences(text)
    if not sentences:
        return {"n_sentences": 0, "entailment_rate": None,
                "contradiction_rate": None}
    scores = [nli_score(tokenizer, model, premise, s) for s in sentences]
    df = pd.DataFrame(scores)
    return {
        "n_sentences": len(sentences),
        "entailment_rate": float(df["entailment"].mean()),
        "contradiction_rate": float(df["contradiction"].mean()),
        "neutral_rate": float(df["neutral"].mean()),
        "min_entailment": float(df["entailment"].min()),
        "n_strong_entailment": int((df["entailment"] > 0.5).sum()),
        "n_contradiction": int((df["contradiction"] > 0.5).sum()),
    }


# ─────────────────────────────────────────────────────────────
# Value match rate — 컨텍스트의 raw value가 explanation에 등장하는 비율
# ─────────────────────────────────────────────────────────────
def extract_context_values(ctx: Dict) -> List[str]:
    """4-mode 모두에서 'value' 필드를 추출."""
    values = []
    for key in ("agreed_drivers", "shap_only_drivers", "attention_only_drivers",
                 "top_drivers_for_default", "top_drivers_against_default"):
        for d in ctx.get(key, []):
            v = d.get("value")
            if v and isinstance(v, str):
                values.append(v)
    # customer_data
    cd = ctx.get("customer_data", {})
    if isinstance(cd, dict):
        for v in cd.values():
            if v and isinstance(v, str):
                values.append(str(v))
    return values


def value_match_rate(text: str, ctx: Dict) -> Dict:
    values = extract_context_values(ctx)
    if not values:
        return {"n_values": 0, "n_matched": 0, "val_match_rate": None}
    matched = 0
    for v in values:
        # value 안에 숫자가 있으면 숫자 부분만, 아니면 전체 문자열로 매칭
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", str(v))
        if nums:
            # 가장 큰 숫자 토큰 (e.g., "4,110 DM" → "4,110")
            biggest = max(nums, key=len)
            if biggest in text or biggest.replace(",", "") in text.replace(",", ""):
                matched += 1
        else:
            # 텍스트 토큰 그대로 검색 (대소문자 무시)
            if v.lower() in text.lower():
                matched += 1
    return {
        "n_values": len(values),
        "n_matched": matched,
        "val_match_rate": matched / len(values),
    }


# ─────────────────────────────────────────────────────────────
# Halluc rate — 영문 대문자 토큰 중 데이터셋 외 비율
# ─────────────────────────────────────────────────────────────
def load_all_features() -> set:
    df = pd.read_parquet(PROCESSED_DIR / "train_scaled.parquet")
    return set(df.columns) - {"TARGET"}


def halluc_rate(text: str, ctx: Dict, all_features: set) -> Dict:
    """German Credit halluc 룰.

    German feature명은 lowercase + 일부 underscore 포함 (e.g., 'credit_amount',
    'checking_status_no_checking'). 일반 영어 단어는 환각 후보에서 제외.

    raw_candidates 정의: underscore 포함 영문 토큰 (변수명 패턴) 또는
    대문자 시작 토큰 (전통적 환각 후보).
    """
    in_context_features = set()
    for key in ("agreed_drivers", "shap_only_drivers", "attention_only_drivers",
                 "top_drivers_for_default", "top_drivers_against_default"):
        for d in ctx.get(key, []):
            if "feature_raw" in d:
                in_context_features.add(d["feature_raw"])

    # underscore 포함 영문 토큰만 (일반 영어 단어 제외)
    underscore_tokens = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b", text))
    # 대문자 시작 토큰 (XGBoost feature 등) — Home Credit과 통일
    upper_tokens = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text))
    raw_candidates = underscore_tokens | upper_tokens

    common_excludes = {"SHAP", "REJECT", "APPROVE", "TARGET", "JSON", "API",
                        "AUC", "AUROC", "AUPRC", "OK", "AI", "LLM", "TabNet",
                        "DTI", "LTV", "DSR", "DM", "EUR", "USD", "MD",
                        "XGBoost", "GENDER", "UCI"}
    raw_candidates -= common_excludes

    # all_features는 기본 in_dataset
    raw_in_dataset = raw_candidates & all_features
    raw_in_ctx = raw_candidates & in_context_features
    raw_outside = raw_candidates - raw_in_dataset

    return {
        "n_raw_candidates": len(raw_candidates),
        "n_raw_outside_dataset": len(raw_outside),
        "halluc_rate_strict": len(raw_outside) / max(len(raw_candidates), 1),
    }


# ─────────────────────────────────────────────────────────────
# 디렉토리 평가
# ─────────────────────────────────────────────────────────────
def evaluate_directory(directory: Path, mode: str, provider: str,
                        all_features: set,
                        nli_tokenizer=None, nli_model=None,
                        geval_client=None, geval_model: str = None,
                        geval_sleep: float = 5.0,
                        skip_nli: bool = False,
                        skip_geval: bool = False) -> List[Dict]:
    rows = []
    files = sorted([p for p in directory.glob("*.json")
                     if p.name != "_index.json"])
    print(f"  [{mode}/{provider}] {len(files)}개 평가")

    for i, exp_path in enumerate(files, start=1):
        with open(exp_path, "r", encoding="utf-8") as f:
            exp = json.load(f)
        ctx = exp["context_sent"]
        text = exp["explanation"]

        row = {
            "sample_id": exp_path.stem,
            "mode": mode, "provider": provider,
            "decision": exp.get("decision"),
            "true_label": exp.get("true_label"),
            "default_proba": ctx.get("default_probability"),
        }
        # halluc + value match
        row.update(halluc_rate(text, ctx, all_features))
        row.update(value_match_rate(text, ctx))

        # NLI
        if not skip_nli and nli_tokenizer is not None:
            row.update(evaluate_nli_one(nli_tokenizer, nli_model, ctx, text))

        # G-Eval (Claude judge)
        if not skip_geval and geval_client is not None:
            ge = None
            last_err = None
            for attempt in range(3):
                try:
                    t0 = time.time()
                    ge = _judge_one(geval_client, geval_model, text, ctx)
                    row["geval_elapsed_sec"] = round(time.time() - t0, 2)
                    break
                except Exception as e:
                    last_err = str(e)
                    backoff = 30 * (2 ** attempt)
                    print(f"    G-Eval ERROR (a={attempt+1}/3): {last_err[:100]}")
                    if attempt < 2:
                        time.sleep(backoff)
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


# ─────────────────────────────────────────────────────────────
# Aggregate + plot
# ─────────────────────────────────────────────────────────────
METRIC_COLS = [
    "halluc_rate_strict", "val_match_rate",
    "entailment_rate", "contradiction_rate", "min_entailment",
    "geval_factual_accuracy", "geval_completeness",
    "geval_sensitive_leak", "geval_style",
]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (mode, provider), grp in df.groupby(["mode", "provider"]):
        for col in METRIC_COLS:
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


def plot_4way(summary: pd.DataFrame, name: str = "40_german_4way") -> Path:
    metrics = [
        ("entailment_rate", "NLI Entailment (↑)"),
        ("val_match_rate", "Value Match Rate (↑)"),
        ("halluc_rate_strict", "Halluc Rate (↓)"),
        ("geval_factual_accuracy", "G-Eval Factual (1-5, ↑)"),
        ("geval_completeness", "G-Eval Completeness (1-5, ↑)"),
        ("geval_sensitive_leak", "G-Eval Sensitive (1-5, ↑)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    mode_order = ["no_shap", "generic_rag", "shaponly", "fusion"]
    palette = {"no_shap": "#A0A0A0", "generic_rag": "#4C72B0",
               "shaponly": "#55A868", "fusion": "#DD8452"}

    for ax, (m, label) in zip(axes.flat, metrics):
        sub = summary[summary["metric"] == m]
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        sub = sub.copy()
        sub["mode_order"] = sub["mode"].map({m: i for i, m in enumerate(mode_order)})
        sub = sub.sort_values(["provider", "mode_order"])
        sns.barplot(data=sub, x="provider", y="mean", hue="mode",
                     hue_order=mode_order, ax=ax, errorbar=None, palette=palette)
        for _, r in sub.iterrows():
            providers = sorted(sub["provider"].unique())
            i = providers.index(r["provider"])
            mode_idx = mode_order.index(r["mode"])
            offset = -0.3 + mode_idx * 0.2
            ax.errorbar(i + offset, r["mean"], yerr=r["std"], fmt="none",
                        ecolor="black", capsize=3, lw=0.8)
        ax.set_title(label)
        ax.set_xlabel("LLM provider")
        ax.set_ylabel("")
        if ax.get_legend():
            ax.legend(loc="best", fontsize=8, ncol=2)
    plt.suptitle("UCI German Credit — 4-mode Comparison (Step 5-D)", fontsize=14, y=0.998)
    plt.tight_layout()
    return savefig(fig, name)


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

    all_features = load_all_features()
    print(f"[data] n_features={len(all_features)}")

    all_rows = []
    for mode in ["no_shap", "generic_rag", "shaponly", "fusion"]:
        for provider in ["anthropic", "gemini"]:
            directory = RESULTS_DIR / f"explanations_german_{mode}_{provider}_{n_samples}"
            if not directory.exists():
                print(f"[skip] {directory} 없음")
                continue
            rows = evaluate_directory(
                directory, mode, provider, all_features,
                nli_tokenizer=nli_tok, nli_model=nli_mod,
                geval_client=geval_client, geval_model=geval_model,
                geval_sleep=geval_sleep,
                skip_nli=skip_nli, skip_geval=skip_geval)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "german_eval.csv", index=False)
    print(f"[saved] results/german_eval.csv ({len(df)} rows)")

    summary = aggregate(df)
    summary.to_csv(RESULTS_DIR / "german_eval_summary.csv", index=False)

    # 보기용 pivot
    print("\n[summary mean]")
    pv = summary.pivot_table(index=["mode", "metric"], columns="provider",
                              values="mean").round(3)
    print(pv.to_string())

    plot_4way(summary)
    print("\n[OK] German Credit 4-mode evaluation 완료")


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
