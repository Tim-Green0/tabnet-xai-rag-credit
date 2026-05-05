"""Step 3-C-2: NLI 기반 Faithfulness 평가.

룰 기반 평가의 sign_match 한계(LLM의 다양한 표현을 키워드 셋으로 못 잡음)를
Natural Language Inference 모델로 보강.

원리:
    Premise = LLM에 주어진 컨텍스트 facts를 자연어로 합친 것
    Hypothesis = LLM이 생성한 설명의 각 문장
    NLI 모델이 (premise, hypothesis) 쌍에 대해 entailment/neutral/contradiction
    확률 분포 출력 → entailment_rate를 instance-level faithfulness로 사용.

NLI 모델: Huffon/klue-roberta-base-nli (한국어 NLI 전용, KLUE-NLI fine-tuned)

산출:
    results/nli_eval.csv          — sample × mode × provider × NLI 점수
    results/nli_summary.csv       — mode × provider × {entailment_rate, contradiction_rate}
    figures/31_nli_vs_rules.png   — 룰/G-Eval/NLI 3-tier 비교

사용:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.nli_eval [--dry-run]
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

from src.utils import FIGURES_DIR, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# KLUE-NLI label 매핑 (모델 config에서 가져오는 게 확실하지만 보통 0=ENT, 1=NEU, 2=CON)
LABEL_ORDER = ["entailment", "neutral", "contradiction"]


# ─────────────────────────────────────────────────────────────
# 컨텍스트 → premise 자연어 변환
# ─────────────────────────────────────────────────────────────
def _driver_to_sentence(d: Dict, group: str = None) -> str:
    """driver 정보를 자연어 문장으로. SHAP 부호 → 방향 표현."""
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


def context_to_premise(ctx: Dict, max_len: int = 1500) -> str:
    """컨텍스트 JSON을 NLI premise용 자연어 단락으로 변환.

    Fusion 컨텍스트와 SHAP-only 컨텍스트 모두 지원.
    너무 길면 max_len에서 자름.
    """
    decision = ctx.get("decision", "")
    prob = ctx.get("default_probability")
    decision_kr = "거절" if decision == "REJECT" else "승인" if decision == "APPROVE" else decision
    parts = [f"이 신청은 {decision_kr} 결정이며 부도 확률은 {prob:.1%}이다."]

    # fusion 컨텍스트
    if "agreed_drivers" in ctx:
        for d in ctx.get("agreed_drivers", []):
            parts.append(_driver_to_sentence(d, group="agreed"))
        for d in ctx.get("shap_only_drivers", []):
            parts.append(_driver_to_sentence(d, group="shap_only"))
        for d in ctx.get("attention_only_drivers", []):
            parts.append(_driver_to_sentence(d, group="attention_only"))
    else:
        # SHAP-only 컨텍스트 (Step 1/2-A 형식)
        for d in ctx.get("top_drivers_for_default", []):
            d2 = dict(d)
            d2["sign_for_default"] = "+"
            parts.append(_driver_to_sentence(d2))
        for d in ctx.get("top_drivers_against_default", []):
            d2 = dict(d)
            d2["sign_for_default"] = "-"
            parts.append(_driver_to_sentence(d2))

    text = " ".join(parts)
    if len(text) > max_len:
        text = text[:max_len]
    return text


# ─────────────────────────────────────────────────────────────
# 한국어 문장 분리
# ─────────────────────────────────────────────────────────────
SECTION_HEADERS = re.compile(
    r"\[(결정 요약|주요 거절 사유|주요 승인 사유|주요 거절/승인 사유.*?|"
    r"긍정적으로 평가된 요인|보완적으로 고려된 요인|개선 권고|면책 고지|.*?신호.*?)\]"
)


def split_sentences(text: str) -> List[str]:
    """LLM 출력에서 평가 대상 문장만 추출.

    [개선 권고]와 [면책 고지] 섹션은 NLI 평가 제외 (조언이라 entailment 의미 약함).
    """
    # 섹션별 분리
    sections = re.split(r"\n\n+", text)
    out = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        # 헤더 라인 추출
        first_line = sec.split("\n", 1)[0].strip()
        if "개선 권고" in first_line or "면책 고지" in first_line:
            continue  # 평가 제외 (advice/disclaimer)
        # 헤더 라인 자체는 제거
        body = sec[len(first_line):].strip() if first_line.startswith("[") else sec
        # 줄 단위 + 마침표 split
        lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
        for line in lines:
            # 번호/대시 prefix 제거
            line = re.sub(r"^[\-\d]+\.?\s*", "", line)
            # 한 줄에 여러 문장 있으면 마침표로 split
            sents = re.split(r"(?<=[.!?])\s+", line)
            for s in sents:
                s = s.strip()
                # 너무 짧은 토막 제거
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
    # label 매핑 확인
    id2label = model.config.id2label
    print(f"     id2label: {id2label}")
    return tokenizer, model


@torch.no_grad()
def nli_score(tokenizer, model, premise: str, hypothesis: str) -> Dict[str, float]:
    """단일 (premise, hypothesis) 쌍 → entailment/neutral/contradiction 확률."""
    inputs = tokenizer(premise, hypothesis, return_tensors="pt",
                        truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    # label index 매핑
    id2label = model.config.id2label
    out = {}
    for i, p in enumerate(probs):
        # config의 label을 우리 표준화된 이름으로
        lbl = id2label[i].lower()
        if "entail" in lbl:
            out["entailment"] = float(p)
        elif "contradict" in lbl:
            out["contradiction"] = float(p)
        else:
            out["neutral"] = float(p)
    return out


# ─────────────────────────────────────────────────────────────
# 평가 루프
# ─────────────────────────────────────────────────────────────
def evaluate_explanation(tokenizer, model,
                          context: Dict, explanation_text: str) -> Dict:
    """단일 LLM 설명을 NLI로 평가.

    각 hypothesis 문장에 대해 entailment 확률 계산 → 평균.
    """
    premise = context_to_premise(context)
    sentences = split_sentences(explanation_text)
    if not sentences:
        return {"n_sentences": 0, "entailment_rate": None,
                "contradiction_rate": None, "neutral_rate": None,
                "min_entailment": None}

    scores = [nli_score(tokenizer, model, premise, s) for s in sentences]
    df_s = pd.DataFrame(scores)
    return {
        "n_sentences": len(sentences),
        "entailment_rate": float(df_s["entailment"].mean()),
        "contradiction_rate": float(df_s["contradiction"].mean()),
        "neutral_rate": float(df_s["neutral"].mean()),
        "min_entailment": float(df_s["entailment"].min()),  # 최악 문장
        "n_strong_entailment": int((df_s["entailment"] > 0.5).sum()),
        "n_contradiction": int((df_s["contradiction"] > 0.5).sum()),
    }


def evaluate_directory(directory: Path, mode: str, provider: str,
                        tokenizer, model,
                        target_idx: List[int] = None) -> List[Dict]:
    rows = []
    files = sorted([p for p in directory.glob("*.json")
                     if p.name != "_index.json"])
    if target_idx is not None:
        target_set = set(target_idx)
        files = [p for p in files if int(p.stem.split("_")[0]) in target_set]
    print(f"  [{mode}/{provider}] {len(files)}개 평가")

    for i, exp_path in enumerate(files, start=1):
        with open(exp_path, "r", encoding="utf-8") as f:
            exp = json.load(f)
        ctx = exp["context_sent"]
        text = exp["explanation"]
        nli_out = evaluate_explanation(tokenizer, model, ctx, text)
        rows.append({
            "sample_id": exp_path.stem,
            "mode": mode, "provider": provider,
            "decision": exp.get("decision"),
            **nli_out,
        })
        if i % 5 == 0:
            print(f"    [{i}/{len(files)}] entailment_rate "
                  f"running mean={np.mean([r['entailment_rate'] for r in rows if r['entailment_rate'] is not None]):.3f}")
    return rows


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (mode, provider), grp in df.groupby(["mode", "provider"]):
        for col in ["entailment_rate", "contradiction_rate", "neutral_rate",
                     "min_entailment"]:
            vals = grp[col].dropna()
            if len(vals) > 0:
                out.append({
                    "mode": mode, "provider": provider, "metric": col,
                    "mean": float(vals.mean()),
                    "std": float(vals.std()) if len(vals) > 1 else 0.0,
                    "n": int(len(vals)),
                })
    return pd.DataFrame(out)


def plot_nli_vs_rules(nli_summary: pd.DataFrame,
                       fusion_summary_path: Path = None,
                       out_name: str = "31_nli_vs_rules") -> Path:
    """NLI entailment_rate를 룰/G-Eval과 비교 시각화."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # NLI entailment
    ax = axes[0]
    sub = nli_summary[nli_summary["metric"] == "entailment_rate"]
    sns.barplot(data=sub, x="provider", y="mean", hue="mode", ax=ax,
                 errorbar=None, palette={"shaponly": "#A0A0A0", "fusion": "#DD8452"})
    for _, r in sub.iterrows():
        i = list(sub["provider"].unique()).index(r["provider"])
        x = i + (-0.2 if r["mode"] == "shaponly" else 0.2)
        ax.errorbar(x, r["mean"], yerr=r["std"], fmt="none", ecolor="black", capsize=4)
    ax.set_title("NLI Entailment Rate (↑ better, KLUE-NLI)")
    ax.set_ylabel("entailment_rate")
    ax.set_xlabel("LLM provider")

    # NLI contradiction
    ax = axes[1]
    sub = nli_summary[nli_summary["metric"] == "contradiction_rate"]
    sns.barplot(data=sub, x="provider", y="mean", hue="mode", ax=ax,
                 errorbar=None, palette={"shaponly": "#A0A0A0", "fusion": "#DD8452"})
    ax.set_title("NLI Contradiction Rate (↓ better)")
    ax.set_ylabel("contradiction_rate")
    ax.set_xlabel("LLM provider")
    if ax.legend_:
        ax.legend_.remove()

    # 룰 sign_match (fusion 결과 csv에서)
    ax = axes[2]
    if fusion_summary_path is not None and fusion_summary_path.exists():
        fs = pd.read_csv(fusion_summary_path)
        sub = fs[fs["metric"] == "sign_match_rate"]
        sns.barplot(data=sub, x="provider", y="mean", hue="mode", ax=ax,
                     errorbar=None, palette={"shaponly": "#A0A0A0", "fusion": "#DD8452"})
        ax.set_title("Rule sign_match (kw-based, ↓ in fusion = limit)")
        ax.set_ylabel("sign_match_rate")
        ax.set_xlabel("LLM provider")
        if ax.legend_:
            ax.legend_.remove()

    plt.suptitle("3-tier 평가 — NLI Entailment / Contradiction / Rule sign_match")
    return savefig(fig, out_name)


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main(dry_run: bool = False, n_samples: int = 30) -> None:
    set_seed(SEED)
    print(f"[device] {DEVICE}")
    tokenizer, model = load_nli_model()

    # 같은 30개 idx (fusion 결과 _index.json에서)
    fusion_idx_path = RESULTS_DIR / f"explanations_fusion_anthropic_{n_samples}" / "_index.json"
    if not fusion_idx_path.exists():
        raise FileNotFoundError(f"fusion 결과 인덱스 없음: {fusion_idx_path}")
    with open(fusion_idx_path, "r", encoding="utf-8") as f:
        fusion_idx = json.load(f)["selected_idx"]

    if dry_run:
        fusion_idx = fusion_idx[:2]
        print(f"[dry-run] {len(fusion_idx)} 인스턴스만")

    plan = [
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
        rows.extend(evaluate_directory(directory, mode, provider,
                                        tokenizer, model,
                                        target_idx=fusion_idx))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "nli_eval.csv", index=False)
    print(f"\n[save] results/nli_eval.csv ({len(df)} rows)")

    summary = aggregate(df)
    summary.to_csv(RESULTS_DIR / "nli_summary.csv", index=False)
    print(f"[save] results/nli_summary.csv")

    # 비교 print
    print("\n[delta] NLI fusion - shaponly:")
    pivot = summary.pivot_table(index=["provider", "metric"], columns="mode",
                                  values="mean").reset_index()
    if "fusion" in pivot.columns and "shaponly" in pivot.columns:
        pivot["delta"] = pivot["fusion"] - pivot["shaponly"]
        print(pivot.to_string(index=False))

    plot_nli_vs_rules(summary, RESULTS_DIR / "fusion_vs_shaponly.csv")
    print(f"[save] figures/31_nli_vs_rules.png")
    print("\n[OK] NLI 평가 완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-samples", type=int, default=30)
    args = ap.parse_args()
    main(dry_run=args.dry_run, n_samples=args.n_samples)
