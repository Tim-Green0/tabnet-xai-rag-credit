"""Day 8 — Counterfactual baseline: SHAP 컨텍스트 없이 LLM 직접 호출.

목적 (계획서 RQ3):
  본 연구의 XAI-RAG vs 일반 LLM 직접 호출 비교.
  SHAP 사실 컨텍스트 없을 때 LLM의 환각률 측정 → 본 구조의 효과 입증.

설계:
  - 동일 10 샘플의 raw feature 값(핵심 변수만) → LLM에 거절 사유 작성 요청
  - SHAP 정보, top driver, fact_only 정책 없음 — LLM이 자유롭게 추론
  - 결과 → results/explanations_baseline_noshap_{provider}/
  - 평가: 동일한 Faithfulness/Hallucination 룰 기반 평가 적용

산출:
  - results/explanations_baseline_noshap_gemini/*.json
  - results/explanations_baseline_noshap_anthropic/*.json
  - results/baseline_comparison.csv (XAI-RAG vs no-SHAP 비교)
  - figures/22_baseline_vs_xairag.png

실행:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.baseline_no_shap
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from src.context_builder import DOMAIN_GLOSSARY, humanize_value
from src.eval_explanation import (
    extract_drivers_from_context,
    faithfulness_per_driver,
    faithfulness_score,
    hallucination_rate,
    load_all_feature_names,
)
from src.llm_explainer import PROVIDER_DEFAULTS, _call_llm, make_client
from src.metrics import find_threshold_youden
from src.utils import FIGURES_DIR, RESULTS_DIR, savefig

load_dotenv("D:/paper/.env", override=True)
sns.set_theme(style="whitegrid", context="notebook")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = RESULTS_DIR / "baseline_models"
TARGET_COL = "TARGET"


# 사용할 핵심 raw 변수 (SHAP 없이도 LLM이 받을 정형 정보)
DISPLAY_FEATURES = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
]


# baseline 프롬프트 — SHAP 없이 raw 데이터만 + 자유 추론 허용
BASELINE_PROMPT_TEMPLATE = """당신은 신용 평가 결과를 고객에게 설명하는 금융 상담사입니다.
아래 [고객 데이터]에 기반하여 [예측 결과]에 대한 자연어 설명 리포트를 작성해주세요.

[예측 결과]
- 부도 확률: {default_prob:.4f}
- 결정: {decision}

[고객 데이터]
{customer_data_text}

[출력 형식]
1. 결정 요약 (1줄)
2. 주요 거절 사유 Top 3 (REJECT인 경우)
3. 긍정적으로 평가된 요인 (최대 3개)
4. 개선 권고
5. 면책 고지

위 형식대로 한국어로 작성해주세요.
"""


def build_baseline_context(idx: int, X_one: pd.DataFrame, proba: float,
                              decision: str, true_label: int) -> Dict:
    """SHAP 없는 baseline context — raw value만 한국어로."""
    customer_data = {}
    for k in DISPLAY_FEATURES:
        if k in X_one.columns:
            v = float(X_one[k].iloc[0])
            kr = DOMAIN_GLOSSARY.get(k, k)
            customer_data[kr] = humanize_value(k, v)

    return {
        "sample_idx": idx,
        "default_probability": round(proba, 4),
        "decision": decision,
        "customer_data": customer_data,
        "true_label": true_label,
        "policy": "no_shap_free_inference",
    }


def fmt_customer_data(cust: Dict[str, str]) -> str:
    lines = []
    for k, v in cust.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def generate_baseline(provider: str, baseline_ctx: Dict) -> Dict:
    model = PROVIDER_DEFAULTS[provider]["model"]
    prompt = BASELINE_PROMPT_TEMPLATE.format(
        default_prob=baseline_ctx["default_probability"],
        decision=baseline_ctx["decision"],
        customer_data_text=fmt_customer_data(baseline_ctx["customer_data"]),
    )
    client_tuple = make_client(provider)
    t0 = time.time()
    text, usage = _call_llm(client_tuple, prompt, model=model)
    elapsed = time.time() - t0
    return {
        "sample_idx": baseline_ctx["sample_idx"],
        "decision": baseline_ctx["decision"],
        "true_label": baseline_ctx["true_label"],
        "provider": provider,
        "model": model,
        "elapsed_sec": round(elapsed, 2),
        "explanation": text,
        "usage_metadata": usage,
        "context_sent": baseline_ctx,
    }


def evaluate_baseline_outputs(out_dir: Path, all_features: set) -> List[Dict]:
    """baseline outputs에 대해 hallucination + (간이) faithfulness 측정.

    참고: SHAP 없는 baseline에서는 'driver'가 없으므로
    Faithfulness 룰 기반은 의미가 작다.
    핵심은 Hallucination Rate — 텍스트의 변수 토큰이 실제 데이터셋에 있는가.
    또한 customer_data에 명시된 변수와 비교.
    """
    rows = []
    for fp in sorted(out_dir.glob("*.json")):
        if fp.name == "_index.json":
            continue
        ex = json.loads(fp.read_text(encoding="utf-8"))
        ctx = ex["context_sent"]
        text = ex["explanation"]
        # customer_data의 변수만 driver처럼 취급
        pseudo_drivers = []
        for kr, val in ctx["customer_data"].items():
            # raw key 추정: DOMAIN_GLOSSARY에서 역으로
            raw = next((k for k, v in DOMAIN_GLOSSARY.items() if v == kr), kr)
            pseudo_drivers.append({
                "feature": kr, "feature_raw": raw,
                "value": val, "value_raw": val, "shap": 0.0,
            })

        # hallucination_rate는 컨텍스트 = customer_data 기준
        # 다만 본 baseline은 SHAP 컨텍스트가 없어 broad는 의미 다름.
        # 그래서 여기서는 strict (데이터셋에 아예 없는 변수 만들어낸 비율) 위주 측정.
        candidates = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text))
        candidates -= {"SHAP", "REJECT", "APPROVE", "TARGET", "JSON",
                        "API", "AUC", "OK", "AI", "LLM"}
        outside_dataset = candidates - all_features

        rows.append({
            "sample_id": fp.stem,
            "decision": ex.get("decision"),
            "true_label": ex.get("true_label"),
            "provider": ex.get("provider"),
            "n_raw_candidates": len(candidates),
            "n_outside_dataset": len(outside_dataset),
            "halluc_rate_strict": (len(outside_dataset)
                                     / max(len(candidates), 1)),
            "outside_dataset_list": sorted(list(outside_dataset)),
        })
    return rows


def main():
    print("[1/4] 기존 컨텍스트 idx 추출 (10명)")
    contexts_dir = RESULTS_DIR / "contexts"
    sample_ids = []
    for fp in sorted(contexts_dir.glob("*.json")):
        if fp.name == "_index.json":
            continue
        ex = json.loads(fp.read_text(encoding="utf-8"))
        sample_ids.append((ex["sample_idx"], fp.stem))

    print(f"  대상: {len(sample_ids)}개 샘플")

    print("[2/4] 데이터·모델 로딩 + threshold")
    test = pd.read_parquet(PROCESSED_DIR / "test_unscaled.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val_unscaled.parquet")
    y_test = test[TARGET_COL].astype(int).values
    X_test = test.drop(columns=[TARGET_COL])
    y_val = val[TARGET_COL].astype(int).values
    X_val = val.drop(columns=[TARGET_COL])
    xgb_model = joblib.load(MODELS_DIR / "xgboost.pkl")
    val_score = xgb_model.predict_proba(X_val)[:, 1]
    thr, _ = find_threshold_youden(y_val, val_score)
    print(f"  threshold = {thr:.4f}")

    print("[3/4] baseline LLM 호출 — Gemini + Claude")
    for prov in ["gemini", "anthropic"]:
        out_dir = RESULTS_DIR / f"explanations_baseline_noshap_{prov}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [provider={prov}]")
        for i, (idx, tag_full) in enumerate(sample_ids, start=1):
            X_one = X_test.iloc[[idx]]
            proba = float(xgb_model.predict_proba(X_one)[0, 1])
            decision = "REJECT" if proba >= thr else "APPROVE"
            ctx = build_baseline_context(idx, X_one, proba, decision,
                                            int(y_test[idx]))
            try:
                res = generate_baseline(prov, ctx)
            except Exception as e:
                print(f"    [{i}/{len(sample_ids)}] {tag_full} ERROR: {e}")
                time.sleep(60)
                res = generate_baseline(prov, ctx)
            with open(out_dir / f"{tag_full}.json", "w",
                       encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            print(f"    [{i}/{len(sample_ids)}] {tag_full}  "
                  f"elapsed={res['elapsed_sec']}s, tokens={res['usage_metadata'].get('total_token_count', '?')}")
            time.sleep(2)

    print("[4/4] 환각률 평가 + XAI-RAG 비교")
    all_features = load_all_feature_names()

    cmp_rows = []
    for prov in ["gemini", "anthropic"]:
        baseline_dir = RESULTS_DIR / f"explanations_baseline_noshap_{prov}"
        rows_b = evaluate_baseline_outputs(baseline_dir, all_features)

        # XAI-RAG 결과 로딩
        if prov == "gemini":
            xai_summary = json.loads(
                (RESULTS_DIR / "explanation_eval_summary_gemini.json").read_text(encoding="utf-8")
            )
        else:
            xai_summary = json.loads(
                (RESULTS_DIR / "explanation_eval_summary_anthropic.json").read_text(encoding="utf-8")
            )

        b_df = pd.DataFrame(rows_b)
        b_df.to_csv(RESULTS_DIR / f"baseline_noshap_eval_{prov}.csv", index=False)

        # 통계
        b_strict_mean = float(b_df["halluc_rate_strict"].mean())
        b_strict_std = float(b_df["halluc_rate_strict"].std())
        b_outside_total = int(b_df["n_outside_dataset"].sum())
        b_total_candidates = int(b_df["n_raw_candidates"].sum())

        x_strict = xai_summary.get("halluc_rate_strict", {})
        x_strict_mean = x_strict.get("mean", 0.0)
        x_strict_std = x_strict.get("std", 0.0)

        cmp_rows.append({
            "provider": prov,
            "xai_rag_halluc_strict_mean": x_strict_mean,
            "xai_rag_halluc_strict_std": x_strict_std,
            "baseline_halluc_strict_mean": b_strict_mean,
            "baseline_halluc_strict_std": b_strict_std,
            "baseline_outside_dataset_total": b_outside_total,
            "baseline_n_candidates_total": b_total_candidates,
        })

    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(RESULTS_DIR / "baseline_comparison.csv", index=False)
    print("\n[비교 표]")
    print(cmp_df.round(4).to_string(index=False))

    # 시각화
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
    out = savefig(fig, "23_baseline_vs_xairag")
    print(f"\n[OK] baseline 비교 완료")
    print(f"     - results/baseline_comparison.csv")
    print(f"     - results/baseline_noshap_eval_{{gemini,anthropic}}.csv")
    print(f"     - results/explanations_baseline_noshap_{{gemini,anthropic}}/")
    print(f"     - {out}")


if __name__ == "__main__":
    main()
