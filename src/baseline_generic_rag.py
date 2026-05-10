"""Step 5-B: Generic RAG baseline — 일반 도메인 지식 chunks vs SHAP-RAG 비교.

동기:
    Step 1의 Counterfactual baseline은 raw 데이터만 (no-SHAP, hard constraint 약함)
    → Claude 45.5% 환각.
    그러나 약점 #3: "정보 적은 쪽이 환각하는 게 trivial 아닌가?"
    Generic RAG로 같은 정보량(raw features + 일반 도메인 지식 chunks + hard constraints)을
    주되 SHAP context만 빼고 비교 → "SHAP-RAG의 차별성"을 직접 입증.

3-way 비교:
    1. no-SHAP baseline (Step 1)         — raw 데이터, 자유 추론
    2. Generic RAG (Step 5-B, NEW)        — raw 데이터 + 도메인 지식 chunks + 동일 hard constraints
    3. SHAP-RAG (Step 1) / Fusion-RAG (Step 3-C-1) — SHAP context, hard constraints

Knowledge chunks (정적, 모든 인스턴스 동일):
    1. 신용평가 핵심 변수 정의
    2. 부도 위험의 일반 원리
    3. Threshold 의미
    4. 민감 변수 마스킹 정책
    5. 금융 용어 가이드
    6. Hard constraints
    7. 출력 형식

산출:
    results/contexts_generic_rag_30/{idx}_{tag}.json
    results/explanations_generic_rag_{anthropic,gemini}_30/

사용:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.baseline_generic_rag \
        --provider anthropic --n-samples 30
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.context_builder import DOMAIN_GLOSSARY, humanize_value
from src.llm_explainer import _call_llm, make_client, PROVIDER_DEFAULTS
from src.utils import RESULTS_DIR, SEED, set_seed

PROCESSED_DIR = Path("data/processed")
TARGET_COL = "TARGET"

# baseline_no_shap과 동일한 raw feature 셋 — fair comparison
RAW_FEATURES = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
]

# 민감 변수 마스킹 (SHAP-RAG와 동일 정책)
SENSITIVE_FEATURES = {"CODE_GENDER", "DAYS_BIRTH"}


# ─────────────────────────────────────────────────────────────
# Knowledge chunks — 일반 도메인 지식
# ─────────────────────────────────────────────────────────────
KNOWLEDGE_CHUNKS = [
    {
        "id": "K1",
        "title": "신용평가 핵심 변수의 의미",
        "content": (
            "신용 평가에서 자주 사용되는 변수의 일반적 의미: "
            "외부 신용평가 점수(EXT_SOURCE_1/2/3)는 외부 신용기관이 평가한 신용도로 "
            "0~1 범위이며 1에 가까울수록 신용이 양호하다. "
            "총 소득(AMT_INCOME_TOTAL)과 대출 신청 금액(AMT_CREDIT)은 평가의 기본 정보이며, "
            "연 환산 상환액(AMT_ANNUITY)은 매년 갚아야 할 금액으로 소득 대비 비율이 부담의 지표가 된다. "
            "재직 일수(DAYS_EMPLOYED, 음수)는 절대값이 클수록 재직 기간이 길어 안정성이 높다는 의미이다."
        ),
    },
    {
        "id": "K2",
        "title": "부도 위험의 일반 원리",
        "content": (
            "부도 위험은 일반적으로 다음 요인이 누적될 때 높아진다: "
            "(1) 외부 신용평가 점수가 낮을수록 위험. "
            "(2) 짧은 재직 기간은 직업 안정성 부족을 시사한다. "
            "(3) 소득 대비 대출 금액(AMT_CREDIT/AMT_INCOME_TOTAL) 비율이 높으면 상환 부담이 크다. "
            "(4) 거주지 등급(REGION_RATING_CLIENT)이 낮은 지역은 통계적으로 위험이 다소 높다. "
            "(5) 사회적 연결망 중 부도 이력자(DEF_*_CNT_SOCIAL_CIRCLE) 수도 위험 지표가 된다. "
            "반대로 높은 외부 신용평가 점수와 안정적인 장기 재직은 부도 위험을 낮춘다."
        ),
    },
    {
        "id": "K3",
        "title": "임계값(threshold)과 결정의 의미",
        "content": (
            "신용 평가 모델은 부도 확률(default probability)을 0~1 범위에서 출력하며, "
            "임계값(threshold)을 넘으면 거절(REJECT), 미만이면 승인(APPROVE)으로 결정한다. "
            "본 시스템에서 사용된 임계값은 약 0.4~0.5 범위이다. "
            "확률이 임계값에서 멀수록(매우 높거나 매우 낮을수록) 결정의 confidence가 높다. "
            "확률 자체는 점수일 뿐이며 절대적 부도 가능성을 의미하지는 않는다."
        ),
    },
    {
        "id": "K4",
        "title": "민감 변수 마스킹 정책",
        "content": (
            "본 시스템은 성별(CODE_GENDER), 연령(DAYS_BIRTH), 인종, 종교, 출신 지역 등 "
            "민감한 보호 속성을 사용자에게 보이는 설명에서 직접 언급하지 않는다. "
            "이런 변수는 모델 학습 단계에서는 사용될 수 있지만, 자연어 설명에서는 마스킹되어 "
            "출력되지 않는다. 설명 작성 시 절대로 성별·연령 등을 언급하지 말 것."
        ),
    },
    {
        "id": "K5",
        "title": "금융 용어 일반 가이드",
        "content": (
            "신용 평가 일반 용어: DTI(Debt-to-Income, 부채상환비율)는 소득 대비 부채 비율이고, "
            "LTV(Loan-to-Value, 담보인정비율)는 담보가치 대비 대출 비율이다. "
            "DSR(Debt Service Ratio)은 소득 대비 모든 대출 원리금 비율이다. "
            "한국에는 햇살론·미소금융·새희망홀씨 등 서민금융 상품이 있지만, "
            "본 평가는 일반 신용 대출에 대한 것이며 특정 상품 추천을 하지 않는다."
        ),
    },
    {
        "id": "K6",
        "title": "출력 형식과 Hard Constraints",
        "content": (
            "설명 작성 시 반드시 지킬 것: "
            "(1) [고객 데이터]에 명시된 변수와 값만 인용한다. 컨텍스트에 없는 변수, 수치, "
            "전화번호, 상품명을 만들어내지 않는다. "
            "(2) 의료·법률 자문, 단정적 미래 예측, 특정 금융 상품 추천을 하지 않는다. "
            "(3) 민감 변수(성별·연령·인종·종교·출신 지역)를 직접 언급하지 않는다. "
            "(4) 추측이나 상상에 기반한 진술 대신, 명시된 숫자를 인용해 설명한다. "
            "(5) 출력은 5개 섹션 순서: [결정 요약] / [주요 사유] / [긍정 요인] / "
            "[개선 권고] / [면책 고지]."
        ),
    },
]


# ─────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────
def build_generic_rag_context(idx: int, X_one: pd.DataFrame, proba: float,
                                 threshold: float, true_label: int) -> Dict:
    """Generic RAG context — raw features + 도메인 지식 chunks (SHAP 없음)."""
    decision = "REJECT" if proba >= threshold else "APPROVE"

    # raw features (민감 속성 마스킹)
    customer_data = {}
    for k in RAW_FEATURES:
        if k in X_one.columns and k not in SENSITIVE_FEATURES:
            v = float(X_one[k].iloc[0])
            kr = DOMAIN_GLOSSARY.get(k, k)
            customer_data[kr] = humanize_value(k, v)

    return {
        "sample_idx": idx,
        "default_probability": round(proba, 4),
        "threshold": round(threshold, 4),
        "decision": decision,
        "customer_data": customer_data,
        "knowledge_chunks": KNOWLEDGE_CHUNKS,
        "policy": "generic_rag_with_hard_constraints",
        "_meta_true_label": true_label,
    }


# ─────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────
GENERIC_RAG_PROMPT = """당신은 고객에게 신용 평가 결과를 친절히 설명하는 금융 상담사입니다.

본 시스템은 일반적인 신용 평가 도메인 지식을 [참조 지식]으로 제공합니다.
[참조 지식]은 일반 원리이며 특정 고객의 의사결정 근거가 아닙니다.
구체적인 의사결정 근거는 [고객 데이터]에서 직접 확인할 수 있는 값에 근거해 설명하세요.

[Hard Constraints — 반드시 준수]
- [고객 데이터]에 명시된 변수와 값만 인용. 컨텍스트에 없는 변수, 수치, 전화번호, 상품명 절대 생성 금지
- 의료·법률 자문, 단정적 미래 예측, 특정 금융 상품 추천 금지
- 민감 변수(성별·연령·인종·종교·출신 지역) 직접 언급 금지
- 추측이나 상상이 아닌, 명시된 숫자만 인용
- DTI/LTV/DSR/햇살론·미소금융 등 외부 용어를 만들어 인용하지 말 것 (참조 지식의 정의 외 사용 금지)

[참조 지식]
{knowledge_text}

[고객 데이터]
{customer_data_text}

[예측 결과]
- 부도 확률: {default_prob:.4f}
- 임계값: {threshold:.4f}
- 결정: {decision}

[출력 형식 — 반드시 이 5개 섹션 순서로]
[결정 요약]   (1줄)
[주요 거절 사유]   (REJECT일 때만, 최대 3개, 고객 데이터의 값 근거)
[긍정적으로 평가된 요인]   (최대 3개, 고객 데이터의 값 근거)
[개선 권고]   (1~3개)
[면책 고지]   (1~2줄)

이제 위 컨텍스트에 기반해 설명 리포트를 작성해주세요. 한국어로.
"""


def fmt_customer_data(cust: Dict[str, str]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in cust.items())


def fmt_knowledge_chunks(chunks: List[Dict]) -> str:
    return "\n\n".join(f"[{c['id']}] {c['title']}\n{c['content']}"
                        for c in chunks)


# ─────────────────────────────────────────────────────────────
# LLM 호출
# ─────────────────────────────────────────────────────────────
def generate_one(client_tuple, context: Dict, model: str) -> Dict:
    ctx_for_prompt = {k: v for k, v in context.items() if not k.startswith("_meta")}
    prompt = GENERIC_RAG_PROMPT.format(
        knowledge_text=fmt_knowledge_chunks(context["knowledge_chunks"]),
        customer_data_text=fmt_customer_data(context["customer_data"]),
        default_prob=context["default_probability"],
        threshold=context["threshold"],
        decision=context["decision"],
    )
    t0 = time.time()
    text, usage = _call_llm(client_tuple, prompt, model=model)
    elapsed = time.time() - t0
    return {
        "sample_idx": context["sample_idx"],
        "decision": context["decision"],
        "true_label": context.get("_meta_true_label"),
        "provider": client_tuple[0],
        "model": model,
        "elapsed_sec": round(elapsed, 2),
        "explanation": text,
        "usage_metadata": usage,
        "context_sent": ctx_for_prompt,
        "context_type": "generic_rag",
    }


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main(provider: str = "anthropic", n_samples: int = 30,
         sleep_sec: float = 4.0,
         shap_examples_path: Optional[Path] = None,
         output_dir: Optional[Path] = None,
         contexts_out_dir: Optional[Path] = None) -> None:
    set_seed(SEED)

    shap_examples_path = shap_examples_path or (
        RESULTS_DIR / "shap_local_examples_100.json")
    contexts_out_dir = contexts_out_dir or (
        RESULTS_DIR / "contexts_generic_rag_30")
    output_dir = output_dir or (
        RESULTS_DIR / f"explanations_generic_rag_{provider}_{n_samples}")
    contexts_out_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 같은 30 idx (fusion 결과 _index.json에서)
    fusion_idx_path = RESULTS_DIR / f"explanations_fusion_anthropic_{n_samples}" / "_index.json"
    if fusion_idx_path.exists():
        with open(fusion_idx_path, "r", encoding="utf-8") as f:
            target_idx = json.load(f)["selected_idx"]
        print(f"[match] {len(target_idx)} idx from fusion _index.json")
    else:
        # fallback: shap_examples에서 random 30
        with open(shap_examples_path, "r", encoding="utf-8") as f:
            shap_examples = json.load(f)
        rnd = random.Random(SEED)
        target_idx = [ex["idx"] for ex in
                       rnd.sample(shap_examples, min(n_samples, len(shap_examples)))]
        print(f"[fallback] random {len(target_idx)} idx from shap_examples")

    # SHAP examples로부터 인스턴스 정보 (predicted_proba, true_label)
    with open(shap_examples_path, "r", encoding="utf-8") as f:
        shap_examples = json.load(f)
    examples_by_idx = {ex["idx"]: ex for ex in shap_examples}

    # test_unscaled 로드 (raw features)
    test = pd.read_parquet(PROCESSED_DIR / "test_unscaled.parquet")

    # threshold (Day 2 XGBoost test threshold)
    threshold = 0.476

    # context build
    print(f"\n[1/3] {len(target_idx)}개 인스턴스 generic RAG context 생성")
    contexts = []
    for idx in sorted(target_idx):
        if idx not in examples_by_idx:
            print(f"  [warn] idx {idx} not in shap examples, skip")
            continue
        ex = examples_by_idx[idx]
        X_one = test.iloc[[idx]].drop(columns=[TARGET_COL], errors="ignore")
        ctx = build_generic_rag_context(
            idx, X_one, ex["predicted_proba"], threshold, ex["true_label"])
        out = contexts_out_dir / f"{idx}_{ex['tag']}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(ctx, f, indent=2, ensure_ascii=False)
        contexts.append((out.stem, ctx))
    print(f"  saved {len(contexts)} contexts to {contexts_out_dir}")

    # _index 저장
    with open(contexts_out_dir / "_index.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_contexts": len(contexts),
            "selected_idx": [int(stem.split("_")[0]) for stem, _ in contexts],
            "n_knowledge_chunks": len(KNOWLEDGE_CHUNKS),
            "policy": "generic_rag_with_hard_constraints",
        }, f, indent=2, ensure_ascii=False)

    # LLM 호출
    print(f"\n[2/3] LLM 호출 (provider={provider})")
    model = PROVIDER_DEFAULTS[provider]["model"]
    client_tuple = make_client(provider)
    print(f"  model={model}, output={output_dir}")

    out_paths = []
    for i, (stem, ctx) in enumerate(contexts, start=1):
        print(f"[{i}/{len(contexts)}] {stem}  decision={ctx['decision']}  prob={ctx['default_probability']:.3f}")
        try:
            result = generate_one(client_tuple, ctx, model=model)
        except Exception as e:
            print(f"  ERROR: {str(e)[:120]}, 60초 대기 후 재시도")
            time.sleep(60)
            result = generate_one(client_tuple, ctx, model=model)

        out = output_dir / f"{stem}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        out_paths.append(out)
        print(f"  elapsed={result['elapsed_sec']}s, "
              f"tokens={result['usage_metadata'].get('total_token_count', '?')}")
        if i < len(contexts):
            time.sleep(sleep_sec)

    # _index
    with open(output_dir / "_index.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_explanations": len(out_paths),
            "provider": provider, "model": model,
            "context_type": "generic_rag",
            "files": [p.name for p in out_paths],
            "selected_idx": [int(p.stem.split("_")[0]) for p in out_paths],
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[3/3] 완료 — {len(out_paths)}개 generic RAG 설명 생성")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "gemini"])
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--sleep", type=float, default=4.0)
    args = ap.parse_args()
    main(provider=args.provider, n_samples=args.n_samples,
          sleep_sec=args.sleep)
