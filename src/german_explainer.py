"""UCI German Credit — 4-mode LLM 설명 생성 (Step 5-D Day 3).

4 modes (Home Credit Step 5-B와 동일):
  1. no_shap     : raw features only, 자유 추론 (counterfactual baseline)
  2. generic_rag : raw + 일반 도메인 지식 chunks + hard constraints (SHAP 없음)
  3. shaponly    : SHAP top-5 + raw values + hard constraints
  4. fusion      : SHAP + TabNet attention agreement-aware (본 연구 메커니즘)

Sensitive features (마스킹):
  - age, personal_status_*, GENDER_*, foreign_worker_*

산출:
  results/contexts_german_{mode}_30/{idx}_{tag}.json
  results/explanations_german_{mode}_{provider}_30/{idx}_{tag}.json

실행:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.german_explainer \\
        --mode fusion --provider anthropic --n-samples 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from src.llm_explainer import _call_llm, make_client, PROVIDER_DEFAULTS
from src.utils import PROJECT_ROOT, RESULTS_DIR, SEED, set_seed

GERMAN_DIR = PROJECT_ROOT / "data" / "german_credit"
PROCESSED_DIR = GERMAN_DIR / "processed"
MODELS_DIR = RESULTS_DIR / "baseline_models"
TARGET_COL = "TARGET"

# ─────────────────────────────────────────────────────────────
# German Credit 도메인 매핑
# ─────────────────────────────────────────────────────────────
SENSITIVE_RAW = {"age", "personal_status", "GENDER", "foreign_worker"}

# Sanitized one-hot 컬럼명에서 SENSITIVE 검출용 prefix
SENSITIVE_ONEHOT_PREFIXES = ("personal_status_", "GENDER_", "foreign_worker_")

# Numeric/bool 컬럼 한국어 라벨
GERMAN_GLOSSARY = {
    "duration": "대출 기간(개월)",
    "credit_amount": "대출 금액(DM)",
    "installment_commitment": "분할 상환 비율(%)",
    "residence_since": "현 거주지 거주 기간(년)",
    "age": "나이",  # SENSITIVE
    "existing_credits": "기존 대출 건수",
    "num_dependents": "부양 가족 수",
}

# One-hot 카테고리 prefix 한국어 라벨
GERMAN_CATEGORY_LABELS = {
    "checking_status": "체크 계좌 상태",
    "credit_history": "신용 이력",
    "purpose": "대출 목적",
    "savings_status": "저축 상태",
    "employment": "재직 기간",
    "personal_status": "개인 신상",  # SENSITIVE
    "other_parties": "보증인/공동신청자",
    "property_magnitude": "보유 자산",
    "other_payment_plans": "기타 지불 계획",
    "housing": "주거 형태",
    "job": "직업 분류",
    "own_telephone": "전화 보유",
    "foreign_worker": "외국인 노동자",  # SENSITIVE
    "GENDER": "성별",  # SENSITIVE
}

# 카테고리 값 한국어 (sanitized 후 컬럼명 suffix 기준)
GERMAN_VALUE_LABELS = {
    # checking_status (sanitized: "_0", "0_=X_200", "_=200", "no_checking")
    "checking_status__0": "잔액 < 0 DM",
    "checking_status_0_=X_200": "0 ≤ 잔액 < 200 DM",
    "checking_status__=200": "잔액 ≥ 200 DM",
    "checking_status_no_checking": "체크 계좌 없음",
    # credit_history
    "credit_history_no_credits/all_paid": "기존 대출 없거나 모두 상환",
    "credit_history_all_paid": "모든 대출 상환 완료",
    "credit_history_existing_paid": "기존 대출 정상 상환 중",
    "credit_history_delayed_previously": "과거 연체 이력",
    "credit_history_critical/other_existing_credit": "신용 이력 critical (위험)",
    # savings_status (sanitized: "_100", "100_=X_500", "500_=X_1000", "_=1000", "no_known_savings")
    "savings_status__100": "저축 < 100 DM",
    "savings_status_100_=X_500": "100 ≤ 저축 < 500 DM",
    "savings_status_500_=X_1000": "500 ≤ 저축 < 1000 DM",
    "savings_status__=1000": "저축 ≥ 1000 DM",
    "savings_status_no_known_savings": "저축 정보 없음",
    # employment (sanitized: "_1", "1_=X_4", "4_=X_7", "_=7", "unemployed")
    "employment__1": "재직 < 1년",
    "employment_1_=X_4": "재직 1~4년",
    "employment_4_=X_7": "재직 4~7년",
    "employment__=7": "재직 ≥ 7년",
    "employment_unemployed": "실업 상태",
    # housing
    "housing_own": "자가",
    "housing_rent": "임대",
    "housing_for_free": "무상 거주",
    # other_payment_plans
    "other_payment_plans_none": "기타 지불 계획 없음",
    "other_payment_plans_bank": "은행 분할 지불",
    "other_payment_plans_stores": "상점 분할 지불",
    # job
    "job_unemp/unskilled_non_res": "실업/비숙련(비거주자)",
    "job_unskilled_resident": "비숙련(거주자)",
    "job_skilled": "숙련직",
    "job_high_qualif/self_emp/mgmt": "고급기술/자영업/관리직",
    # purpose
    "purpose_radio/tv": "라디오/TV",
    "purpose_education": "교육",
    "purpose_furniture/equipment": "가구/장비",
    "purpose_new_car": "신차",
    "purpose_used_car": "중고차",
    "purpose_business": "사업",
    "purpose_domestic_appliance": "가전",
    "purpose_repairs": "수리",
    "purpose_other": "기타",
    "purpose_retraining": "재교육",
    # property_magnitude
    "property_magnitude_real_estate": "부동산",
    "property_magnitude_life_insurance": "생명보험",
    "property_magnitude_car": "자동차",
    "property_magnitude_no_known_property": "보유 자산 없음",
    # other_parties
    "other_parties_none": "보증인 없음",
    "other_parties_co_applicant": "공동 신청자",
    "other_parties_guarantor": "보증인 있음",
    # own_telephone
    "own_telephone_yes": "전화 보유",
    "own_telephone_none": "전화 없음",
}


def humanize_feature(col: str) -> str:
    """Sanitized 컬럼명 → 한국어 라벨."""
    if col in GERMAN_VALUE_LABELS:
        return GERMAN_VALUE_LABELS[col]
    if col in GERMAN_GLOSSARY:
        return GERMAN_GLOSSARY[col]
    # 일반 fallback
    for prefix, label in GERMAN_CATEGORY_LABELS.items():
        if col.startswith(prefix + "_"):
            suffix = col[len(prefix) + 1:]
            return f"{label}: {suffix}"
    return col


def humanize_value(col: str, val) -> str:
    """one-hot은 활성/비활성으로, numeric은 단위 포함."""
    if col in GERMAN_VALUE_LABELS:
        # one-hot 컬럼 — 활성 여부
        return "해당" if float(val) > 0.5 else "비해당"
    if col == "duration":
        return f"{int(val)}개월"
    if col == "credit_amount":
        return f"{int(val):,} DM"
    if col == "installment_commitment":
        return f"{val}%"
    if col == "residence_since":
        return f"{int(val)}년"
    if col == "age":
        return f"{int(val)}세"
    if col in {"existing_credits", "num_dependents"}:
        return f"{int(val)}건" if col == "existing_credits" else f"{int(val)}명"
    return str(val)


def is_sensitive(col: str) -> bool:
    if col in SENSITIVE_RAW:
        return True
    return any(col.startswith(p) for p in SENSITIVE_ONEHOT_PREFIXES)


# ─────────────────────────────────────────────────────────────
# 도메인 지식 chunks (Generic RAG mode용) — UCI German Credit 도메인
# ─────────────────────────────────────────────────────────────
GERMAN_KNOWLEDGE_CHUNKS = [
    {
        "id": "K1",
        "title": "독일 신용평가 핵심 변수의 의미 (UCI 표준)",
        "content": (
            "본 데이터는 독일 신용 평가 표준 변수를 사용한다. "
            "체크 계좌 상태(checking_status)는 신청자의 활성 체크 계좌 잔액 범위로, '체크 계좌 없음(no_checking)'이거나 음수 잔액은 부도 위험 신호로 본다. "
            "대출 기간(duration)은 개월 단위로, 길수록 위험이 누적된다. "
            "대출 금액(credit_amount)은 독일 마르크(DM, 1990년대 화폐) 단위. "
            "신용 이력(credit_history)은 기존 대출 상환 패턴으로, 'critical/other_existing_credit'(critical)은 위험, 'all_paid' 또는 'existing_paid'는 양호 신호. "
            "저축 상태(savings_status)는 잔액 범위로 클수록 안정적."
        ),
    },
    {
        "id": "K2",
        "title": "부도 위험의 일반 원리",
        "content": (
            "신용 평가 모델은 다음 요인이 누적될 때 부도 확률을 높게 평가한다: "
            "(1) 체크 계좌가 없거나 음수 잔액. "
            "(2) 신용 이력이 critical 등급. "
            "(3) 짧은 재직 기간(<1년) 또는 실업 상태. "
            "(4) 저축 < 100 DM 등 자산 부족. "
            "(5) 긴 대출 기간(>24개월). "
            "(6) 대출 목적이 신차·사업 등 변동성 큰 항목. "
            "반대로 장기 재직, 충분한 저축, 자가 보유, 신용 이력 양호는 위험을 낮춘다."
        ),
    },
    {
        "id": "K3",
        "title": "임계값(threshold)과 결정",
        "content": (
            "본 모델은 부도 확률(0~1)을 출력하며 임계값을 넘으면 거절(REJECT), 미만이면 승인(APPROVE). "
            "본 시스템 임계값은 약 0.45~0.55 범위. "
            "확률이 임계값에서 멀수록 결정의 confidence가 높다. "
            "확률 자체는 점수일 뿐이며 절대적 부도 가능성을 의미하지는 않는다."
        ),
    },
    {
        "id": "K4",
        "title": "민감 변수 마스킹 정책",
        "content": (
            "본 시스템은 나이(age), 성별(personal_status에 결합된 sex 정보), 외국인 여부(foreign_worker)를 "
            "사용자에게 보이는 설명에서 직접 언급하지 않는다. "
            "이런 변수는 모델 학습에는 사용될 수 있지만, 자연어 설명에서는 마스킹된다. "
            "설명 작성 시 절대로 나이·성별·외국인 여부 등을 언급하지 말 것."
        ),
    },
    {
        "id": "K5",
        "title": "독일 화폐와 도메인 단위",
        "content": (
            "본 데이터의 금액 단위는 독일 마르크(DM, Deutsche Mark) — 1990년대 표준 통화. "
            "대출 금액 1000 DM은 약 500 EUR 또는 한화 70만원 정도(1990년대 환율 기준). "
            "재직 기간은 연 단위 카테고리(<1년, 1~4년, 4~7년, ≥7년)로 표기. "
            "대출 목적(purpose)은 radio/tv, education, business 등 11종으로 분류."
        ),
    },
    {
        "id": "K6",
        "title": "출력 형식과 Hard Constraints",
        "content": (
            "설명 작성 시 반드시 지킬 것: "
            "(1) [고객 데이터]에 명시된 변수와 값만 인용. 컨텍스트에 없는 변수, 수치, 전화번호, 상품명 절대 생성 금지. "
            "(2) 의료·법률 자문, 단정적 미래 예측, 특정 금융 상품 추천 금지. "
            "(3) 민감 변수(나이·성별·외국인) 직접 언급 금지. "
            "(4) 추측 대신, 명시된 숫자만 인용. "
            "(5) 출력은 5개 섹션 순서: [결정 요약] / [주요 사유] / [긍정 요인] / [개선 권고] / [면책 고지]."
        ),
    },
    {
        "id": "K7",
        "title": "Reject/Approve 인지 가이드",
        "content": (
            "REJECT 결정 시 부정 요인이 긍정 요인보다 강하다는 의미. "
            "사유는 [고객 데이터]의 값에서만 골라 설명. "
            "APPROVE 결정 시 긍정 요인을 강조하고 위험 요인은 '주의해야 할 점'으로 부드럽게 안내. "
            "어느 경우든 단정적 미래 표현 ('반드시 부도가 난다', '확실히 갚을 수 있다')은 사용 금지."
        ),
    },
]


# ─────────────────────────────────────────────────────────────
# Threshold (XGBoost val에서 결정) — german_xgb.pkl 로드
# ─────────────────────────────────────────────────────────────
def load_threshold() -> float:
    bundle = joblib.load(MODELS_DIR / "german_xgb.pkl")
    return float(bundle["threshold"])


def load_raw_test() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "test_raw.parquet")


# ─────────────────────────────────────────────────────────────
# Mode 1: no_shap — raw features only, 자유 추론
# ─────────────────────────────────────────────────────────────
def build_no_shap_context(idx: int, raw_row: pd.Series, proba: float,
                            threshold: float, true_label: int) -> Dict:
    decision = "REJECT" if proba >= threshold else "APPROVE"
    customer_data = {}
    for col in raw_row.index:
        if col in {"row_id", "TARGET"}:
            continue
        if col in SENSITIVE_RAW:
            continue  # 마스킹
        v = raw_row[col]
        # raw row의 컬럼은 sanitize 안 된 원본
        # 단순 fallback: 컬럼명 그대로 + 값 그대로 (또는 humanize 시도)
        label = GERMAN_GLOSSARY.get(col) or GERMAN_CATEGORY_LABELS.get(col, col)
        if col in {"duration", "credit_amount", "installment_commitment",
                   "residence_since", "existing_credits", "num_dependents"}:
            value_str = humanize_value(col, v)
        else:
            value_str = str(v)
        customer_data[label] = value_str

    return {
        "sample_idx": idx,
        "default_probability": round(proba, 4),
        "threshold": round(threshold, 4),
        "decision": decision,
        "customer_data": customer_data,
        "policy": "no_shap_raw_only",
        "_meta_true_label": true_label,
    }


NO_SHAP_PROMPT = """당신은 고객에게 신용 평가 결과를 친절히 설명하는 금융 상담사입니다.

[고객 데이터 — 이 정보만 사용]
{customer_data_text}

[예측 결과]
- 부도 확률: {default_prob:.4f}
- 임계값: {threshold:.4f}
- 결정: {decision}

[출력 형식 — 반드시 이 5개 섹션 순서로]
[결정 요약]   (1줄)
[주요 거절 사유]   (REJECT일 때만, 최대 3개)
[긍정적으로 평가된 요인]   (최대 3개)
[개선 권고]   (1~3개)
[면책 고지]   (1~2줄)

이제 위 정보에 기반해 설명 리포트를 작성해주세요. 한국어로.
"""


# ─────────────────────────────────────────────────────────────
# Mode 2: generic_rag — raw + 도메인 chunks + hard constraints
# ─────────────────────────────────────────────────────────────
def build_generic_rag_context(idx: int, raw_row: pd.Series, proba: float,
                                threshold: float, true_label: int) -> Dict:
    ctx = build_no_shap_context(idx, raw_row, proba, threshold, true_label)
    ctx["knowledge_chunks"] = GERMAN_KNOWLEDGE_CHUNKS
    ctx["policy"] = "generic_rag_with_hard_constraints"
    return ctx


GENERIC_RAG_PROMPT = """당신은 고객에게 신용 평가 결과를 친절히 설명하는 금융 상담사입니다.

본 시스템은 일반적인 독일 신용 평가 도메인 지식을 [참조 지식]으로 제공합니다.
[참조 지식]은 일반 원리이며 특정 고객의 의사결정 근거가 아닙니다.
구체적 의사결정 근거는 [고객 데이터]에 명시된 값에 근거해 설명하세요.

[Hard Constraints — 반드시 준수]
- [고객 데이터]에 명시된 변수와 값만 인용. 컨텍스트에 없는 변수, 수치, 전화번호, 상품명 절대 생성 금지
- 의료·법률 자문, 단정적 미래 예측, 특정 금융 상품 추천 금지
- 민감 변수(나이·성별·외국인 여부) 직접 언급 금지
- 추측이 아닌, 명시된 숫자만 인용

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


# ─────────────────────────────────────────────────────────────
# Mode 3: shaponly — SHAP top-5 + raw values + hard constraints
# ─────────────────────────────────────────────────────────────
def build_shaponly_context(shap_example: Dict, threshold: float) -> Dict:
    proba = float(shap_example["predicted_proba"])
    decision = "REJECT" if proba >= threshold else "APPROVE"

    drivers_for_default = []  # SHAP > 0
    drivers_against = []       # SHAP < 0
    for d in shap_example["top_k_shap"]:
        if is_sensitive(d["feature"]):
            continue
        entry = {
            "feature": humanize_feature(d["feature"]),
            "feature_raw": d["feature"],
            "value": humanize_value(d["feature"], d["value"]),
            "shap": round(float(d["shap_value"]), 4),
            "rank": d["rank"],
        }
        if d["shap_value"] > 0:
            drivers_for_default.append(entry)
        else:
            drivers_against.append(entry)

    return {
        "sample_idx": shap_example["idx"],
        "default_probability": round(proba, 4),
        "threshold": round(threshold, 4),
        "decision": decision,
        "top_drivers_for_default": drivers_for_default,
        "top_drivers_against_default": drivers_against,
        "fairness_flags": [],
        "model": "XGBoost + SHAP",
        "explanation_policy": "fact_only_shap_top5",
        "_meta_true_label": int(shap_example["true_label"]),
    }


SHAP_PROMPT = """당신은 고객에게 신용 평가 결과를 친절히 설명하는 금융 상담사입니다.

[Hard Constraints]
- 아래 [컨텍스트] JSON에 명시된 변수, 값, SHAP 부호만 사용해 설명을 작성하세요.
- 컨텍스트에 없는 변수, 수치, 추론은 절대 생성하지 마세요.
- 의료·법률 자문, 단정적 미래 예측 금지.
- 민감 변수(나이·성별·외국인 여부) 직접 언급 금지.
- 거절 사유는 top_drivers_for_default에서, 긍정 요인은 top_drivers_against_default에서만 선택.
- SHAP 부호: 양수=부도 가능성↑, 음수=부도 가능성↓.

[출력 형식 — 반드시 이 5개 섹션 순서로]
[결정 요약]   (1줄)
[주요 거절 사유]   (REJECT일 때만, 최대 3개)
[긍정적으로 평가된 요인]   (최대 3개)
[개선 권고]   (1~3개)
[면책 고지]   (1~2줄)

[컨텍스트 — 이 사실만 사용]
{context_json}

이제 위 컨텍스트에 대한 자연어 설명 리포트를 위 출력 형식 그대로 작성해주세요. 한국어로.
"""


# ─────────────────────────────────────────────────────────────
# Mode 4: fusion — SHAP + Attention agreement-aware
# ─────────────────────────────────────────────────────────────
def build_fusion_context(shap_example: Dict, att_example: Dict,
                          threshold: float) -> Dict:
    assert shap_example["idx"] == att_example["idx"]
    proba = float(shap_example["predicted_proba"])
    decision = "REJECT" if proba >= threshold else "APPROVE"

    # SHAP top-k → dict (sign, value, shap)
    shap_dict = {}
    for d in shap_example["top_k_shap"]:
        if is_sensitive(d["feature"]):
            continue
        shap_dict[d["feature"]] = {
            "value_raw": d["value"],
            "shap": float(d["shap_value"]),
            "sign_for_default": "+" if d["shap_value"] > 0 else "-",
        }
    # Attention top-k → dict
    att_dict = {}
    for d in att_example["top_k_attention"]:
        if is_sensitive(d["feature"]):
            continue
        att_dict[d["feature"]] = {
            "value_raw": d["value"],
            "attention": float(d["attention_score"]),
        }

    shap_set = set(shap_dict)
    att_set = set(att_dict)
    agreed = sorted(shap_set & att_set,
                     key=lambda x: -abs(shap_dict[x]["shap"]))
    shap_only = sorted(shap_set - att_set,
                        key=lambda x: -abs(shap_dict[x]["shap"]))
    att_only = sorted(att_set - shap_set,
                        key=lambda x: -att_dict[x]["attention"])

    def make_entry(f, group, rank):
        s = shap_dict.get(f, {})
        a = att_dict.get(f, {})
        ent = {
            "feature": humanize_feature(f),
            "feature_raw": f,
            "value": humanize_value(f, s.get("value_raw", a.get("value_raw"))),
            "rank": rank, "group": group,
        }
        if s:
            ent["shap"] = round(s["shap"], 4)
            ent["sign_for_default"] = s["sign_for_default"]
        if a:
            ent["attention"] = round(a["attention"], 6)
        return ent

    agreed_entries = [make_entry(f, "agreed", i + 1) for i, f in enumerate(agreed)]
    shap_only_entries = [make_entry(f, "shap_only", i + 1) for i, f in enumerate(shap_only)]
    att_only_entries = [make_entry(f, "attention_only", i + 1) for i, f in enumerate(att_only)]

    return {
        "sample_idx": shap_example["idx"],
        "decision": decision,
        "default_probability": round(proba, 4),
        "threshold": round(threshold, 4),
        "agreed_drivers": agreed_entries,
        "shap_only_drivers": shap_only_entries,
        "attention_only_drivers": att_only_entries,
        "model_predict": "XGBoost",
        "model_explain": ["SHAP_xgb_local", "TabNet_attention_local"],
        "explanation_policy": "fact_only_with_agreement_labels",
        "n_agreed": len(agreed_entries),
        "n_shap_only": len(shap_only_entries),
        "n_attention_only": len(att_only_entries),
        "_meta_true_label": int(shap_example["true_label"]),
    }


FUSION_PROMPT = """당신은 고객에게 신용 평가 결과를 친절히 설명하는 금융 상담사입니다.

[Hard Constraints]
- 아래 [컨텍스트] JSON에 명시된 변수, 값, SHAP 부호만 사용해 설명을 작성하세요.
- 컨텍스트에 없는 변수, 수치, 추론은 절대 생성하지 마세요.
- 의료·법률 자문, 단정적 미래 예측 금지.
- 민감 변수(나이·성별·외국인 여부) 직접 언급 금지.
- 사유 선택은 다음 그룹을 우선순위 순서로 활용:
  (1) agreed_drivers — XGBoost SHAP과 TabNet attention이 모두 주목한 강한 신호 (sign_for_default 부호 사용)
  (2) shap_only_drivers — SHAP 부호 정보가 있는 보완 신호 (sign 사용)
  (3) attention_only_drivers — TabNet의 sparse 보완 신호 (부호 없으므로 '주목된 정보'로만 인용)
- SHAP 부호: 양수='+'=부도 가능성↑, 음수='-'=부도 가능성↓.

[출력 형식 — 반드시 이 5개 섹션 순서로]
[결정 요약]   (1줄, agreement 정보 1줄 포함 가능)
[주요 거절 사유]   (REJECT일 때만, 최대 3개, agreed → shap_only 순서로)
[긍정적으로 평가된 요인]   (최대 3개, sign='-'인 항목 위주)
[개선 권고]   (1~3개)
[면책 고지]   (1~2줄)

[컨텍스트 — 이 사실만 사용]
{context_json}

이제 위 컨텍스트에 대한 자연어 설명 리포트를 작성해주세요. 한국어로.
"""


# ─────────────────────────────────────────────────────────────
# 통합 builder + LLM 호출
# ─────────────────────────────────────────────────────────────
def build_contexts(mode: str) -> List[Dict]:
    threshold = load_threshold()
    raw_test = load_raw_test()
    with open(RESULTS_DIR / "german_shap_local.json", "r", encoding="utf-8") as f:
        shap_examples = json.load(f)
    att_examples = None
    if mode == "fusion":
        with open(RESULTS_DIR / "german_tabnet_attention.json", "r", encoding="utf-8") as f:
            att_list = json.load(f)
        att_examples = {e["idx"]: e for e in att_list}

    contexts = []
    for ex in shap_examples:
        idx = ex["idx"]
        if mode in ("no_shap", "generic_rag"):
            raw_row = raw_test.iloc[idx]
            if mode == "no_shap":
                ctx = build_no_shap_context(
                    idx, raw_row, ex["predicted_proba"], threshold, ex["true_label"])
            else:
                ctx = build_generic_rag_context(
                    idx, raw_row, ex["predicted_proba"], threshold, ex["true_label"])
        elif mode == "shaponly":
            ctx = build_shaponly_context(ex, threshold)
        elif mode == "fusion":
            att_ex = att_examples[idx]
            ctx = build_fusion_context(ex, att_ex, threshold)
        else:
            raise ValueError(f"unknown mode: {mode}")
        ctx["_tag"] = ex["tag"]
        contexts.append(ctx)
    return contexts


def fmt_customer_data(d: Dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in d.items())


def fmt_knowledge(chunks: List[Dict]) -> str:
    return "\n\n".join(f"[{c['id']}] {c['title']}\n{c['content']}" for c in chunks)


def make_prompt(mode: str, ctx: Dict) -> str:
    ctx_clean = {k: v for k, v in ctx.items() if not k.startswith("_")}
    if mode == "no_shap":
        return NO_SHAP_PROMPT.format(
            customer_data_text=fmt_customer_data(ctx["customer_data"]),
            default_prob=ctx["default_probability"],
            threshold=ctx["threshold"], decision=ctx["decision"])
    if mode == "generic_rag":
        return GENERIC_RAG_PROMPT.format(
            knowledge_text=fmt_knowledge(ctx["knowledge_chunks"]),
            customer_data_text=fmt_customer_data(ctx["customer_data"]),
            default_prob=ctx["default_probability"],
            threshold=ctx["threshold"], decision=ctx["decision"])
    if mode == "shaponly":
        return SHAP_PROMPT.format(
            context_json=json.dumps(ctx_clean, ensure_ascii=False, indent=2))
    if mode == "fusion":
        return FUSION_PROMPT.format(
            context_json=json.dumps(ctx_clean, ensure_ascii=False, indent=2))
    raise ValueError(f"unknown mode: {mode}")


def main(mode: str, provider: str, n_samples: int = 30,
         sleep_sec: float = 3.0) -> None:
    set_seed(SEED)

    print(f"[1/3] {mode} 컨텍스트 빌드")
    contexts = build_contexts(mode)
    if n_samples < len(contexts):
        contexts = contexts[:n_samples]
    print(f"  {len(contexts)}개 컨텍스트 준비")

    contexts_dir = RESULTS_DIR / f"contexts_german_{mode}_{n_samples}"
    contexts_dir.mkdir(parents=True, exist_ok=True)
    for ctx in contexts:
        out = contexts_dir / f"{ctx['sample_idx']}_{ctx['_tag']}.json"
        ctx_save = {k: v for k, v in ctx.items() if k != "_tag"}
        with open(out, "w", encoding="utf-8") as f:
            json.dump(ctx_save, f, indent=2, ensure_ascii=False)
    print(f"  saved → {contexts_dir}")

    output_dir = RESULTS_DIR / f"explanations_german_{mode}_{provider}_{n_samples}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/3] LLM 호출 (provider={provider}, mode={mode})")
    model = PROVIDER_DEFAULTS[provider]["model"]
    client_tuple = make_client(provider)
    print(f"  model={model}, output={output_dir}")

    out_paths = []
    for i, ctx in enumerate(contexts, start=1):
        tag = ctx["_tag"]
        idx = ctx["sample_idx"]
        prompt = make_prompt(mode, ctx)
        print(f"[{i}/{len(contexts)}] {idx}_{tag}  decision={ctx['decision']}  prob={ctx['default_probability']:.3f}")
        for retry in range(3):
            try:
                t0 = time.time()
                text, usage = _call_llm(client_tuple, prompt, model=model)
                elapsed = time.time() - t0
                break
            except Exception as e:
                wait = 30 * (retry + 1)
                print(f"  ERROR: {str(e)[:120]}, {wait}s 대기 후 재시도 {retry+1}/3")
                time.sleep(wait)
        else:
            print(f"  3회 실패 — skip")
            continue

        result = {
            "sample_idx": idx,
            "decision": ctx["decision"],
            "true_label": ctx.get("_meta_true_label"),
            "provider": provider, "model": model, "mode": mode,
            "elapsed_sec": round(elapsed, 2),
            "explanation": text,
            "usage_metadata": usage,
            "context_sent": {k: v for k, v in ctx.items() if not k.startswith("_")},
            "context_type": f"german_{mode}",
        }
        out = output_dir / f"{idx}_{tag}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        out_paths.append(out)
        tot = usage.get("total_token_count", "?")
        print(f"  elapsed={elapsed:.1f}s, tokens={tot}")
        if i < len(contexts):
            time.sleep(sleep_sec)

    with open(output_dir / "_index.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_explanations": len(out_paths),
            "provider": provider, "model": model, "mode": mode,
            "selected_idx": [int(p.stem.split("_")[0]) for p in out_paths],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[3/3] 완료 — {len(out_paths)}개 설명 → {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["no_shap", "generic_rag", "shaponly", "fusion"])
    ap.add_argument("--provider", required=True, choices=["anthropic", "gemini"])
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--sleep", type=float, default=3.0)
    args = ap.parse_args()
    main(mode=args.mode, provider=args.provider,
         n_samples=args.n_samples, sleep_sec=args.sleep)
