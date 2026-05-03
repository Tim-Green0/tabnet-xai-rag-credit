"""Day 6 — XAI-RAG 컨텍스트 빌더.

SHAP local explanation → 구조화된 JSON 컨텍스트 (계획서 3.6).

원칙:
  - 정합성: 컨텍스트의 변수명·값·SHAP 부호는 모델 출력과 1:1 매칭
  - 최소성: 상위 K개 핵심 기여 변수만 포함 → LLM의 산만한 추론 차단
  - 민감 변수 마스킹: CODE_GENDER, AGE 등은 컨텍스트에 직접 노출하지 않음
  - 도메인 용어집: DTI 등 약어 풀어쓰기 매핑

부호 의미:
  positive SHAP → P(default) ↑ → "거절 측 요인" (top_drivers_for_default)
  negative SHAP → P(default) ↓ → "승인 측 요인" (top_drivers_against_default)

산출물:
  - results/contexts/{idx}_{tag}.json   : 인스턴스별 컨텍스트
  - results/contexts_index.json         : 전체 인덱스
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils import RESULTS_DIR

CONTEXTS_DIR = RESULTS_DIR / "contexts"
CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)

# 계획서 3.6 — 민감 변수 마스킹 대상 (전처리 후 컬럼 기준)
SENSITIVE_FEATURES = {
    "CODE_GENDER_F", "CODE_GENDER_M", "CODE_GENDER_XNA",
    "DAYS_BIRTH",  # 연령
}

# 도메인 용어집 — Home Credit 변수 풀어쓰기 (한국어)
DOMAIN_GLOSSARY: Dict[str, str] = {
    "EXT_SOURCE_1": "외부 신용평가 점수 1",
    "EXT_SOURCE_2": "외부 신용평가 점수 2",
    "EXT_SOURCE_3": "외부 신용평가 점수 3",
    "AMT_INCOME_TOTAL": "총 소득",
    "AMT_CREDIT": "대출 신청 금액",
    "AMT_ANNUITY": "연 환산 상환액",
    "AMT_GOODS_PRICE": "구매 대상 상품 가격",
    "DAYS_EMPLOYED": "재직 일수(음수: 신청일 기준)",
    "DAYS_REGISTRATION": "거주지 등록 후 경과 일수",
    "DAYS_ID_PUBLISH": "신분증 발급 후 경과 일수",
    "DAYS_LAST_PHONE_CHANGE": "최근 휴대폰 변경 후 경과 일수",
    "OWN_CAR_AGE": "보유 차량 연식",
    "CNT_CHILDREN": "자녀 수",
    "CNT_FAM_MEMBERS": "가족 구성원 수",
    "EMPLOYED_FLAG": "재직 여부 플래그(1=재직)",
    "REGION_RATING_CLIENT": "거주지 등급(낮을수록 양호)",
    "REGION_RATING_CLIENT_W_CITY": "거주지(도시 가중) 등급",
    "REGION_POPULATION_RELATIVE": "거주지 인구 밀도(상대값)",
    "REG_CITY_NOT_LIVE_CITY": "등록 도시와 실거주 도시 불일치",
    "REG_CITY_NOT_WORK_CITY": "등록 도시와 근무 도시 불일치",
    "LIVE_CITY_NOT_WORK_CITY": "실거주 도시와 근무 도시 불일치",
    "OCCUPATION_TYPE_TE": "직업 분류 (타깃 인코딩 값)",
    "ORGANIZATION_TYPE_TE": "근무 조직 유형 (타깃 인코딩 값)",
    "NAME_CONTRACT_TYPE_Cash loans": "신청 대출 유형: 현금대출",
    "NAME_CONTRACT_TYPE_Revolving loans": "신청 대출 유형: 리볼빙(한도 대출)",
    "NAME_EDUCATION_TYPE_Higher education": "학력: 고등교육 이상",
    "NAME_EDUCATION_TYPE_Secondary / secondary special": "학력: 중등(전문중등)",
    "FLAG_OWN_CAR_Y": "차량 보유: 있음",
    "FLAG_OWN_REALTY_Y": "부동산 보유: 있음",
    "FLAG_EMP_PHONE": "직장 전화 등록 여부",
    "FLAG_WORK_PHONE": "근무 전화 등록 여부",
    "EXT_SOURCE_1_MISSING_FLAG": "외부 신용평가 1 결측 여부",
    "EXT_SOURCE_2_MISSING_FLAG": "외부 신용평가 2 결측 여부",
    "EXT_SOURCE_3_MISSING_FLAG": "외부 신용평가 3 결측 여부",
}


def humanize_feature(feature: str) -> str:
    """변수명을 사람이 이해할 수 있는 한국어로."""
    return DOMAIN_GLOSSARY.get(feature, feature)


def humanize_value(feature: str, value) -> str:
    """변수 + 값을 보기 좋게 표기. 일부 변수만 단위 변환."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if feature == "DAYS_BIRTH":
        return f"{int(-v / 365.25)}세"
    if feature == "DAYS_EMPLOYED":
        return f"{int(-v / 365.25)}년 재직"
    if feature == "DAYS_REGISTRATION":
        return f"{int(-v / 365.25)}년 전 등록"
    if feature.startswith("AMT_"):
        return f"{v:,.0f}"
    if feature.endswith("_FLAG") or "_FLAG" in feature:
        return "예" if v >= 0.5 else "아니오"
    if feature.startswith("FLAG_"):
        return "예" if v >= 0.5 else "아니오"
    if "MISSING_FLAG" in feature:
        return "결측" if v >= 0.5 else "값 있음"
    if feature.endswith("_TE"):
        return f"{v:.4f} (타깃 인코딩 값)"
    # one-hot 변수
    if any(feature.startswith(prefix) for prefix in
            ["NAME_", "CODE_", "FLAG_OWN_", "WEEKDAY_", "ORGANIZATION_",
             "OCCUPATION_", "EMERGENCYSTATE_", "FONDKAPREMONT_",
             "WALLSMATERIAL_", "HOUSETYPE_"]):
        return "예" if v >= 0.5 else "아니오"
    return f"{v:g}"


def build_context(
    sample_idx: int,
    y_score: float,
    threshold: float,
    feature_values: Dict[str, float],
    shap_values: Dict[str, float],
    top_k: int = 5,
    model_name: str = "XGBoost",
    fairness_flags: Optional[List[str]] = None,
    mask_sensitive: bool = True,
    true_label: Optional[int] = None,
) -> Dict:
    """SHAP 결과를 LLM 컨텍스트로 변환.

    Parameters
    ----------
    sample_idx : 인스턴스 식별자 (test set 인덱스)
    y_score    : 모델의 부도 확률 P(default)
    threshold  : 분류 임계치
    feature_values : 변수 → 값 dict
    shap_values    : 변수 → SHAP value dict (positive: P(default) ↑)
    top_k          : 양/음 각각 상위 몇 개 변수
    fairness_flags : 공정성 경고 (있으면 list of str)
    mask_sensitive : True면 SENSITIVE_FEATURES 마스킹
    true_label     : 실제 정답 (검증 용도, LLM에는 비공개 권장)
    """
    decision = "REJECT" if y_score >= threshold else "APPROVE"

    contribs = pd.Series(shap_values)
    if mask_sensitive:
        masked = [c for c in contribs.index if c in SENSITIVE_FEATURES]
        contribs = contribs.drop(masked, errors="ignore")
    else:
        masked = []

    pos = contribs[contribs > 0].sort_values(ascending=False).head(top_k)
    neg = contribs[contribs < 0].sort_values().head(top_k)

    drivers_for = []
    for rank, (name, sv) in enumerate(pos.items(), start=1):
        drivers_for.append({
            "feature": humanize_feature(name),
            "feature_raw": name,
            "value": humanize_value(name, feature_values.get(name)),
            "value_raw": feature_values.get(name),
            "shap": round(float(sv), 4),
            "rank": rank,
        })
    drivers_against = []
    for rank, (name, sv) in enumerate(neg.items(), start=1):
        drivers_against.append({
            "feature": humanize_feature(name),
            "feature_raw": name,
            "value": humanize_value(name, feature_values.get(name)),
            "value_raw": feature_values.get(name),
            "shap": round(float(sv), 4),
            "rank": rank,
        })

    ctx = {
        "sample_idx": sample_idx,
        "decision": decision,
        "default_probability": round(float(y_score), 4),
        "threshold": round(float(threshold), 4),
        "top_drivers_for_default": drivers_for,
        "top_drivers_against_default": drivers_against,
        "fairness_flags": fairness_flags or [],
        "model": model_name,
        "explanation_policy": "fact_only",
        "masked_sensitive_features": masked,
    }
    if true_label is not None:
        ctx["_meta_true_label"] = int(true_label)
    return ctx


def save_context(ctx: Dict, tag: str = "") -> Path:
    fn = f"{ctx['sample_idx']}_{tag}.json" if tag else f"{ctx['sample_idx']}.json"
    out = CONTEXTS_DIR / fn
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2, ensure_ascii=False)
    return out


def build_from_local_examples_file(
    local_examples_path: Path,
    threshold: float,
    model_name: str = "XGBoost",
    top_k: int = 5,
) -> List[Path]:
    """results/shap_local_examples.json을 읽어 컨텍스트 일괄 생성."""
    with open(local_examples_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    paths = []
    for ex in examples:
        # SHAP local examples 포맷 → context_builder 입력으로 변환
        feature_values = {}
        shap_values = {}
        for d in ex["top_5_positive_drivers"] + ex["top_5_negative_drivers"]:
            feature_values[d["feature"]] = d["value"]
            shap_values[d["feature"]] = d["shap"]

        ctx = build_context(
            sample_idx=ex["idx"],
            y_score=ex["predicted_proba"],
            threshold=threshold,
            feature_values=feature_values,
            shap_values=shap_values,
            top_k=top_k,
            model_name=model_name,
            true_label=ex["true_label"],
        )
        paths.append(save_context(ctx, tag=ex["tag"]))

    # index 파일
    index = {
        "n_contexts": len(paths),
        "files": [str(p) for p in paths],
        "model": model_name,
        "threshold": threshold,
    }
    with open(CONTEXTS_DIR / "_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-examples",
                    default=str(RESULTS_DIR / "shap_local_examples.json"),
                    help="SHAP local examples JSON 경로")
    ap.add_argument("--threshold", type=float, default=0.476,
                    help="모델 임계치 (Day 2 XGBoost test threshold 기본값)")
    ap.add_argument("--model", default="XGBoost")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    paths = build_from_local_examples_file(
        Path(args.local_examples),
        threshold=args.threshold,
        model_name=args.model,
        top_k=args.top_k,
    )
    print(f"[OK] {len(paths)}개 컨텍스트 생성 → {CONTEXTS_DIR}")
    for p in paths:
        print(f"  {p}")
