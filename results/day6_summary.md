# Day 6 요약 — XAI-RAG 컨텍스트 + Gemini LLM 자연어 설명

날짜: 2026-05-03
미팅까지: D-7

---

## TL;DR

1. **XAI-RAG 파이프라인 end-to-end 작동 확인**: SHAP local → JSON 컨텍스트 → Gemini 2.5 Flash → 한국어 자연어 설명 리포트.
2. **10명 샘플 모두 성공** (REJECT 5 + APPROVE 5). 평균 12.7초/호출, 평균 4150 토큰.
3. **수치 충실성 시각적 확인**: 생성된 텍스트의 모든 변수명·값·SHAP 부호가 컨텍스트와 1:1 매칭. Day 7 Faithfulness 정량 평가에서 높은 점수 예상.
4. **본 연구 핵심 메시지 입증**: "LLM이 임의 추론 안 하고 SHAP 사실만 받아쓰는 XAI-RAG"가 데모 가능한 형태로 작동.
5. **무료 옵션 작동**: gemini-2.5-flash는 free tier에서 사용 가능 (gemini-2.0-flash는 limit 0). 미팅 시연 비용 0원.

---

## A. 파이프라인 구성

```
SHAP local examples (10명, 5 REJECT + 5 APPROVE)
       ↓
context_builder.py
   - 변수명 한국어화 (EXT_SOURCE_2 → "외부 신용평가 점수 2")
   - 값 단위 변환 (DAYS_BIRTH → "X세", AMT_* → "X,XXX")
   - 민감 변수 마스킹 (DAYS_BIRTH, CODE_GENDER_*)
   - 도메인 용어집(DOMAIN_GLOSSARY) 적용
       ↓
results/contexts/{idx}_{tag}.json (10개)
       ↓
llm_explainer.py
   - 프롬프트 템플릿: Role + Hard Constraints + Few-shot + Context + Output Schema
   - Gemini 2.5 Flash API 호출
   - free tier RPM 회피 (4초 간격)
       ↓
results/explanations/{idx}_{tag}.json (10개)
   - 한국어 자연어 설명 + 메타데이터(토큰, 시간)
```

---

## B. 프롬프트 정책 (계획서 3.7 정합)

### Role Specification
> "당신은 고객에게 신용 평가 결과를 친절히 설명하는 금융 상담사입니다."

### Hard Constraints (5가지)
1. 컨텍스트에 명시된 변수·값·SHAP 부호만 사용
2. 컨텍스트에 없는 변수·수치·추론 절대 생성 금지
3. 의료/법률 자문 + 단정적 미래 예측 금지
4. 민감 변수(성별·연령·인종·종교·출신) 직접 언급 금지
5. SHAP 부호 정확히 반영 (양수=부도↑, 음수=부도↓)

### Output Schema (5섹션 강제)
1. [결정 요약] (1줄)
2. [주요 거절 사유] (REJECT만, 최대 3개)
3. [긍정적으로 평가된 요인] (최대 3개)
4. [개선 권고] (1~3개)
5. [면책 고지] (1~2줄)

### Few-shot Examples
1개 REJECT 예시로 출력 톤·구조 가이드.

---

## C. 결과 — 10명 샘플 분석

### C-1. 처리 통계
| 항목 | 값 |
|---|---|
| 성공률 | 10/10 (100%) |
| 평균 elapsed/호출 | 12.7초 |
| 평균 토큰/호출 | 4,150 (input ~1,500, output ~270, reasoning ~2,400) |
| 총 토큰 | 41,547 |
| 무료 tier 비용 | $0 |

### C-2. True label 분포 (예측 vs 실제)
| | true=0 (정상) | true=1 (부도) |
|---|---|---|
| **APPROVE (5)** | 5 (TN ✅) | 0 |
| **REJECT (5)** | **3 (FP ⚠️)** | 2 (TP ✅) |

흥미롭게도 high-confidence reject 5명 중 **3명이 false positive**. 이는 미팅에서 "모델 자동 거절을 자연어로 설명 + 인간 심사역 검토 필수"라는 메시지로 활용 가능.

### C-3. 출력 품질 — REJECT 예시 (idx 54529, false positive)

**컨텍스트**:
- decision=REJECT, default_probability=0.9044
- top_drivers_for_default[0]: EXT_SOURCE_3=0.0633754, shap=1.1114
- top_drivers_for_default[1]: EXT_SOURCE_2=0.0444795, shap=0.8850
- top_drivers_for_default[2]: AMT_GOODS_PRICE=337,500, shap=0.2072

**LLM 출력 (발췌)**:
> [주요 거절 사유]
> 1. 외부 신용평가 점수 3이 **0.0633754**로 매우 낮게 평가되어 부도 가능성을 높였습니다.
> 2. 외부 신용평가 점수 2 또한 **0.0444795**로 낮게 평가되어 부도 가능성에 부정적인 영향을 주었습니다.
> 3. 구매 대상 상품 가격이 **337,500원**으로 책정되어 부도 가능성을 높이는 요인으로 작용했습니다.

**검증 — Faithfulness 4가지**:
- ✅ 변수명: 컨텍스트의 한국어 변수명 그대로 인용
- ✅ 수치: 0.0633754, 0.0444795, 337,500 — 컨텍스트와 픽셀 단위 일치
- ✅ SHAP 부호: 양수 → "부도 가능성을 높였습니다" 정확히 반영
- ✅ 컨텍스트 외 변수 생성: 없음

**검증 — 정책 준수**:
- ✅ 민감 변수(성별, 연령) 직접 언급 없음
- ✅ 단정적 미래 예측 없음 ("일정 기간 후 재신청을 검토하실 수 있습니다" — 권유형)
- ✅ 의료/법률 자문 없음

---

## D. 핵심 발견

### D-1. **gemini-2.0-flash는 free tier에서 limit=0**
첫 시도 `gemini-2.0-flash`에서 `RESOURCE_EXHAUSTED, limit: 0` 에러. 정책 변경된 것으로 보임.
**해결**: `gemini-2.5-flash`로 전환. 무료 사용 가능, 출력 품질도 우수.

### D-2. **gemini-2.5-flash는 reasoning model**
토큰 소비의 약 60%가 reasoning(thinking) 토큰. 13초/호출 중 대부분이 추론 시간.
- 장점: 컨텍스트 정확 인용, 정책 준수도 높음
- 단점: 호출당 시간 길고 토큰 큼 (free tier RPD 제한 도달 빠름)
- 미팅용 10건이면 충분 — 1분 내 처리

### D-3. **Faithfulness 시각 확인**
10건 모두 컨텍스트의 변수·값·부호를 정확히 인용. Day 7 정량 평가에서 Faithfulness Score ≥ 0.95, Hallucination Rate ≤ 5% 예상.

---

## E. 산출물

```
src/
├─ context_builder.py     # SHAP local → JSON 컨텍스트 (Day 4 commit)
└─ llm_explainer.py       # Gemini 호출 + 프롬프트 + 결과 저장

results/
├─ contexts/              # 10개 + _index.json
│  ├─ 22226_accept.json
│  ├─ 36556_reject.json
│  ├─ 39518_reject.json
│  ├─ 40525_accept.json
│  ├─ 45114_reject.json
│  ├─ 54421_reject.json
│  ├─ 54529_reject.json
│  ├─ 57037_accept.json
│  ├─ 5886_accept.json
│  └─ 60717_accept.json
├─ explanations/          # 10개 + _index.json (자연어 설명 + 토큰 메타)
└─ day6_summary.md
```

`.env`는 .gitignore 처리. GEMINI_API_KEY 노출 없음.

---

## F. 다음 단계 — Day 7 (정량 평가)

`src/eval_explanation.py` 작성:
- **Faithfulness Score**: 생성 텍스트의 변수명·수치 주장을 추출 → 컨텍스트 ground truth와 매칭
- **Hallucination Rate**: 컨텍스트에 근거 없는 주장 비율
- **G-Eval (LLM-as-a-Judge)**: GPT-4o 또는 Gemini로 정확성·완결성·민감변수 노출·문체 5점 평가
- **Counterfactual Test**: SHAP 변수 제거 후 재생성 → 설명이 변하는지 측정
- **(선택) LLM 비교**: Anthropic API 키 등록 시 Claude 추가 비교

미팅 직전이므로 G-Eval은 Gemini self-evaluation으로 진행 (별도 API 키 불필요). Counterfactual Test는 시간 봐서 결정.

**Day 7 진행 OK인지 알려줘.** 또는 Day 6 결과 검토 후 보강할 사항 있으면 알려줘.
