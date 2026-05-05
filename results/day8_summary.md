# Day 8 요약 — Demo + 미팅 보고서 + Counterfactual Baseline

날짜: 2026-05-03
미팅까지: D-7

---

## TL;DR (미팅 결정타 4가지)

1. **🎯 Counterfactual Baseline 비교** — Claude는 SHAP 없는 baseline에서 환각률 **45.5%**, XAI-RAG에서는 **0.0%**.
   "SHAP 컨텍스트 유무가 환각률에 결정적 차이를 만든다"는 직접 증명.
2. **End-to-end Demo (idx 59291)** — True Positive 케이스. 정형 데이터 → XGBoost → SHAP → JSON → Gemini/Claude 자연어 설명, 전 단계 라이브 시연 가능.
3. **미팅 보고서 docx** — 9개 섹션(연구 목적 → 데이터 → 모델 비교 → SHAP·Attention → 공정성 → Demo → 평가 → Baseline 비교 → Future Work).
4. **Baseline 환각 패턴**: Claude가 학습된 도메인 지식(DTI, LTV, DSR, '햇살론, 미소금융', 가짜 전화번호 1588-XXXX)을 자유 추론으로 가져옴 → 본 연구가 차단하고자 했던 정확한 케이스.

---

## A. Demo (`src/demo.py`)

샘플 idx 59291 — **True Positive** (실제 부도, 모델도 REJECT).
- 연령 44.9세, 남성
- AMT_INCOME_TOTAL 234,000 / AMT_CREDIT 592,560
- EXT_SOURCE_3 = 0.0634 (매우 낮음)
- DAYS_EMPLOYED -145 (≈ 0.4년, 신규 고용)

**파이프라인**:
1. XGBoost: P(default) = 0.9484 → REJECT
2. SHAP Top 5 거절 측: EXT_SOURCE_3 (+1.04), EXT_SOURCE_2 (+0.49), AMT_GOODS_PRICE (+0.23), REG_CITY_NOT_LIVE_CITY (+0.16), DAYS_EMPLOYED (+0.14)
3. JSON 컨텍스트 (CODE_GENDER_*, DAYS_BIRTH 마스킹)
4. Gemini 14.75s / 4959 tokens → 자연어 설명 생성
5. Claude 13.48s / 2694 tokens → 자연어 설명 생성

산출:
- `results/demo_walkthrough.md` (사람이 읽는 워크스루)
- `results/demo_walkthrough.json` (전 중간 산출물)
- `figures/22_demo_walkthrough.png` (SHAP 막대 + decision summary)

---

## B. 중간 보고 docx (`paper/midterm_report.docx`, 456 KB)

9개 섹션:
1. 연구 목적과 기여 (계획서 1.2 회복)
2. 데이터 및 전처리
3. 모델 비교 (5-fold CV)
4. SHAP × 어텐션 일관성 (RQ2)
5. 공정성 진단 + Mitigation
6. XAI-RAG 자연어 설명 (Demo idx 59291)
7. 정량 평가 (RQ3 — 환각 차단)
8. **Counterfactual Baseline (XAI-RAG vs no-SHAP)** ← 신설
9. 향후 계획

5개 그림 임베드: `13_cv_comparison`, `16_attention_vs_shap_scatter`, `19_fairness_mitigation`, `21_llm_comparison`, `23_baseline_vs_xairag`.

---

## C. Counterfactual Baseline (`src/baseline_no_shap.py`)

### 실험 설계
- 동일 11 샘플(10 + demo idx 59291)에 raw feature 값(핵심 17개 변수)만 LLM에 제공
- SHAP 사실 컨텍스트, top driver, fact_only 정책 모두 제거
- LLM에 거절 사유 5단 구조로 작성 요청 → 자유 추론

### 결과 — 환각률 (Hallucination Rate strict, 데이터셋 외 변수 만들어낸 비율)

| LLM | XAI-RAG (SHAP context) | Baseline (no SHAP) | 차이 |
|---|---|---|---|
| Gemini | 0.000 | 0.000 | 동일 (측정 한계) |
| **Claude** | **0.000** | **0.4545 (45.5%)** | **결정적 차이** |

### Claude baseline 환각 사례

| 환각 변수 | 의미 | 출처 |
|---|---|---|
| **DTI** | Debt-to-Income, 소득 대비 대출 비율 | 일반 금융 도메인 약어 |
| **LTV** | Loan-to-Value, 담보 가치 비율 | 일반 금융 도메인 약어 |
| **DSR** | Debt Service Ratio, 총부채상환비율 | 일반 금융 도메인 약어 |
| 햇살론, 미소금융 | 한국 특정 정부 지원 금융상품 | 학습 데이터에서 끌어옴 |
| 1588-XXXX | 가짜 고객센터 번호 | LLM 자체 생성 |
| DEF_30_CNT | 변수명 잘림 (실제 DEF_30_CNT_SOCIAL_CIRCLE) | 부정확 인용 |

**해석**:
- Claude는 SHAP 컨텍스트 없으면 **학습된 일반 금융 도메인 지식**을 끌어와 그럴듯한 거절 사유 생성
- 본 연구의 XAI-RAG가 정확히 이 패턴을 차단하는 것을 직접 증명
- 미팅 메시지: **"SHAP 컨텍스트 = LLM의 자유 추론을 차단하는 가드레일"**

### Gemini의 측정 한계
Gemini baseline 환각률 0.0%는 환각이 **없는** 게 아니라:
- Gemini는 raw 영문 변수명(`DTI`, `LTV` 등) 거의 사용 안 함
- 한국어 자연어로 의역 → 영문 정규식 룰로 안 잡힘
- 즉 측정 한계: strict 룰은 영문 변수명 환각만 잡음
- → **future work**: Cross-LLM judge로 의미 단위 환각 측정

---

## D. 산출물 종합

```
src/
├─ demo.py                    # 1명 샘플 end-to-end 시연
├─ gen_report.py              # 미팅 docx 보고서 생성
└─ baseline_no_shap.py        # Counterfactual baseline 비교

paper/
└─ midterm_report.docx        # ★ 미팅 보고서 (456 KB, 9 섹션, 5 그림)

results/
├─ demo_walkthrough.{md,json}
├─ explanations_baseline_noshap_gemini/      (11 파일)
├─ explanations_baseline_noshap_anthropic/   (11 파일)
├─ baseline_noshap_eval_{gemini,anthropic}.csv
├─ baseline_comparison.csv                   ← 핵심 비교 결과
└─ day8_summary.md (이 파일)

figures/
├─ 22_demo_walkthrough.png
└─ 23_baseline_vs_xairag.png   ← 미팅 결정타 그림
```

---

## E. 미팅 흐름 권장 (15~20분 발표 가정)

| # | 슬라이드 / 섹션 | 시간 | 핵심 메시지 |
|---|---|---|---|
| 1 | 연구 목적 | 1분 | TabNet + SHAP + LLM XAI-RAG 통합 |
| 2 | 데이터 + EDA | 2분 | 307K 샘플, 8% 불균형, EXT_SOURCE 핵심 |
| 3 | 모델 비교 (5-fold CV) | 2분 | XGBoost 0.7587 ± 0.0008 (안정적 우위) |
| 4 | SHAP × Attention | 2분 | ρ=0.117, Top-20 9개 교집합 (부분 일관 + 부분 상보) |
| 5 | 공정성 | 2분 | 8/8 케이스 4/5 rule 위반, GENDER ablation 효과적 |
| 6 | **XAI-RAG Demo** | 3분 | idx 59291 라이브 시연 |
| 7 | **평가 결과 (RQ3)** | 3분 | 두 LLM 모두 Halluc 0%, G-Eval 5.0 |
| 8 | **Baseline 비교** | 2분 | **Claude baseline 45.5% vs XAI-RAG 0%** ⭐ |
| 9 | Future Work | 1분 | Counterfactual, fairness-aware, 보조 테이블 등 |

미팅 결정타: **8번 슬라이드**. baseline 비교가 본 연구 가치를 직접 입증.

---

## F. 검증 부탁

1. `paper/midterm_report.docx` — 9개 섹션, 모든 그림 임베드 확인
2. `figures/23_baseline_vs_xairag.png` — Claude 45.5% vs 0% 차이 시각적
3. `results/explanations_baseline_noshap_anthropic/45114_reject.json` — DTI, 햇살론 등 환각 사례 직접 보기
4. `results/demo_walkthrough.md` — 미팅 라이브 시연용 워크스루

수정 요청 있으면 알려줘. 없으면 commit/push로 마무리.
