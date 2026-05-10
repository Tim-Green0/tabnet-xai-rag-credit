# 학위논문 초안 — Outline (Phase 1)

> 작업 분기: `claude/thesis-draft`
> 학교 양식: 서강대학교 AI·SW대학원 학위논문 작성지침 (2026년)
> 본 outline은 사용자 검토 후 각 chapter 본문 작성으로 진행.

---

## 논문 정보

| 항목 | 값 |
|---|---|
| 논문 제목 | **정형 데이터 특화 딥러닝(TabNet)과 거대언어모델(LLM) 기반 XAI-RAG를 활용한 설명 가능한 신용 평가 및 사용자 맞춤형 리포트 생성** |
| 부제목 (선택) | 어텐션-SHAP 융합 컨텍스트와 다중 평가 프레임워크 적용 |
| 저자 | 오현택 (학번 A70067) |
| 지도교수 | 박운상 |
| 전공 | 데이터사이언스·인공지능 |
| 제출 학기 | 2026학년도 (1학기/2학기 미정) |
| 본문 언어 | 한국어 (영문 abstract 앞에 배치) |
| 분량 목표 | 본문 ~60~80 페이지 + 부록 |

---

## 표지 / 머리말 (학위논문지침 별지)

1. **겉표지** (별지 3호)
2. **속표지** (겉표지 동일)
3. **제출문** (별지 4호 — "이 논문을 ○○ 석사 학위논문으로 제출함")
4. **논문 인준서** (별지 5호 — 주심/부심 3인)
5. **감사의 글** (Acknowledgement)
6. **목차** (Table of Contents)
7. **표 차례** (List of Tables)
8. **그림 차례** (List of Figures)
9. **국문 초록** (Abstract — 본문이 한국어이므로 영문 초록 앞에 배치) ← **수정**: 영문 초록 앞 배치
   - 학위논문지침 ⑺: "본문이 국문일 경우에는 외국어로 된 초록을 앞세우고"
10. **영문 Abstract** ← 영문이 먼저
11. (수정 후) **국문 초록**

---

## 본문 구성 (8장 + 참고문헌 + 부록)

### **제1장 서론** (5~7p)
- 1.1 연구 배경
  - 신용평가의 사회적 중요성과 자동화 동향
  - 블랙박스 AI 도입에 따른 설명 책임 (GDPR, 금융감독원 가이드라인)
  - LLM 등장과 자연어 설명의 가능성, 환각 위험
- 1.2 연구 동기
  - 정형 데이터 + XAI + LLM RAG 결합의 학술적 공백
  - TabNet 어텐션과 SHAP의 보완 가능성 (선행연구 한계)
  - 신용평가 도메인의 공정성 / customer-facing 제약
- 1.3 연구 질문
  - RQ1: TabNet 어텐션과 SHAP 부호 정보를 융합한 컨텍스트가 LLM 자연어 설명의 사실성을 향상시키는가?
  - RQ2: 본 메커니즘이 다른 데이터셋에 일반화되는가?
  - RQ3: 사람 관점에서 fusion 설명이 어떻게 평가되는가? (clarity / actionability)
  - RQ4: 공정성 mitigation이 모델 성능 손실 없이 가능한가?
- 1.4 본 논문의 기여
  - C1: Agreement-aware fusion context (SHAP × TabNet attention) 신규 설계
  - C2: 3-tier 평가 (Rules + G-Eval Cross-judge + NLI) + Pilot Human-Proxy
  - C3: 4-mode 비교 (no_shap / generic_rag / shaponly / fusion) — 환각 차단 vs fact-grounding 차별 입증
  - C4: 일반화 검증 (UCI German Credit) — 약점 #5 해소
  - C5: Reweighing 기반 공정성 mitigation 4/4 통과
- 1.5 논문 구성

### **제2장 관련 연구** (8~10p)
- 2.1 신용평가 모델
  - 2.1.1 전통 ML (Logistic, XGBoost, LightGBM) — 표 1: 모델 비교
  - 2.1.2 정형 데이터 특화 딥러닝 (TabNet, FT-Transformer)
  - 2.1.3 본 연구의 위치 (XGBoost 메인 + TabNet 보조)
- 2.2 설명 가능한 인공지능 (XAI)
  - 2.2.1 SHAP (TreeSHAP)
  - 2.2.2 Attention 기반 해석 (TabNet, Transformer)
  - 2.2.3 SHAP vs Attention 일관성 선행 연구
- 2.3 거대언어모델 기반 자연어 설명
  - 2.3.1 RAG (Retrieval-Augmented Generation)
  - 2.3.2 환각 (Hallucination) 정의 및 측정
  - 2.3.3 LLM 환각 차단 기법 (hard constraint, fact-grounding)
- 2.4 공정성 (Fairness)
  - 2.4.1 4/5 rule, Disparate Impact, Equal Opportunity
  - 2.4.2 Pre/in/post-processing mitigation (Reweighing, Fairlearn)
- 2.5 LLM 평가 방법론
  - 2.5.1 G-Eval (LLM-as-a-Judge), Cross-judge 필요성
  - 2.5.2 NLI 기반 Faithfulness
  - 2.5.3 Persona pilot (Human evaluation의 LLM proxy)
- 2.6 본 연구와의 차별성

### **제3장 데이터 및 전처리** (5~7p)
- 3.1 데이터셋 개요
  - 3.1.1 Home Credit Default Risk (Kaggle, 307,511 × 122)
  - 3.1.2 UCI German Credit (일반화 검증용, 1,000 × 21)
- 3.2 탐색적 데이터 분석 (EDA)
  - 결측률, 부도율, 외부 신용평가 점수 분포, 보호 속성 분포
- 3.3 전처리 정책 (Day 1 결정 사항)
  - 3.3.1 결측치 (A1: 50%+ missing flag, EXT_SOURCE_* median + flag)
  - 3.3.2 범주형 인코딩 (B1: cardinality ≤8 → one-hot, OCCUPATION/ORGANIZATION → target encoding)
  - 3.3.3 데이터 분할 (D: 60/20/20 stratified, SEED=42)
  - 3.3.4 보호 속성 (F: 학습 feature로 포함, 평가 별도)
  - 3.3.5 DAYS_EMPLOYED sentinel 처리 (365243 → NaN + EMPLOYED_FLAG)

### **제4장 방법론** (12~15p)
- 4.1 시스템 전체 구조 — 그림 1: 4단계 파이프라인
- 4.2 예측 모델 학습
  - 4.2.1 XGBoost (메인 예측기)
  - 4.2.2 TabNet (보조 + 어텐션 추출, Optuna 최적화)
  - 4.2.3 5-fold CV 평가 + 임계값 결정 (Youden's J)
- 4.3 모델 해석
  - 4.3.1 SHAP local (XGBoost native pred_contribs 우회)
  - 4.3.2 TabNet attention (instance-level mask)
  - 4.3.3 SHAP × Attention 일관성 분석 (Spearman ρ, Top-K overlap)
- 4.4 Agreement-aware Fusion Context (★ 본 연구 핵심)
  - 4.4.1 3-그룹 분류: agreed / shap_only / attention_only
  - 4.4.2 민감 속성 마스킹 (CODE_GENDER, DAYS_BIRTH 등)
  - 4.4.3 JSON 컨텍스트 구조 — 그림 2
- 4.5 LLM-RAG 자연어 설명 생성
  - 4.5.1 Hard Constraints (5개 항목)
  - 4.5.2 4-mode 정의: no_shap / generic_rag / shaponly / fusion
  - 4.5.3 LLM 선택 — Gemini 2.5 Flash, Claude Sonnet 4.5 (cross-LLM)
  - 4.5.4 5-section 출력 형식
- 4.6 공정성-aware 학습
  - 4.6.1 Reweighing (Kamiran-Calders)
  - 4.6.2 Fairlearn ExpGrad (DP / EO 비교)

### **제5장 평가 프레임워크** (8~10p)
- 5.1 모델 성능 지표 (AUROC, AUPRC, KS, F1, Precision, Recall)
- 5.2 룰 기반 평가
  - 5.2.1 Faithfulness Score (변수/값/부호 토큰 매칭, fusion-aware)
  - 5.2.2 Hallucination Rate (strict / broad)
- 5.3 NLI 기반 Faithfulness (★ 본 연구 신규)
  - 5.3.1 mDeBERTa-v3-base-xnli-multilingual-nli
  - 5.3.2 컨텍스트 → premise 자연어 변환
  - 5.3.3 Entailment / Contradiction 측정
- 5.4 G-Eval (LLM-as-a-Judge)
  - 5.4.1 4 차원 척도: factual_accuracy / completeness / sensitive_leak / style
  - 5.4.2 Cross-judge (Anthropic ↔ Gemini) 검증
- 5.5 Pilot Human-Proxy 평가 (★ 본 연구 신규)
  - 5.5.1 3 Personas: Credit Expert / Customer / Regulator
  - 5.5.2 3 Metrics × 5점 척도: trustworthiness / clarity / actionability
- 5.6 공정성 평가
  - 5.6.1 Demographic Parity, Equal Opportunity
  - 5.6.2 4/5 rule, Disparate Impact ratio

### **제6장 실험 결과** (15~20p, 가장 분량 큼)
- 6.1 모델 성능 비교
  - 6.1.1 4 모델 5-fold CV (Home Credit) — 표 2
  - 6.1.2 보조 테이블 추가 효과 (AUROC 0.7587 → 0.7755, +2.22%)
- 6.2 SHAP × Attention 일관성
  - 6.2.1 Spearman ρ = 0.117 (전체), Top-50 ρ = -0.195 — 그림 3
  - 6.2.2 "부분 일관 + 부분 상보" 해석 → fusion 정당성
- 6.3 4-mode 자연어 설명 비교 — 표 3, 그림 4 (Step 5-B 통합)
  - 6.3.1 Hallucination 0% (모든 mode) — hard constraints 견고
  - 6.3.2 NLI Entailment 4-mode 단조증가: no_shap (0.27~0.43) < generic_rag (0.36~0.37) < shaponly (0.41~0.51) < fusion (0.62)
  - 6.3.3 G-Eval (Cross-judge) — Completeness 양 judge 일관 양수
  - 6.3.4 Value Match Rate: 0.59 → 0.73 → 0.85 → 0.90
- 6.4 Pilot Persona 평가 결과 (Step 5-C)
  - 6.4.1 사실성 (NLI/G-Eval) 1위 = fusion
  - 6.4.2 사람 친화성 (Persona trust) 1위 = generic_rag (4.91 vs fusion 4.31)
  - 6.4.3 ⚠ Customer clarity: generic_rag 4.93 vs fusion 2.67 (큰 격차)
  - 6.4.4 Trade-off 정량 입증
- 6.5 공정성 mitigation (Step 5-A)
  - 6.5.1 Reweighing 4/4 통과 (GENDER + AGE × baseline + aux)
  - 6.5.2 baseline AUROC -0.003, aux AUROC +0.003
  - 6.5.3 Fairlearn DP는 AUROC 큰 손실, EO는 DP 악화 안티 패턴
- 6.6 일반화 검증 (Step 5-D)
  - 6.6.1 UCI German Credit AUROC: Logistic 0.797, XGBoost 0.771, TabNet 0.750
  - 6.6.2 SHAP × Attention ρ = 0.114 (Home 0.117과 거의 동일) ★
  - 6.6.3 4-mode NLI 단조증가 패턴 일관 (fusion 0.711 1위)
  - 6.6.4 G-Eval Completeness 데이터셋별 차이 — Home (fusion 4.82 1위) vs German (generic_rag 4.58 1위)

### **제7장 논의** (5~7p)
- 7.1 핵심 발견 정리
  - 7.1.1 Fusion 메커니즘의 NLI 사실성 일관 1위 (양 데이터셋)
  - 7.1.2 환각 차단은 hard constraints 충분, fusion 차별성은 fact-grounded 정확성
  - 7.1.3 Mode 선택 = 응용 시나리오 × 데이터 복잡도 trade-off
- 7.2 한계 (Honest Reporting)
  - 7.2.1 표본 크기 30 인스턴스 (4-mode × 2 LLM, 통계 의미는 trends 수준)
  - 7.2.2 Persona pilot은 LLM proxy — 정식 IRB 인간평가 부재 (약점 #2 부분 해소)
  - 7.2.3 Customer clarity 약점 (fusion 2.67) — 본 연구의 정직한 한계
  - 7.2.4 Hard constraint 의존성 (제거 시 환각 가능성 증가, no_shap baseline 45.5% 입증)
  - 7.2.5 양 데이터셋 모두 한국어 humanize 매핑 의존 (자동 한국어 변환 framework 부재)
- 7.3 응용 시사점
  - 7.3.1 Audit / Regulation 시나리오: fusion mode (사실성 + agreement 라벨)
  - 7.3.2 Customer-facing 시나리오: generic_rag mode (clarity 우월)
  - 7.3.3 도메인 복잡도에 따른 mode 선택 가이드라인
- 7.4 향후 연구 방향
  - 7.4.1 정식 IRB 인간평가 (약점 #2 완전 해소)
  - 7.4.2 추가 데이터셋 (Australian Credit, Lending Club)
  - 7.4.3 3-way ablation (Attention 단독 검증)
  - 7.4.4 Customer-friendly 표현 정제 (fusion clarity 보완)

### **제8장 결론** (3~4p)
- 8.1 본 연구 요약
- 8.2 학술적 기여 (C1~C5 재정리)
- 8.3 산업적 시사점
- 8.4 마무리

### **참고문헌** (~30~50개)
- 인용 순 또는 가나다순 + abc순 (학위논문지침 ⑼ 가운데 결정 — **인용 순 추천**)
- 분야별 분포: 신용평가 (5~10), TabNet/XAI (5~7), LLM/RAG (5~7), 공정성 (3~5), 평가방법론 (5~7)

### **부록**
- A. 전체 SHAP global importance (Top 50)
- B. 보호 속성별 분포 표
- C. LLM 프롬프트 템플릿 전문 (4-mode 모두)
- D. G-Eval 루브릭 + NLI 모델 라벨
- E. 4-mode 설명 예시 (1 인스턴스 × 4 mode 직접 비교)
- F. 코드 저장소 안내 (GitHub repository)

---

## 표/그림 List (작성 시 채울 자료 매핑)

| ID | 제목 | 출처 |
|---|---|---|
| 표 1 | 모델별 5-fold CV (Home Credit) | results/cv_summary.csv |
| 표 2 | 보조 테이블 추가 효과 | results/cv_aux_vs_baseline.csv |
| 표 3 | 4-mode 평가 (Halluc/Faith/NLI/G-Eval) | results/generic_rag_summary.csv |
| 표 4 | Persona × mode trade-off | results/human_proxy_summary.csv |
| 표 5 | Reweighing 4/4 결과 | results/fairness_mitigation_v2.csv |
| 표 6 | UCI German Credit 4-mode | results/german_eval_summary.csv |
| 표 7 | Home vs German 비교 | results/step5d_comparison.csv |
| 그림 1 | 시스템 전체 파이프라인 | 새로 그릴 (또는 paper/midterm 활용) |
| 그림 2 | Fusion context 구조 | 새로 그릴 |
| 그림 3 | SHAP × Attention scatter | figures/16_attention_vs_shap_scatter.png |
| 그림 4 | 4-mode NLI 비교 | figures/35_generic_rag_4way.png |
| 그림 5 | Persona × mode 막대 | figures/36_human_proxy_personas.png |
| 그림 6 | Fairness mitigation | figures/33-34_fairness_mitigation_*.png |
| 그림 7 | Home vs German 일반화 | figures/41_generalization.png |

---

## 초록 Draft (영문 + 국문)

### English Abstract (~250 words)
> **Title**: Explainable Credit Scoring and Customer-Tailored Reports via Tabular-Specialized Deep Learning (TabNet) and LLM-based XAI-RAG
>
> Machine learning credit scoring models often face the dual demand of high predictive accuracy and transparent justification. While SHAP and attention-based explanations have emerged separately, their fusion in a single context for downstream LLM-based natural-language reports remains under-explored. This thesis proposes an *agreement-aware fusion context* that combines local SHAP values from XGBoost with TabNet attention masks, classifying drivers into three groups (agreed, shap_only, attention_only) for an LLM-RAG explanation pipeline.
>
> We evaluate the system on Home Credit Default Risk (307K samples, 214 features) and validate generalization on UCI German Credit (1K samples, 63 features). A three-tier evaluation framework — rule-based faithfulness, mDeBERTa-NLI entailment, and cross-judge G-Eval (Claude × Gemini) — combined with a 3-persona LLM-proxy pilot, demonstrates: (1) fusion explanations rank first in NLI entailment across both datasets (0.625 / 0.711), (2) hallucination is eliminated in all modes via hard constraints, but only fusion achieves fact-grounded faithfulness, (3) reweighing-based fairness mitigation passes the 4/5 rule on all four protected-attribute combinations with negligible AUROC change, and (4) mode selection follows an *application × data-complexity trade-off* — fusion fits audit/regulation, generic_rag fits customer-facing.
>
> The work contributes (a) a novel agreement-aware context schema, (b) a 3-tier + persona evaluation framework, (c) generalization evidence on a second dataset, and (d) honest reporting of customer-clarity weaknesses, identifying directions for future formal IRB human evaluation.
>
> **Keywords**: XAI, SHAP, TabNet, Credit Scoring, LLM, RAG, Fairness, Cross-judge, NLI

### 국문 초록 (~500자)
> 본 연구는 정형 데이터 특화 딥러닝 모델(TabNet)과 트리 기반 모델(XGBoost)의 해석 결과를 융합하여, 거대언어모델 기반 검색 증강 생성(LLM-RAG)으로 신용평가 설명 리포트를 자동 생성하는 시스템을 제안한다. SHAP local 기여도와 TabNet 어텐션 마스크를 인스턴스 수준에서 통합한 *동의 기반(agreement-aware) 융합 컨텍스트*를 설계하고, 동의 그룹·SHAP 단독 그룹·어텐션 단독 그룹 세 가지로 분해하여 LLM에 전달한다.
>
> Kaggle Home Credit Default Risk 데이터셋(307,511명) 위에서 4가지 컨텍스트 모드(no_shap, generic_rag, shaponly, fusion)를 비교하고, UCI German Credit(1,000명)에 동일 파이프라인을 이식하여 일반화 가능성을 검증한다. 평가 체계는 룰 기반 충실성, mDeBERTa-NLI 함의도, 교차 LLM G-Eval(Claude × Gemini), 3-페르소나 LLM 대리 평가의 4-tier 구조를 갖춘다.
>
> 주요 결과는 다음과 같다. (1) 융합 컨텍스트가 양 데이터셋에서 NLI 함의도 1위(0.625 / 0.711)를 달성한다. (2) hard constraint 조건에서 모든 모드의 환각률은 0%이지만, 사실 기반 정확성은 융합 모드에서만 우월하다. (3) Reweighing 기반 공정성 보정이 모델 성능 손실 거의 없이 보호 속성 4가지 조합 모두에서 4/5 규칙을 통과한다. (4) Persona 평가 결과 사실성 1위는 fusion이지만 고객 친화성 1위는 generic_rag로, *응용 시나리오 × 데이터 복잡도* 두 차원에 따른 모드 선택 trade-off가 정량 입증된다.
>
> **주제어**: 설명 가능한 인공지능, SHAP, TabNet, 신용평가, 거대언어모델, 검색 증강 생성, 공정성, 교차 평가, 자연어 추론

---

## 작업 단계 매핑 (Phase별)

| Phase | Chapter | 분량 | 예상 일수 |
|---|---|---|---|
| **Phase 1** ★ 현재 | Outline + Abstract draft | — | 0.5 |
| Phase 2 | 1장 서론 + 2장 관련 연구 | ~15p | 2~3일 |
| Phase 3 | 3장 데이터 + 4장 방법론 | ~20p | 2~3일 |
| Phase 4 | 5장 평가 + 6장 실험 결과 | ~25p | 3~4일 |
| Phase 5 | 7장 논의 + 8장 결론 + 참고문헌 + 부록 | ~15p | 2일 |
| Phase 6 | 표지/제출문/인준서 + 전체 검토 | — | 1~2일 |

**총 예상**: 12~15일 (사용자 검토 시간 별도)

---

## ★ Phase 1 결정사항 (사용자 확정)

| 항목 | 결정 |
|---|---|
| 논문 제목 | 정형 데이터 특화 딥러닝(TabNet)과 거대언어모델(LLM) 기반 XAI-RAG를 활용한 설명 가능한 신용 평가 및 사용자 맞춤형 리포트 생성 |
| 부제목 | 어텐션-SHAP 융합 컨텍스트와 다중 평가 프레임워크 적용 |
| 제출 학기 | 2026학년도 1학기 (2026년 6월) |
| 본문 구조 | **8장** (사용자 결정 — 세세한 구조 선호) |
| 참고문헌 양식 | IEEE [1] [2] 스타일 (학위논문지침 예시 일치) |
| Abstract | 영문 ~250 words + 국문 ~500자 (앞에 영문, 뒤에 국문) |
| 표/그림 | 표 7개 + 그림 6개 + 본문 추가 가능 |
| 코드 저장소 | 부록 F에 GitHub `Tim-Green0/tabnet-xai-rag-credit` 명시 |
| 참고 패턴 | 손지민 (2024, 박운상 교수 직지도) 학교 표준 패턴 + 본 연구 8장 확장 |

## 학교 표준 참조 (손지민 2024)

- 한국어 본문, 영문 + 국문 abstract (영문 먼저)
- 중심어 / Keywords
- IEEE [1] [2] 인용
- 1.1 연구 배경 및 목적, 1.2 논문의 구성, 1.3 연구 방법 및 범위 패턴

## Phase 진행 단계

- [x] Phase 0: 환경 셋업 + PDF 변환
- [x] Phase 1: outline + abstract draft + 결정사항 확정
- [ ] **Phase 2 (다음)**: 1장 서론 + 2장 관련 연구 작성 (markdown)
- [ ] Phase 3: 3장 데이터 + 4장 방법론
- [ ] Phase 4: 5장 평가 + 6장 실험 결과
- [ ] Phase 5: 7장 논의 + 8장 결론 + 참고문헌 + 부록
- [ ] Phase 6: docx 통합 (python-docx) + 학교 양식 적용 + 별지 (표지/제출문/인준서) + commit/merge/push
