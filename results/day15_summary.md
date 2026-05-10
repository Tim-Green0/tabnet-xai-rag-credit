# Day 15 / Step 5-C 요약 — Pilot Human-Proxy Evaluation

날짜: 2026-05-10 (미팅 일정 연기 후)
이전 시점: Step 5-B 완료 (tag `step5b`, Generic RAG 4-way)

---

## TL;DR — Step 5-C의 핵심 5가지

1. **🎯 약점 #2 (LLM 평가 객관성) 부분 해소** — 정식 IRB 인간평가의 pilot 대안으로 3 페르소나(Credit Expert / Customer / Regulator) LLM-proxy 평가. 3 metrics × 3 personas × 4 modes × 2 LLM target × 15 instances = 276개 평가 완료.
2. **🎯🎯 의외의 발견 — Trade-off 명확**: Persona 관점에서 **Generic RAG가 fusion/shaponly를 압도**. 본 연구의 fusion 메커니즘이 사실성 1위지만 친근함은 Generic RAG가 우위.
3. **🎯 Customer perspective의 약점**: SHAP-RAG와 Fusion 모두 customer clarity 2.67~2.80 (5점 만점) — SHAP 부호/agreement 라벨 같은 기술적 표기가 일반 고객에게 낯섦. honest 인정 필요한 한계.
4. **🎯 다층 평가 메시지**: "본 연구의 가치 = 응용 시나리오에 따른 mode trade-off 정량 입증" — 단순 우월성 주장이 아니라 적합한 응용 식별 가능.
5. **Case study 6 instances** — reject 3 + accept 3, 4 mode 출력 정성 비교로 정량 결과 보강.

---

## A. 동기

정식 IRB 인간평가는 1.5~2개월 행정 부담으로 미팅 시점 진행 어려움. Pilot 대안:
- **3 LLM personas** = 다양한 stakeholder perspective의 사람 대리(human proxy)
- 5점 척도 + rationale로 정량 측정
- "informal pilot" 형태로 논문에 포함 (정식 인간평가는 future work)

---

## B. 설계

### B.1 Personas
| Persona | 평가 관점 |
|---|---|
| **Credit Expert** (10년 경력) | 전문가 신뢰성 — 결정 근거의 정확성, 분야 일반 원리 부합성 |
| **Customer** (대출 신청자 본인) | 일반 고객 납득도 — 전문 용어 없이 이해 가능한가, 행동 지침 명확한가 |
| **Regulator** (금융감독원 평가자) | 규제 요건 만족도 — 사실성, 추적 가능성, 보호 변수 마스킹 |

각 persona × 5점 척도:
- **trustworthiness** (신뢰할 만한가)
- **clarity** (이해 가능한가)
- **actionability** (행동 지침이 명확한가)

### B.2 표본
- 30 idx 중 random 15개 (SEED=42, fusion `_index.json`에서)
- 4 modes (no_shap / generic_rag / shaponly / fusion) × 2 LLM target (Anthropic + Gemini)
- 4 × 2 × 15 × 3 personas = **360 평가** (no_shap는 표본 작아 실제 276)
- Judge: Claude (Anthropic, Step 3-C-2-f에서 안정성 검증됨)

---

## C. 결과 (Anthropic target, n=15 each, no_shap n=1)

### C.1 Persona별 mean (5점 만점)

| Persona | Metric | no_shap* | generic_rag | shaponly | fusion |
|---|---|---:|---:|---:|---:|
| **Credit Expert** | trustworthiness | (2.0) | **4.80** ★ | 4.33 | 4.73 |
| | clarity | (4.0) | **4.93** ★ | 4.67 | 4.73 |
| | actionability | (3.0) | **4.73** ★ | 3.40 | 3.73 |
| **Customer** | trustworthiness | (5.0) | **4.93** ★ | 3.53 | 3.67 |
| | clarity | (5.0) | **4.93** ★ | 2.80 ⚠️ | 2.67 ⚠️ |
| | actionability | (5.0) | **4.33** ★ | 2.33 ⚠️ | 2.80 |
| **Regulator** | trustworthiness | (5.0) | **5.00** ★ | 4.53 | 4.53 |
| | clarity | (5.0) | **5.00** ★ | 4.40 | 4.53 |
| | actionability | (5.0) | **5.00** ★ | 3.33 | 3.60 |

*no_shap n=1 (target 30 idx와 Step 1 baseline_no_shap 표본의 교집합), 통계적 의미 없음

### C.2 Trustworthiness running mean (그룹별)

| Mode | Anthropic 최종 | Gemini 최종 |
|---|---:|---:|
| generic_rag | **4.91** ★ | **4.53** |
| fusion | 4.31 | 4.20 |
| shaponly | 4.13 | 3.89 |
| no_shap | (n=1) | (n=1) |

양 LLM target에서 일관: **generic_rag > fusion > shaponly**.

---

## D. ★ 충격적 발견 — Trade-off 명확

### D.1 Persona 관점 vs NLI/G-Eval 관점 차이

| 평가 차원 | 1위 | 2위 | 비고 |
|---|---|---|---|
| **사실성 (NLI Entailment)** | fusion 0.62 | shaponly 0.41~0.51 | Step 3-C-2 결과 |
| **충실성 (G-Eval Completeness)** | fusion 4.97 | generic_rag 4.83 | Step 5-B 결과 |
| **사람 친화성 (Persona trustworthiness)** | **generic_rag 4.91** | fusion 4.31 | **Step 5-C 결과** |
| **Customer clarity** | **generic_rag 4.93** | shaponly 2.80 | **Step 5-C 발견** |

### D.2 발견의 의미

**본 연구의 fusion 메커니즘은 사실성 1위지만 사람 친화성은 1위 아님.**

이는 단순한 약점이 아니라 **trade-off 정량 입증**:
- **Fusion (SHAP + Attention agreement-aware)**: fact-grounded 정확성에 강함, audit/regulation 적합
- **Generic RAG (도메인 지식 chunks)**: 친근한 표현, customer-facing UI 적합
- **응용 시나리오에 따라 mode 선택 가이드 가능**

### D.3 Customer Clarity의 SHAP-RAG/Fusion 약점

특히 흥미로운 발견 — Customer perspective에서 SHAP-RAG/Fusion clarity 2.67~2.80 (만점 5):
- "두 모델이 동의한 강한 신호", "부도 가능성을 높이는 요인" 같은 기술적 표기
- SHAP 부호의 의미를 일반 고객이 직관적으로 이해 어려움
- Generic RAG는 도메인 chunks의 자연스러운 한국어 설명 + raw value만 인용

→ **본 연구의 한계 honest 인정**: customer-facing 시나리오에서는 추가 표현 정제 필요

---

## E. Case Study (6 instances 정성 비교)

`results/case_study.md`에 별도 저장:
- reject 3 (idx 10629, 18074, 45114) + accept 3 (idx 7687, 28952, 29583)
- 4 mode 출력 비교 (Anthropic target)
- 정성 노트 (자동 분석):

| Mode | clarity (전문 용어 적음) | specificity (수치 인용) | actionability (개선 권고 구체성) |
|---|---|---|---|
| no_shap | 4 (자유, 환각 위험) | 3 | 4 (자유 권고, 환각 동반) |
| generic_rag | **5** (자연스러움) | 4 | **5** (chunks 가이드 풍부) |
| shaponly | 3 (SHAP 부호 기술적) | **5** | 3 (driver 중심) |
| fusion | 3 (agreement 라벨 기술적) | **5** | 4 (다층 정보) |

→ 정량(persona scores)과 정성(case study) 결과 일관.

---

## F. 핵심 메시지 종합

### F.1 약점 #2 (LLM 평가 객관성) 부분 해소
- 자동 평가 (Rules + G-Eval Cross-judge + NLI) ≠ 인간평가
- Pilot human-proxy로 정식 IRB 전 단계 평가 진행
- 본 평가는 informal pilot, 정식 인간평가는 future work로 명시

### F.2 본 연구 메시지 재정의
이전 단순 메시지: "fusion이 모든 메트릭에서 최고"
Step 5-C 후 정교한 메시지:
> "본 XAI-RAG fusion 메커니즘은 사실성·충실성에서 압도적이지만, customer-facing 친근함에서는 generic RAG가 우위. **응용 시나리오에 따른 mode 선택 trade-off가 정량 입증됨** — audit/regulation은 fusion, customer UI는 generic RAG, 또는 향후 두 표현을 결합하는 hybrid 형태."

### F.3 Honest reporting 강화
- "환각 차단" (Step 1 메시지) → "환각 차단은 hard constraints로도 가능, 차별성은 fact-grounded 정확성" (Step 5-B)
- "fact-grounded 정확성 1위" (Step 5-B) → "사실성 1위 + 친근함은 trade-off, 응용에 따라 선택" (Step 5-C)

각 step마다 메시지 정교화 + 한계 정직 인정 → 학술 논문의 표준 진행.

---

## G. 한계 (Pilot 평가의 본질적 한계)

| 한계 | 영향 |
|---|---|
| LLM persona는 사람 proxy | 진짜 인간 사고는 다를 수 있음. 정식 IRB 평가가 최종 검증. |
| Judge LLM은 Claude 단일 | Cross-judge cross-validation은 future work. |
| n=15 표본 | 95% CI 큼. 효과 크기는 명확하지만 정밀도는 제한. |
| no_shap n=1 | 통계적 의미 없음. 표본 확장 필요. |
| Customer persona의 "일반 고객" 정의 | 실제 고객 다양성 못 반영. |

→ **모든 한계 day15_summary와 미팅 자료에 명시** (honest reporting)

---

## H. 산출물

### 새 코드
- `src/human_proxy_eval.py` — 3 persona × 5점 척도 평가
- `src/case_study.py` — 정성 case study 자동 생성

### 새 결과
- `results/human_proxy_eval.csv` (276 rows)
- `results/human_proxy_summary.csv`
- `results/case_study.md` (6 instances 정성 비교)

### 새 figure
- `figures/36_human_proxy_personas.png` — 3 metrics × 3 personas × 4 modes × 2 LLM

---

## I. 다음 단계

5가지 약점 중 #1, #2(부분), #3, #4 해소. 남은 #5 (UCI German Credit, 데이터 다양성).

| 순위 | 작업 | 약점 | 기간 |
|---|---|---|---|
| 1 | **UCI German Credit 일반화** | #5 해소 | 3~4일 |
| 2 | 정식 IRB 인간평가 | #2 완전 해소 | 1.5~2개월 (장기) |
| 3 | 3-way ablation (SHAP-only / Attn-only / Fusion) | 보강 | 3~4일 |
| 4 | Bureau ablation | 보강 | 1~2일 |

미팅 후 또는 미팅 일정 연장 추가 시 1~3순위 진행 가능.
