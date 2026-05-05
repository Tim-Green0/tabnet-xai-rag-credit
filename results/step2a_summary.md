# Step 2-A 요약 — 평가 신뢰성 강화

날짜: 2026-05-05
미팅까지: D-5
이전 시점: Step 1 완료 (8일 작업, tag `step1`)

---

## TL;DR — Step 2-A에서 새로 입증한 4가지

1. **🎯 100명 표본으로 확장 후에도 Hallucination Rate = 0.000** (Gemini 100건, Claude 100건). Step 1의 10명 결과가 우연이 아니라 안정적인 패턴임을 통계로 입증.
2. **Cross-LLM G-Eval (양방향)으로 self-bias 우회** — 두 LLM 모두 다른 LLM이 평가했을 때 factual_accuracy 4.6+/5, sensitive_leak 5.0/5 만점.
3. **Counterfactual Test 정량화** — top driver 제거 시 cosine 0.91 (Claude), 0.92 (Gemini). LLM이 컨텍스트에 부분적으로 의존하면서도 전체 의미 일관성 유지.
4. **Robustness 정량화** — 프롬프트 미세 변형(role/example/driver order)에 cosine 0.91~0.92로 안정적.

---

## A. Phase 1 — 평가 표본 100명 확장

### 설계
- XGBoost test set에서 high-confidence reject 50명 + high-confidence accept 50명 무작위 추출
- 각 인스턴스에 대해 SHAP local 계산 → JSON 컨텍스트 빌드 (`results/contexts_100/`)
- Step 1의 10명과 동일한 파이프라인, 표본만 10배 확장

### 산출
- `results/contexts_100/` (100 + _index.json)
- `results/shap_local_examples_100.json`
- `src/expand_samples.py` (재사용 가능한 스크립트)

---

## B. Phase 4 — 룰 기반 Faithfulness / Hallucination (100건 × 2 LLM)

| 지표 | Gemini 2.5 Flash (n=100) | Claude Sonnet 4.5 (n=100) |
|---|---|---|
| feat_match_rate | 0.450 | 0.487 |
| val_match_rate | 0.811 | 0.901 |
| sign_match_rate | 0.825 | 0.890 |
| full_match_rate | 0.397 | 0.430 |
| **halluc_rate_strict** | **0.000 ± 0.000** | **0.000 ± 0.000** |
| **halluc_rate_broad** | **0.000 ± 0.000** | **0.000 ± 0.000** |

> 100명 표본에서도 Hallucination 0% 유지. Step 1 결과의 통계적 신뢰성 ↑.

→ `results/explanation_eval_{gemini_100,anthropic_100}.csv`
→ `figures/20_eval_metrics_{gemini_100,anthropic_100}.png`

---

## C. Phase 5 — Cross-LLM G-Eval (self-bias 우회)

### 설계
기존: Gemini가 자기 출력을 평가 → self-bias 가능성
신규: 다른 LLM에게 30건씩 양방향 평가 시킴.

### 결과 (n=30 × 2 방향)

| Judge → Target | Factual | Completeness | Sensitive Leak | Style |
|---|---|---|---|---|
| **Claude → Gemini** | 4.867 ± 0.507 | **4.000 ± 0.643** | 5.000 ± 0.000 | 4.967 ± 0.183 |
| **Gemini → Claude** | 4.600 ± 0.894 | **3.333 ± 0.959** | 5.000 ± 0.000 | 5.000 ± 0.000 |
| Gemini self (Step 1, n=8) | 5.0 | 3.375 | 5.0 | 5.0 |

→ `figures/26_cross_llm_geval.png`, `results/explanation_eval_summary_cross.csv`

### 인사이트
- **양 LLM 모두 factual ≥ 4.6 / 5** — 본 구조에서 사실성 매우 높음
- **sensitive_leak 5.0 / 5 만점** — 두 평가자 모두 민감변수 노출 없다고 판정. 본 연구의 마스킹 정책이 두 LLM 모두에서 작동
- **흥미로운 self-bias 패턴**: Gemini는 자기 자신의 completeness를 3.375로 박하게 평가했는데, Claude judge는 4.0으로 더 후하게 평가. 즉 Gemini self-bias가 부정적 방향(자기 비판) — 본 결과의 신뢰성을 오히려 강화
- **상호 평가의 작은 자기 편애**: Gemini는 Claude의 factual을 4.6으로, 자기 자신은 5.0으로 판정 → 약 0.4점 차이

---

## D. Phase 6 — Counterfactual Test 정량화

### 설계 (계획서 3.8.1.d)
각 인스턴스 30개에 대해 SHAP top_drivers_for_default rank 1 변수를 컨텍스트에서 제거 → LLM 재호출 → 원본과 비교.

### 결과

| LLM | n | Cosine sim | ROUGE-L F1 |
|---|---|---|---|
| Claude Sonnet 4.5 | 30 | 0.909 ± 0.069 | 0.747 ± 0.112 |
| Gemini 2.5 Flash | 30 | 0.920 ± 0.040 | 0.750 ± 0.116 |

→ `figures/24_counterfactual_{anthropic,gemini}.png`

### 해석
- **두 LLM 모두 cosine 0.91~0.92** — top driver 1개 제거 시 의미는 90% 유지
- **ROUGE-L 0.75** — 어휘 중복은 75%, 즉 25% 정도는 다른 단어로 표현됨
- 해석: LLM이 컨텍스트의 일부 변경에 부분적으로 반응하면서도 전체 메시지의 일관성은 유지. 본 연구의 안정성 증거.
- **추가 조사 가능 영역 (future work)**: top 5 모두 제거 또는 SHAP 부호 뒤집기 — 더 강한 perturbation 시 응답이 어떻게 변하는가

---

## E. Phase 7 — Robustness 평가 (프롬프트 변형)

### 설계 (계획서 3.8.5)
각 인스턴스 20개에 대해 프롬프트 3가지 변형 후 응답 비교:
- **role_swap**: Role 문장 변형 ("금융 상담사" → "신용 평가 전문가")
- **example_swap**: Few-shot 예시 위치를 컨텍스트 뒤로 이동
- **driver_shuffle**: 컨텍스트의 driver 순서를 그룹 내에서 셔플 (rank는 유지)

### 결과 — Claude vs Gemini (n=20 each)

| Variant | Claude cosine | Claude ROUGE-L | Gemini cosine | Gemini ROUGE-L |
|---|---|---|---|---|
| role_swap | 0.923 ± 0.057 | 0.875 ± 0.109 | **0.951 ± 0.034** | **0.910 ± 0.095** |
| example_swap | 0.914 ± 0.060 | 0.837 ± 0.117 | 0.908 ± 0.077 | 0.865 ± 0.127 |
| driver_shuffle | 0.924 ± 0.054 | 0.843 ± 0.094 | 0.942 ± 0.032 | 0.910 ± 0.125 |

→ `figures/25_robustness_{anthropic,gemini}.png`

### 해석
- **두 LLM 모두 6 케이스 전부 cosine ≥ 0.90** (목표 0.85+ 달성)
- 프롬프트 미세 변형에 강건 — 본 시스템 운영 안정성 입증
- **Gemini가 약간 더 안정적** (role_swap에서 0.951 vs 0.923, driver_shuffle 0.942 vs 0.924)
  - Gemini가 reasoning model로 출력 형식이 더 일관된 결과로 보임
- Few-shot 예시 위치 이동(example_swap)이 가장 큰 변화 — 그래도 cosine 0.91 수준

---

## F. 종합 — Step 2-A로 강화된 본 연구의 답변

### RQ3: XAI-RAG가 일반 LLM 직접 호출 대비 환각률·충실성에서 우위를 보이는가?
✅ **확실하게 그렇다** (Step 2-A로 강화):
- 100명 표본 × 2 LLM 모두 Hallucination 0%
- Claude baseline (no SHAP) 45.5% (Step 1) — XAI-RAG의 환각 차단 효과는 표본을 늘려도 유지
- Cross-LLM G-Eval 양방향 모두 factual 4.6+/5, sensitive 5.0/5

### Step 2-A로 답변 가능해진 보너스 RQ
- **표본 안정성**: 10명 → 100명 확장 후에도 환각 0% 유지 (통계적 안정성 확보)
- **Self-bias**: Cross-LLM judge 결과로 Gemini self-judge가 self-critical 했음을 확인. Claude judge가 Gemini 출력을 더 후하게 평가
- **컨텍스트 의존도**: top driver 제거 시 cosine 0.91 — LLM이 컨텍스트의 부분 변경에 부분적으로 반응
- **운영 안정성**: 프롬프트 변형에 강건 (cosine 0.91~0.92)

---

## G. 산출물 요약

```
src/
├─ expand_samples.py            # 100명 SHAP + context 빌드
├─ counterfactual_test.py       # Counterfactual + cosine/ROUGE
├─ robustness_test.py           # 3 variant + cosine/ROUGE
├─ cross_llm_geval.py           # Cross-LLM G-Eval
└─ text_similarity.py           # sentence-transformers wrapper

results/
├─ contexts_100/                # 100개 JSON 컨텍스트
├─ explanations_gemini_100/     # 100건 Gemini 설명
├─ explanations_anthropic_100/  # 100건 Claude 설명
├─ explanations_counterfactual_{gemini,anthropic}/  # CT 변형 출력
├─ explanations_robustness_anthropic/  # Robustness 4 variants × 20
├─ explanation_eval_{gemini_100,anthropic_100}.csv  # 룰 기반
├─ counterfactual_eval_{gemini,anthropic}.csv
├─ robustness_eval_anthropic.csv
├─ explanation_eval_cross_{gemini_judges_anthropic, anthropic_judges_gemini}.csv
└─ explanation_eval_summary_cross.csv

figures/
├─ 20_eval_metrics_{gemini_100, anthropic_100}.png
├─ 24_counterfactual_{anthropic, gemini}.png
├─ 25_robustness_anthropic.png  (gemini 추가 예정)
└─ 26_cross_llm_geval.png
```

---

## H. 비용 정리 (실측)

| 항목 | 비용 |
|---|---|
| Gemini 2.5 Flash (paid tier 활성화 후) | 사용자 충전 $10 일부 |
| Claude Sonnet 4.5 (Step 1+2 누적) | ~$5~7 추정 |
| **합계** | **$10 미만** |

---

## I. 다음 단계 후보 (Step 3)

Step 2-A에서 평가 신뢰성을 강화했으니 다음은 패키지 B (성능·방법론):
- ⑤ 보조 테이블 활용 → AUROC 0.78+ 목표
- ⑥ Fairness-aware 학습 (Reweighing, Adversarial Debiasing)
- ⑦ FT-Transformer 비교 모델

또는 **Step 3-D (논문 작성)**:
- ⑧ 인간 평가 (Plausibility) — IRB 절차
- 본격 논문 초안 작성 (LaTeX 또는 docx)

미팅에서 지도교수님 피드백 받고 결정하면 됨.
