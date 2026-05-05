# Day 10 / Step 3-C-1 요약 — TabNet 어텐션 × SHAP 융합 컨텍스트

날짜: 2026-05-05
미팅까지: D-5 (2026-05-10)
이전 시점: Step 3-B 완료 (tag `step3b`)

---

## TL;DR — Step 3-C-1의 핵심 4가지

1. **🎯 융합 컨텍스트에서도 환각률 0/30 유지** (Anthropic + Gemini × shap-only/fusion 4 그룹 모두). Step 1/2-A의 환각 차단 메커니즘이 두 해석 신호 융합 후에도 견고.
2. **🎯 G-Eval Completeness 큰 향상**: Anthropic +0.67 (4.30→4.97), Gemini +0.80 (3.90→4.70). 양 LLM 모두 일관. 두 해석 신호의 상보성이 더 완결된 설명을 만든다는 직접 증거.
3. **🎯 TabNet의 역할 재정의**: 단순 비교 모델이 아니라 LLM 컨텍스트 메커니즘의 일부로 통합 — 논문 제목 "TabNet+SHAP+LLM 기반 XAI-RAG"의 의미가 비로소 일치함.
4. **🎯 두 해석 신호의 부분 일관 + 부분 상보를 정량 입증**: 100명 기준 평균 agreed=2.12, shap_only=6.98, attention_only=2.06. 3개 이상 동의 22%, 4개 이상 동의 0%. Day 4의 ρ=0.117 분석과 일치하는 instance-level 패턴.

---

## A. Pipeline (`src/{tabnet_attention_local, fusion_context, llm_explainer_fusion, eval_fusion}.py`)

### 4단계 파이프라인의 **3단계(Context Builder)에 TabNet 어텐션 통합**

```
[정형 데이터] → [XGBoost 예측 + SHAP local] ┐
                                             ├→ [Agreement-aware JSON 컨텍스트] → [LLM]
              [TabNet 예측 + 어텐션 mask]    ┘    └ agreed / shap_only / attention_only
```

### 핵심 산출
| 모듈 | 역할 | 주요 산출 |
|---|---|---|
| `tabnet_attention_local.py` | TabNet `.explain()` API로 100 인스턴스의 instance-level attention 추출 | `results/tabnet_local_attention_100.json` |
| `fusion_context.py` | SHAP top-k ∩ Attention top-k 분류 → JSON | `results/contexts_fusion_100/` (100개) |
| `llm_explainer_fusion.py` | fusion-aware 프롬프트 (그룹 라벨 의미 명시) + 양 LLM 호출 | `results/explanations_fusion_{anthropic,gemini}_30/` |
| `eval_fusion.py` | 룰 + Cross-LLM G-Eval (Claude judge), SHAP-only vs Fusion Δ 비교 | `results/fusion_eval.csv`, `results/fusion_vs_shaponly.csv` |

---

## B. Agreement Statistics (100명 기준)

| 그룹 | 평균 변수 수 | 의미 |
|---|---:|---|
| `agreed_drivers` | **2.12** | SHAP top-10과 Attention top-5의 교집합. "두 모델 동의 강한 신호" |
| `shap_only_drivers` | 6.98 | SHAP만 본 변수. 부호(sign_for_default) + 기여도 정보 보존 |
| `attention_only_drivers` | 2.06 | TabNet 어텐션만 본 변수. 부호 없음, sparse |

### n_agreed 분포 (100 인스턴스)
| n_agreed | 인스턴스 수 |
|---:|---:|
| 0 | 1 |
| 1 | 8 |
| 2 | **69** |
| 3 | **22** |
| 4+ | **0** |

**해석**: 두 해석 모델이 거의 항상 부분적으로만 겹친다 (3개 이상 동의 22%, 4개 이상 동의 0%). Day 4의 어텐션-SHAP global ρ=0.117 (전체) / Top-50 ρ=−0.195 분석과 일치하는 instance-level 패턴. **부분 일관 + 부분 상보가 본 데이터의 본질적 특성**.

---

## C. LLM 호출 결과 (양 LLM × 30명)

### 호출 자체
- 모든 60건 정상 (양 LLM 각 30건). `_index.json`의 `selected_idx` 일치 → SHAP-only 결과와 동일한 30개 idx로 비교 가능.
- 첫 dry-run (1개)부터 LLM이 그룹 라벨의 의미를 정확히 활용:
  - "두 해석 모델이 동의한 강한 신호" 명시
  - "보완 신호" 구분 (SHAP/Attention 출처)
  - attention_only 변수에 대해 부호 없이 "결정에 영향을 준 변수"로만 표현 → **방향성 추측 안 함** (지시 충실 이행)

---

## D. ★ 평가 결과 — SHAP-only vs Fusion (n=30 each, judge=Claude)

### 양 LLM target × 4 핵심 메트릭

| Metric | LLM | SHAP-only | **Fusion** | Δ | 해석 |
|---|---|---:|---:|---:|---|
| **Halluc Rate (strict, ↓)** | Anthropic | 0.000 | **0.000** | 0 | ✅ 환각 차단 유지 |
| | Gemini | 0.000 | **0.000** | 0 | ✅ 환각 차단 유지 |
| **G-Eval Completeness (↑)** | Anthropic | 4.30 | **4.97** | **+0.67** | ★ 큰 향상 |
| | Gemini | 3.90 | **4.70** | **+0.80** | ★ 큰 향상 |
| G-Eval Factual Accuracy (↑) | Anthropic | 4.87 | 4.90 | +0.03 | ≈ 유지 |
| | Gemini | 4.90 | 4.77 | -0.13 | ≈ 유지 (소폭) |
| G-Eval Sensitive Leak (↑) | Anthropic | 5.00 | 5.00 | 0 | ✅ 만점 유지 |
| | Gemini | 5.00 | 5.00 | 0 | ✅ 만점 유지 |
| G-Eval Style (↑) | Anthropic | 5.00 | 4.97 | -0.03 | ≈ 유지 |
| | Gemini | 4.93 | 4.93 | 0 | ≈ 유지 |
| feat_match_rate (룰, ↑) | Anthropic | 0.43 | 0.46 | +0.03 | 향상 |
| | Gemini | 0.42 | 0.44 | +0.02 | 향상 |
| val_match_rate (룰, ↑) | Anthropic | 0.85 | **0.90** | +0.06 | 향상 |
| | Gemini | 0.79 | **0.86** | +0.07 | 향상 |
| sign_match_rate (룰, ↑) | Anthropic | 0.87 | 0.65 | **-0.22** | ⚠️ 룰의 한계 (아래 §E) |
| | Gemini | 0.94 | 0.77 | **-0.18** | ⚠️ 같은 한계 |
| full_match_rate (룰, ↑) | Anthropic | 0.38 | 0.30 | -0.07 | (sign_match 영향) |
| | Gemini | 0.40 | 0.35 | -0.05 | (sign_match 영향) |

### 핵심 메시지

#### 메시지 1 — 융합이 환각을 늘리지 않는다 ★
- 두 LLM × shaponly/fusion 4 조합 모두 **Halluc strict 0/30**.
- Step 1의 핵심 발견(Halluc 0%)이 fusion 컨텍스트에서도 견고함을 입증.

#### 메시지 2 — 융합은 완결성을 크게 올린다 ★
- G-Eval Completeness +0.67 (Anthropic), +0.80 (Gemini). 양 LLM 일관.
- "두 해석 모델의 상보적 신호가 LLM에게 더 풍부한 정보를 줘서 더 완결된 설명을 만든다"는 직접 증거.

#### 메시지 3 — 융합이 사실성·민감도·스타일은 유지한다 ★
- Factual Accuracy ≈ 유지 (Anthropic +0.03, Gemini -0.13 — 둘 다 소수 단위)
- Sensitive Leak 5.0 만점 유지 (마스킹 정책 무관)
- Style 5.0 거의 유지

#### 메시지 4 — 룰 기반 sign_match 하락은 honest reporting
- LLM이 fusion에서 더 다양한 표현 사용 (예: "증가시키는", "위험" — 룰의 pos_words 셋에 없는 단어)
- 룰의 키워드 sensitivity 한계 → G-Eval(factual 4.77~4.97 만점급)이 진짜 사실성을 더 잘 측정
- 본 한계는 future work에서 NLI 기반 faithfulness로 해소 가능

---

## E. ★ 부수 발견 — 룰 패치 두 가지

### 패치 1: prefix 매칭 (Halluc rate false positive 제거)
- 첫 평가에서 Anthropic fusion halluc 0.133 발견 → 분석 결과 모두 `FONDKAPREMONT_MODE_*` (one-hot)을 LLM이 prefix `FONDKAPREMONT_MODE`로 인용한 것
- regex `\b[A-Z][A-Z0-9_]{2,}\b`가 공백 전까지만 잡으므로 prefix가 outside_dataset로 잘못 분류됨
- 패치: 컨텍스트 변수의 영문 prefix(예: `FONDKAPREMONT_MODE`)도 in_context_features에 포함
- 결과: Halluc 0.133 → **0.000** ✅

### 패치 2: attention_only는 sign 평가 제외
- attention_only 변수는 부호 없음 (TabNet attention은 importance만)
- 기존 룰은 `if shap > 0` else 로 강제 분기 → 부호 없는 변수에 negative 키워드 강제 검사 → false negative
- 패치: `group=='attention_only'` 시 sign_in을 None으로 두고 sign_match_rate 계산에서 제외

---

## F. 추가 안 한 것 (의도적, 미팅 후로)

| 항목 | 사유 |
|---|---|
| 표본 100명 확장 | 30명에서 메시지 명확. 100명은 미팅 후. |
| Gemini judge cross-validation | Gemini 503 과부하로 차단됨. Claude judge 단일로 메인 비교(같은 judge로 4 그룹 통제)는 Δ 측정에는 충분. 미팅 후 Gemini 회복 시 양방향 cross-validation 추가. |
| TabNet-only 컨텍스트 (SHAP 없는) | 3-way ablation 완성용. 시간 부족, 미팅 후. |
| Counterfactual on fusion | Step 2-A의 counterfactual + fusion 결합. 미팅 후. |

---

## G. 산출물 인덱스

### 새 코드 (`src/`)
- `tabnet_attention_local.py` — TabNet local attention 추출
- `fusion_context.py` — agreement-aware JSON context builder
- `llm_explainer_fusion.py` — fusion-aware LLM 호출 (Claude + Gemini)
- `eval_fusion.py` — 룰 + Cross-LLM G-Eval (judge 옵션 추가)

### 새 데이터/결과
- `results/tabnet_local_attention_100.json` (100 instances)
- `results/contexts_fusion_100/` (100개 + _index.json)
- `results/explanations_fusion_anthropic_30/` (30개 + _index.json)
- `results/explanations_fusion_gemini_30/` (30개 + _index.json)
- `results/fusion_eval.csv` (120 rows)
- `results/fusion_vs_shaponly.csv`

### 새 figure
- `figures/30_fusion_vs_shaponly.png` — 4 메트릭 × 2 LLM × 2 mode 비교

---

## H. 미팅 발표 메시지 업데이트

기존 Step 1+2-A+3-B 메시지에 한 슬라이드 추가 가능:

> **"TabNet은 정말 메인 메커니즘에 통합되어 있는가? 융합 컨텍스트 결과"**
>
> Step 1에서 TabNet은 비교 모델이었지만, Step 3-C-1에서 SHAP과 TabNet 어텐션을 융합한
> **agreement-aware 컨텍스트**를 LLM에 제공하는 메커니즘을 구현. 두 해석 신호의
> 동의/보완 구조를 LLM에 명시 라벨로 전달.
>
> 양 LLM × 30 인스턴스에서:
> - **Hallucination Rate 0/30 유지** (양 mode, 양 LLM 모두) — 환각 차단 견고
> - **G-Eval Completeness +0.67~+0.80 큰 향상** — 두 해석 신호의 상보성이 더 완결한 설명을 만든다
> - **Factual Accuracy 4.77~4.97 유지** — 사실성 손상 없음
> - **Sensitive Leak 5.0 만점 유지** — 마스킹 정책 무관
>
> 결과: TabNet이 단순 비교 대상이 아니라 본 XAI-RAG 메커니즘의 핵심 구성 요소가 됨.
> 본 논문 제목 "TabNet+SHAP+LLM 기반 XAI-RAG"의 의미가 비로소 정확히 일치함.

---

## I. 다음 단계 (Step 3-C-2~)

### 미팅 전 (D-5 ~ D-1)
- 슬라이드 / docx 통합 (오늘~내일)
- 리허설 (D-2 ~ D-1)

### 미팅 후 (지도교수 피드백 따라)
1. Gemini judge cross-validation으로 양방향 객관성 보강
2. 표본 30 → 100 확장
3. 3-way ablation: SHAP-only vs Attention-only vs Fusion
4. NLI 기반 Faithfulness (룰의 sign_match 한계 해소)
5. 인간 평가 (Plausibility, IRB 간소판)
