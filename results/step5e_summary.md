# Step 5-E 요약 — Customer-friendly fusion vs original fusion

## 동기 (Step 5-C+5-D 일관 약점)
- Step 5-C Customer persona clarity: fusion 2.67 (vs generic_rag 4.93)
- Step 5-D G-Eval Factual: fusion 3.97/2.97 (vs generic_rag 5.0/5.0)
→ fusion의 SHAP 부호 raw + agreement 라벨이 over-explain으로 평가됨.

## 변경 사항
- 같은 fusion context 그대로 사용
- Prompt만 변경: SHAP 부호 자연어화, agreement 라벨 직관 표현, 정성 표현 추가, 친근한 톤
- 새 mode: `fusion_friendly`

## NLI Entailment (사실성 유지 여부, ↑)

| Dataset | fusion | fusion_friendly | Δ |
|---|---|---|---|
| home | 0.625 | 0.506 | -0.118 |
| german | 0.711 | 0.541 | -0.170 |

## G-Eval Factual Accuracy (★ Step 5-D fusion 약점 차원)

| Dataset | fusion | fusion_friendly | Δ |
|---|---|---|---|
| home | 4.833 | 4.700 | -0.133 |
| german | 3.467 | 3.933 | +0.467 |

## G-Eval Completeness (충실성)

| Dataset | fusion | fusion_friendly | Δ |
|---|---|---|---|
| home | 4.817 | 4.600 | -0.217 |
| german | 3.767 | 4.250 | +0.483 |

## G-Eval Style (친근함)

| Dataset | fusion | fusion_friendly | Δ |
|---|---|---|---|
| home | 4.967 | 4.967 | 0.000 |
| german | 4.683 | 4.850 | +0.167 |

## Value Match Rate (값 정확 인용)

| Dataset | fusion | fusion_friendly | Δ |
|---|---|---|---|
| home | 0.882 | 0.608 | -0.274 |
| german | 0.695 | 0.441 | -0.254 |

## ★ 핵심 발견 — 데이터셋 복잡도에 따른 trade-off

### 1. 사실성 (NLI / Value Match) → friendly가 일관 하락
- NLI: -0.12 (Home) / -0.17 (German)
- Val Match: -0.27 (Home) / -0.25 (German)
- 원인 분석:
  1. **정성 표현 추가**("매우 큰 영향", "결정적인 요인")가 raw 수치 + sign 표현과 매칭 약화
  2. **출력 길이 증가** (~600 → ~1000 tokens)로 value 인용 비율 감소 (절대 정확도는 비슷)
  3. NLI premise는 fusion context의 raw 표현 자연어화 → friendly의 자연어화된 표현이 entailment에서 약해짐 (modeling artifact 측면 포함)

### 2. G-Eval (LLM judge) → 데이터셋별 정반대 결과
| 차원 | Home Credit (복잡, 214 f) | German Credit (단순, 63 f) |
|---|---|---|
| Factual | -0.13 (4.83→4.70) | **+0.47** (3.47→3.93) ⬆️ |
| Completeness | -0.22 (4.82→4.60) | **+0.48** (3.77→4.25) ⬆️ |
| Style | 0 (4.97→4.97) | **+0.17** (4.68→4.85) ⬆️ |

→ **German Credit (단순 데이터)에서 friendly가 Step 5-D fusion 약점 보완 성공**
→ Home Credit (복잡 데이터)에서는 fusion이 이미 우수, friendly 개선 효과 없음

### 3. 비용 (실측 + 추정)
- Friendly explanation 120 호출 (Anth+Gem): ~$1.45
- G-Eval friendly 120 호출 (Anth judge): ~$1.30
- 합계: ~$2.75

## 결론 — 본 연구 메시지 3차원 정교화

### 종전 메시지 (Step 5-C+5-D 통합)
> "fusion = NLI 사실성 1위, but customer clarity / G-Eval factual 약점"

### 새 메시지 (Step 5-E 통합)
> "**Mode 선택 trade-off는 3 차원 결합**:
> (1) 응용 시나리오 (audit/regulation vs customer-facing)
> (2) 평가 차원 (NLI fact-grounding vs LLM judge satisfaction)
> (3) **데이터 복잡도** (단순한 도메인에서만 friendly 효과 큼)
>
> Customer-facing + 단순 도메인 → **fusion_friendly** 권장
> Audit/regulation + 복잡 도메인 → **fusion** 유지
> Cf. 단순 generic_rag도 단순 도메인에서 강력 (Step 5-D)"

### Honest Findings 추가
- friendly는 사실성을 trade-off하지 않으면 customer clarity 개선 불가
- 한쪽 mode가 모든 차원에서 1위인 경우 없음 (Step 5-C, 5-D, 5-E 일관)
- 데이터셋 복잡도가 fusion 우월성에 결정적 영향 (Step 5-D 최초 발견 → 5-E에서 재확인)

## 다음 단계 후보

### 우선순위 갱신
- ~~1순위 Customer-friendly~~ ✅ Step 5-E (fact-grounding과의 명확한 trade-off 정량 입증)
- 새 1순위: **3-way ablation** (Attention-only 단독 검증) — fusion 우월성 attention 기여 isolate
- 새 2순위: **하이브리드 prompt 실험** — 사실성 손실 최소화하면서 friendly 일부 효과만 유지 (e.g., 정성 표현 옵션화)
- 새 3순위: **Australian Credit 추가 데이터셋** — 1000~5000 샘플, 복잡도 중간 → friendly 효과 검증

## 산출 파일

- `results/explanations_friendly_{home,german}_{anthropic,gemini}_30/` (총 120 explanations)
- `results/step5e_friendly_eval.csv` (120 rows)
- `results/step5e_comparison.csv` (70 rows: friendly + fusion 양 데이터셋)
- `figures/42_friendly_vs_fusion.png`
