# Day 12 / Step 3-C-2-f 요약 — Gemini judge G-Eval Cross-Validation

날짜: 2026-05-05
미팅까지: D-5
이전 시점: Step 3-C-2 (NLI) 완료 (commit `09fda26`)

---

## TL;DR — Step 3-C-2-f의 핵심 4가지

1. **🎯 양 judge에서 fusion Completeness 큰 향상 일관**: Claude judge +0.67/+0.80, Gemini judge +0.90/+1.10. fusion 효과가 judge 종속 아님을 명확히 입증.
2. **🎯 Cross-judge가 필요한 이유 직접 증거**: Gemini target의 G-Eval Factual에서 Claude judge는 -0.13 (사실상 동등), Gemini judge는 **+0.47**. 단일 judge였다면 잘못된 결론을 낼 뻔.
3. **🎯 Sensitive Leak 5.0/5.0 양 judge 만점 유지**: 마스킹 정책의 LLM 종속성 없음을 다층 검증.
4. **🎯 Step 2-A의 Cross-LLM 패턴이 fusion 평가에서도 재현**: future work 항목 1개 해소 + 본 연구의 평가 신뢰성 한 단계 더 강화.

---

## A. 동기

Step 3-C-2까지의 평가는 Claude judge 단독으로 4 그룹(SHAP-only/Fusion × Anthropic/Gemini target)을 평가했음. 이는 다음 위험을 안고 있었음:
- **Self-bias**: Claude judge가 Anthropic target(자기 출력)을 더 호의적으로 평가할 가능성
- **Single-judge 종속**: 한 LLM의 평가 성향에 의존하는 결과

미팅까지 시간 여유 있고 Gemini API 503 회복(dry-run으로 확인) → cross-validation 추가 진행.

---

## B. Pipeline

### 평가 흐름
1. 기존 Claude judge 결과 백업 (step3c1 시점 csv 복원)
2. `eval_fusion.py --judge gemini --geval-sleep 6` 실행
   - 4 그룹 × 30 인스턴스 = 120 평가
   - retry 로직: 30s → 60s → 120s → 240s 백오프
3. `cross_judge_analysis.py`로 두 judge 결과 머지 + 비교 figure

### 503 처리
- 총 transient retry: **40회** (40/120 = 33% 호출에서 retry 발생)
- 진짜 fail (4/4): **0회** (모든 호출 회복)
- 결과 누락: **0건**, 양 judge 모두 n=30/30

---

## C. 결과 — Cross-Judge G-Eval

### 핵심 표 (Δ = fusion - shaponly, n=30 each)

| Target LLM | Metric | shaponly | fusion | Δ Claude judge | Δ Gemini judge | 차이 |
|---|---|---:|---:|---:|---:|---:|
| **Anthropic** | Completeness | 4.30 / 3.50 | **4.97 / 4.40** | **+0.667** | **+0.900** | 0.23 |
| Anthropic | Factual | 4.87 / 4.83 | 4.90 / 4.97 | +0.033 | +0.133 | 0.10 |
| Anthropic | Sensitive | 5.0 / 5.0 | 5.0 / 5.0 | 0 | 0 | 0 |
| Anthropic | Style | 5.0 / 5.0 | 4.97 / 5.0 | -0.033 | 0 | 0.03 |
| **Gemini** | Completeness | 3.90 / 3.27 | **4.70 / 4.37** | **+0.800** | **+1.100** | 0.30 |
| **Gemini** | Factual | 4.90 / 4.47 | 4.77 / 4.93 | **-0.133** | **+0.467** | **0.60** |
| Gemini | Sensitive | 5.0 / 5.0 | 5.0 / 5.0 | 0 | 0 | 0 |
| Gemini | Style | 4.93 / 5.0 | 4.93 / 5.0 | 0 | 0 | 0 |

(각 셀 "X / Y": Claude judge / Gemini judge)

### 핵심 메시지

#### 메시지 1 — Completeness 양 judge 일관 큰 향상 ★
- Anthropic: Claude judge +0.67, Gemini judge +0.90 (둘 다 큰 향상)
- Gemini: Claude judge +0.80, Gemini judge +1.10 (둘 다 큰 향상)
- → **fusion 효과의 핵심 메시지**가 judge 종속 아님 입증

#### 메시지 2 — Gemini target Factual에서 Cross-judge 가치 ★★
- Claude judge: -0.13 (fusion이 약간 떨어짐)
- Gemini judge: **+0.467** (fusion이 명확히 더 사실적)
- 차이 **0.60** — 단일 judge였으면 잘못된 결론 낼 뻔
- → Cross-LLM judge가 본 연구 평가 신뢰성에 필수임을 직접 증거

#### 메시지 3 — Sensitive Leak 5.0/5.0 양 judge 만점 ★
- 두 judge 모두 sensitive_leak에서 양 mode 5.0/5.0
- 마스킹 정책 (CODE_GENDER, DAYS_BIRTH 등)이 LLM 종속성 없음을 다층 검증

#### 메시지 4 — Self-bias 패턴 약함
- Self-judge가 더 호의적이라는 일반 가정과 약간 다름
- Gemini judge가 self(Gemini target)와 cross(Anthropic target) 모두에 더 큰 Δ를 줌
- → Gemini judge는 fusion 효과를 더 강하게 인식하는 경향
- → "fusion이 정말 향상이다"라는 메시지를 양 judge가 다른 강도로 동의

### 주의 — 룰 메트릭은 양 judge에서 동일
룰 기반 메트릭(feat_match, val_match, sign_match, halluc)은 LLM judge와 무관하게 텍스트만 분석하므로 두 csv에서 동일. cross-judge 분석은 **G-Eval 4 메트릭에 한정**.

---

## D. Step 2-A 패턴과의 일치성

Step 2-A에서 Cross-LLM G-Eval (양방향, n=30)을 진행했었음:
- Claude → Gemini: factual 4.87, completeness 4.0
- Gemini → Claude: factual 4.6, completeness 3.33

이번 Step 3-C-2-f 결과:
- 양 judge 모두 fusion에서 큰 향상 (Step 2-A는 양 LLM이 서로의 출력을 fair하게 평가)
- **Self-bias 패턴 약함** (Step 2-A에서도 비슷한 결론)
- Gemini judge가 일반적으로 더 큰 Δ를 인식하는 경향 (Step 2-A에서는 Gemini self-judge가 자기 비판이었음과 약간 다름. 이번엔 fusion 효과를 더 크게 봄)

→ **본 연구의 평가 패턴이 step별로 일관됨**.

---

## E. 산출물

### 새 코드
- `src/cross_judge_analysis.py` — Claude vs Gemini judge 비교 + figure

### 새 데이터
- `results/fusion_eval_claude_judge.csv` — Claude judge raw (step3c1 시점 복원, n=30 깨끗)
- `results/fusion_eval_gemini_judge.csv` — Gemini judge raw (n=30, retry 회복)
- `results/fusion_vs_shaponly_claude_judge.csv` — Claude judge summary
- `results/fusion_vs_shaponly_gemini_judge.csv` — Gemini judge summary
- `results/cross_judge_comparison.csv` — wide format cross-judge 비교

### 새 figure
- `figures/32_cross_judge_geval.png` — 4 메트릭 × 2 target × 2 judge × 2 mode

---

## F. 미팅 자료 통합 결정

이 결과는 메시지 강화에 결정적:
- "Cross-LLM judge에서도 fusion 효과 일관" → 평가 신뢰성 강화
- "Cross-judge가 본 연구에 필수임을 직접 증거" → 방법론적 정당성

**미팅 자료에 추가 권장**:
- 슬라이드: 새 슬라이드 또는 Step 3-C-2 슬라이드에 cross-judge figure 한 줄 추가
- docx: 11.6절 또는 14.5절로 cross-judge 검증 섹션 추가
- 종합 슬라이드 / 한계 슬라이드의 G-Eval 측면 갱신

---

## G. Future Work 갱신

기존 future work에서 "Gemini judge cross-validation"이 1순위였는데 이번에 부분 해소:
- ✅ G-Eval Cross-validation (Claude + Gemini judge) 완료
- ⏳ 잔여: 인간평가 (Plausibility) — 미팅 후 IRB 진행
