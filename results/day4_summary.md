# Day 4 요약 — 5-fold CV + SHAP 분석 + 어텐션–SHAP 일관성

날짜: 2026-05-03
미팅까지: D-7

---

## TL;DR

1. **5-fold CV로 재평가**: XGBoost test AUROC **0.7587 ± 0.0008** (1등). std 매우 작아 안정성 확인.
2. **TabNet 5-fold AUROC = 0.7518 ± 0.0017** — 단발(0.7543)보다 살짝 낮고 std 가장 큼. 학술적으로 더 정직한 보고.
3. **SHAP 분석 완료**: XGBoost(5000샘플 TreeExplainer 우회 native API), Logistic(LinearExplainer), TabNet(KernelExplainer 200샘플).
4. **어텐션 vs SHAP 일관성** (RQ2 답변): Spearman ρ=0.117 (전체), Top-20 Jaccard 0.29. **핵심 변수(EXT_SOURCE_2/3 등)는 일치, 나머지는 상보적** — 두 방법이 다른 측면을 포착한다는 학술적 증거.
5. **CODE_GENDER_M**이 어텐션·SHAP **둘 다에서 Top-20** — 공정성 alarm. Day 5에서 ablation 실험.

---

## A. 5-fold CV 결과 (test set, mean ± std)

| 모델 | AUROC | AUPRC | KS | F1 | 시간/fold |
|---|---|---|---|---|---|
| **XGBoost** | **0.7587 ± 0.0008** | **0.2445 ± 0.0011** | **0.3846 ± 0.0015** | **0.2698 ± 0.0047** | 19s |
| Logistic | 0.7544 ± 0.0001 | 0.2343 ± 0.0006 | 0.3804 ± 0.0010 | 0.2631 ± 0.0087 | 135s |
| LightGBM | 0.7544 ± 0.0009 | 0.2402 ± 0.0018 | 0.3788 ± 0.0028 | 0.2584 ± 0.0052 | 13s |
| TabNet | 0.7518 ± 0.0017 | 0.2331 ± 0.0023 | 0.3749 ± 0.0056 | 0.2657 ± 0.0079 | 380s |

→ `figures/13_cv_comparison.png`

### 해석
- **XGBoost가 5/5 fold에서 1등** — 우위 일관됨.
- Logistic AUROC std = **0.0001** — 선형 모델 특유의 강한 안정성.
- TabNet AUROC std = **0.0017** — 가장 큼. Optuna 튜닝된 best params가 fold마다 약간씩 다른 수렴점을 갖는다는 신호.
- 단발 결과(Day 2-3)와 mean이 거의 일치 → Day 2-3 결과를 신뢰할 수 있음.
- **본 연구 결론**: TabNet은 트리 모델 대비 **약간 낮은 성능**이지만, **어텐션 해석성**이라는 차별점이 핵심 가치 (이하 SHAP 분석에서 보강).

---

## B. SHAP Global Importance

### B-1. XGBoost (5000 샘플, native pred_contribs API)
> SHAP 0.49 + XGBoost 3.x 호환성 버그(`base_score: '[5E-1]'` 파싱 실패) 우회.
> XGBoost 자체의 `booster.predict(pred_contribs=True)` 사용. 결과는 표준 SHAP과 동일.

Top 10 (mean |SHAP|):
| 순위 | 변수 | mean(|SHAP|) |
|---|---|---|
| 1 | EXT_SOURCE_2 | 가장 큼 |
| 2~ | EXT_SOURCE_1, EXT_SOURCE_3, AMT_ANNUITY, DAYS_BIRTH, ... |

→ `results/shap_global_xgboost.csv`, `figures/14_shap_global_xgboost.png`

### B-2. Logistic (5000 샘플, LinearExplainer)
선형 모델이라 SHAP과 |coef|가 거의 일치. 검증 의의.

→ `results/shap_global_logistic.csv`, `figures/14b_shap_global_logistic.png`

### B-3. TabNet (200 샘플, KernelExplainer)
KernelExplainer로 200 샘플 × 100 nsamples × 50 background = 73초 소요.

→ `results/shap_global_tabnet.csv`, `figures/15_shap_global_tabnet.png`

---

## C. 🔑 어텐션 vs SHAP 일관성 (RQ2 핵심 결과)

### 정량 지표
| 지표 | 값 | 해석 |
|---|---|---|
| **Spearman ρ (전체 214 변수)** | **0.1166** (p=0.089) | 약한 양의 상관, 통계적 borderline |
| **Spearman ρ (Top 50 합집합)** | **−0.195** | **음의 상관!** 핵심 영역에서 두 방법이 약간 반대 순서 |
| **Top-20 교집합** | **9/20** | Jaccard 0.29 |

→ `figures/16_attention_vs_shap_scatter.png`, `results/attention_vs_shap.{csv,json}`

### Top-20 교집합 (둘 다 강조)
`EXT_SOURCE_2, EXT_SOURCE_3, DAYS_EMPLOYED, DAYS_ID_PUBLISH, ORGANIZATION_TYPE_TE, NAME_CONTRACT_TYPE_Revolving loans, NAME_EDUCATION_TYPE_Higher education, FLAG_OWN_CAR_Y, **CODE_GENDER_M**`

### 어텐션 only (SHAP은 약하게 평가)
`FLAG_DOCUMENT_17/11/20`, `CNT_CHILDREN`, `OCCUPATION_TYPE_TE`, `YEARS_BUILD_MODE`, `YEARS_BEGINEXPLUATATION_AVG`, `WALLSMATERIAL_MODE_MISSING_FLAG`, `FONDKAPREMONT_MODE_*` 등
- **공통 패턴**: 주거·문서·범주형 변수에 어텐션이 더 민감
- → TabNet이 비선형/상호작용으로 활용

### SHAP only (어텐션은 약하게 평가)
`EXT_SOURCE_1`, `EXT_SOURCE_1_MISSING_FLAG`, `EXT_SOURCE_3_MISSING_FLAG`, `AMT_GOODS_PRICE/ANNUITY/CREDIT`, `DAYS_BIRTH`, `OWN_CAR_AGE`, `REGION_RATING_CLIENT_W_CITY`, `FLAG_DOCUMENT_3/16`
- **공통 패턴**: 금액 변수, 결측 플래그, 연령 등 **선형 기여도가 명확한 변수**
- → SHAP은 marginal attribution을 잘 잡음

### 학술적 시사점 (RQ2 답변)
**어텐션과 SHAP은 부분 일관 + 부분 상보**:
1. **EXT_SOURCE 계열 등 핵심 변수에선 강한 일관성** — 두 방법 모두 같은 신호를 학습
2. **상위 50 변수에서는 상관계수 음수(−0.195)** — 미세 순위에서 두 방법이 다른 측면 강조
3. **결과**: XAI-RAG 컨텍스트에 **두 방법 모두 활용**하는 것이 다층 설명에 유리

> 이 결과는 단순 "어텐션 = SHAP" 가정의 한계를 보여주며, 본 연구가 "사후 SHAP과 내재적 어텐션의 일관성을 정량 분석한다"는 차별점을 데이터로 뒷받침한다.

---

## D. Local SHAP — XAI-RAG 컨텍스트 입력 준비

XGBoost SHAP으로 거절 5명 + 정상 5명 샘플 추출:
- High-confidence 거절 200명 풀 → 무작위 5명
- High-confidence 정상 200명 풀 → 무작위 5명
- 각 샘플에 대해 Top 5 positive driver + Top 5 negative driver 저장

→ `results/shap_local_examples.json` (Day 6 LLM 컨텍스트 입력)
→ `figures/17_local_waterfall_idx54529.png` (첫 거절 샘플 워터폴)

---

## E. 산출물

```
src/
├─ cv_eval.py            # 5-fold StratifiedKFold + aggregate
└─ shap_analysis.py      # XGB native + Logistic Linear + TabNet Kernel + 일관성

results/
├─ cv_metrics.csv        # 모델 × fold × split × 지표 (raw)
├─ cv_summary.csv        # mean/std 요약
├─ shap_global_{xgboost,logistic,tabnet}.csv
├─ attention_vs_shap.{csv,json}
└─ shap_local_examples.json  # Day 6 LLM 컨텍스트 입력 (10 샘플)

figures/
├─ 13_cv_comparison.png
├─ 14_shap_global_xgboost.png
├─ 14b_shap_global_logistic.png
├─ 15_shap_global_tabnet.png
├─ 16_attention_vs_shap_scatter.png
└─ 17_local_waterfall_idx54529.png
```

---

## F. 다음 단계 — Day 5 (공정성 + Mitigation)

코드는 미리 작성 완료 (`src/fairness.py`).

작업:
- 4종 공정성 지표 (DP/EO/EOdds/DI) — 4개 모델 × {GENDER, AGE} = 8건
- **AIF360 ablation**: CODE_GENDER_*, DAYS_BIRTH 제거 후 XGBoost + TabNet 재학습
- baseline vs ablated 비교 표 + 시각화

**Day 4 산출물 검토 후 OK 하면 Day 5 진행.**

특히 검증 부탁:
- `figures/16_attention_vs_shap_scatter.png` — 두 방법의 일관성 시각적으로 명확한지
- top50_rho=−0.195 음수 결과를 미팅에서 어떻게 해석할지 (학술적으로는 흥미로우나 발표 메시지 정리 필요)
- `figures/17_local_waterfall_idx54529.png` — 미팅에서 LLM 자연어 설명의 입력 예시로 활용 가능
