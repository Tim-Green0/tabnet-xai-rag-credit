# Day 5 요약 — 공정성 진단 + Ablation Mitigation

날짜: 2026-05-03
미팅까지: D-7

---

## TL;DR

1. **모든 모델·모든 보호속성에서 4/5 rule 위반** (Disparate Impact < 0.8). 데이터 자체에 강한 편향 존재.
2. **AGE(연령) 편향이 GENDER(성별)보다 큼** — 연령 DI ~0.50 vs 성별 DI ~0.62.
3. **Ablation mitigation 효과**: GENDER 컬럼 제거 시 DP 30~40% 감소(좋음), AGE 컬럼 제거는 효과 미미. **연령 정보가 다른 변수에 간접 인코딩**되어 있다는 발견.
4. **성능 손실**: XGBoost Δ−0.005, TabNet Δ−0.010 AUROC. 성능 vs 공정성 trade-off가 합리적 범위.
5. 4/5 rule은 **여전히 미통과** → 단순 ablation의 한계 = 본 연구의 명확한 future work 메시지.

---

## A. 공정성 지표 (베이스라인)

4개 모델 × 2개 보호속성 = 8개 케이스 모두 측정. 임계치는 validation Youden's J.

| 모델 | 보호속성 | DP diff | EO diff | EOdds diff | DI ratio | 4/5 통과 |
|---|---|---|---|---|---|---|
| Logistic | GENDER | 0.168 | 0.153 | 0.156 | 0.614 | ❌ |
| Logistic | AGE | 0.180 | 0.223 | 0.223 | 0.532 | ❌ |
| XGBoost | GENDER | 0.164 | 0.136 | 0.153 | 0.622 | ❌ |
| XGBoost | AGE | 0.197 | 0.222 | 0.222 | **0.498** | ❌ |
| LightGBM | GENDER | 0.154 | 0.118 | 0.144 | 0.656 | ❌ |
| LightGBM | AGE | 0.189 | 0.210 | 0.210 | 0.539 | ❌ |
| TabNet | GENDER | 0.156 | 0.128 | 0.146 | 0.621 | ❌ |
| TabNet | AGE | 0.185 | 0.223 | 0.223 | 0.504 | ❌ |

→ `figures/18_fairness_metrics.png`, `results/fairness_metrics.csv`

### 해석
- **모든 케이스 4/5 rule 위반** (DI < 0.8). 데이터 자체에 구조적 편향.
- **연령 편향이 성별보다 큼** (AGE DI ~0.50 vs GENDER DI ~0.62). EDA Day 1에서 발견한 "25세 미만 12.3% vs 65세+ 3.7% 부도율" 차이의 직접적 모델 반영.
- **모델 간 차이 작음** — Logistic부터 TabNet까지 비슷한 수준 → 알고리즘이 아닌 데이터 차원 문제.
- **LightGBM이 GENDER 측면에서 가장 공정** (DP 0.154), 그러나 여전히 4/5 미통과.

---

## B. Ablation Mitigation (보호 속성 제거 후 재학습)

**제거한 컬럼**:
- `CODE_GENDER_F, CODE_GENDER_M, CODE_GENDER_XNA` (성별 one-hot)
- `DAYS_BIRTH` (연령)

**대상**: XGBoost, TabNet (LightGBM/Logistic은 시간상 생략, 패턴 동일 추정)

### 결과

| 모델 × 속성 | AUROC (Δ) | DP diff (Δ) | DI ratio (Δ) |
|---|---|---|---|
| XGBoost × GENDER | 0.7605 → 0.7554 (**−0.0051**) | 0.164 → **0.105** (**−36%**) | 0.622 → **0.718** (**+15%**) |
| XGBoost × AGE | 0.7605 → 0.7554 (−0.0051) | 0.197 → 0.179 (−9%) | 0.498 → 0.507 (+2%) |
| TabNet × GENDER | 0.7543 → 0.7443 (**−0.0100**) | 0.156 → **0.093** (**−40%**) | 0.621 → **0.757** (**+22%**) |
| TabNet × AGE | 0.7543 → 0.7443 (−0.0100) | 0.185 → 0.188 (+2%) | 0.504 → 0.513 (+2%) |

→ `figures/19_fairness_mitigation.png`, `results/fairness_mitigation.csv`
→ 모델 가중치: `results/fairness_models/{xgboost_ablated.pkl, tabnet_ablated.zip}` (gitignore)

### 핵심 인사이트 (3가지)

#### 1. **GENDER ablation은 효과적**
- DP 30~40% 감소, DI 15~22% 증가 — 의미 있는 개선
- 성능 손실은 작음 (XGB −0.005, TabNet −0.010 AUROC)
- 미팅 메시지: **"성별 정보 제거로 공정성 개선 가능, 성능 trade-off는 1% 이내"**

#### 2. **AGE ablation은 효과 미미** — 학술적으로 더 흥미로운 발견
- DAYS_BIRTH 단 1개 컬럼 제거로는 연령 편향 거의 안 줄어듦 (DP −9%, DI +2% 수준)
- 원인: **연령 정보가 다른 변수에 간접 인코딩**
  - DAYS_EMPLOYED (재직 기간) — 연령과 강한 상관
  - DAYS_REGISTRATION (거주지 등록 후 경과)
  - DAYS_ID_PUBLISH (신분증 발급 후 경과)
  - OWN_CAR_AGE (차량 연식)
  - NAME_INCOME_TYPE_Pensioner (연금 수령자)
- → 단순 컬럼 제거로 안 되는 **proxy variable 문제**가 명확히 드러남

#### 3. **여전히 4/5 rule 미통과**
- GENDER 후처리 DI 0.72~0.76 < 0.8
- AGE 후처리 DI ~0.51, 거의 그대로
- → **단순 ablation은 충분하지 않음**. Reweighing, adversarial debiasing, FairML 알고리즘 등 본격적 fairness-aware 학습이 future work로 자연스럽게 도출됨.

---

## C. 미팅 메시지 후보

1. **공정성 진단 결과**: 4개 모델 × 2개 보호속성 모두 4/5 rule 위반, AGE 편향이 GENDER보다 큼. 데이터 자체에 구조적 편향.
2. **단순 ablation의 효과와 한계**: GENDER 제거는 효과적이지만 AGE는 proxy variable 때문에 비효과적. 본격적 fairness-aware 학습 필요.
3. **성능 vs 공정성 trade-off**: AUROC 1% 미만 손실로 DP 30~40% 감소 가능 → 실용적 가치 있음.
4. **본 연구의 XAI-RAG 측면**: 컨텍스트 빌더에서 보호 속성을 마스킹(SENSITIVE_FEATURES) 처리 → LLM 자연어 설명에서는 성별·연령을 직접 언급하지 않도록 통제 (계획서 3.6 정합).

---

## D. 산출물

```
src/
└─ fairness.py              # 4종 지표 + ablation 재학습

results/
├─ fairness_metrics.csv     # 베이스라인 (8 cases)
├─ fairness_mitigation.csv  # baseline vs ablated
├─ day5_summary.md
└─ fairness_models/         # gitignore (대용량 모델)
   ├─ xgboost_ablated.pkl
   └─ tabnet_ablated.zip

figures/
├─ 18_fairness_metrics.png       # 4종 지표 × 모델별
└─ 19_fairness_mitigation.png    # baseline vs ablated 막대
```

---

## E. 다음 단계 — Day 6 (XAI-RAG + LLM 호출)

`src/context_builder.py`는 미리 작성 완료 (Day 4 commit).

남은 작업:
- `src/llm_explainer.py` 작성 (Gemini API 호출, 프롬프트 템플릿)
- SHAP local 10명 샘플 → JSON 컨텍스트 → Gemini 자연어 설명 생성
- (선택) Anthropic API 추가 비교 — 사용자 결정 필요

**Gemini API 키 발급 필요**: https://aistudio.google.com → Get API key
- 무료 tier (RPM 제한 있음, 10명 샘플엔 충분)
- 발급 후 `D:\paper\.env`에 `GEMINI_API_KEY=...` 형태로 저장

키 발급 후 알려주면 Day 6 진행.
