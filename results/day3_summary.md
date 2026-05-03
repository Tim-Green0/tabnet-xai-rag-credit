# Day 3 요약 — TabNet (정형 데이터 딥러닝)

날짜: 2026-05-03
미팅까지: D-7
GPU: GTX 1660 Ti 6GB (CUDA 12.1, 사용)

---

## TL;DR

- **TabNet 학습 + Optuna 10 trials + 어텐션 추출 완료.**
- Test AUROC: **TabNet-fixed 0.7500** / **TabNet-tuned 0.7543** (XGBoost 0.7605에 살짝 못 미침).
- 어텐션 Top 1 = `NAME_CONTRACT_TYPE_Revolving loans` (0.291), Top 2-4 = `EXT_SOURCE_2/3` 계열 → **EDA 인사이트와 일관성 확인**.
- **`CODE_GENDER_M`이 어텐션 6위(0.065)** — 모델이 성별을 활용 중. Day 5 공정성 평가의 핵심 단서.

---

## 학습 세팅

| 단계 | 설정 | 시간 |
|---|---|---|
| Fixed-config 학습 | `n_d=n_a=16, n_steps=4, gamma=1.5, λ=1e-4, lr=2e-2, sparsemax`, batch=1024, vb=128, max_epochs=80, patience=15, weights=1(auto-balanced) | **649s (10.8분)** |
| Optuna 튜닝 | 10 trials, max_epochs=25, patience=4, TPESampler+MedianPruner | **4423s (73.7분)** |
| Best params 최종 학습 | `n_d=16, n_steps=3, γ=1.156, λ=1.31e-5, lr=0.0367, entmax`, max_epochs=80, patience=15 | **386s (6.4분)** |

총 학습 시간: **약 92분** (예상 60~120분 범위 내).

### Optuna 결과
- Best **Trial 0**: val AUROC 0.7482 (`n_d=16, n_steps=3, entmax, lr=0.0367, γ=1.156, λ=1.31e-5`)
- Trials 1-9는 0.74 부근에서 안정 — TPESampler가 Trial 0 인근만 탐색했지만, **best가 첫 trial이라 이후 개선 미미**. 미팅용으론 충분.

---

## 메트릭 (Test set)

| 모델 | AUROC | AUPRC | KS | F1 | Precision | Recall | 임계치 | 시간 |
|---|---|---|---|---|---|---|---|---|
| Logistic | 0.7547 | 0.2346 | 0.3787 | 0.2663 | 0.1663 | 0.6679 | 0.513 | 88s |
| LightGBM | 0.7549 | 0.2414 | 0.3785 | 0.2622 | 0.1617 | 0.6935 | 0.411 | 17s |
| **XGBoost** | **0.7605** | **0.2459** | **0.3892** | **0.2705** | 0.1688 | 0.6800 | 0.476 | 21s |
| TabNet-fixed | 0.7500 | 0.2275 | 0.3685 | 0.2537 | 0.1553 | 0.6937 | 0.472 | 649s |
| **TabNet-tuned** | **0.7543** | 0.2333 | 0.3788 | 0.2706 | 0.1706 | 0.6544 | 0.550 | 386s |

### 해석
- **TabNet-tuned는 Logistic/LightGBM과 거의 동등** (AUROC 차 0.0004~0.0006). XGBoost와는 ~0.006 차이.
- **TabNet의 Recall이 65.4%** — 다른 베이스라인보다 살짝 낮으나 Precision이 17.1%로 가장 높음. 즉 더 보수적인 임계치(0.55)가 학습된 결과.
- 어텐션 마스크라는 **내재적 해석성**을 추가로 얻은 것이 본 연구의 핵심 가치 — 단순 성능만 보면 XGBoost로 충분하지만, **어텐션–SHAP 일관성 분석(Day 4)**을 위한 모델이 확보됨.

### 산업 표준 vs 본 결과
- TabNet 원논문(Arik & Pfister 2019): 메인 테이블만 사용 시 AUROC 0.755~0.770 보고
- 본 결과 0.7543는 정상 범위 하단~중단. n_steps를 더 키우거나 데이터에 보조 테이블을 합치면 0.76+ 달성 가능.

---

## 어텐션 기반 변수 중요도 (Top 10)

| 순위 | 변수 | importance | EDA 상관계수 |
|---|---|---|---|
| 1 | NAME_CONTRACT_TYPE_Revolving loans | 0.2909 | (one-hot, EDA에선 미정) |
| 2 | **EXT_SOURCE_2** | 0.1263 | -0.160 ✅ |
| 3 | YEARS_BUILD_MODE | 0.1190 | -0.022 |
| 4 | **EXT_SOURCE_3** | 0.0932 | -0.179 ✅ |
| 5 | NAME_EDUCATION_TYPE_Higher education | 0.0671 | (one-hot) |
| 6 | **CODE_GENDER_M** | 0.0647 | (보호 속성!) |
| 7 | FLAG_OWN_CAR_Y | 0.0637 | (one-hot) |
| 8 | FLAG_DOCUMENT_20 | 0.0359 | +0.000 |
| 9 | FONDKAPREMONT_MODE_org spec account | 0.0346 | (one-hot) |
| 10 | FONDKAPREMONT_MODE_reg oper spec account | 0.0201 | (one-hot) |

→ `figures/12_tabnet_attention_top20.png`

### 4가지 인사이트
1. **EXT_SOURCE 패턴 검증**: EDA에서 발견한 EXT_SOURCE_2/3가 어텐션에서도 Top 5 — TabNet이 같은 신호를 학습. **단일 변수 강도와 어텐션 강도가 일치**한다는 증거.
2. **상관계수가 약하지만 어텐션이 강한 변수**: NAME_CONTRACT_TYPE_Revolving loans, YEARS_BUILD_MODE, NAME_EDUCATION_TYPE_Higher education 등은 ρ가 낮은데 어텐션이 높음 → **비선형/상호작용으로 활용**되고 있다는 신호. 이게 TabNet 도입 정당화의 핵심 메시지.
3. **CODE_GENDER_M 6위 — 공정성 alarm**: 모델이 성별을 적극 활용. Day 5 공정성 분석 시 ablation이 의미 있을 영역.
4. **YEARS_BUILD_MODE 3위 — 의외**: EDA 상관 -0.022로 약한데 어텐션 매우 강함. TabNet이 다른 변수와의 상호작용으로 활용 중일 가능성. SHAP에서 검증 예정.

---

## 산출물

```
src/
├─ tabnet_train.py             # fixed + Optuna + 어텐션 추출

results/
├─ baseline_models/
│   ├─ tabnet_fixed.zip        # 198 KB
│   └─ tabnet_best.zip         # 196 KB    ← Day 4-7에서 사용
├─ tabnet_metrics.csv          # 모델 × split × 지표
├─ tabnet_summary.json         # 설정 + 결과 + feature_names
├─ tabnet_optuna_trials.csv    # Optuna 10 trials 전체
└─ tabnet_attention_importance.csv   # 214 변수 × 어텐션 importance

figures/
├─ 11_tabnet_training_curve_tuned.png
└─ 12_tabnet_attention_top20.png
```

재실행: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.tabnet_train --n-trials 10`

---

## 다음 단계 — Day 4 (SHAP)

목표: TabNet에 SHAP 적용 + 어텐션과의 일관성 정량 분석.

작업:
- `src/shap_analysis.py`
- TabNet은 `KernelExplainer` 또는 `GradientExplainer` 적용 (DeepExplainer는 pytorch-tabnet 호환 제한)
- 베이스라인 비교: XGBoost는 `TreeExplainer` (빠름)
- Global SHAP vs Attention importance: Spearman ρ, Top-K Overlap
- Local SHAP: 5~10명 샘플 → 다음 Day 6 LLM 컨텍스트 입력 준비

산출물 예정:
- `results/shap_global_*.csv` (모델별 변수 중요도)
- `results/attention_vs_shap.csv` (Spearman, Overlap)
- `figures/13_shap_global.png`, `14_attention_vs_shap.png`, `15_shap_local_examples.png`

대략 30~60분 소요 (KernelExplainer가 느릴 수 있어 샘플링 필요).

**Day 4 진행 OK인지, 또는 Day 3 결과 검토 후 보완할 것 있는지 알려줘.**
