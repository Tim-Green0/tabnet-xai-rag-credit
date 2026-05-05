# Day 9 / Step 3-B 요약 — 보조 테이블 활용 (성능 확장)

날짜: 2026-05-05
미팅까지: D-5 (2026-05-10)
이전 시점: Step 2-A 완료 (tag `step2a`)

---

## TL;DR — Step 3-B의 핵심 4가지

1. **🎯 보조 테이블 2개(`bureau`, `previous_application`) 추가만으로 XGBoost test AUROC 0.7587 → 0.7755 (+0.0168, +2.22%)**.
   5-fold CV mean ± std 모두 baseline 신뢰구간 밖.
2. **불균형 데이터에서 더 큰 임팩트**: AUPRC +8.21%, KS +7.81%. positive 8% 환경에서 분류력 자체 강화.
3. **새로 도입된 보조 feature 중 5개가 SHAP top 20에 진입** — 모두 `previous_application` 기반.
   `bureau`(외부 신용기관)는 top 20 진입 못함 → "우리 회사 자체 이력"이 더 결정적이라는 의외의 발견.
4. **메모리 효율적 파이프라인 입증**: dtype downcast로 보조 데이터 -89% 압축 (950MB raw → ~388MB DataFrame),
   1161 features XGBoost 5-fold CV가 138s/fold 안에 완료.

---

## A. Feature Engineering (`src/aux_data.py`, `src/aux_features.py`)

### 데이터 사용 범위
| 테이블 | 행 | 메모리(downcast) | main 커버리지 |
|---|---:|---:|---:|
| `bureau` | 1,716,428 | 85 MB | 85.69% |
| `bureau_balance` | 27,299,925 | 156 MB | (bureau의 45.11%) |
| `previous_application` | 1,670,214 | 147 MB | 94.65% |

원본 csv 합계 ~950MB → optimize_dtypes 적용 후 388MB (-89%).

### 집계 전략
- **bureau_balance → SK_ID_BUREAU** (월별 STATUS 비율 + MONTHS_BALANCE 통계, 9 features)
- **bureau (with bb merge) → SK_ID_CURR** (전체 / Active / Closed 분리 집계, **346 features**)
- **previous_application → SK_ID_CURR** (전체 / Approved / Refused 분리 집계, **410 features**)
- 두 결과 SK_ID_CURR outer merge → **`data/processed/aux_features.parquet` (756 features, 256 MB)**

### 결측 정책
- 보조 테이블 미커버 신청자(bureau 14.31%, prev 5.35%)는 join 후 NaN
- 기존 `HomeCreditPreprocessor` (A1 정책)가 자동으로 `_MISSING_FLAG` 생성
- 결측 50%+ aux feature 191개도 같은 정책 적용

---

## B. 통합 전처리 (`src/preprocess_with_aux.py`)

기존 `src/preprocess.py`의 `HomeCreditPreprocessor`를 그대로 재사용.
즉, 전처리 정책(A1/B1/C1/D/E1/F)에 변경 없음 → Step 1 비교 공정성 유지.

| | input | output |
|---|---:|---:|
| n_features | 122 (main) → 877 (after aux merge) | **1161** (after preprocessing) |
| n_rows train/val/test | — | 184,506 / 61,502 / 61,503 (60/20/20 stratified, SEED=42) |

산출:
- `data/processed/{train,val,test}_{scaled,unscaled}_aux.parquet`
- `results/preprocessor_aux.pkl`

---

## C. 5-fold CV — XGBoost (`src/cv_eval_aux.py`)

### Test set 메트릭 (5-fold mean ± std)

| Metric | Baseline (Step 1) | **+ Aux (Step 3-B)** | Δ | Δ% |
|---|---|---|---|---|
| **AUROC** | 0.7587 ± 0.0008 | **0.7755 ± 0.0011** | **+0.0168** | **+2.22%** |
| AUPRC | 0.2445 ± 0.0011 | **0.2646 ± 0.0015** | +0.0201 | **+8.21%** |
| KS | 0.3846 ± 0.0015 | **0.4146 ± 0.0040** | +0.0301 | **+7.81%** |
| F1 | 0.2698 ± 0.0047 | **0.2813 ± 0.0096** | +0.0115 | +4.27% |
| Time/fold | 19.1 s | 138.1 s | — | (×7.2) |

Fold별 test AUROC: 0.7771 / 0.7747 / 0.7754 / 0.7743 / 0.7760 — 매우 안정적.

**해석**:
- AUROC +0.0168은 baseline std (0.0008)의 21배 → 통계적으로 명확한 향상.
- AUPRC가 AUROC보다 큰 비율로 개선된 것은 positive class(부도)에 대한 식별력이 강화됐다는 의미. 신용 평가 도메인에서 가장 중요한 metric.
- KS +0.030은 Kolmogorov-Smirnov 분리력 향상으로, 거절/승인 분포 간 거리가 더 멀어짐.

산출:
- `results/cv_metrics_aux.csv`, `results/cv_summary_aux.csv`, `results/cv_aux_vs_baseline.csv`
- `figures/27_cv_aux_comparison.png` — baseline vs aux 막대 비교 (AUROC/AUPRC/KS/F1)

---

## D. SHAP 재실행 — top 20 변화 (`src/shap_aux.py`)

### 단일 모델 (train+val 학습) test 메트릭
- AUROC 0.7739 / AUPRC 0.2627 / KS 0.4087 / F1 0.2821
- CV mean(0.7755)와 -0.0016 차이 — fold 결합 효과 vs 단일 fit 차이로 정상 범위.

### Top 20 진입 — 보조 테이블 feature 5개 (모두 PREV_)

| rank | feature | mean(\|SHAP\|) | 의미 |
|---:|---|---:|---|
| 12 | `PREV_NAME_YIELD_GROUP_high_mean` | 0.0499 | 이전 신청 중 high-yield 그룹 비율 |
| 13 | `PREV_CNT_PAYMENT_std` | 0.0486 | 이전 결제 횟수 변동성 |
| 14 | `PREV_NAME_YIELD_GROUP_low_action_mean` | 0.0450 | 이전 low-yield 그룹 비율 |
| 16 | `PREV_DAYS_LAST_DUE_1ST_VERSION_max` | 0.0440 | 이전 만기일 최댓값 |
| 20 | **`PREV_NAME_CONTRACT_STATUS_Refused_mean`** | 0.0360 | **이전 거절 비율** ★ |

### 발견
1. Top 1~10은 **baseline과 동일** (EXT_SOURCE_2/3/1, AMT_*, ORGANIZATION_TYPE_TE, DAYS_EMPLOYED, AMT_CREDIT, CODE_GENDER_F, NAME_EDUCATION_TYPE_Higher education).
   → 핵심 신호는 main 테이블에 이미 존재.
2. **rank 11~20 구간에서 baseline의 약한 feature 5개가 PREV_* 5개로 대체**.
   특히 `PREV_NAME_CONTRACT_STATUS_Refused_mean` (rank 20)은 직관적으로 가장 강력 — "예전에 자주 거절된 사람은 다시 거절될 확률 높음"이라는 신용 평가 상식과 일치.
3. **bureau (외부 신용기관) feature는 top 20 진입 못함** — 의외의 결과.
   - 가능한 이유: 외부 신용기관 데이터는 main 테이블의 EXT_SOURCE_1/2/3과 정보 중복 가능성.
   - bureau 자체가 무용한 게 아니라 **이미 EXT_SOURCE에 응축된 신호** 일 가능성. 추가 ablation 필요 (future work).

산출:
- `results/shap_global_xgboost_aux.csv`
- `results/shap_top20_diff.csv`
- `figures/28_shap_global_xgb_aux.png` — aux 모델 SHAP top 20 막대
- `figures/29_shap_top20_overlap.png` — baseline vs aux top 20 비교 (overlap, only-baseline, only-aux 색상 구분)

---

## E. 추가 안 한 것 (의도적 축소)

### 미진행 항목과 이유
| 항목 | 진행 여부 | 사유 |
|---|---|---|
| LightGBM CV with aux | ❌ | XGBoost가 Step 1 메인 비교 모델. 미팅 메시지엔 XGBoost가 결정적. LightGBM 추가는 미팅 후. |
| TabNet 재학습 with aux | ❌ | 1161 features에서 TabNet 학습 시간 크게 증가 + 메모리 부담. Step 3-B 이후 검토. |
| LLM 컨텍스트 재생성 (XAI-RAG with aux SHAP) | ❌ | Halluc 0/100은 모델 변경에 둔감 가능성 높음 (XAI-RAG 메커니즘이 SHAP의 형식만 의존). API 비용 + 시간 부담 큼. **future work**. |
| Bureau ablation (왜 top 20에 안 들었나) | ❌ | bureau만 빼고 prev만 추가한 비교 실험 — 시간 부담. 미팅 후. |
| Fairness 재진단 with aux | ❌ | 보호 속성 자체는 변하지 않음. 미팅 발표용 메시지엔 영향 적음. 미팅 후. |
| Fairness-aware 학습 (Reweighing 등) | ❌ | 별도 한 사이클 필요. Step 3-B 후속 (Step 3-B-2). |

---

## F. 미팅 발표 메시지 업데이트

기존 Step 1+2-A 메시지에 한 슬라이드/문단 추가 가능:

> **"보조 테이블 2개를 추가하면 모델 성능이 어떻게 달라지나?"**
>
> Step 1의 application_train만 사용한 XGBoost는 5-fold CV test AUROC 0.7587 ± 0.0008 이었음.
> Home Credit 보조 테이블 중 가장 임팩트가 큰 두 개(외부 신용기관 `bureau`,
> 자체 신청 이력 `previous_application`)에 대해 SK_ID_CURR 단위로 756개의 집계 변수를 생성하고
> 동일 전처리 정책으로 추가하면, **AUROC 0.7755 ± 0.0011 (+0.0168, +2.22%)**, AUPRC +8.21%,
> KS +7.81% 의 향상을 얻음.
>
> SHAP top 20 분석 결과, 새로 진입한 5개의 보조 변수는 모두 `previous_application` 기반이며,
> 가장 직관적인 신호는 **이전 신청 거절 비율(PREV_NAME_CONTRACT_STATUS_Refused_mean)** 이었음.
> 외부 신용기관 변수는 top 20에 진입하지 못했는데, 이는 main 테이블의 EXT_SOURCE_1/2/3에
> 외부 신용 정보가 이미 응축돼 있을 가능성을 시사함 — 추가 ablation으로 검증할 future work.

---

## G. 산출물 인덱스

### 새 코드
- `src/aux_data.py` — 보조 테이블 로더 + dtype downcast
- `src/aux_eda.py` — EDA 보고서 생성
- `src/aux_features.py` — 집계 feature 생성 (756개)
- `src/preprocess_with_aux.py` — 통합 전처리 (1161 features)
- `src/cv_eval_aux.py` — XGBoost 5-fold CV + baseline 비교
- `src/shap_aux.py` — SHAP 재실행 + top 20 변화 분석

### 새 데이터
- `data/processed/aux_features.parquet` (256 MB, 353,577 rows × 756 cols)
- `data/processed/{train,val,test}_{scaled,unscaled}_aux.parquet`
- `data/processed/test_protected_attrs_aux.parquet`

### 새 결과/모델
- `results/aux_eda.md`, `results/aux_eda_summary.json`
- `results/aux_features_summary.json`
- `results/preprocessor_aux.pkl`, `results/preprocess_aux_summary.json`
- `results/cv_metrics_aux.csv`, `results/cv_summary_aux.csv`, `results/cv_aux_vs_baseline.csv`
- `results/shap_global_xgboost_aux.csv`, `results/shap_top20_diff.csv`
- `results/baseline_models/xgboost_aux.pkl`

### 새 figure
- `figures/27_cv_aux_comparison.png`
- `figures/28_shap_global_xgb_aux.png`
- `figures/29_shap_top20_overlap.png`

---

## H. 다음 단계 제안 (미팅 전/후)

### 미팅 전 (D-5 ~ D-1)
1. **발표 자료 업데이트** — `paper/midterm_slides.pptx`에 Step 3-B 슬라이드 1~2장 추가
   - 슬라이드: AUROC bar chart (baseline vs aux), SHAP top 20 변화 figure
2. **midterm_report.docx 부록 추가** (선택) — Step 3-B 한 섹션
3. **리허설**

### 미팅 후 (지도교수 피드백 따라)
1. Bureau ablation (bureau 단독 / prev 단독 / 둘 다 — 어느 게 임팩트 주력인지)
2. LightGBM, TabNet으로 aux 효과 일반화 확인
3. LLM 컨텍스트 재생성 (XAI-RAG with aux SHAP) — Halluc 변화 측정
4. Fairness-aware 학습 — Reweighing / Adversarial Debiasing
