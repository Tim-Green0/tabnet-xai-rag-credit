# Day 1 EDA — Home Credit Default Risk

작성일: 2026-05-03
데이터: `data/home_credit/application_train.csv`
스크립트: `src/eda.py`
재현: `.venv/Scripts/python.exe -m src.eda`

---

## TL;DR (핵심 인사이트)

1. **307,511행 × 122컬럼**, 메모리 537 MB. 메인 테이블만 사용해도 충분히 큼.
2. **TARGET 불균형 8.07%** (24,825 / 307,511). class_weight 또는 SMOTE 필요.
3. **결측 심각** — 67개 컬럼에 결측, 그중 41개가 50% 초과. 대부분 주거/건물 정보 변수.
4. **EXT_SOURCE_3 / EXT_SOURCE_2 가 핵심 신호** (|ρ|=0.18, 0.16). 다른 모든 변수는 |ρ|<0.06.
   → 본 데이터는 단일 변수 선형 신호가 약하고 **비선형·상호작용**이 핵심. TabNet/GBM 우위 영역.
5. **공정성 시그널이 이미 데이터에 있음** — 남성 부도율 10.1% vs 여성 7.0% (1.45배), 25세 미만 12.3% vs 65세+ 3.7% (3.4배). 본 연구 공정성 챕터의 좋은 출발점.
6. **데이터 품질 이슈**: DAYS_EMPLOYED에 sentinel 값 365243(=999.99년)이 18%. 무직(NaN) 의미. 전처리 필수.

---

## 1. 데이터 개요

| 항목 | 값 |
|---|---|
| 행 수 | 307,511 |
| 컬럼 수 | 122 |
| 메모리 | 536.7 MB (Pandas, deep) |
| dtype 분포 | float64 65 / int64 41 / object 16 |

## 2. TARGET 분포

| 클래스 | count | 비율 |
|---|---|---|
| 0 (정상) | 282,686 | 91.93% |
| 1 (부도) | **24,825** | **8.07%** |

→ `figures/01_target_distribution.png`

**의미:** Accuracy는 무의미 (91.93%만 맞춰도 되니까). **AUROC, AUPRC, KS** 위주 평가 필요. 학습 시 `class_weight='balanced'` 또는 SMOTE 비교 실험 권장.

## 3. 결측 분석

| 항목 | 값 |
|---|---|
| 결측 있는 컬럼 | 67 / 122 |
| 결측 없는 컬럼 | 55 / 122 |
| 결측률 max | 69.87% |
| 결측률 median (결측 컬럼만) | 50.75% |
| 50% 초과 결측 컬럼 | 41개 |

→ `figures/02_missing_top30.png`, `results/eda_missing_table.csv`

**관찰:** 50%+ 결측 컬럼 41개는 대부분 `*_AVG / _MODE / _MEDI`의 주거/건물 통계 — 주거 정보를 안 받은 신청 건이 많다는 의미. **변수 그룹 전체를 통째로 제거하거나, "주거 정보 입력 여부" 자체를 새 binary feature로 만드는 것** 두 옵션 검토 가치.

## 4. 보호 속성 (성별·연령)

→ `figures/03_protected_attrs.png`

### 성별 (CODE_GENDER)
| | count | 부도율 |
|---|---|---|
| F | 202,448 (65.8%) | **7.00%** |
| M | 105,059 (34.2%) | **10.14%** |
| XNA | 4 | 0.00% |

남성 부도율이 여성 대비 **1.45배**. Demographic Parity 위반 가능성 있음.

### 연령 (DAYS_BIRTH 변환)
| 연령대 | 부도율 |
|---|---|
| ~25 | **12.29%** |
| 25–35 | 10.66% |
| 35–45 | 8.41% |
| 45–55 | 7.05% |
| 55–65 | 5.42% |
| 65+ | **3.66%** |

**3.4배 차이**. 본 연구 공정성 분석에서 **연령 이슈가 성별보다 큰 신호**. AIF360 4/5 rule 위반 거의 확실.

## 5. 핵심 변수 (TARGET 상관)

→ `figures/05_correlation_top.png`, `figures/06_target_vs_features.png`, `results/eda_target_corr.csv`

### |상관계수| Top 10
| 변수 | ρ | 비고 |
|---|---|---|
| **EXT_SOURCE_3** | **-0.1789** | 외부 신용평가 점수. 핵심. |
| **EXT_SOURCE_2** | **-0.1605** | 동일 계열. |
| DAYS_BIRTH | +0.0782 | 어릴수록 부도↑ (음수일수록 어려서 부호 +) |
| REGION_RATING_CLIENT_W_CITY | +0.0609 | 거주지 등급(낮을수록 좋음) |
| REGION_RATING_CLIENT | +0.0589 | |
| DAYS_LAST_PHONE_CHANGE | +0.0552 | 최근 폰 변경 ↑ → 부도↑ |
| DAYS_ID_PUBLISH | +0.0515 | 신분증 발급 후 경과일 |
| REG_CITY_NOT_WORK_CITY | +0.0510 | 등록·근무지 불일치 |
| FLAG_EMP_PHONE | +0.0460 | 직장 폰 등록 |
| REG_CITY_NOT_LIVE_CITY | +0.0444 | |

**해석:**
- `EXT_SOURCE_*` 두 변수가 강력. 이들이 SHAP에서도 주요 driver가 될 가능성 높음 → 자연어 리포트에서도 주요 사유로 등장 예상.
- 그 외 변수는 모두 |ρ|<0.06 수준 → **선형 모델(Logistic) 천장이 낮음**. TabNet/GBM이 비선형으로 끌어올릴 여지가 큼.

## 6. 범주형 변수 (16개)

| cardinality | 컬럼 |
|---|---|
| 2 | NAME_CONTRACT_TYPE, FLAG_OWN_CAR, FLAG_OWN_REALTY, EMERGENCYSTATE_MODE |
| 3 | HOUSETYPE_MODE, CODE_GENDER |
| 4–8 | NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, WEEKDAY_APPR_PROCESS_START, NAME_TYPE_SUITE, WALLSMATERIAL_MODE, NAME_INCOME_TYPE, FONDKAPREMONT_MODE |
| **18** | **OCCUPATION_TYPE** |
| **58** | **ORGANIZATION_TYPE** |

→ `figures/07_categorical_cardinality.png`

**전처리 방향:**
- ≤8 cardinality 14개 → 원-핫 인코딩
- OCCUPATION_TYPE / ORGANIZATION_TYPE → **타깃 인코딩** 또는 임베딩 (원-핫 시 76 컬럼 추가됨)

## 7. 데이터 품질 이슈

### DAYS_EMPLOYED sentinel 365243
```
DAYS_EMPLOYED == 365243  →  55,374건 (18.01%)
```
- 365243일 ≈ 1000년. 사실상 **무직(unemployed) 의미의 sentinel**.
- 그대로 두면 모델이 거대 양수로 인식해 패턴이 왜곡됨.
- **전처리에서 NaN 변환 + `EMPLOYED_FLAG=0` 새 binary 변수 생성** 권장.

### 기타 의심 변수 (전처리 모듈에서 점검 예정)
- `XNA` 같은 unknown placeholder가 있는 범주형
- `AMT_INCOME_TOTAL`의 극단치 (예: 117M 같은 outlier)

## 8. 전처리 권장 방안 (Day 1-D에서 구현 예정)

| 단계 | 처리 |
|---|---|
| 1. sentinel 처리 | `DAYS_EMPLOYED == 365243 → NaN`, `EMPLOYED_FLAG` 신규 생성 |
| 2. 결측치 | 수치형 median, 범주형 최빈값 또는 'Missing' 카테고리 |
| 3. 50%+ 결측 컬럼 | 일단 유지하되 `MISSING_FLAG` 변수 생성 옵션 |
| 4. 범주형 인코딩 | cardinality ≤ 8 → one-hot, 그 외 → target encoding (CV-safe) |
| 5. 이상치 | AMT_*, DAYS_* 변수에 1~99% 분위수 클리핑 |
| 6. 스케일링 | 트리계열은 불필요, TabNet은 RobustScaler 적용 |
| 7. 클래스 불균형 | class_weight='balanced' (기본) + SMOTE 비교 (선택) |
| 8. 분할 | Train/Valid/Test = 60/20/20, stratified by TARGET, SEED=42 |

## 9. 산출물 위치

```
figures/
├─ 01_target_distribution.png       # 8.07% 불균형 시각화
├─ 02_missing_top30.png              # 결측률 막대
├─ 03_protected_attrs.png            # 성별·연령 분포 + 부도율
├─ 04_numeric_distributions.png      # 핵심 14개 수치형 히스토그램
├─ 05_correlation_top.png            # |ρ| top 20 vs TARGET 히트맵
├─ 06_target_vs_features.png         # TARGET별 KDE 6개
└─ 07_categorical_cardinality.png    # 범주형 cardinality

results/
├─ eda_summary.json                  # 본 보고서의 모든 수치 (재사용 용)
├─ eda_missing_table.csv             # 67개 결측 컬럼 전수
└─ eda_target_corr.csv               # 모든 수치형 변수의 TARGET 상관
```

## 10. 다음 단계

- [ ] **Day 1-D**: `src/preprocess.py` 작성 (위 8번 표 기반) — **사용자 검증 후 진행**
- [ ] **Day 2**: 베이스라인 모델 (Logistic, XGBoost, LightGBM)
- [ ] **Day 3**: TabNet + Optuna 축소 튜닝

전처리 진입 전 사용자 결정 필요한 항목 → Day 1 요약 보고에서 별도 정리.
