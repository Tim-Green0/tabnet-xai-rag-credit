# Day 1 요약 — 환경 셋업 + EDA + 전처리

날짜: 2026-05-03
다음 미팅까지: D-7 (2026-05-10)

---

## 완료한 것

### 1. 환경 셋업
- venv: `D:\paper\.venv\` (Python 3.10.11)
- PyTorch 2.5.1+cu121 + CUDA 인식 OK (GTX 1660 Ti 6GB, compute 7.5)
- 17개 ML/LLM/유틸 패키지 설치 (anthropic, google-genai 양쪽 모두 둠)
- 상세: [results/environment.md](environment.md)

### 2. EDA (보통 깊이)
- 7개 figures + 3개 결과 파일
- 핵심 인사이트 6가지:
  1. 307,511행 × 122컬럼, TARGET 8.07% 불균형
  2. 67/122 컬럼에 결측, 그중 41개가 50%+
  3. **EXT_SOURCE_2/3 만 강한 신호** (|ρ|=0.16~0.18). 나머지 |ρ|<0.06 → 비선형이 핵심
  4. 공정성 사전 신호: 남녀 부도율 1.45배, 연령대 3.4배 차이
  5. DAYS_EMPLOYED에 sentinel 365243 18% (무직)
  6. 고-cardinality 범주형 2개 (OCCUPATION_TYPE 18, ORGANIZATION_TYPE 58)
- 상세: [results/eda.md](eda.md)

### 3. 전처리 모듈 + 데이터 가공
- `src/preprocess.py` — leakage 방지 fit/transform 분리
- 적용된 정책 (사용자 결정 A1/B1/C1/D/E1/F):
  - 결측 50%+ 컬럼 유지 + `*_MISSING_FLAG` 추가 → flag 컬럼 43개
  - 범주형: ≤8 cardinality → one-hot (14개), OCCUPATION/ORGANIZATION → target encoding (CV-safe smoothing=10)
  - 수치형: median imputation + 1~99% 분위수 클리핑
  - DAYS_EMPLOYED sentinel → NaN + `EMPLOYED_FLAG` 신규 변수
  - EXT_SOURCE_*: median + `_MISSING_FLAG` 추가
  - RobustScaler: 학습셋 fit, scaled 버전과 unscaled 버전 둘 다 저장
  - Train/Val/Test = 60/20/20 stratified, SEED=42

### 4. 데이터 처리 결과
| 단계 | shape | 비고 |
|---|---|---|
| 입력 | (307,511, 122) | application_train.csv |
| Train | (184,506, 214) | pos_rate=0.0807 |
| Val | (61,502, 214) | pos_rate=0.0807 |
| Test | (61,503, 214) | pos_rate=0.0807 |

122 → 214 features (one-hot 확장 + flag/EMPLOYED_FLAG 신규).
NaN 0개.

---

## 산출물 위치

```
src/
├─ utils.py           # SEED, paths, matplotlib config
├─ data_loader.py     # load_application_train(), basic_info()
├─ eda.py             # 재실행 가능한 EDA 스크립트
└─ preprocess.py      # split_data() + HomeCreditPreprocessor

data/
└─ processed/
   ├─ train_scaled.parquet      (17 MB)  ← TabNet/Logistic용
   ├─ val_scaled.parquet
   ├─ test_scaled.parquet
   ├─ train_unscaled.parquet    (17 MB)  ← XGBoost/LightGBM용
   ├─ val_unscaled.parquet
   ├─ test_unscaled.parquet
   └─ test_protected_attrs.parquet      ← 공정성 평가용 (CODE_GENDER, AGE 보존)

results/
├─ environment.md
├─ eda.md
├─ eda_summary.json
├─ eda_missing_table.csv
├─ eda_target_corr.csv
├─ preprocess_summary.json
├─ preprocessor.pkl              ← 학습된 전처리기 (재사용)
└─ day1_summary.md (이 파일)

figures/
└─ 01~07_*.png  (7개)
```

---

## 다음 단계 — Day 2 (베이스라인)

목표: 3개 베이스라인 학습 + AUROC/AUPRC/KS/F1 평가 + threshold 튜닝.

대상 모델:
- Logistic Regression (scaled 입력 + class_weight='balanced')
- XGBoost (unscaled 입력 + scale_pos_weight)
- LightGBM (unscaled 입력 + class_weight='balanced')

산출물 예정:
- `src/baselines.py`
- `results/baseline_metrics.csv` (모델 × 지표)
- `figures/08_roc_pr_curves.png`
- `figures/09_threshold_sweep.png`
- `results/baseline_models/{logistic,xgb,lgbm}.pkl`

대략 1~2시간이면 충분 (Optuna 안 쓰고 디폴트로 한 번 + 약간 튜닝).

Day 2 시작해도 되는지 사용자 확인.
