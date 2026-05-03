# Day 2 요약 — 베이스라인 (Logistic / XGBoost / LightGBM)

날짜: 2026-05-03 (Day 1 즉일 진행)
미팅까지: D-7

---

## TL;DR

- **3개 베이스라인 학습 + 평가 완료. XGBoost가 모든 지표에서 1등.**
- 풀 데이터(184K train, 61K val, 61K test) 기준 **test AUROC 0.755~0.761**, **AUPRC 0.234~0.246**, **KS 0.379~0.389**
- 학습 시간: LGBM 17s < XGB 21s < Logistic 88s (Logistic이 가장 느림 — 모든 변수 dense scaled로 lbfgs 수렴이 오래 걸림)
- Day 3 TabNet의 비교 기준점 확보. **TabNet이 의미 있으려면 test AUROC 0.760+ 또는 KS 0.39+ 가 필요.**

---

## 메트릭 표 (Test 셋, threshold=Youden's J on validation)

| 모델 | AUROC | AUPRC | KS | F1 | Precision | Recall | 임계치 | 학습 |
|---|---|---|---|---|---|---|---|---|
| Logistic | 0.7547 | 0.2346 | 0.3787 | 0.2663 | 0.1663 | 0.6679 | 0.513 | 88s |
| **XGBoost** | **0.7605** | **0.2459** | **0.3892** | **0.2705** | 0.1688 | 0.6800 | 0.476 | 21s |
| LightGBM | 0.7549 | 0.2414 | 0.3785 | 0.2622 | 0.1617 | 0.6935 | 0.411 | 17s |

> Validation 셋 결과는 `results/baseline_metrics.csv` 참고.

### 해석
- **XGBoost가 일관되게 최고**. AUROC 격차는 크지 않으나(약 0.005~0.006) AUPRC 격차가 분명함 (0.2459 vs 0.2346).
- **Recall 우선 모델**: 모두 임계치를 낮춰 recall 65~70%까지 끌어올린 상태. 반대로 precision은 16~17%대.
  - 신용평가 도메인에서는 "부도 의심을 충분히 걸러내고 + 사람이 2차 검토" 방식과 정합. 다만 미팅에선 KS/AUROC가 메인 메시지.
- **Logistic이 단일 변수 신호가 약한 데이터(EDA Day 1 인사이트)에서도 0.755** — class_weight 균형 + RobustScaler + 214 feature(원-핫 + flag) 효과. 단, 학습 시간 4배.
- LightGBM은 best_iteration=500에 도달(early stopping 미발동) → n_estimators 1000으로 늘리면 약간 더 향상 여지 있음. 미팅용으로는 현재로 충분.

### 산업 표준 비교
- Home Credit Kaggle 대회 1등 솔루션: 0.8053 (private LB) — **단, 7개 보조 테이블까지 모두 활용 + 광범위한 feature engineering**.
- 메인 테이블만 사용한 구간 솔루션: 0.745~0.770 수준이 일반적.
- → 본 결과 0.755~0.761은 **메인 테이블만 사용한 작업으로 정상 범위 상단**.

---

## 시각화 (Day 2 figures)

- `figures/08_roc_pr_curves.png` — ROC + PR 곡선 비교 (Test)
- `figures/09_threshold_sweep.png` — threshold sweep, validation에서 정한 점 표시
- `figures/10_feature_importance.png` — Logistic |coef| / XGB gain / LGBM gain 각 Top 20

> Day 4 SHAP 분석에서 이 importance 결과와 일관성 비교 예정.

---

## 코드 + 산출물

```
src/
├─ metrics.py                # AUROC/AUPRC/KS/F1, Youden's J 임계치
└─ baselines.py              # 학습 + 평가 + 시각화

results/
├─ baseline_metrics.csv      # 모델 × split × 지표
├─ baseline_summary.json     # 설정/결과/threshold 전부
└─ baseline_models/
    ├─ logistic.pkl
    ├─ xgboost.pkl           # ★ 현 시점 best
    └─ lightgbm.pkl

figures/
├─ 08_roc_pr_curves.png
├─ 09_threshold_sweep.png
└─ 10_feature_importance.png
```

재실행: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.baselines`

---

## 알아둘 만한 작은 결정

1. **임계치 정책**: validation에서 Youden's J(=KS 달성 임계치)로 결정 → test에 적용. 데이터 누수 차단 + 신용평가 표준.
2. **LightGBM 컬럼명 sanitize**: pd.get_dummies가 만든 `NAME_TYPE_SUITE_Spouse, partner` 같은 콤마 포함 컬럼명을 LGBM이 거부 → 학습 직전 `_sanitize_columns`로 치환. (XGBoost/Logistic에는 영향 없음.)
3. **하이퍼파라미터**: 미팅 데드라인 고려해 휴리스틱 디폴트(n_estimators=500, max_depth=6/num_leaves=63, lr=0.05, early_stopping=30). Optuna 튜닝은 Day 3 TabNet에서 작은 budget으로만.

---

## 다음 단계 — Day 3 (TabNet)

목표: TabNet 학습 + Optuna 축소 튜닝 + 어텐션 마스크 추출.

대상:
- pytorch-tabnet 4.1.0 사용
- scaled 입력 (RobustScaler 결과)
- batch_size=1024, virtual_batch_size=128 (6GB VRAM 안전 영역)
- Optuna 20~30 trials, 5-fold CV는 시간 부족하니 hold-out + 1회 CV로 축소
- 어텐션 마스크 step별로 추출 → 평균하여 변수 중요도 계산 → SHAP과 비교 준비

산출물 예정:
- `src/tabnet_train.py`
- `results/tabnet_metrics.csv`, `tabnet_optuna_trials.csv`
- `results/baseline_models/tabnet.pkl`
- `figures/11_tabnet_training_curve.png`
- `figures/12_tabnet_attention_vs_importance.png`

대략 30~90분 소요 (Optuna trial 수에 비례).

**Day 3 들어가도 되는지 확인 부탁.** 또는 베이스라인 결과 검토 후 Day 2 보완할 사항 있으면 알려줘.
