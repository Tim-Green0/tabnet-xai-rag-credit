# CLAUDE.md — 석사학위 논문 프로젝트 지침

> 이 파일은 본 프로젝트에서 Claude가 작업할 때 따라야 할 핵심 컨텍스트와 규칙이다.
> 사용자(석사 과정 학생)가 일주일 후 지도교수 미팅용 **프로토타입**을 준비하기 위한 작업 환경.

---

## 1. 프로젝트 개요

**논문 제목:** 정형 데이터 특화 딥러닝(TabNet)과 거대언어모델(LLM) 기반 XAI-RAG를 활용한 설명 가능한 신용 평가 및 사용자 맞춤형 리포트 생성 연구

**전공:** 데이터사이언스 · 인공지능 (석사)
**지도교수:** 박운상
**계획서 일자:** 2026-01-23 (6개월 일정)

### 핵심 아이디어
1. **TabNet**: 정형 신용 데이터에서 트리 기반 모델과 동등 이상의 분류 성능 + 내재적 어텐션 해석
2. **SHAP**: 모델의 예측 근거를 변수별 기여도로 정량화
3. **XAI-RAG**: SHAP 결과를 "신뢰 가능한 컨텍스트(retrieved evidence)"로 재정의하여 LLM 프롬프트에 주입 → LLM이 임의 추론하지 않고 사실 기반 자연어 설명 생성 → **환각(Hallucination) 원천 차단**
4. **다층 평가**: Faithfulness · Hallucination Rate · Plausibility · G-Eval로 설명 품질을 정량 검증

### 본 연구의 차별점 (계획서 2.7 표 기준)
| 차원 | 기존 연구 | 본 연구 |
|---|---|---|
| 예측 모델 | XGB/LGBM 중심 | TabNet + 어텐션–SHAP 일관성 분석 |
| 설명 산출물 | 그래프/수치 (전문가용) | 자연어 리포트 (소비자용) |
| LLM 활용 | 분류기 / 단순 변환기 | XAI-RAG로 환각 차단, 사실 기반 보장 |
| 설명 평가 | 정성적 사례 제시 | Faithfulness + Hallucination + G-Eval 정량 평가 |

---

## 2. 일주일 데드라인 (절대 준수)

- **D-day: 2026-05-10 (지도교수 중간 미팅)**
- 오늘: 2026-05-03 (Day 1)
- 목표: **완벽한 논문이 아닌, "작동하는 파이프라인" 프로토타입**
- 계획서의 6개월 일정은 미팅 이후 본 진행 → 미팅용으로는 의도적으로 축소·단순화

---

## 3. 실제 확인된 환경 (2026-05-03 점검 기준)

자세한 내용: `results/environment.md`

| 항목 | 값 | 비고 |
|---|---|---|
| OS | Windows 10 Pro 19045 | |
| Python | 3.10.11 (시스템) | venv 미사용. **venv 또는 conda 환경 생성 권장** |
| GPU | GTX 1660 Ti, **6 GB VRAM** | 계획서(RTX 3090 24GB) 대비 매우 제한적 |
| 드라이버 | 460.89 (CUDA 11.2) | **구버전**. 최신 PyTorch 호환성 의문 |
| 패키지 | pandas, sklearn, numpy, matplotlib만 설치 | torch/tabnet/xgb/lgbm/shap/optuna 미설치 |
| 데이터 | `data/home_credit/` 비어있음 | Kaggle에서 다운로드 필요 |

### 환경 제약이 연구 설계에 미치는 영향
- **6 GB VRAM**: TabNet은 작은 batch + virtual_batch로 학습 가능. 그러나 **Llama 3 8B/70B 로컬 실행 불가**.
- **LLM 호출은 외부 API(OpenAI GPT-4o, Anthropic Claude API 등) 사용**이 현실적.
- 계획서에 명시된 Llama 3 70B 비교는 **future work로 이관**.

---

## 4. 데이터

- **메인 데이터셋: Kaggle Home Credit Default Risk** (`application_train.csv`)
  - 약 307,511 행 × 122 컬럼, 약 166 MB
  - **메인 테이블만 사용** (bureau, previous_application 등 보조 테이블은 미사용 — 미팅용 단순화)
  - 위치: `data/home_credit/application_train.csv`
- 부가 데이터셋(UCI German Credit): **이번 일주일에서는 보류**, 시간 여유 시 추가
- 타깃: `TARGET` (1=default, 0=정상)
- 클래스 비율: 약 8% positive (불균형) → AUPRC, KS 병행 평가

---

## 5. 모델 비교 대상 (일주일 미팅용)

### 필수
- Logistic Regression (해석 가능 베이스라인)
- LightGBM (트리 SOTA)
- XGBoost
- TabNet (제안 모델)

### 선택 (시간 여유 시)
- MLP (DNN 베이스라인)

### 미팅용 제외 (future work)
- Random Forest (LGBM과 중복 성격)
- FT-Transformer
- Llama 3 70B 비교

모든 모델: 동일 split, 동일 SEED, 동일 평가 지표.

---

## 6. TabNet 하이퍼파라미터 권장 (6 GB VRAM 기준)

```python
TabNetClassifier(
    n_d=16, n_a=16,           # 의사결정/어텐션 차원 (8~32 범위 탐색)
    n_steps=4,                # 의사결정 단계 (3~6 범위)
    gamma=1.5,                # 마스크 재사용 패널티
    lambda_sparse=1e-4,       # 희소성 정규화
    optimizer_fn=torch.optim.AdamW,
    optimizer_params=dict(lr=2e-2),
    scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
    mask_type='sparsemax',    # 또는 'entmax'
    seed=42,
)
# fit:
clf.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=['auc'],
    max_epochs=100,
    patience=15,
    batch_size=1024,          # 6GB에서 안전. 64 차원 모델이면 512로 축소
    virtual_batch_size=128,
    num_workers=0,            # Windows에서는 0 권장
    drop_last=False,
)
```

- Optuna 탐색은 **20~30 trials**로 축소 (계획서의 100 trials는 시간 부족)
- 5-fold CV는 **3-fold로 축소 가능** (또는 hold-out + 1회 CV)

---

## 7. 평가 지표

### 예측 성능 (필수)
- AUROC (메인)
- AUPRC (불균형 대비)
- F1, Precision, Recall (threshold 0.5 기준 + KS 통계량으로 결정한 threshold)
- KS statistic

### 공정성 (간략)
- Demographic Parity (DP)
- Equal Opportunity (EO)
- 보호 속성: 연령(`DAYS_BIRTH`로 50세 기준 이분), 성별(`CODE_GENDER`)
- AIF360 미설치 시 **수동 계산** (SciPy/numpy로 충분)

### 설명 품질 (간략)
- **Faithfulness Score**: 생성 텍스트의 변수·부호·수치 주장이 SHAP 컨텍스트와 일치하는 비율 (LLM 보조 추출 또는 정규식)
- **Hallucination Rate**: 컨텍스트에 없는 변수가 생성 텍스트에 등장한 비율
- 미팅용 제외: G-Eval, 인간 평가, BERTScore Robustness, Counterfactual Test

---

## 8. 폴더 구조

```
D:\paper\
├─ paper/             # 계획서 docx
├─ data/home_credit/  # 데이터 (Kaggle)
├─ src/
│   ├─ data_loader.py
│   ├─ preprocess.py
│   ├─ baselines.py   # LR, XGB, LGBM
│   ├─ tabnet_train.py
│   ├─ shap_analysis.py
│   ├─ fairness.py
│   ├─ context_builder.py  # SHAP → JSON
│   ├─ llm_explainer.py    # XAI-RAG
│   └─ eval_explanation.py # Faithfulness/Hallucination
├─ thesis/            # CLAUDE.md, 작업 노트
├─ results/           # 실험 결과(.json, .csv, .md)
├─ figures/           # 시각화 산출물
└─ notebooks/         # (선택) 탐색용 jupyter
```

---

## 9. 코딩 규칙

### 절대 규칙
- **`SEED = 42`** 고정. numpy, random, torch, sklearn `random_state` 모두 동일.
- **모든 결과는 파일로 저장.** `results/` 또는 `figures/`에 적절한 이름으로.
  - 표/메트릭: `.csv` 또는 `.json`
  - 그림: `.png` (dpi ≥ 150)
  - 모델 가중치: `results/models/`
- **중요한 의사결정은 멈추고 사용자에게 확인.** 예: 모델 구조 변경, 데이터 분할 전략 변경, 외부 API 호출, 패키지 추가 설치.
- **에러는 추측 말고 작은 테스트로 검증.** 풀데이터 돌리기 전 1,000행 샘플로 dry-run.
- **백업되지 않은 데이터/결과 덮어쓰기 금지.** 기존 파일 있으면 사용자 확인.

### 작업 흐름
- 매 단계마다 짧은 요약(`results/day_N_summary.md`)을 남긴다.
- 코드는 `src/`에 모듈화. 노트북은 탐색 전용, 재현은 `src/` 스크립트로.
- 사용자 검증 단계: 코드 실행 후 결과 파일 위치와 핵심 수치를 사용자에게 보고하고 다음 단계로 진행 여부 확인.

### 패키지 설치
- 새 패키지는 사용자 확인 후 설치. 설치 명령은 `pip install`로 통일.
- venv 만들기: 사용자 결정 후 진행.

---

## 10. 7일 일정 (미팅용 프로토타입)

| 날짜 | Day | 작업 | 산출물 |
|---|---|---|---|
| 05-03 (오늘) | 1 | 환경/계획서 점검 (완료) → 데이터 다운로드 확인, EDA, 전처리 모듈 | `results/eda.md`, `figures/eda_*.png`, `src/data_loader.py`, `src/preprocess.py` |
| 05-04 | 2 | 베이스라인 모델 (Logistic, XGBoost, LightGBM) 학습 + 평가 | `results/baseline_metrics.csv`, 모델 weight |
| 05-05 | 3 | TabNet 구현 + Optuna 축소 튜닝 + 학습 | `results/tabnet_metrics.csv`, 어텐션 마스크 분석 |
| 05-06 | 4 | SHAP 분석 (global/local) + 어텐션–SHAP 일관성 비교 | `figures/shap_*.png`, `results/attention_vs_shap.csv` |
| 05-07 | 5 | 공정성 지표 측정 + XAI-RAG JSON 컨텍스트 빌더 + 프롬프트 템플릿 | `src/context_builder.py`, `results/fairness.csv` |
| 05-08 | 6 | LLM API 호출(GPT-4o or Claude) → 설명 리포트 생성 → Faithfulness/Hallucination 평가 | `results/explanations/*.json`, `results/explanation_eval.csv` |
| 05-09 | 7 | 결과 종합, 발표 자료(슬라이드 또는 docx), 데모 시나리오 정리 | `paper/midterm_slides.pptx` 또는 `.docx` |
| 05-10 | — | 미팅 |

여유분: 매일 마지막 1~2시간은 다음날 준비 + 막힘 해결.

---

## 11. 미팅용 의도적 축소 (계획서 → 프로토타입)

| 계획서 항목 | 프로토타입 처리 |
|---|---|
| UCI German Credit + Home Credit | Home Credit만 (메인 테이블만) |
| 5-fold CV × Optuna 100 trials | 3-fold or hold-out × Optuna 20~30 trials |
| Llama 3 8B / 70B 비교 | API 1종(예: GPT-4o)만, 로컬 LLM 미실행 |
| AIF360 기반 fairness 완화 재학습 | 측정만, 완화는 future work |
| Counterfactual Test, Robustness, BERTScore, ROUGE-L | future work |
| 인간 평가 (Plausibility) | future work, 대신 G-Eval 1회 시도 가능 |
| FT-Transformer | future work |
| MLflow 실험 추적 | 단순 csv/json 로깅으로 대체 |

---

## 12. Claude 작업 시 추가 주의

- 본 프로젝트는 **첫 세션**으로 git 저장소가 아님. 코드 변경은 파일 시스템에 직접 적용.
- 사용자는 검증·해석에 집중하고 코드는 Claude가 짠다.
- 미팅 일정이 빠듯하므로 "완벽한 모듈화"보다 "작동 → 결과 → 다음 단계"를 우선.
- 결과가 예상과 다르면 즉시 사용자에게 보고하고 원인 분석.
- 외부 API 호출(LLM, Kaggle API 등)은 비용/키 관리 필요 → **항상 사용자 확인 후 호출**.
