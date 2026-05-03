# TabNet + SHAP + LLM 기반 XAI-RAG 신용 평가

석사학위 논문 프로젝트 (DS·AI). **TabNet** 어텐션 기반 정형 딥러닝 + **SHAP** XAI를 LLM의 검색된 컨텍스트(retrieved evidence)로 재정의하는 **XAI-RAG** 프레임워크. 신용 평가 결과의 자연어 설명 리포트를 환각 없이 자동 생성하고, 그 신뢰성을 정량 평가한다.

> 이 저장소는 진행 중인 연구의 작업 기록이다. 미팅 데드라인까지의 빠른 프로토타이핑이 목표이며, 계획서의 6개월 일정 중 일부는 future work로 미뤘다 (`thesis/CLAUDE.md` 참조).

---

## 핵심 기여

1. 정형 데이터 딥러닝(TabNet)과 사후 해석(SHAP)을 결합한 신용 예측 + 해석 일관성 분석
2. SHAP 결과를 **신뢰 가능한 컨텍스트**로 재정의하는 XAI-RAG 메커니즘 → LLM 환각 차단
3. 자연어 설명 품질을 **Faithfulness / Hallucination Rate / Plausibility / G-Eval**로 다층 정량 평가
4. 보호 속성(연령·성별)에 대한 4종 공정성 지표 측정 + ablation 기반 완화 비교

---

## 프레임워크 개요

```
[정형 데이터]
    ↓
[Prediction]   TabNet ─ baseline 비교(Logistic, XGBoost, LightGBM)
    ↓
[Interpretation] SHAP (global/local) + TabNet 어텐션 마스크
    ↓
[Context Builder] SHAP 결과 → 구조화된 JSON 컨텍스트 (사실 단위)
    ↓
[Generation]   LLM API (Gemini / Claude) — 사실 기반 자연어 리포트
    ↓
[Evaluation]   Faithfulness · Hallucination Rate · G-Eval · Robustness
```

---

## 환경

| 항목 | 값 |
|---|---|
| OS | Windows 10 |
| Python | 3.10.11 |
| GPU | NVIDIA GTX 1660 Ti (6 GB VRAM, CUDA 12.1) |
| 주요 패키지 | `requirements.txt` 참조 |

상세 환경 점검: [results/environment.md](results/environment.md)

---

## 데이터

- **Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)** 메인 테이블 (`application_train.csv`, 약 307K 행 × 122 컬럼)
- 보조 테이블(bureau, previous_application 등)은 **future work**
- 데이터는 라이선스/용량 문제로 저장소에 포함하지 않음 — 다운로드 가이드: [thesis/kaggle_data_setup.md](thesis/kaggle_data_setup.md)

---

## 폴더 구조

```
.
├─ src/                  # 학습/평가 파이프라인 코드
│  ├─ utils.py           # SEED, paths, matplotlib config
│  ├─ data_loader.py     # 메인 테이블 로딩
│  ├─ eda.py             # Day 1 EDA (재실행 가능)
│  ├─ preprocess.py      # 결측/인코딩/스케일링/분할
│  ├─ metrics.py         # AUROC/AUPRC/KS/F1/Youden's J
│  ├─ baselines.py       # Logistic/XGBoost/LightGBM
│  └─ tabnet_train.py    # TabNet + Optuna + 어텐션 추출
├─ thesis/               # 프로젝트 지침 문서
│  ├─ CLAUDE.md          # 작업 컨텍스트, 코딩 규칙, 7일 일정
│  ├─ kaggle_data_setup.md
│  └─ llm_options.md
├─ results/              # 메트릭, 보고서, 요약
│  ├─ environment.md
│  ├─ eda.md / eda_summary.json
│  ├─ baseline_metrics.csv / baseline_summary.json
│  ├─ tabnet_metrics.csv / tabnet_summary.json
│  └─ day{1,2,3}_summary.md
├─ figures/              # 시각화 (PNG)
├─ data/                 # 원본/전처리 데이터 (gitignored)
└─ paper/                # 학번 포함 docx (gitignored)
```

---

## 재현 절차

```bash
# 1. 가상환경 생성 (Windows)
python -m venv .venv
.\.venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt
# PyTorch는 CUDA 버전에 맞춰 별도 설치 권장:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. 데이터 다운로드 (thesis/kaggle_data_setup.md 참조)
#    → data/home_credit/application_train.csv 위치에 둠

# 4. EDA + 전처리 → data/processed/*.parquet 생성
python -m src.eda
python -m src.preprocess

# 5. 베이스라인 학습 (Logistic / XGBoost / LightGBM)
python -m src.baselines

# 6. TabNet 학습 + Optuna 튜닝 + 어텐션 추출
python -m src.tabnet_train --n-trials 10
```

전 과정 SEED=42 고정.

---

## 진행 상황

| Day | 작업 | 상태 |
|---|---|---|
| 1 | 환경 / EDA / 전처리 모듈 | ✅ ([day1_summary](results/day1_summary.md)) |
| 2 | 베이스라인 (Logistic/XGB/LGBM) — best test AUROC **0.7605** (XGBoost) | ✅ ([day2_summary](results/day2_summary.md)) |
| 3 | TabNet + Optuna + 어텐션 분석 | ⏳ 진행 중 |
| 4 | SHAP global/local + 어텐션–SHAP 일관성 | 예정 |
| 5 | 공정성 (DP/EO/EOdds/DI) + XAI-RAG 컨텍스트 빌더 | 예정 |
| 6 | LLM(Gemini API) 설명 생성 + Faithfulness/Hallucination 평가 | 예정 |
| 7 | 결과 종합, 발표자료 | 예정 |

---

## 라이선스 / 주의

- 본 저장소의 코드/문서는 학술 연구용.
- 데이터셋은 [Kaggle Home Credit Default Risk 라이선스](https://www.kaggle.com/competitions/home-credit-default-risk/rules)를 따름. 본 저장소에 포함되지 않으므로 사용자 본인이 동의 후 다운로드.
- 학번/실명 포함 docx 등 식별 정보는 의도적으로 .gitignore 처리.
