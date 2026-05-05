# CLAUDE.md — 석사 논문 프로젝트 지침

> 이 파일은 본 프로젝트에서 Claude가 작업할 때 따라야 할 핵심 컨텍스트.
> **새 세션을 시작하면 이 파일을 가장 먼저 읽고, 진행 흐름(§3)을 따라 현재 위치를 파악할 것.**

---

## 0. 새 세션 진입 시 회복 절차

```bash
cd D:\paper
git log --oneline -10              # 최근 작업 확인
git tag -l                         # step1, step2a 등 마일스톤 태그
ls results/day*_summary.md         # Day별 보고서 (Day 1~8 + Step 2-A)
ls results/step*_summary.md        # Step 보고서
cat thesis/CLAUDE.md                # ← 본 파일
```

**현재 위치**: Step 2-A 완료 (tag `step2a`, 2026-05-05).
**다음 후보**: Step 3-B (성능·방법론) 또는 Step 3-C (논문 작성) — §4 참조.

---

## 1. 프로젝트 개요

**논문 제목**: 정형 데이터 특화 딥러닝(TabNet)과 거대언어모델(LLM) 기반 XAI-RAG를 활용한 설명 가능한 신용 평가 및 사용자 맞춤형 리포트 생성

**전공**: DS·AI 석사 / **학번**: A70067 / **이름**: 오현택 / **지도교수**: 박운상
**계획서**: 2026-01-23 / **중간 미팅 데드라인**: **2026-05-10**

### 핵심 아이디어 (한 줄)
> "SHAP의 수치 기반 설명을 LLM의 검색된 근거(retrieved evidence)로 재정의하면, LLM 종속성 없이 환각이 0%인 자연어 설명 리포트를 생성할 수 있다."

### 4단계 파이프라인
```
[정형 데이터] → [TabNet/XGBoost 예측] → [SHAP local + Attention]
              → [JSON 컨텍스트(민감변수 마스킹)] → [LLM 자연어 설명] → [정량 평가]
```

---

## 2. 환경 정보 (실측 — 변경되면 갱신)

| 항목 | 값 |
|---|---|
| OS | Windows 10 Pro 19045 |
| Python | 3.10.11, venv: `D:\paper\.venv\` |
| GPU | GTX 1660 Ti, 6 GB VRAM, CUDA 12.1 (드라이버 업데이트 후) |
| 주요 패키지 | torch 2.5.1+cu121, pytorch-tabnet 4.1.0, xgboost 3.2.0, lightgbm 4.6.0, shap 0.49.1, optuna 4.8.0, sentence-transformers, anthropic, google-genai |
| LLM | Gemini 2.5 Flash (paid, $10 충전), Claude Sonnet 4.5 (paid) |
| API 키 | `D:\paper\.env`: `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` (gitignored) |

### 데이터
- **메인**: Kaggle Home Credit Default Risk `application_train.csv` (307,511 × 122)
- **위치**: `data/home_credit/` (gitignored, 사용자 다운로드)
- **타깃**: `TARGET` 8.07% positive (불균형)
- **보조 테이블**: 미사용 (Step 3-B에서 추가 검토)

### 처리된 데이터
- `data/processed/{train,val,test}_{scaled,unscaled}.parquet`
- 60/20/20 stratified split (SEED=42), 122 → 214 features
- `results/preprocessor.pkl` (학습된 전처리기)

---

## 3. 진행 흐름 — 단계별 정리 ★

본 프로젝트는 **Step 0(환경) → Step 1(미팅 프로토타입 8일) → Step 2-A(평가 강화) → ...** 식으로 진행. 각 Step 끝에 git tag로 마킹.

### Step 0 — 환경·데이터 셋업 (Day 0)

| 한 일 | 산출물 |
|---|---|
| venv + 17개 패키지 설치 | `.venv/`, `requirements.txt` |
| Kaggle 데이터 다운로드 가이드 | `thesis/kaggle_data_setup.md` |
| LLM 옵션 정리 | `thesis/llm_options.md` |
| 환경 점검 보고서 | `results/environment.md` |

### Step 1 — 미팅 프로토타입 (8일, Day 1-8) ★ tag `step1`

> **목표**: D-7 미팅용 작동 프로토타입. 완벽함보다 "작동하는 파이프라인" 우선.

#### Day 1 — EDA + 전처리 (`results/day1_summary.md`)
- **산출**: `figures/01~07_*.png`, `data/processed/*.parquet`, `src/{utils,data_loader,eda,preprocess}.py`
- **핵심 발견**: TARGET 8.07% 불균형, EXT_SOURCE_2/3가 핵심 신호 (|ρ|=0.16~0.18), 결측 50%+ 41컬럼, DAYS_EMPLOYED sentinel 365243(18%), 공정성 신호 이미 데이터에 존재 (성별 1.45배, 연령 3.4배)
- **사용자 결정**: A1 (결측 컬럼 유지+flag), B1 (cardinality≤8 one-hot, OCC/ORG target encoding), C1 (class_weight balanced), D (60/20/20), E1 (EXT_SOURCE median+flag), F (보호속성 학습 포함, 공정성 별도)

#### Day 2 — 베이스라인 (`results/day2_summary.md`)
- **산출**: `src/{baselines,metrics}.py`, `results/baseline_metrics.csv`, `figures/08~10_*.png`
- **핵심 결과**: XGBoost test AUROC **0.7605** (1등), LightGBM 0.7549, Logistic 0.7547
- **결정**: threshold = Youden's J on validation, applied to test

#### Day 3 — TabNet (`results/day3_summary.md`)
- **산출**: `src/tabnet_train.py`, `figures/11_12_*.png`, `tabnet_best.zip`
- **핵심 결과**: TabNet-tuned test AUROC **0.7543** (XGBoost보다 0.006 부족), 어텐션 Top 1 = NAME_CONTRACT_TYPE_Revolving loans, EXT_SOURCE_2/3 Top 2/4
- **공정성 alarm**: CODE_GENDER_M이 어텐션 Top 6 → Day 5 공정성 분석 동기

#### Day 4 — 5-fold CV + SHAP + Attention 일관성 (`results/day4_summary.md`)
- **산출**: `src/{cv_eval,shap_analysis}.py`, `figures/13~17_*.png`, SHAP global csv
- **핵심 결과**:
  - 5-fold CV: XGBoost **0.7587 ± 0.0008** (5/5 fold 1등, std 매우 작음)
  - **Attention vs SHAP** (RQ2 답변): Spearman ρ=0.117, Top-50 ρ=−0.195, Top-20 Jaccard 0.29 → 핵심 변수는 일관, 미세는 상보적
- **버그 우회**: SHAP 0.49 + XGBoost 3.x 호환성 → `_XgbNativeExplainer` 클래스 (XGBoost native pred_contribs API)
- **Local SHAP**: 거절 5 + 정상 5 샘플 → `results/shap_local_examples.json` (Day 6 LLM 입력용)

#### Day 5 — 공정성 (`results/day5_summary.md`)
- **산출**: `src/fairness.py`, `figures/18_19_*.png`
- **핵심 결과**:
  - 베이스라인: 4 모델 × {GENDER, AGE} = **8/8 케이스 4/5 rule 위반** (DI<0.8)
  - GENDER ablation 효과적 (DP −36~40%, AUROC −0.005~−0.010)
  - AGE ablation 효과 미미 — proxy variable (DAYS_EMPLOYED 등에 간접 인코딩)

#### Day 6 — XAI-RAG + LLM 호출 (`results/day6_summary.md`)
- **산출**: `src/{context_builder,llm_explainer}.py`, `results/contexts/`, `results/explanations/` (Gemini), `results/explanations_anthropic/` (Claude)
- **핵심 결과**: 10 샘플 모두 성공, 컨텍스트 변수·값·SHAP 부호 정확 인용, 민감변수 마스킹 작동
- **발견**: gemini-2.0-flash free tier limit=0 → **gemini-2.5-flash로 전환**, RPD 20건/일 제한 (paid 전환으로 해결)

#### Day 7 — 정량 평가 (`results/day7_summary.md`)
- **산출**: `src/{eval_explanation,compare_llms}.py`, `figures/20~21_*.png`
- **핵심 결과** (10 샘플 기준):
  - **Hallucination Rate = 0.000** (Gemini, Claude 모두) ⭐
  - G-Eval (Gemini self): factual 5.0, sensitive 5.0, style 5.0, completeness 3.4
  - Claude는 더 빠르고 효율적 (8.4s vs 12.7s, 2500토큰 vs 4155)

#### Day 8 — Demo + 보고서 + Counterfactual baseline (`results/day8_summary.md`)
- **산출**: `src/{demo,gen_report,baseline_no_shap}.py`, `paper/midterm_report.docx`, `figures/22_23_*.png`
- **핵심 결과**:
  - **Demo idx 59291 (True Positive)**: end-to-end 시연 가능
  - **★ Counterfactual baseline**: Claude 0% (XAI-RAG) vs **45.5%** (no-SHAP) — DTI/LTV/DSR/햇살론 등 환각 사례 발견
  - 미팅 결정타 메시지

#### Step 1 추가 산출물 (사용자 요청)
- `paper/midterm_report_friendly.docx` (697 KB) — 친절판
- `paper/midterm_slides.pptx` (483 KB, 14 슬라이드) — 미팅용
- `src/{gen_friendly_report,gen_slides,regen_figures}.py`
- 모든 figure 텍스트 영어화 (한글 폰트 fallback 문제 해결)

---

### Step 2-A — 평가 신뢰성 강화 ★ tag `step2a`

> **목표**: Step 1의 가장 큰 약점(평가 표본 10명) 해소 + Cross-LLM/Counterfactual/Robustness 정량화

#### Phase 1 — 100명 표본 확장
- **스크립트**: `src/expand_samples.py`
- **산출**: `results/contexts_100/` (100개), `results/shap_local_examples_100.json`

#### Phase 2-3 — LLM 자연어 설명 100건 × 2 LLM
- Gemini 100건: `results/explanations_gemini_100/`
- Claude 100건: `results/explanations_anthropic_100/`

#### Phase 4 — 룰 기반 평가 100건
- **결과**: **Halluc 0/100 (양 LLM 모두)** — Step 1의 10건 결과가 표본 확장에도 견고

#### Phase 5 — Cross-LLM G-Eval (self-bias 우회)
- **스크립트**: `src/cross_llm_geval.py`
- **결과** (n=30 each):
  - Claude → Gemini: factual 4.87, comp 4.0, sens 5.0, style 4.97
  - Gemini → Claude: factual 4.6, comp 3.33, sens 5.0, style 5.0
- **발견**: Gemini self-judge는 자기 비판 방향 (3.375 < Claude judge의 4.0)

#### Phase 6 — Counterfactual Test 정량화
- **스크립트**: `src/counterfactual_test.py`, `src/text_similarity.py` (multilingual sentence-transformers)
- **결과** (n=30, top driver 1개 제거):
  - Claude: cosine 0.909, ROUGE-L 0.747
  - Gemini: cosine 0.920, ROUGE-L 0.750
- **해석**: 부분 perturbation에서도 의미 일관성 유지

#### Phase 7 — Robustness (3 변형: role/example/driver shuffle)
- **스크립트**: `src/robustness_test.py`
- **결과** (n=20):
  - Claude cosine: 0.914 ~ 0.924
  - Gemini cosine: 0.908 ~ 0.951 (약간 더 안정)
- **해석**: 프롬프트 미세 변형에 강건

#### Step 2-A 보고서
- `results/step2a_summary.md`

---

## 4. 다음 단계 후보 (Step 3 ~)

### Step 3-B — 성능·방법론 확장 (4~5일)
- **⑤ 보조 테이블 활용** (bureau, previous_application 등 6개) → AUROC 0.78+ 목표
  - feature engineering (집계 변수)
  - 메모리 관리 (전체 합치면 ~2.6GB)
- **⑥ Fairness-aware 학습**:
  - Reweighing (sample weight 조정)
  - Adversarial Debiasing
  - 4/5 rule 통과 시도
- **⑦ FT-Transformer 비교 모델 추가**

### Step 3-C — 논문 작성 모드 (Step 3-B 후 또는 병행)
- **⑧ 인간 평가 (Plausibility)** — IRB 절차, 5점 척도, Cohen's κ
- **본격 논문 초안**: LaTeX 또는 docx
  - 5장 구성 (서론/관련연구/방법론/실험결과/결론)
  - 도식 (4단계 파이프라인 그림)
  - 모든 표·figure 영문 캡션

### Step 3-D — 본 결정 사항 (계획서 항목 중 미진행)
- ⑨ 한국어 도메인 특화 LLM QLoRA 미세조정 — GPU 부담
- 보조 데이터셋 (UCI German Credit) 추가
- MLflow 실험 추적 시스템화

### 미팅 후 진행할 가능성 높은 것
- 지도교수 피드백에 따라 Step 3-B 또는 Step 3-C 우선순위 결정
- 보조 테이블이 가장 큰 임팩트 — 추천 1순위

---

## 5. 코드 / 폴더 구조

```
D:\paper\
├─ .venv/                       # gitignored
├─ .env                         # gitignored, GEMINI/ANTHROPIC 키
├─ data/
│   ├─ home_credit/             # 원본 csv (gitignored, 2.6GB)
│   └─ processed/               # 학습용 parquet (gitignored)
├─ paper/
│   ├─ 석사학위_논문계획서_*.docx   # gitignored (학번 노출)
│   ├─ midterm_report.docx        # ✅ commit (압축판)
│   ├─ midterm_report_friendly.docx  # ✅ commit (친절판)
│   └─ midterm_slides.pptx        # ✅ commit (14 슬라이드)
├─ src/                         # ✅ 모든 코드
│   ├─ utils.py                 # SEED=42, paths, matplotlib config
│   ├─ data_loader.py / eda.py / preprocess.py
│   ├─ metrics.py               # AUROC/AUPRC/KS/F1, Youden's J
│   ├─ baselines.py / cv_eval.py
│   ├─ tabnet_train.py
│   ├─ shap_analysis.py         # _XgbNativeExplainer (호환성 우회)
│   ├─ fairness.py
│   ├─ context_builder.py       # SHAP → JSON, DOMAIN_GLOSSARY 한국어 매핑
│   ├─ llm_explainer.py         # Gemini + Claude provider 추상화
│   ├─ eval_explanation.py      # Faithfulness/Hallucination/G-Eval 룰
│   ├─ compare_llms.py / baseline_no_shap.py / demo.py
│   ├─ gen_report.py            # midterm_report.docx 생성기
│   ├─ gen_friendly_report.py   # midterm_report_friendly.docx 생성기
│   ├─ gen_slides.py            # midterm_slides.pptx 생성기
│   ├─ regen_figures.py         # csv/json에서 figure만 재생성
│   ├─ expand_samples.py        # Step 2-A: 100명 SHAP+context
│   ├─ text_similarity.py       # Step 2-A: multilingual sentence-transformers
│   ├─ counterfactual_test.py   # Step 2-A
│   ├─ robustness_test.py       # Step 2-A
│   └─ cross_llm_geval.py       # Step 2-A
├─ results/                     # 메트릭/보고서/아티팩트
│   ├─ environment.md / day{1..8}_summary.md / step2a_summary.md
│   ├─ baseline_models/         # *.pkl/zip (gitignored, 5MB)
│   ├─ contexts/ (10) / contexts_100/ (100)
│   ├─ explanations{,_anthropic}/ (10) / explanations_{gemini,anthropic}_100/ (100)
│   ├─ explanations_counterfactual_{gemini,anthropic}/
│   ├─ explanations_robustness_{gemini,anthropic}/
│   ├─ explanations_baseline_noshap_{gemini,anthropic}/
│   └─ *.csv / *.json (메트릭, summary)
├─ figures/                     # ✅ 26개 png (Step 1: 1-23, Step 2-A: 20_*_100, 24-26)
└─ thesis/
    ├─ CLAUDE.md (이 파일)
    ├─ kaggle_data_setup.md
    └─ llm_options.md
```

---

## 6. 자주 쓰는 명령어

```bash
# 가상환경 (Windows bash)
.venv/Scripts/python.exe -m <module>

# 인코딩 (한글 출력 시 필수)
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -m src.eda

# 핵심 파이프라인 재실행
python -m src.eda                    # EDA
python -m src.preprocess             # 전처리
python -m src.baselines              # 베이스라인 학습
python -m src.cv_eval --skip-tabnet  # 5-fold CV (베이스라인만)
python -m src.cv_eval --only-tabnet  # 5-fold CV (TabNet만 추가)
python -m src.tabnet_train --n-trials 10
python -m src.shap_analysis
python -m src.fairness
python -m src.context_builder        # SHAP → JSON 컨텍스트
python -m src.llm_explainer --provider gemini   # 또는 anthropic
python -m src.eval_explanation       # 룰 기반 + G-Eval

# Step 2-A
python -m src.expand_samples
python -m src.llm_explainer --provider gemini --contexts-dir results/contexts_100 --output-dir results/explanations_gemini_100
python -m src.counterfactual_test --provider anthropic --n-samples 30
python -m src.robustness_test --provider anthropic --n-samples 20
python -m src.cross_llm_geval --n-samples 30

# 미팅 자료 재생성
python -m src.gen_report               # midterm_report.docx
python -m src.gen_friendly_report      # midterm_report_friendly.docx
python -m src.gen_slides               # midterm_slides.pptx
python -m src.regen_figures            # 모든 figure 재생성

# Git
git log --oneline -10
git tag -l                             # step1, step2a
git diff step1..step2a --stat          # Step 2-A에서 추가/변경된 파일
```

---

## 7. 코딩 규칙 (절대 준수)

1. **`SEED = 42`** 고정 (numpy, random, torch, sklearn random_state)
2. **모든 결과는 `results/` 또는 `figures/`에 파일로 저장** (재실행 가능 + 미팅 데모 대비)
3. **중요한 의사결정은 멈추고 사용자에게 확인** — 모델 구조 변경, 데이터 분할 변경, 외부 API 호출, 패키지 추가 설치
4. **에러는 추측 말고 작은 테스트로 검증** — 풀데이터 전 1,000행 dry-run
5. **백업되지 않은 데이터/결과 덮어쓰기 금지**
6. **자동 push 정책**: Step 단위 또는 사용자 요청 시 commit + push (CLAUDE 자동, 사용자가 "지금 커밋해줘" 가능)
7. **시크릿 절대 commit 안 함**: `.env`, kaggle.json은 .gitignore. 채팅에 토큰 노출 금지

### Figure 작성 규칙
- 텍스트는 **영어**로 (한글 폰트 fallback 문제로 □□ 발생 이력 있음)
- docx/ppt 본문은 한국어 OK (Malgun Gothic 명시 적용됨)

### Windows 콘솔 인코딩
- 한글 print 시 `PYTHONIOENCODING=utf-8` 필수
- `PYTHONUNBUFFERED=1`로 백그라운드 stdout 즉시 flush

---

## 8. 사용자가 결정한 의사결정 이력

| 시점 | 결정 사항 |
|---|---|
| Day 1 | 전처리 정책 A1/B1/C1/D/E1/F |
| Day 6 | LLM = Gemini 2.5 Flash (`gemini-2.0-flash` free tier limit=0 발견 후 전환) |
| Day 7 | Anthropic API 추가 (Claude Sonnet 4.5), $5 충전 |
| Step 1 | GitHub repo `Tim-Green0/tabnet-xai-rag-credit`, public, 자동 push |
| Step 1 | paper/midterm_report.docx, midterm_report_friendly.docx, midterm_slides.pptx 모두 commit (예외 in .gitignore) |
| Step 2-A | Gemini paid tier 활성화, $10 충전 (free tier RPD 20건/일 제한 때문에) |

---

## 9. 알려진 이슈 / 주의사항

| 이슈 | 해결 |
|---|---|
| `gemini-2.0-flash` free tier limit = 0 | `gemini-2.5-flash`로 전환 (paid 또는 자정 quota 리셋 후) |
| SHAP 0.49 + XGBoost 3.x base_score 파싱 버그 | `src/shap_analysis._XgbNativeExplainer`로 우회 |
| LightGBM 컬럼명에 콤마 거부 | `src/baselines._sanitize_columns` 적용 |
| Windows 콘솔 한글 print 깨짐 | `PYTHONIOENCODING=utf-8` 필수 |
| matplotlib `set_theme()`이 한글 폰트 override | figure 텍스트는 영어로 통일 |
| TabNet `feature_importances_`가 load_model 후 None | csv에서 직접 로딩 (`tabnet_attention_importance.csv`) |
| pickle된 preprocessor의 module 경로 mismatch | `train_scaled.parquet` 컬럼명에서 직접 가져옴 |

---

## 10. 다음 세션 첫 메시지 권장 형식

새 세션 시작 시 사용자가 입력하면 좋은 첫 메시지 예:

> "CLAUDE.md 읽고 step2a까지의 진행 상황 확인해줘. 그 다음 Step 3-B (보조 테이블 활용) 시작."

또는

> "이어서 Step 3-C 논문 초안 작성 시작해줘."

또는

> "현재 어디까지 진행됐는지 요약해주고, 다음 후보들 다시 보여줘."

이 파일(§3 진행 흐름)이 최신이면 새 세션이 컨텍스트를 즉시 회복할 수 있다.
