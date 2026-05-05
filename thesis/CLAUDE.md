# CLAUDE.md — 석사 논문 프로젝트 지침

> 이 파일은 본 프로젝트에서 Claude가 작업할 때 따라야 할 핵심 컨텍스트.
> **새 세션을 시작하면 이 파일을 가장 먼저 읽고, 진행 흐름(§3)을 따라 현재 위치를 파악할 것.**

---

## 0. 새 세션 진입 시 회복 절차

```bash
cd D:\paper
git log --oneline -10              # 최근 작업 확인
git tag -l                         # 마일스톤 태그 (step1 → step4)
ls results/day*_summary.md         # Day별 보고서 (Day 1~12)
ls results/step*_summary.md        # Step 보고서
cat thesis/CLAUDE.md                # ← 본 파일
```

**현재 위치**: Step 4 완료 (tag `step4`, 2026-05-06).
- Step 1 (`step1`) — 미팅 프로토타입
- Step 2-A (`step2a`) — 평가 신뢰성
- Step 3-B (`step3b`) — 보조 테이블
- Step 3-C-1 (`step3c1` = `step3`) — TabNet 어텐션 × SHAP 융합
- Step 3-C-2 (`step3c2`) — NLI 평가
- **Step 3-C-2-f (`step4`) — Cross-Judge G-Eval ★ 현재 마일스톤**

**미팅 데드라인**: 2026-05-10 (D-4 from 2026-05-06).
**미팅 자료**: `paper/midterm_slides.pptx` (20 슬라이드), `paper/midterm_report.docx` (12 섹션), `paper/midterm_report_friendly.docx` (15 섹션) 모두 step4 결과 통합 완료.

**다음 후보**: 미팅 후 Step 5 — §4 참조. 1순위는 **인간평가 (Plausibility)** 또는 **Fairness-aware 학습**.

---

## 1. 프로젝트 개요

**논문 제목**: 정형 데이터 특화 딥러닝(TabNet)과 거대언어모델(LLM) 기반 XAI-RAG를 활용한 설명 가능한 신용 평가 및 사용자 맞춤형 리포트 생성

**전공**: DS·AI 석사 / **학번**: A70067 / **이름**: 오현택 / **지도교수**: 박운상
**계획서**: 2026-01-23 / **중간 미팅 데드라인**: **2026-05-10**

### 핵심 아이디어 (Step 3-C-1 이후 갱신)
> "SHAP과 TabNet 어텐션의 결합 결과를 LLM의 retrieved evidence로 재정의하면, LLM 종속성 없이 환각이 0%인 자연어 설명 리포트를 생성할 수 있다."

### 4단계 파이프라인 (Step 3-C-1 이후)
```
[정형 데이터] → [XGBoost 예측 + SHAP local] + [TabNet 어텐션]
              → [Agreement-aware JSON 컨텍스트 (민감변수 마스킹)]
              → [LLM 자연어 설명 (Gemini + Claude)]
              → [정량 평가: Rules + G-Eval(Cross-Judge) + NLI]
```

---

## 2. 환경 정보 (실측 — 변경되면 갱신)

| 항목 | 값 |
|---|---|
| OS | Windows 10 Pro 19045 |
| Python | 3.10.11, venv: `D:\paper\.venv\` |
| GPU | GTX 1660 Ti, 6 GB VRAM, CUDA 12.1 |
| 주요 패키지 | torch 2.5.1+cu121, pytorch-tabnet 4.1.0, xgboost 3.2.0, lightgbm 4.6.0, shap 0.49.1, optuna 4.8.0, transformers 5.7.0, sentence-transformers, anthropic, google-genai |
| LLM | Gemini 2.5 Flash (paid, $10 충전), Claude Sonnet 4.5 (paid) |
| NLI 모델 | MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (다국어, 한국어 학습 데이터 100M+ 포함) |
| API 키 | `D:\paper\.env`: `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` (gitignored) |

### 데이터
- **메인**: Kaggle Home Credit Default Risk `application_train.csv` (307,511 × 122)
- **보조** (Step 3-B 추가): `bureau.csv` (1.7M), `bureau_balance.csv` (27.3M), `previous_application.csv` (1.7M)
- **위치**: `data/home_credit/` (gitignored, 사용자 다운로드)
- **타깃**: `TARGET` 8.07% positive (불균형)
- **잔여 보조 테이블**: POS_CASH_balance, credit_card_balance, installments_payments + bureau_balance 추가 활용 — Step 5 후보

### 처리된 데이터
- **Step 1 (main only, 214 features)**: `data/processed/{train,val,test}_{scaled,unscaled}.parquet`
- **Step 3-B (main + aux, 1161 features)**: `data/processed/{train,val,test}_{scaled,unscaled}_aux.parquet`
- 60/20/20 stratified split (SEED=42)
- 전처리기: `results/preprocessor.pkl` (Step 1), `results/preprocessor_aux.pkl` (Step 3-B)

### Worktree 환경 (Claude Code 작업 시)
- worktree에서 작업 시 `data/`, `.venv/`는 main checkout(D:\paper)에 있음 → junction 필요:
  ```bash
  cmd //c "mklink /J D:\paper\.claude\worktrees\<worktree>\data D:\paper\data"
  cmd //c "mklink /J D:\paper\.claude\worktrees\<worktree>\.venv D:\paper\.venv"
  ```
- `results/baseline_models/` 는 .gitignored라 worktree에 없을 수 있음. 필요 시 `cp D:/paper/results/baseline_models/* results/baseline_models/`

---

## 3. 진행 흐름 — 단계별 정리 ★

본 프로젝트는 **Step 0(환경) → Step 1(미팅 프로토타입) → Step 2-A(평가 강화) → Step 3-B(성능) → Step 3-C-1(TabNet 통합) → Step 3-C-2(NLI) → Step 3-C-2-f(Cross-Judge)** 식으로 진행. 각 Step 끝에 git tag로 마킹.

### Step 0 — 환경·데이터 셋업

| 한 일 | 산출물 |
|---|---|
| venv + 패키지 설치 | `.venv/`, `requirements.txt` |
| Kaggle 데이터 가이드 | `thesis/kaggle_data_setup.md` |
| 환경 점검 보고서 | `results/environment.md` |

### Step 1 — 미팅 프로토타입 (8일, Day 1~8) ★ tag `step1`

#### Day 1 — EDA + 전처리 ([day1](results/day1_summary.md))
- 사용자 결정: A1 (결측 flag) / B1 (one-hot ≤ 8 + target encoding for OCC/ORG) / C1 (class_weight balanced) / D (60/20/20) / E1 (EXT_SOURCE median+flag) / F (보호속성 학습 포함)
- 핵심 발견: TARGET 8.07% 불균형, EXT_SOURCE_2/3 핵심 신호, DAYS_EMPLOYED sentinel 365243 (18%)

#### Day 2 — 베이스라인 ([day2](results/day2_summary.md))
- XGBoost test AUROC **0.7605** (1등), LightGBM 0.7549, Logistic 0.7547

#### Day 3 — TabNet ([day3](results/day3_summary.md))
- TabNet-tuned **0.7543** (XGBoost보다 0.006 부족)
- 어텐션 Top 6에 CODE_GENDER_M → Day 5 공정성 동기

#### Day 4 — 5-fold CV + SHAP ([day4](results/day4_summary.md))
- XGBoost CV **0.7587 ± 0.0008** (5/5 fold 1등)
- ★ Attention vs SHAP: ρ=0.117 (전체), Top-50 ρ=−0.195, Top-20 Jaccard 0.29 — "부분 일관 + 부분 상보"
- 버그 우회: `_XgbNativeExplainer` (SHAP 0.49 + XGBoost 3.x 호환)

#### Day 5 — 공정성 ([day5](results/day5_summary.md))
- 베이스라인: 8/8 케이스 4/5 rule 위반 (DI < 0.8)
- GENDER ablation 효과적, AGE는 proxy variable

#### Day 6 — XAI-RAG + LLM ([day6](results/day6_summary.md))
- `results/contexts/` (10) → `results/explanations{,_anthropic}/`
- 발견: gemini-2.0-flash limit=0 → gemini-2.5-flash 전환

#### Day 7 — 정량 평가 ([day7](results/day7_summary.md))
- ★ Halluc Rate **0.000** (10 샘플, 양 LLM)

#### Day 8 — Demo + Counterfactual ([day8](results/day8_summary.md))
- Demo idx 59291 (TP) end-to-end
- ★★ **Counterfactual baseline**: Claude **45.5%** (no-SHAP) vs **0%** (XAI-RAG) — 미팅 결정타

### Step 2-A — 평가 신뢰성 강화 ★ tag `step2a` ([step2a](results/step2a_summary.md))

| Phase | 결과 |
|---|---|
| 1. 100명 표본 확장 | `results/contexts_100/`, `shap_local_examples_100.json` |
| 2-3. 양 LLM × 100 호출 | `explanations_{gemini,anthropic}_100/` |
| 4. 룰 평가 100건 | **Halluc 0/100** (양 LLM) ★ |
| 5. Cross-LLM G-Eval (n=30) | Claude→Gemini factual 4.87, Gemini→Claude 4.6 |
| 6. Counterfactual 정량화 (n=30) | cosine 0.91~0.92, ROUGE-L 0.75 |
| 7. Robustness (n=20, 3 변형) | cosine 0.91~0.95 |

### Step 3-B — 성능 확장 (보조 테이블) ★ tag `step3b` ([day9](results/day9_summary.md))

> 보조 테이블 2개(bureau, previous_application) → AUROC +2.22%

#### 데이터 + Feature engineering
- `src/aux_data.py` — dtype downcast (-89%, 950MB → 388MB)
- `src/aux_features.py` — SK_ID_CURR 단위 집계 (756 features)
  - bureau (with bb merge): 346 (전체/Active/Closed 분리)
  - previous_application: 410 (전체/Approved/Refused 분리)
- `src/preprocess_with_aux.py` — main + aux merge → **1161 features**

#### CV 결과 (5-fold, test, mean ± std)
| | Baseline | Aux | Δ |
|---|---|---|---|
| AUROC | 0.7587 ± 0.0008 | **0.7755 ± 0.0011** | +0.0168 (+2.22%) |
| AUPRC | 0.2445 | **0.2646** | +8.21% |
| KS | 0.3846 | **0.4146** | +7.81% |

#### SHAP top 20 변화 (의외)
- 신규 진입 5개 모두 **PREV_*** (자체 이력) — 최강: `PREV_NAME_CONTRACT_STATUS_Refused_mean` (이전 거절 비율)
- ★ Bureau (외부 신용기관)은 진입 못함 → main의 EXT_SOURCE_1/2/3에 응축됐을 가능성 (future work에서 ablation)

### Step 3-C-1 — TabNet 어텐션 × SHAP 융합 ★ tag `step3c1` (= `step3`) ([day10](results/day10_summary.md))

> Step 1의 1순위 약점 해결: TabNet이 비교 모델 → 메커니즘 핵심으로 격상

#### Agreement-aware 컨텍스트 (3 그룹)
- `agreed_drivers`: SHAP top-10 ∩ Attention top-5 — 두 모델 동의 강한 신호
- `shap_only_drivers`: SHAP만 (부호 + 기여도 보존)
- `attention_only_drivers`: TabNet만 (sparse, 부호 없음)

#### Agreement 통계 (n=100)
- 평균 agreed=2.12, shap_only=6.98, attention_only=2.06
- n_agreed 분포: 0개=1, 1개=8, 2개=69, 3개=22, **4개+=0** — 부분 상보 instance-level 입증

#### 결과 (n=30 each, Claude judge)
- ★ Halluc 0/30 (양 LLM × 양 mode 모두) — 환각 차단 견고
- ★ G-Eval Completeness +0.67 (Anthropic) / +0.80 (Gemini) — 큰 향상
- Factual 4.77~4.97 ≈ 유지, Sensitive 5.0/5.0 만점
- 룰 sign_match -0.18~-0.22 (룰 키워드 한계) → Step 3-C-2에서 NLI로 입증

### Step 3-C-2 — NLI 평가 ★ tag `step3c2` ([day11](results/day11_summary.md))

> 룰 sign_match 하락이 키워드 한계임을 의미적 측정으로 입증

#### 알고리즘
- Premise = 컨텍스트 facts 자연어 단락
- Hypothesis = LLM 설명 문장 (advice/disclaimer 제외)
- mDeBERTa-v3-xnli (다국어 NLI) → entailment/neutral/contradiction 확률
- 인스턴스별 entailment_rate 평균 → faithfulness score

#### 결과 (n=30 each)
| Provider | entailment Δ | contradiction Δ |
|---|---|---|
| Anthropic | +0.212 ★ | -0.175 ★ |
| Gemini | +0.115 ★ | -0.140 ★ |

★ **3-tier 평가 체계 완성**: Rules + G-Eval(Cross-LLM) + NLI

### Step 3-C-2-f — Cross-Judge G-Eval ★ tag `step4` ([day12](results/day12_summary.md))

> 같은 4 그룹을 Gemini judge로 재평가 → cross-judge 일관성 정량화

#### 결과 (Δ = fusion - shaponly, n=30 each)
| Target | Metric | Claude judge | Gemini judge |
|---|---|---|---|
| Anthropic | Completeness | +0.67 ★ | +0.90 ★ |
| Anthropic | Factual | +0.03 | +0.13 |
| **Gemini** | **Factual** | **-0.13** | **+0.47** ★★ |
| Gemini | Completeness | +0.80 ★ | +1.10 ★ |
| Both | Sensitive | 5.0/5.0 | 5.0/5.0 |

#### 핵심 발견
1. Completeness 양 judge 일관 큰 향상 — fusion 효과 judge 종속 아님
2. **Gemini target Factual cross-judge 가치 입증** — Claude -0.13 vs Gemini +0.47, 차이 0.60 — 단일 judge였으면 잘못된 결론
3. Sensitive 5.0/5.0 양 judge 만점 — 마스킹 정책 LLM 무관 견고
4. Self-bias 약함 — Gemini judge가 self/cross 모두 더 큰 Δ 부여

#### Gemini API 503 처리
- retry 로직 30s/60s/120s/240s 백오프 (4회) 추가 — `eval_fusion.py`
- 첫 시도 503 누적 시 30분~1시간 대기 후 재시도 (timing-dependent)

---

## 4. 다음 단계 후보 (Step 5 ~) — 미팅 후

석사 논문 본 심사 약점 5가지 중 1번만 해소됨 (Step 3-C-1로). 나머지 우선순위:

### 🥇 1순위 — 본 논문 심사 핵심 약점

| 번호 | 작업 | 기간 | 임팩트 |
|---|---|---|---|
| 1.1 | **인간평가 (Plausibility)** | 1.5~2주 | 약점 #2 완전 해소. IRB 간소판 + 5점 척도 + Cohen's κ |
| 1.2 | **Fairness-aware 학습** | 3~4일 | 약점 #4 해소. Reweighing/Adversarial Debiasing → 4/5 rule 통과 시도 |
| 1.3 | **Generic RAG baseline** | 3~4일 | 약점 #3 해소. Counterfactual baseline 정당성 보강 ("trivial" 반박 대비) |

### 🥈 2순위 — 논문 강화

- **UCI German Credit** (3~4일) — 데이터 다양성, 일반화 입증
- **3-way ablation** (3~4일) — SHAP-only / Attention-only / Fusion
- **Bureau ablation** (1~2일) — EXT_SOURCE 응축 가설 검증
- **잔여 보조 테이블 4개** (1주) — AUROC 0.78+ 도전

### 🥉 3순위 — 보조

- TabNet/LightGBM에 aux 효과 일반화 (3~4일)
- FT-Transformer 비교 모델 (1주)
- Fusion 표본 30 → 100 (반나절)
- 한국어 native NLI 검증 (1~2일, torch 환경 정비 후 KLUE-roberta-NLI)

### 4순위 — 장기

- 한국어 도메인 LLM QLoRA
- MLflow 실험 추적

### 권장 진행 순서
```
미팅 → 피드백 반영 → 1.2 Fairness (3~4일, 빠른 win)
                  → 1.3 Generic RAG (3~4일)
                  → 1.1 인간평가 IRB 신청 + 평가 (1.5~2주, 병행)
                  → 2.1 UCI German Credit (3~4일)
                  → 2.2 3-way ablation
                  → 본 논문 초안 작성
```

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
│   ├─ midterm_report.docx        # ✅ commit (12 섹션, 1075KB)
│   ├─ midterm_report_friendly.docx  # ✅ commit (15 섹션, 1167KB)
│   └─ midterm_slides.pptx        # ✅ commit (20 슬라이드, 793KB)
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
│   ├─ gen_report.py / gen_friendly_report.py / gen_slides.py / regen_figures.py
│   │   # Step 2-A
│   ├─ expand_samples.py        # 100명 SHAP+context
│   ├─ text_similarity.py       # multilingual sentence-transformers
│   ├─ counterfactual_test.py / robustness_test.py / cross_llm_geval.py
│   │   # Step 3-B
│   ├─ aux_data.py / aux_eda.py / aux_features.py
│   ├─ preprocess_with_aux.py / cv_eval_aux.py / shap_aux.py
│   │   # Step 3-C-1
│   ├─ tabnet_attention_local.py    # TabNet local M_explain
│   ├─ fusion_context.py            # agreement-aware JSON
│   ├─ llm_explainer_fusion.py      # fusion-aware 프롬프트
│   ├─ eval_fusion.py               # 룰 + G-Eval (judge 옵션, 503 retry)
│   │   # Step 3-C-2 / 3-C-2-f
│   ├─ nli_eval.py                  # mDeBERTa-NLI faithfulness
│   └─ cross_judge_analysis.py      # Claude vs Gemini judge 비교
├─ results/
│   ├─ environment.md / day{1..12}_summary.md / step2a_summary.md
│   ├─ baseline_models/         # *.pkl/zip (gitignored)
│   ├─ contexts/ (10) / contexts_100/ (100) / contexts_fusion_100/ (100)
│   ├─ explanations{,_anthropic}/ / explanations_{gemini,anthropic}_100/
│   ├─ explanations_counterfactual_{gemini,anthropic}/
│   ├─ explanations_robustness_{gemini,anthropic}/
│   ├─ explanations_baseline_noshap_{gemini,anthropic}/
│   ├─ explanations_fusion_{anthropic,gemini}_30/   # Step 3-C-1
│   ├─ tabnet_local_attention_100.json
│   ├─ fusion_eval{,_claude_judge,_gemini_judge}.csv
│   ├─ fusion_vs_shaponly{,_claude_judge,_gemini_judge}.csv
│   ├─ cross_judge_comparison.csv
│   ├─ nli_eval.csv / nli_summary.csv
│   └─ cv_aux_vs_baseline.csv / cv_summary_aux.csv / shap_global_xgboost_aux.csv ...
├─ figures/                     # ✅ 32개 png
│   # 1-23: Step 1, 24-26: Step 2-A, 27-29: Step 3-B
│   # 30: fusion vs shaponly (Step 3-C-1)
│   # 31: NLI vs rules (Step 3-C-2)
│   # 32: cross-judge G-Eval (Step 3-C-2-f)
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
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -W ignore -m <module>

# Step 1 핵심 파이프라인
python -m src.eda
python -m src.preprocess
python -m src.baselines
python -m src.cv_eval
python -m src.tabnet_train --n-trials 10
python -m src.shap_analysis
python -m src.fairness
python -m src.context_builder
python -m src.llm_explainer --provider gemini   # 또는 anthropic
python -m src.eval_explanation

# Step 2-A
python -m src.expand_samples
python -m src.llm_explainer --provider gemini --contexts-dir results/contexts_100 --output-dir results/explanations_gemini_100
python -m src.counterfactual_test --provider anthropic --n-samples 30
python -m src.robustness_test --provider anthropic --n-samples 20
python -m src.cross_llm_geval --n-samples 30

# Step 3-B (보조 테이블)
python -m src.aux_eda
python -m src.aux_features            # full 또는 --dry-run
python -m src.preprocess_with_aux
python -m src.cv_eval_aux --only-xgb
python -m src.shap_aux --n-test 5000

# Step 3-C-1 (TabNet 융합)
python -m src.tabnet_attention_local
python -m src.fusion_context
python -m src.llm_explainer_fusion --provider anthropic --n-samples 30
python -m src.llm_explainer_fusion --provider gemini --n-samples 30
python -m src.eval_fusion --judge anthropic --geval-sleep 4   # 또는 --judge gemini, --skip-geval

# Step 3-C-2 / 3-C-2-f
python -m src.nli_eval                # 또는 --dry-run
python -m src.cross_judge_analysis    # 두 judge 결과 비교

# 미팅 자료 재생성 (현재 step4 결과 통합됨)
python -m src.gen_report               # midterm_report.docx (12 섹션)
python -m src.gen_friendly_report      # midterm_report_friendly.docx (15 섹션)
python -m src.gen_slides               # midterm_slides.pptx (20 슬라이드)

# Git
git log --oneline -10
git tag -l                              # step1 ~ step4
git diff step3..step4 --stat            # Step 3-C-2 → 3-C-2-f 변경
git -C D:/paper merge --ff-only claude/<worktree-branch>   # worktree에서 main 병합
```

---

## 7. 코딩 규칙 (절대 준수)

1. **`SEED = 42`** 고정 (numpy, random, torch, sklearn random_state)
2. **모든 결과는 `results/` 또는 `figures/`에 파일로 저장**
3. **중요한 의사결정은 멈추고 사용자에게 확인** — 모델 구조, 데이터 분할, 외부 API 호출, 패키지 추가
4. **에러는 추측 말고 작은 테스트로 검증** — 풀데이터 전 1,000행 dry-run
5. **백업되지 않은 데이터/결과 덮어쓰기 금지**
6. **Step 단위 commit + tag + push**: Step 끝나면 자동 / 사용자 요청 시
7. **시크릿 절대 commit 안 함**: `.env`, kaggle.json은 .gitignore. 채팅에 토큰 노출 금지

### Figure 작성 규칙
- 텍스트는 **영어**로 (한글 폰트 fallback 문제로 □□ 발생 이력)
- docx/ppt 본문은 한국어 OK (Malgun Gothic 명시 적용)

### Windows 콘솔 인코딩
- 한글 print 시 `PYTHONIOENCODING=utf-8` 필수
- `PYTHONUNBUFFERED=1`로 백그라운드 stdout 즉시 flush

### CSV 덮어쓰기 주의 (Step 3-C-2-f에서 발견)
- `eval_fusion.py`는 매 실행마다 `fusion_eval.csv` / `fusion_vs_shaponly.csv` 덮어씀
- judge별 비교 시 `cp results/fusion_eval.csv results/fusion_eval_<judge>_judge.csv`로 백업 후 다른 judge 실행
- `--skip-geval`로 룰만 다시 돌리면 G-Eval 컬럼이 NaN되니 주의 (G-Eval 결과 포함된 csv 백업 필수)

---

## 8. 사용자가 결정한 의사결정 이력

| 시점 | 결정 사항 |
|---|---|
| Day 1 | 전처리 정책 A1/B1/C1/D/E1/F |
| Day 6 | LLM = Gemini 2.5 Flash (gemini-2.0-flash limit=0 발견 후 전환) |
| Day 7 | Anthropic API 추가 (Claude Sonnet 4.5), $5 충전 |
| Step 1 | GitHub repo `Tim-Green0/tabnet-xai-rag-credit`, public, 자동 push |
| Step 1 | paper/midterm_*.docx/pptx 모두 commit (.gitignore 화이트리스트) |
| Step 2-A | Gemini paid tier $10 충전 (RPD 20건/일 제한 때문) |
| Step 3-B | 보조 테이블 2개만 활용 (bureau + previous_application) — 잔여 4개는 future work |
| Step 3-C-1 | Fusion 설계 옵션 A1 (agreement-aware) 채택 — Day 4 분석과 직접 연결 |
| Step 3-C-1 | Fusion 표본 30, 양 LLM (Claude+Gemini), 기존 TabNet (baseline 데이터) |
| Step 3-C-2 | NLI 모델 Huffon/klue-roberta-base-nli 시도 → torch 2.5/transformers 5.7 보안 충돌 → mDeBERTa-multilingual-NLI 전환 |
| Step 3-C-2-f | Gemini 503 회복 후 cross-judge 진행, Claude judge 데이터를 step3c1 시점으로 복원 |

---

## 9. 알려진 이슈 / 주의사항

| 이슈 | 해결 |
|---|---|
| `gemini-2.0-flash` free tier limit = 0 | `gemini-2.5-flash`로 전환 (paid 또는 자정 quota 리셋 후) |
| Gemini 2.5 Flash 503 UNAVAILABLE 다발 | retry 30s/60s/120s/240s 백오프; 회복 안 되면 30분~1시간 대기 후 재시도 |
| SHAP 0.49 + XGBoost 3.x base_score 파싱 버그 | `src/shap_analysis._XgbNativeExplainer`로 우회 |
| LightGBM 컬럼명에 콤마 거부 | `src/baselines._sanitize_columns` 적용 |
| Windows 콘솔 한글 print 깨짐 | `PYTHONIOENCODING=utf-8` 필수 |
| matplotlib `set_theme()`이 한글 폰트 override | figure 텍스트는 영어로 통일 |
| TabNet `feature_importances_`가 load_model 후 None | csv에서 직접 로딩 (`tabnet_attention_importance.csv`) |
| pickle된 preprocessor module 경로 mismatch | `train_scaled.parquet` 컬럼명에서 직접 가져옴 |
| torch 2.5 + transformers 5.7 NLI 모델 .bin 차단 | safetensors 형식 모델 사용 (CVE-2025-32434) |
| eval_fusion.py csv 덮어쓰기 | judge별 백업 필수 (`fusion_eval_{judge}_judge.csv`) |
| eval_fusion.py G-Eval JSON parse_error 시 NaN | retry 로직 4회로 patch 완료. NaN 있으면 csv 분석 시 dropna 필요 |
| One-hot 컬럼명 prefix가 raw_outside_dataset로 false positive | `eval_fusion.hallucination_rate_fusion`에서 prefix 매칭 추가 |
| Worktree에 data/, .venv/ 없음 | junction 또는 main checkout(D:\paper) 직접 작업 |
| Worktree에 results/baseline_models/ 없음 | `cp D:/paper/results/baseline_models/* results/baseline_models/` 또는 mkdir 후 직접 복사 |

---

## 10. 다음 세션 첫 메시지 권장 형식

새 세션 시작 시 사용자가 입력하면 좋은 첫 메시지 예:

> "CLAUDE.md 읽고 step4까지의 진행 상황 확인해줘. 그 다음 인간평가 IRB 절차부터 시작."

또는

> "이어서 Step 5 — Fairness-aware 학습 (Reweighing) 시작."

또는

> "현재 어디까지 진행됐는지 요약해주고, 다음 후보들 다시 보여줘. 미팅 결과 반영해서 우선순위 조정 필요."

또는 (미팅 후)

> "지도교수가 [X]를 강조했어. step5는 거기서 시작."

이 파일(§3 진행 흐름)이 최신이면 새 세션이 컨텍스트를 즉시 회복할 수 있다.
