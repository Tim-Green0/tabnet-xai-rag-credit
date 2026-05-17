# CLAUDE.md — 석사 논문 프로젝트 지침

> 이 파일은 본 프로젝트에서 Claude가 작업할 때 따라야 할 핵심 컨텍스트.
> **새 세션을 시작하면 이 파일을 가장 먼저 읽고, 진행 흐름(§3)을 따라 현재 위치를 파악할 것.**

---

## 0. 새 세션 진입 시 회복 절차

```bash
cd D:\paper
git log --oneline -10              # 최근 작업 확인
git tag -l                         # 마일스톤 태그 (step1 → step5d)
ls results/day*_summary.md         # Day별 보고서 (Day 1~15)
ls results/step*_summary.md        # Step 보고서
ls thesis/draft/                   # 학위논문 초안 (★ 현재 작업)
cat thesis/CLAUDE.md                # ← 본 파일
```

**현재 위치**: ★ **학위논문 초안 작성 중** (2026-05-11, branch `claude/thesis-draft`).
- Step 1~5-D 모든 실험 완료 (5가지 약점 해소)
- Step 5-E (Customer-friendly fusion) 시도 → **revert** (negative result, 핵심 메시지 약화 우려)
- **학위논문 5장 구조 초안 완성**: `thesis/draft/thesis_draft.docx` (~815 KB, 한국어 본문 ~36,000 단어)
- 교수님 1차 미팅 완료 → 환각률 0% 정의 명확화 + AUROC 0.76 실무 수준 평가 절 추가
- 다음: 교수님 2차 검토 대기

**완료된 Step 마일스톤** (git tag):
- Step 1 (`step1`) — 미팅 프로토타입
- Step 2-A (`step2a`) — 평가 신뢰성
- Step 3-B (`step3b`) — 보조 테이블 (AUROC +2.22%)
- Step 3-C-1 (`step3c1` = `step3`) — TabNet 어텐션 × SHAP 융합
- Step 3-C-2 (`step3c2`) — NLI 평가
- Step 3-C-2-f (`step4`) — Cross-Judge G-Eval
- Step 5-A (`step5a`) — Fairness-aware Learning (Reweighing 4/4 통과)
- Step 5-B (`step5b`) — Generic RAG Baseline (4-way 비교, 약점 #3 해소)
- Step 5-C (`step5c`) — Pilot Human-Proxy Evaluation (3 personas, trade-off 발견)
- Step 5-D (`step5d`) — UCI German Credit 일반화 (약점 #5 해소, NLI fusion 0.711 1위)
- ~~Step 5-E~~ — revert됨 (Customer-friendly fusion, NLI/value_match 손실 vs G-Eval 일부 개선 trade-off — 본 contribution 아님)

**미팅 데드라인**: 2026-05-10 → 연기됨 → 2026-05-11 1차 미팅 완료, 2차 미정.
**미팅 자료**: `paper/midterm_slides.pptx` (23 슬라이드), `paper/midterm_report.docx` (15 섹션), `paper/midterm_report_friendly.docx` (18 섹션) — step5c 통합 완료, step5d/학위논문 초안은 별도.

**5가지 약점 진척** (★ 모두 해소 / 부분 해소):
| 약점 | 상태 | 해소 step |
|---|---|---|
| #1 TabNet 역할 | ✅ 해소 | Step 3-C-1 (어텐션 × SHAP 융합) |
| #2 LLM 평가 객관성 | ◐ 부분 해소 | Step 3-C-2 (NLI) + 5-C (pilot persona) |
| #3 Counterfactual 정당성 | ✅ 해소 | Step 5-B (Generic RAG 비교) |
| #4 Fairness mitigation | ✅ 해소 | Step 5-A (Reweighing 4/4 통과) |
| #5 데이터 다양성 | ✅ 해소 | **Step 5-D (UCI German Credit 일반화)** |

**다음 후보**: §4 참조. **교수님 2차 검토 대기 중**. 검토 후보강 방향:
1. 교수님 추가 피드백 반영
2. (선택) 3-way ablation (Attention 단독 검증)
3. (선택) 잔여 보조 테이블 4개 활용 (AUROC 0.81+ 가능)
4. (선택) 정식 IRB 인간평가 (장기, 심사 직전)

---

## 1. 프로젝트 개요

**논문 제목**: 정형 데이터 특화 딥러닝(TabNet)과 거대언어모델(LLM) 기반 XAI-RAG를 활용한 설명 가능한 신용 평가 및 사용자 맞춤형 리포트 생성

**전공**: DS·AI 석사 / **학번**: A70067 / **이름**: 오현택 / **지도교수**: 박운상

### 핵심 메시지 (학위논문 초안 기준)
- Step 1: "환각 0% vs 45.5%" — 강력하지만 trivial 반박 가능
- Step 5-B: "환각 차단은 hard constraints로도 가능, 차별성은 fact-grounded 정확성"
- Step 5-C: "사실성 1위 + 친근함은 trade-off, 응용에 따라 mode 선택"
- Step 5-D: "데이터 복잡도가 fusion 우월성에 영향 — 복잡 도메인 fusion ↑, 단순 도메인 generic_rag ↑"
- ~~Step 5-E~~: revert됨 (사실성 vs 친화성 trade-off 입증했으나 본 contribution 아님)
- **학위논문 메시지** ★ 현재:
  > "본 연구는 *예측 정확도 갱신* 이 아니라 *Fusion XAI + LLM RAG + 공정성 보정의 통합 시스템* 을 구축.
  > Baseline 예측 (XGBoost AUROC 0.76, mainstream)을 honest 하게 보고하고,
  > Fusion의 차별성을 **NLI Entailment 0.625** + **Value Match 0.88** 에서 정량 입증."

### 환각률 0%의 정확한 의미 (★ 교수 미팅 피드백 후 명확화, §4.3.2)
본 시스템은 3 가지 정확도 차원을 *분리하여* 측정하고 보고한다:

| 차원 | 측정 대상 | 도구 | 본 연구 결과 |
|---|---|---|---|
| A. 부도 예측 정확도 | XGBoost가 실제 부도/정상 맞춤 | AUROC, AUPRC, KS | 0.7587 (Home), 0.7714 (German) |
| **B. LLM 변수명 환각률** | LLM이 없는 변수명 생성 | 룰 기반 영문 토큰 매칭 | **0%** (모든 4-mode) |
| C. LLM 의미적 충실성 | 설명이 입력과 의미 일치 | NLI Entailment | fusion 0.625, no_shap 0.30 |

"환각률 0%"는 차원 B만 보장. 차원 A는 모델 성능 (§4.1), 차원 C는 의미 충실성 (§4.3.3) 별도 측정.

### 4단계 파이프라인
```
[정형 데이터] → [XGBoost 예측 + SHAP local] + [TabNet 어텐션]
              → [Agreement-aware JSON 컨텍스트 (민감변수 마스킹)]
              → [LLM 자연어 설명 (Gemini + Claude)]
              → [정량 평가: Rules + G-Eval(Cross-Judge) + NLI + Pilot Human-Proxy]
```

---

## 2. 환경 정보 (실측)

| 항목 | 값 |
|---|---|
| OS | Windows 10 Pro 19045 |
| Python | 3.10.11, venv: `D:\paper\.venv\` |
| GPU | GTX 1660 Ti, 6 GB VRAM, CUDA 12.1 |
| 주요 패키지 | torch 2.5.1+cu121, pytorch-tabnet 4.1.0, xgboost 3.2.0, lightgbm 4.6.0, shap 0.49.1, optuna 4.8.0, transformers 5.7.0, sentence-transformers, anthropic, google-genai, **fairlearn 0.13.0** |
| LLM | Gemini 2.5 Flash (paid), Claude Sonnet 4.5 (paid) |
| NLI 모델 | MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 |
| API 키 | `D:\paper\.env`: `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` (gitignored) |

### 데이터
- **메인**: Kaggle Home Credit Default Risk `application_train.csv` (307,511 × 122)
- **보조** (Step 3-B 추가): bureau, bureau_balance, previous_application
- **잔여 보조 테이블**: POS_CASH_balance, credit_card_balance, installments_payments + bureau_balance 추가 활용 — future work

### 처리된 데이터
- Step 1 (214 features): `data/processed/{train,val,test}_{scaled,unscaled}.parquet`
- Step 3-B (1161 features): `data/processed/{train,val,test}_{scaled,unscaled}_aux.parquet`
- 60/20/20 stratified split (SEED=42)

### Worktree 환경
- worktree에서 작업 시 `data/`, `.venv/`는 main에 있음 → junction:
  ```bash
  cmd //c "mklink /J D:\paper\.claude\worktrees\<wt>\data D:\paper\data"
  cmd //c "mklink /J D:\paper\.claude\worktrees\<wt>\.venv D:\paper\.venv"
  ```
- `results/baseline_models/`은 .gitignored이라 worktree에 없을 수 있음. 필요 시 main에서 복사

---

## 3. 진행 흐름 — 단계별 정리 ★

### Step 0~2-A (요약, 자세히는 day1~8 + step2a_summary 참조)

#### Step 1 (tag `step1`) — 미팅 프로토타입 8일
- XGBoost CV 0.7587 ± 0.0008 (1등), TabNet 0.7543
- ★ Attention vs SHAP: ρ=0.117 (전체), Top-50 ρ=−0.195 — "부분 일관 + 부분 상보"
- ★ 공정성 8/8 4/5 rule 위반 진단
- ★ XAI-RAG (10 샘플): Halluc 0/10, ★ **Counterfactual baseline: Claude 0% vs 45.5%** (no-SHAP)

#### Step 2-A (tag `step2a`) — 평가 신뢰성 강화
- ★ 100명 Halluc 0/100 (양 LLM)
- Cross-LLM G-Eval (n=30): factual 4.6~4.87, sensitive 5.0 만점
- Counterfactual + Robustness 정량화 (cosine 0.91~0.95)

### Step 3-B (tag `step3b`) — 보조 테이블 (`day9_summary`)
- bureau + previous_application 추가 → **AUROC 0.7587 → 0.7755 (+2.22%)**
- AUPRC +8.21%, KS +7.81%
- ★ Bureau는 SHAP top 20 진입 못함 → main의 EXT_SOURCE에 응축됐을 가능성 (future work)

### Step 3-C-1 (tag `step3c1` = `step3`) — TabNet 어텐션 × SHAP 융합 (`day10_summary`)
> 약점 #1 (TabNet 역할) 해소

- Agreement-aware 컨텍스트: agreed / shap_only / attention_only 3 그룹
- Agreement 통계 (n=100): mean(agreed)=2.12, n_agreed 분포 0~3 (4+ 동의 0%)
- ★ Halluc 0/30 (양 LLM × 양 mode), ★ G-Eval Completeness +0.67 (A) / +0.80 (G)
- 룰 sign_match -0.18~-0.22 (룰 키워드 한계 추정 → Step 3-C-2에서 입증)

### Step 3-C-2 (tag `step3c2`) — NLI 평가 (`day11_summary`)
> 약점 #2 부분 해소

- mDeBERTa-multilingual-NLI (다국어, 한국어 학습 데이터 포함)
- ★ Fusion entailment +0.21 (A) / +0.12 (G), contradiction -0.18 (A) / -0.14 (G)
- 룰 sign_match 하락이 키워드 한계임을 의미적 측정으로 입증
- ★ 3-tier 평가 체계 완성: Rules + G-Eval + NLI

### Step 3-C-2-f (tag `step4`) — Cross-Judge G-Eval (`day12_summary`)
- Gemini judge로 양방향 cross-validation
- ★ Completeness 양 judge 일관 (Claude +0.67~+0.80, Gemini +0.90~+1.10)
- ★ Gemini target Factual: Claude judge -0.13 vs Gemini judge +0.47, **차이 0.60** — Cross-judge 가치 직접 입증
- Sensitive 5.0/5.0 양 judge 만점

### Step 5-A (tag `step5a`) — Fairness-aware Learning (`day13_summary`)
> 약점 #4 해소

- Reweighing (Kamiran-Calders) + Fairlearn ExpGrad (DP/EO) 비교
- ★ **Reweighing 4/4 통과** (GENDER + AGE × baseline + aux 데이터)
  - baseline: AUROC -0.003, DI 0.62→0.90 (G), 0.56→0.90 (A)
  - **aux: AUROC +0.003 (오히려 향상)**, DI 0.64→0.87 (G), 0.57→0.83 (A)
- Day 5의 "AGE proxy" 결론 갱신 — 정식 mitigation은 proxy variable에서도 효과적
- Fairlearn DP는 AUROC 큰 손실, EO는 DP 악화 안티 패턴

### Step 5-B (tag `step5b`) — Generic RAG Baseline (`day14_summary`)
> 약점 #3 정량 해소

- 4-mode 비교: no_shap / generic_rag / shaponly / fusion
- Generic RAG = raw features + 7 도메인 지식 chunks + 동일 hard constraints (SHAP X)
- ★ Halluc 0/30 — Generic RAG도 환각 차단 가능 (SHAP은 충분조건이지만 필요조건 X)
- ★ NLI Entailment 4-mode 일관 4단계 차이: no_shap (0.27~0.43) < generic_rag (0.36~0.37) < shaponly (0.41~0.51) < **fusion (0.62~0.62)**
- ★ val_match_rate: 0.59 → 0.73 → 0.85 → 0.90 (4단계)
- 흥미로운 발견: Generic RAG가 G-Eval Completeness에서 SHAP-only보다 약간 더 높음 (chunks 효과)
- 본 연구 차별성 재정의: **환각 차단은 hard constraints로도 가능, 차별성은 fact-grounded 정확성 (의미적 충실성 + 값 정확 인용)**

### Step 5-C (tag `step5c`) — Pilot Human-Proxy Evaluation (`day15_summary`)
> 약점 #2 부분 해소 (정식 IRB는 future work)

- 3 personas LLM-proxy: Credit Expert / Customer / Regulator
- 3 metrics × 5점 척도: trustworthiness / clarity / actionability
- 4 modes × 2 LLM target × 15 instances × 3 personas = 360 평가 (실제 276)
- Judge: Claude (안정적)

#### ★ 충격적 발견 — Trade-off 명확
| 평가 차원 | 1위 |
|---|---|
| 사실성 (NLI) | **fusion** ★ |
| 충실성 (G-Eval) | **fusion** ★ |
| 사람 친화성 (Persona trust) | **generic_rag** ★ (4.91 vs fusion 4.31) |
| Customer clarity ⚠️ | **generic_rag 4.93** (fusion 2.67, shaponly 2.80) |

- ★ 본 연구의 fusion 메커니즘이 **모든 차원에서 1위 아님** — honest reporting
- ★ Customer perspective에서 SHAP-RAG/Fusion clarity 약점 (전문 용어, agreement 라벨)
- ★ 응용 시나리오에 따른 **mode 선택 trade-off 정량 입증** — fusion = audit/regulation, generic_rag = customer-facing

#### Case study (정성)
- 6 instances (reject 3 + accept 3) × 4 modes × Anthropic target
- `results/case_study.md`에 정성 비교 노트
- 정량(persona)과 정성(case study) 결과 일관

### Step 5-D (tag `step5d`) — UCI German Credit 일반화 (`step5d_summary`)
> 약점 #5 해소

- **데이터**: sklearn `fetch_openml('credit-g')`, 1000 × 20 features (+target), 60/20/20 split
- **포팅**: 메인 메커니즘 (XGBoost + SHAP + TabNet attention + LLM 4-mode RAG) 그대로 이식
- 보호 속성: age, personal_status(sex), foreign_worker
- 4 mode × 2 LLM × 30 instances = 240 explanations (Anth+Gem) + 240 G-Eval (Anth judge)

#### ★ 일반화 입증 결과
| 지표 | Home Credit | German Credit | 결과 |
|---|---|---|---|
| AUROC (XGBoost test, 5-fold) | 0.759 | 0.771 | 일관 |
| SHAP × Attention ρ (full) | 0.117 | 0.114 | **거의 동일** ★ |
| NLI Entailment fusion | 0.625 | **0.711** | German에서 더 강함 ★ |
| Halluc rate | 0% | 0% | hard constraints 견고 |
| Sensitive Leak | 5.0 | 5.0 | 마스킹 일관 |

#### ★ 새 발견 — G-Eval Completeness 데이터셋별 차이
- Home: **fusion 4.82** ★ 1위 (vs generic_rag 4.48)
- German: **generic_rag 4.58** ★ 1위 (vs fusion 3.77) ⚠ 역전!
- → 복잡한 도메인(Home, 214 features)은 fusion 우월, **단순한 도메인(German, 63 features)은 generic_rag 우월**

#### ★ G-Eval Factual Accuracy — fusion 일관 약점
- no_shap, generic_rag: 5.0 (만점)
- shaponly, fusion: 3~4 (Gemini fusion 2.97 매우 낮음)
- → fusion이 agreement 라벨 / SHAP 부호를 over-explain하는 경향 (Step 5-C customer clarity 약점과 일관)

#### NLI Entailment 4-mode 단조증가 (양 데이터셋 동일 패턴)
| Mode | Home | German |
|---|---|---|
| no_shap | 0.350 | 0.393 |
| generic_rag | 0.367 | 0.410 |
| shaponly | 0.461 | 0.628 |
| **fusion** | **0.625** | **0.711** ★ |

#### LLM 비용 (실측, $3.7)
| 호출 | 수 | 비용 |
|---|---|---|
| Anthropic explanation | 120 | $1.82 |
| Gemini explanation | 120 | $0.02 |
| Anthropic G-Eval (judge) | 240 | ~$1.87 |

#### 메시지 정교화 (Step 5-C+5-D 통합)
> **fusion 메커니즘은 NLI 사실성에서 일관 1위**지만, **데이터 복잡도에 따라 G-Eval 충실성 우월성이 달라지고**, customer clarity·LLM judge factual에서 over-explain 약점.
> → **응용 시나리오 + 데이터 복잡도** 두 차원에 따른 mode 선택 trade-off.

### ~~Step 5-E~~ (revert됨, 2026-05-11) — Customer-friendly fusion

- 시도: fusion 컨텍스트 그대로 + prompt만 친근하게 변경 (SHAP 부호 자연어화, agreement 라벨 직관 표현, 정성 표현)
- 양 데이터셋 × 2 LLM × 30 instances = 120 explanations + 120 G-Eval (Anthropic judge)
- 결과:
  - 사실성 (NLI/Value Match): 양 데이터셋 일관 **손실** (NLI -0.12~0.17, Val Match -0.25~0.27)
  - G-Eval: 데이터셋별 정반대 (Home 변화 미미, German factual +0.47/completeness +0.48)
- **결론**: trade-off 정량 입증했으나 본 contribution의 핵심 메시지 (NLI 1위) 약화 우려 → **revert** (commit `916a3ac` revert)
- LLM 비용: explanation $1.45 + G-Eval ~$1.30 = ~$2.75 (학습 비용)

### 학위논문 초안 작성 (2026-05-11 진행 중, branch `claude/thesis-draft`)

- **5장 구조** (한국 석사 표준, 손지민 2024 패턴 일부 참조하되 직접 인용 제거)
  - 1장 서론 / 2장 이론적 배경 / 3장 연구 방법론 / 4장 분석 결과 / 5장 결론 및 시사점
- **본문 분량**: 한국어 ~36,000 단어, ~70-80 페이지
- **양식**: 학위논문지침 준수 (Malgun Gothic, 11pt, 줄간격 160%, IEEE [N] 인용)
- **별지**: 3호 겉표지 / 4호 제출문 (지도교수 박운상) / 5호 인준서 + 목차/표차례/그림차례 (Word ToC field 자동 갱신)
- **figures**: 그림 4-1 시스템 파이프라인 (matplotlib 신규) + 4-2~4-6 실험 결과 (6개)
- **참고문헌**: IEEE 양식 24개
- **산출**: `thesis/draft/thesis_draft.docx` (815 KB, 5장 구조)

#### 교수 미팅 1차 (2026-05-11) 피드백 반영
1. **환각률 0% 정의 명확화** (§4.3.2) — A/B/C 3차원 분리:
   - A. 부도 예측 정확도 (XGBoost AUROC, 별도)
   - B. LLM 변수명 환각률 (0%, 본 절)
   - C. LLM 의미적 충실성 (NLI Entailment, §4.3.3)
2. **AUROC 0.76 실무 수준 평가** (§4.1.4) — Lessmann et al. 2015 벤치마크 비교, 실무 도입 보강 5항목

---

## 4. 다음 단계 후보 (학위논문 초안 → 교수 검토 후)

### 🥇 1순위 — 교수님 2차 검토 피드백 반영
- 미정 (검토 후 결정)
- **현재**: thesis/draft/thesis_draft.docx 작성 완료, 교수 검토 대기

### 🥈 2순위 — 3-way ablation (Attention-only 단독 검증)
- **근거**: 현재 `shaponly vs fusion` 비교만 있고 Attention-only 단독 mode 미검증. Fusion 우월성이 attention의 추가적 기여인지 단순 union 효과인지 isolate 안 됨
- 작업: TabNet attention top-k만 → LLM 4-mode 비교 (no_shap / attention_only / shaponly / fusion)
- **3~4일**

### 🥉 3순위 — Step 5-D 보강 (표본 확장)
- 표본 30 → 100 (양 데이터셋)
- Step 5-C persona pilot을 German에도 적용 (trade-off 일반화 입증)
- **3~5일**

### 🎖 4순위 — 보강 작업 (Home Credit 차원)
- **Bureau ablation** (EXT_SOURCE 응축 가설) — 1~2일
- **잔여 보조 테이블 4개** (POS_CASH, credit_card, installments + bureau_balance) — 1주, AUROC 0.81+ 가능
- **TabNet/LightGBM에 aux 효과 일반화** — 3~4일

### 🏅 5순위 — 추가 데이터셋 (확장 일반화)
- **Australian Credit** (UCI, 690 × 14, 복잡도 중간) — 3~4일
- **Lending Club** (Kaggle, ~1.3M × 150) — 5~7일

### ⚠️ 6순위 — 정식 IRB 인간평가 (장기, 심사 직전)
- 약점 #2 완전 해소. IRB 간소판 + 5점 척도 + 다수 평가자 + Cohen's κ
- **1.5~2개월**

### 권장 진행 순서
```
교수님 검토 (2차 미팅) 대기
→ 2차 피드백 반영 (논문 본문 수정)
→ (선택) 보강 실험 (3-way ablation 또는 보조 테이블 활용)
→ 추가 검토
→ 정식 IRB 인간평가 (심사 전)
→ 최종 제출
```

---

## 5. 코드 / 폴더 구조

```
D:\paper\
├─ .venv/                       # gitignored
├─ .env                         # gitignored, GEMINI/ANTHROPIC 키
├─ data/                        # gitignored
├─ paper/
│   ├─ midterm_report.docx        # ✅ 15 섹션, 1.5MB
│   ├─ midterm_report_friendly.docx  # ✅ 18 섹션, 1.6MB
│   └─ midterm_slides.pptx        # ✅ 23 슬라이드, 1.1MB
├─ src/                         # ✅ 모든 코드
│   # Step 1 / 2-A
│   ├─ utils.py / data_loader.py / eda.py / preprocess.py
│   ├─ metrics.py / baselines.py / cv_eval.py / tabnet_train.py
│   ├─ shap_analysis.py (_XgbNativeExplainer 우회)
│   ├─ fairness.py
│   ├─ context_builder.py / llm_explainer.py / eval_explanation.py
│   ├─ compare_llms.py / baseline_no_shap.py / demo.py
│   ├─ gen_report.py / gen_friendly_report.py / gen_slides.py / regen_figures.py
│   ├─ expand_samples.py / text_similarity.py
│   ├─ counterfactual_test.py / robustness_test.py / cross_llm_geval.py
│   # Step 3-B
│   ├─ aux_data.py / aux_eda.py / aux_features.py
│   ├─ preprocess_with_aux.py / cv_eval_aux.py / shap_aux.py
│   # Step 3-C-1
│   ├─ tabnet_attention_local.py / fusion_context.py
│   ├─ llm_explainer_fusion.py / eval_fusion.py
│   # Step 3-C-2 / 3-C-2-f
│   ├─ nli_eval.py / cross_judge_analysis.py
│   # Step 5-A
│   ├─ fairness_mitigation.py
│   # Step 5-B
│   ├─ baseline_generic_rag.py / eval_generic_rag.py
│   # Step 5-C
│   ├─ human_proxy_eval.py / case_study.py
│   # Step 5-D (UCI German Credit)
│   ├─ german_data.py / german_train.py / german_explainer.py / german_eval.py / german_compare.py
├─ data/
│   ├─ home_credit/ ...
│   ├─ german_credit/raw.parquet, processed/{train,val,test}_*.parquet
├─ results/
│   ├─ environment.md / day{1..15}_summary.md / step2a_summary.md
│   ├─ baseline_models/         # *.pkl/zip (gitignored)
│   ├─ contexts/ contexts_100/ contexts_fusion_100/ contexts_generic_rag_30/
│   ├─ explanations/ explanations_anthropic/ explanations_{gemini,anthropic}_100/
│   ├─ explanations_counterfactual_/ explanations_robustness_/
│   ├─ explanations_baseline_noshap_{anthropic,gemini}/
│   ├─ explanations_fusion_{anthropic,gemini}_30/      # Step 3-C-1
│   ├─ explanations_generic_rag_{anthropic,gemini}_30/ # Step 5-B
│   ├─ tabnet_local_attention_100.json
│   ├─ fusion_eval{,_claude_judge,_gemini_judge}.csv
│   ├─ fusion_vs_shaponly{,_claude_judge,_gemini_judge}.csv
│   ├─ cross_judge_comparison.csv
│   ├─ nli_eval.csv / nli_summary.csv
│   ├─ fairness_mitigation_v2.csv               # Step 5-A
│   ├─ generic_rag_eval.csv / generic_rag_summary.csv  # Step 5-B
│   ├─ human_proxy_eval.csv / human_proxy_summary.csv  # Step 5-C
│   ├─ case_study.md                            # Step 5-C
│   ├─ german_eda.json / german_cv_{metrics,summary}.csv      # Step 5-D
│   ├─ german_shap_{global,local}.* / german_tabnet_attention.json
│   ├─ german_attention_vs_shap.json
│   ├─ contexts_german_{no_shap,generic_rag,shaponly,fusion}_30/
│   ├─ explanations_german_{mode}_{anthropic,gemini}_30/
│   ├─ german_eval.csv / german_eval_summary.csv
│   ├─ step5d_comparison.csv / step5d_summary.md
│   └─ ... (cv_aux_*, shap_global_*aux*, etc.)
├─ figures/                     # ✅ 41개 png
│   # 1-23: Step 1, 24-26: Step 2-A, 27-29: Step 3-B
│   # 30: Step 3-C-1 fusion, 31: Step 3-C-2 NLI, 32: Step 3-C-2-f cross-judge
│   # 33-34: Step 5-A fairness mitigation
│   # 35: Step 5-B generic RAG 4-way
│   # 36: Step 5-C human-proxy personas
│   # 37-41: Step 5-D German Credit (EDA / CV / SHAP / 4-way / generalization)
│   # 42: 학위논문 그림 4-1 시스템 파이프라인 (matplotlib 신규)
├─ thesis/
│   ├─ CLAUDE.md (이 파일) / kaggle_data_setup.md / llm_options.md
│   └─ draft/                       # ★ 학위논문 초안 (branch claude/thesis-draft)
│       ├─ outline.md                  # Phase 진행 노트, 결정사항
│       ├─ abstract.md                 # 영문 (~250 words) + 국문 (~600자)
│       ├─ chapter1_introduction.md    # 1장 서론
│       ├─ chapter2_related_work.md    # 2장 이론적 배경
│       ├─ chapter3_data.md            # 3.1 데이터셋 및 전처리
│       ├─ chapter4_methodology.md     # 3.2 시스템 구조 (★ Fusion §4.4)
│       ├─ chapter5_evaluation.md      # 3.3 평가 프레임워크
│       ├─ chapter6_results.md         # 4장 분석 결과 (§4.1.4 AUROC 실무 평가)
│       ├─ chapter7_discussion.md      # 5.1 논의 (5장 build에서 합쳐짐)
│       ├─ chapter8_conclusion.md      # 5.2 결론
│       ├─ references.md               # IEEE 24개 (손지민 제거됨)
│       ├─ appendix.md                 # 부록 A~F
│       ├─ build_thesis_docx.py        # 8장 build (현재 unused, 호환성 유지)
│       ├─ build_thesis_5ch_docx.py    # ★ 5장 build (현재 사용)
│       ├─ make_pipeline_figure.py     # 그림 4-1 생성
│       ├─ thesis_draft.docx           # ★ 최종 docx (815 KB, 5장 구조)
│       └─ refs/                       # 학위논문지침 + 학교 발간 6개 논문 텍스트
```

---

## 6. 자주 쓰는 명령어

```bash
# 가상환경 + 인코딩
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -W ignore -m <module>

# Step 1 핵심 파이프라인
python -m src.eda / preprocess / baselines / cv_eval / tabnet_train
python -m src.shap_analysis / fairness / context_builder
python -m src.llm_explainer --provider gemini   # 또는 anthropic
python -m src.eval_explanation

# Step 2-A
python -m src.expand_samples
python -m src.counterfactual_test --provider anthropic --n-samples 30
python -m src.robustness_test --provider anthropic --n-samples 20
python -m src.cross_llm_geval --n-samples 30

# Step 3-B
python -m src.aux_eda / aux_features / preprocess_with_aux
python -m src.cv_eval_aux --only-xgb
python -m src.shap_aux --n-test 5000

# Step 3-C-1 (TabNet 융합)
python -m src.tabnet_attention_local
python -m src.fusion_context
python -m src.llm_explainer_fusion --provider anthropic --n-samples 30
python -m src.eval_fusion --judge anthropic --geval-sleep 4

# Step 3-C-2 / 3-C-2-f
python -m src.nli_eval                # 또는 --dry-run
python -m src.cross_judge_analysis    # Claude vs Gemini judge 비교

# Step 5-A (Fairness mitigation)
python -m src.fairness_mitigation     # baseline + aux × GENDER + AGE × 4 methods

# Step 5-B (Generic RAG)
python -m src.baseline_generic_rag --provider anthropic --n-samples 30
python -m src.baseline_generic_rag --provider gemini --n-samples 30
python -m src.eval_generic_rag --judge anthropic   # 4-way 비교

# Step 5-C (Pilot human-proxy)
python -m src.human_proxy_eval --n-samples 15 --judge anthropic --sleep 3
python -m src.case_study              # 6 instances 정성 분석

# Step 5-D (UCI German Credit 일반화)
python -m src.german_data --step all   # load + EDA + preprocess
python -m src.german_train --step cv   # 5-fold CV (Logistic+XGB+LGBM+TabNet)
python -m src.german_train --step xgb  # 단발 학습 → SHAP용
python -m src.german_train --step tabnet
python -m src.german_train --step shap        # SHAP global + local 30
python -m src.german_train --step attention   # TabNet attention 30
python -m src.german_train --step consistency # ρ + Top-K overlap
python -m src.german_explainer --mode {no_shap,generic_rag,shaponly,fusion} --provider {anthropic,gemini} --n-samples 30
python -m src.german_eval --skip-geval        # NLI + value_match만 (빠름)
python -m src.german_eval --geval-judge anthropic --geval-sleep 3   # G-Eval 추가 (~60분)
python -m src.german_compare                  # Home vs German + step5d_summary.md

# 미팅 자료 재생성
python -m src.gen_report               # 15 섹션
python -m src.gen_friendly_report      # 18 섹션
python -m src.gen_slides               # 23 슬라이드

# 학위논문 (★ 현재 작업)
# 1) 그림 4-1 시스템 파이프라인 재생성 (matplotlib)
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe thesis/draft/make_pipeline_figure.py
# 2) 5장 구조 docx 빌드 (★ 사용 중)
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -W ignore -c "import sys; sys.path.insert(0, 'thesis/draft'); import build_thesis_5ch_docx; build_thesis_5ch_docx.main()"
# → 출력: thesis/draft/thesis_draft.docx
# Word에서 [F9]로 목차/표차례/그림차례 자동 갱신

# Git
git log --oneline -10
git tag -l                              # step1 ~ step5d
git diff step5c..step5d --stat          # 단계 간 변경
git -C D:/paper merge --ff-only claude/<worktree-branch>
```

---

## 7. 코딩 규칙 (절대 준수)

1. **`SEED = 42`** 고정
2. **모든 결과는 `results/` 또는 `figures/`에 파일로 저장**
3. **중요한 의사결정은 멈추고 사용자 확인** — 모델 구조, 데이터 분할, 외부 API, 패키지 추가
4. **에러는 추측 말고 작은 테스트로 검증** — 풀데이터 전 dry-run
5. **백업되지 않은 데이터/결과 덮어쓰기 금지**
6. **Step 단위 commit + tag + push**: Step 끝나면 자동 / 사용자 요청 시
7. **시크릿 절대 commit 안 함**: `.env`, kaggle.json은 .gitignore

### Figure 작성 규칙
- 텍스트는 **영어**로 (한글 폰트 fallback 문제로 □□ 발생 이력)
- docx/ppt 본문은 한국어 OK (Malgun Gothic 명시)

### Windows 콘솔 인코딩
- 한글 print 시 `PYTHONIOENCODING=utf-8` 필수
- `PYTHONUNBUFFERED=1`로 백그라운드 stdout 즉시 flush

### CSV 덮어쓰기 주의 (Step 3-C-2-f에서 발견)
- `eval_fusion.py`는 매 실행마다 csv 덮어씀
- judge별 비교 시 `cp results/fusion_eval.csv results/fusion_eval_<judge>_judge.csv`로 백업 후 다른 judge 실행
- `--skip-geval`로 룰만 다시 돌리면 G-Eval 컬럼 NaN — 백업 필수

### Worktree 작업 시 주의 (Step 5-C에서 발견)
- main checkout에서 docx/pptx 파일 열려 있으면 git unlink 실패 → merge 차단
- 사용자에게 파일 닫아달라고 요청 후 재시도
- Worktree commit 후 main에 partial 변경 잔여 가능 — `git -C D:/paper checkout -- <files>` 또는 `rm <untracked>` 후 재 merge

---

## 8. 사용자가 결정한 의사결정 이력

| 시점 | 결정 사항 |
|---|---|
| Day 1 | 전처리 정책 A1/B1/C1/D/E1/F |
| Day 6 | LLM = Gemini 2.5 Flash |
| Day 7 | Anthropic API 추가 (Claude Sonnet 4.5) |
| Step 1 | GitHub repo `Tim-Green0/tabnet-xai-rag-credit`, public, 자동 push |
| Step 1 | paper/midterm_*.docx/pptx 모두 commit (.gitignore 화이트리스트) |
| Step 2-A | Gemini paid tier $10 충전 |
| Step 3-B | 보조 테이블 2개만 활용 (bureau + previous_application) |
| Step 3-C-1 | Fusion 설계 옵션 A1 (agreement-aware) — Day 4 분석과 직접 연결 |
| Step 3-C-1 | Fusion 표본 30, 양 LLM, 기존 TabNet (baseline 데이터) |
| Step 3-C-2 | NLI 모델 mDeBERTa-multilingual (KLUE-roberta-NLI는 torch 보안 충돌) |
| Step 3-C-2-f | Gemini 503 회복 후 cross-judge 진행 |
| **Step 5-A** | Fairness mitigation: Reweighing + Fairlearn (DP/EO), 보호속성 GENDER+AGE, baseline+aux 데이터 |
| **Step 5-B** | Generic RAG 정적 chunks 7개 (인스턴스별 retrieval은 future work) |
| **Step 5-C** | IRB 어렵다는 사용자 결정 → LLM persona pilot으로 proxy 평가 |
| **Step 5-C** | 3 personas (Credit Expert / Customer / Regulator), 15 instances, Claude judge |
| **Step 5-C** | 미팅 일정 연기 — 추가 작업 가능 (UCI German Credit 등) |
| **Step 5-D** | UCI German Credit (sklearn fetch_openml `credit-g` v1), 30 instances × 4 mode × 2 LLM |
| **Step 5-D** | TabNet hyperparams 재조정 (n_d=8, batch_size=128 등 — 1000 샘플 작은 데이터) |
| **Step 5-D** | German custom humanize 매핑 (GERMAN_GLOSSARY/VALUE_LABELS) — 한국어 컬럼/값 라벨 |
| **Step 5-D** | G-Eval Anthropic judge로 Anthropic+Gemini 양 출력 평가 (cross-judge는 future work) |
| **Step 5-E** | revert (negative result, 핵심 메시지 약화 우려) — fusion_friendly mode 폐기, tag 제거 |
| **학위논문 v1** | 8장 구조로 작성 시도 → 사용자 결정: 5장 구조 (학교 표준, 손지민 2024 패턴) |
| **학위논문 v2** | 손지민 직접 인용 제거 (학과/교수 명시 우려) + reference 번호 재정렬 [16~25]→[15~24] |
| **학위논문 v3** | 줄간격 200% → 160% (학위논문지침 최소값) + 감사의 글 제거 |
| **학위논문 v4** | 그림 4-1 시스템 파이프라인 (matplotlib 자체 작성) + Word ToC 자동 필드 |
| **학위논문 v5** | 교수 미팅 후 §4.3.2 환각률 A/B/C 3차원 분리 + §4.1.4 AUROC 실무 평가 추가 |
| **Poppler 설치** | pdftotext 활용 위해 winget으로 Poppler 25.07 설치 (PATH 등록) |
| **Worktree 워크플로** | 매 step은 `claude/<branch>`에서 commit → main fast-forward → push (이전 세션부터 채택) |

---

## 9. 알려진 이슈 / 주의사항

| 이슈 | 해결 |
|---|---|
| `gemini-2.0-flash` free tier limit = 0 | `gemini-2.5-flash`로 전환 |
| Gemini 2.5 Flash 503 UNAVAILABLE 다발 | retry 30s/60s/120s/240s 백오프; 회복 안 되면 30분~1시간 대기 |
| SHAP 0.49 + XGBoost 3.x base_score 파싱 버그 | `src/shap_analysis._XgbNativeExplainer` 우회 |
| LightGBM 컬럼명에 콤마 거부 | `src/baselines._sanitize_columns` 적용 |
| Windows 콘솔 한글 print 깨짐 | `PYTHONIOENCODING=utf-8` 필수 |
| matplotlib `set_theme()`이 한글 폰트 override | figure 텍스트는 영어로 통일 |
| TabNet `feature_importances_`가 load_model 후 None | csv에서 직접 로딩 |
| pickle preprocessor module 경로 mismatch | parquet 컬럼명에서 직접 가져옴 |
| torch 2.5 + transformers 5.7 NLI 모델 .bin 차단 | safetensors 형식 모델 사용 (CVE-2025-32434) |
| eval_fusion.py csv 덮어쓰기 | judge별 백업 필수 (`fusion_eval_{judge}_judge.csv`) |
| eval_fusion.py G-Eval JSON parse_error | retry 4회 로직, NaN 있으면 dropna 처리 |
| One-hot prefix raw_outside_dataset false positive | `eval_fusion.hallucination_rate_fusion` prefix 매칭 |
| Worktree에 data/, .venv/ 없음 | junction 또는 main에서 직접 작업 |
| Worktree에 results/baseline_models/ 없음 | main에서 복사 |
| **docx/pptx 잠금 → git unlink 실패** | **사용자가 Word/PowerPoint 닫고 재시도** |
| **Worktree merge 시 main 디렉토리에 partial 변경 잔여** | **`git -C D:/paper checkout -- <files>` + untracked rm 후 재시도** |
| Fairlearn ExpGrad EO+AGE 안티 패턴 | EO는 DP를 보장 안 함, 본 데이터의 4/5 rule(DI 기반) 통과엔 부적합 |
| no_shap 표본 작음 (Step 1 데이터 30 idx 중 1~2개만) | 표본 확장 future work, 현재는 통계 의미 약함 명시 |
| **python-docx 수식($) 미지원** | LaTeX 수식 → 자연어/유니코드 (ρ, ŷ 등) + `> **수식 N-N**: ...` 박스로 표현 |
| **python-docx Word ToC 미지원** | OXML 필드 (`w:fldChar` + `TOC \\o "1-3"`) 직접 삽입 → 사용자가 [F9]로 갱신 |
| **build_thesis_5ch_docx cascade 매핑 버그** | substring replace 시 깊은 헤딩(### X.Y.Z)부터 먼저, ## X.Y 그 다음 — 일반 `## ` 패턴 금지 |
| **Read tool PDF 변환 실패** | Poppler `pdftoppm` 필요 → winget 설치 후 새 Claude Code 세션 재시작 (PATH 갱신) |
| **한국어 PDF 우회 추출** | `pdftotext.exe -layout <pdf> <out.txt>` → Read tool로 텍스트 읽기 |

---

## 10. 다음 세션 첫 메시지 권장 형식

**★ 현재 상태**: 학위논문 초안(5장 구조, 815 KB docx) 완성, 교수 2차 미팅 대기.

새 세션 시작 시 사용자가 입력하면 좋은 첫 메시지 예:

> "CLAUDE.md 읽고 현재 상태 확인해줘. thesis/draft/thesis_draft.docx 학위논문 초안이 최신 상태야."

또는 (교수님 2차 미팅 후)

> "교수님 2차 피드백: [내용]. 본문 어디 수정할지 정리해줘."

또는 (보강 실험 진행 시)

> "3-way ablation 시작 — Attention-only 단독 검증. Fusion 우월성이 attention 기여인지 isolate."

또는 (추가 데이터셋 진행 시)

> "Australian Credit 추가 — 690 × 14, 복잡도 중간. Step 5-D 패턴 그대로 이식."

또는 (논문 수정)

> "thesis/draft/chapter[1~8]_*.md 수정 후 build_thesis_5ch_docx로 재빌드해줘."

이 파일(§3 진행 흐름)이 최신이면 새 세션이 컨텍스트를 즉시 회복할 수 있다.

### 학위논문 docx 재빌드 빠른 명령

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -W ignore -c \
  "import sys; sys.path.insert(0, 'thesis/draft'); import build_thesis_5ch_docx; build_thesis_5ch_docx.main()"
```

### 환각률 / AUROC 등 교수 미팅 자주 나오는 질문

§3 학위논문 초안 작성 부분 + §1.5 환각률 0%의 정확한 의미 참조. 핵심:

- "환각률 0%" = LLM 변수명 환각률 (차원 B)만 보장. 부도 예측 정확도 (차원 A, AUROC 0.76)와 LLM 의미 충실성 (차원 C, NLI 0.625)은 별도 측정
- AUROC 0.76은 신용평가 mainstream (Lessmann et al. 2015 0.70~0.80 범위), 실무 가능 수준
- 본 연구 contribution은 *예측 정확도 갱신* 이 아니라 *통합 시스템 + 4-tier 평가 프레임워크*

---

## 부록 — Step 5-C 충격적 발견의 학술 가치

본 연구의 가장 정직한 메시지는 Step 5-C의 trade-off 발견:

| 평가 차원 | 1위 | 2위 |
|---|---|---|
| 사실성 (NLI Entailment) | fusion 0.62 | shaponly 0.41~0.51 |
| 충실성 (G-Eval Completeness) | fusion 4.97 | generic_rag 4.83 |
| **사람 친화성 (Persona trust)** | **generic_rag 4.91** | fusion 4.31 |
| **Customer clarity** | **generic_rag 4.93** | fusion 2.67 ⚠️ |

→ **본 연구의 fusion 메커니즘은 사실성 1위지만 친근함은 1위 아님**. 단순 우월성 주장 X, 응용에 따른 mode 선택 trade-off 정량 입증.

이 발견은:
1. 약점 #2 부분 해소 (LLM 평가 객관성 → pilot human proxy)
2. 본 연구의 한계 honest 인정 (Customer clarity 약점)
3. 메시지 정교화 ("환각 차단"만이 아닌 "fact-grounded faithfulness with trade-off")
4. Future work 명확화 (Customer-friendly 표현 정제, hybrid 표현)

학술 논문에서 honest reporting + trade-off 정량 = 좋은 연구의 표준 진행. 본 연구가 이 단계에 도달함.
