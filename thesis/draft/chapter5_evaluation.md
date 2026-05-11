# 제 5 장. 평가 프레임워크

본 장은 본 연구의 4-tier 평가 체계를 정의하고, 각 평가 도구의 측정 원리·구현·해석 방법을 상세히 기술한다. 4-tier 체계의 구성은 (1) 룰 기반 표면적 측정, (2) NLI 기반 의미적 측정, (3) LLM 기반 다차원 측정, (4) 페르소나 기반 이해관계자 측정으로 정의된다. 또한 모델 성능 지표와 공정성 지표도 함께 정의한다.

## 5.1 모델 성능 지표

본 연구의 예측 모델 평가는 신용평가 도메인의 표준 지표 6개로 수행된다.

- **AUROC** (Area Under the ROC Curve): 임계값 무관한 모델 분류 능력. 0.5는 무작위, 1.0은 완벽 분류를 의미한다.
- **AUPRC** (Area Under the Precision-Recall Curve): 클래스 불균형 환경에서 양성 클래스의 검출 능력. AUROC 대비 부도와 같은 소수 클래스의 분류 품질을 더 민감하게 반영한다.
- **KS** (Kolmogorov-Smirnov 통계량): KS = max_t |TPR(t) − FPR(t)| 로 정의되며, 신용평가 도메인의 전통 표준 지표이다.
- **F1, Precision, Recall**: 분류 임계값 적용 후의 표준 분류 지표.

본 연구는 검증 셋에서 Youden's J 통계량(`J = TPR - FPR`)을 최대화하는 임계값을 결정한 후, 동일 임계값을 테스트 셋에 적용하여 모든 지표를 산출한다. 5-fold CV를 통해 매 fold의 mean ± std를 보고하며, fold별 지표 변동성을 함께 제공한다.

## 5.2 룰 기반 평가

### 5.2.1 Faithfulness Score (충실성 점수)

Faithfulness Score는 LLM 출력이 주어진 컨텍스트의 변수·값·SHAP 부호를 정확히 인용하였는지를 토큰 매칭으로 측정한다. 본 연구는 컨텍스트의 각 driver(변수·값·부호 entry)에 대해 다음 4가지 매칭 비율을 산출한다.

- **feat_match_rate**: 컨텍스트의 변수명이 LLM 출력 텍스트에 등장한 비율
- **val_match_rate**: 컨텍스트의 값이 LLM 출력 텍스트에 등장한 비율 (숫자 토큰 우선 매칭)
- **sign_match_rate**: 컨텍스트의 SHAP 부호가 LLM 출력에 정확히 반영된 비율 (부도 가능성을 높이는/낮추는 표현)
- **full_match_rate**: 변수·값·부호 셋 모두가 매칭된 driver의 비율

룰 기반 평가의 한계는 표현의 다양성을 충분히 잡지 못한다는 점이다. 예컨대 LLM이 "외부 신용평가 점수 3은 매우 낮은 수준"이라고 표현하면, 이는 의미적으로는 정확하지만 토큰 수준에서는 부분 매칭으로 산출될 수 있다. 이러한 룰 기반의 한계를 보완하기 위해 다음 5.3절의 NLI 기반 평가를 도입한다.

### 5.2.2 Hallucination Rate (환각률)

Hallucination Rate는 LLM 출력에 등장한 변수 토큰 중 컨텍스트에 명시되지 않은 비율을 측정한다. 본 연구는 두 가지 정의를 함께 보고한다.

- **strict**: 데이터셋의 변수 집합 외부에서 인용된 토큰 비율
- **broad**: 컨텍스트 내부에 없으나 데이터셋 전체에는 있는 토큰 + strict 환각의 합산 비율

영문 대문자 토큰(예: `EXT_SOURCE_3`, `AMT_INCOME_TOTAL`)을 정규표현식 `\b[A-Z][A-Z0-9_]{2,}\b`로 추출한 후, 일반 약어(SHAP, REJECT, APPROVE 등)를 제외하고 매칭을 수행한다. UCI German Credit과 같이 변수명이 소문자인 경우 underscore가 포함된 영문 토큰(`\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b`)으로 패턴을 확장한다.

이 룰은 LLM이 *지어낸 변수* 또는 *컨텍스트에 없는 변수* 의 인용을 직접적으로 측정하므로, 환각의 가장 명확한 지표로 활용된다.

## 5.3 NLI 기반 의미적 충실성 평가

### 5.3.1 mDeBERTa 다국어 NLI 모델

본 연구는 NLI 평가용 모델로 `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` [23]을 활용한다. 이는 mDeBERTa-v3 기반 다국어 NLI 모델로 2.7M개 다국어 NLI 표본으로 fine-tuning되었으며, 한국어를 포함한 다국어 추론 능력을 보유한다. 본 연구의 자연어 설명은 한국어로 작성되므로 한국어 추론 능력이 필수적이며, 본 모델은 이 요건을 충족한다.

원래 한국어 전용 NLI 모델(예: `Huffon/klue-roberta-base-nli`)도 후보였으나, 최신 PyTorch 2.5와 transformers 5.7 환경에서 보안 이슈(CVE-2025-32434, .bin 파일 차단)로 인해 safetensors 형식의 mDeBERTa-v3로 최종 결정하였다.

### 5.3.2 컨텍스트 → premise 자연어 변환

NLI 평가는 (premise, hypothesis) 쌍에 대해 함의(entailment)/중립(neutral)/모순(contradiction) 확률을 산출한다. 본 연구는 fusion 컨텍스트의 각 driver를 다음과 같은 자연어 문장으로 변환하여 premise를 구성한다.

- *agreed* 그룹: "두 모델이 동의: <변수>이(가) <값>로, 부도 가능성을 높이는/낮추는 요인이다."
- *shap_only* 그룹: "SHAP 분석: <변수>이(가) <값>로, 부도 가능성을 높이는/낮추는 요인이다."
- *attention_only* 그룹: "TabNet 어텐션: <변수>이(가) <값>로, 결정에 영향을 준 변수이다."

전체 premise는 *결정 요약 문장* 과 위 driver 문장들을 연결한 단락으로 구성되며, 길이가 1,500자를 초과하면 잘라서 NLI 모델의 입력 한계 이내로 조정한다. SHAP-only 컨텍스트와 generic_rag 컨텍스트도 동일한 변환 규칙으로 premise를 구성하며, 4-mode 평가에서 일관된 측정 기준을 유지한다.

### 5.3.3 평가 지표

LLM 출력 텍스트를 문장 단위로 분리한 후(섹션 헤더, [개선 권고], [면책 고지] 섹션은 제외), 각 문장을 hypothesis로 하여 premise와의 NLI 확률을 산출한다. 인스턴스 수준의 지표는 다음 4가지이다.

- **entailment_rate**: 문장별 함의 확률의 평균
- **contradiction_rate**: 모순 확률의 평균
- **min_entailment**: 최악 문장의 함의 확률 (worst-case)
- **n_strong_entailment**: 함의 확률 > 0.5인 문장의 개수

본 연구의 핵심 지표는 entailment_rate로, 이는 LLM 출력의 *의미적 충실성* 을 측정하는 단일 수치이다. contradiction_rate는 LLM이 컨텍스트와 *모순* 되는 진술을 생성한 정도로, 0에 가까울수록 좋다.

## 5.4 G-Eval (LLM-as-a-Judge)

### 5.4.1 4차원 척도

G-Eval [18]은 LLM이 평가자(judge) 역할을 수행하여 다른 LLM의 출력을 다차원 척도로 점수화하는 프레임워크이다. 본 연구는 신용평가 도메인에 맞춰 4가지 척도를 1~5점 정수 척도로 정의한다.

- **factual_accuracy**: 설명의 모든 변수명·수치·SHAP 부호가 컨텍스트와 일치하는 정도. (1=다수 불일치, 5=완벽 일치)
- **completeness**: 컨텍스트의 top driver를 충분히 다루며 결정 사유를 명확히 전달하는 정도. (1=핵심 누락, 5=완결)
- **sensitive_leak**: 성별·연령·인종·종교 등 민감 변수를 직접 언급하지 않은 정도. (1=직접 언급, 5=완벽 마스킹)
- **style**: 차주에게 전달하기 적절한 친절·중립적 톤을 갖춘 정도. (1=부적절, 5=매우 적절)

평가자(LLM judge)는 평가 프롬프트(rubric)를 통해 위 4차원에 대해 각각 1~5점을 출력하며, 사유(rationale)를 한 줄로 함께 출력한다. 출력 형식은 JSON으로 강제되며, JSON 파싱 실패 시 4회까지 재시도한다.

### 5.4.2 Cross-Judge 평가

LLM-as-a-Judge 패러다임의 self-bias 문제를 완화하기 위해 본 연구는 *교차 평가(cross-judge)* 를 도입한다. 즉 Claude judge가 Gemini target의 출력을 평가하고, Gemini judge가 Claude target의 출력을 평가한다. 두 judge의 점수가 일관되게 나오는 차원은 신뢰할 수 있는 결과로, 차이가 큰 차원은 self-bias 또는 모델별 평가 기준의 차이를 시사한다.

본 연구의 Step 3-C-2-f 단계에서 cross-judge를 적용한 결과, *completeness* 차원에서는 두 judge가 일관된 양수 차이를 보였으나, *factual_accuracy* 차원에서는 Gemini target에 대해 Claude judge -0.13 vs Gemini judge +0.47의 큰 차이(0.60)가 관찰되었다. 이는 단일 judge 평가의 한계를 직접적으로 입증하며, cross-judge의 가치를 정량적으로 보여준다.

## 5.5 Pilot Human-Proxy 평가

### 5.5.1 동기와 한계

자연어 설명은 동일한 내용이라도 신용 전문가·일반 고객·규제기관 등 이해관계자에 따라 적합성이 달라진다. 정식 인간 평가(human evaluation)는 IRB(Institutional Review Board) 승인이 필요하며, 다수의 평가자 모집과 평가 시간 확보 측면에서 본 연구의 시간 범위 내 적용이 어려웠다. 따라서 본 연구는 *LLM 페르소나 대리 평가(LLM-proxy persona evaluation)* 를 *pilot* 으로 도입한다.

이는 정식 IRB 인간 평가의 대체가 아니라 *예비 신호 수집* 의 위상을 가지며, 본 연구의 한계 항목으로 명시적으로 보고된다. 정식 IRB 평가는 향후 연구 방향으로 제시한다.

### 5.5.2 3 Personas와 3 Metrics

본 연구는 3개의 페르소나를 다음과 같이 정의한다.

- **Credit Expert** (신용 전문가): 부도 모델의 정확성과 변수 인용의 fact-grounding을 중시. 통계적 정확성과 도메인 지식 기반 평가를 수행한다.
- **Customer** (고객): 본인의 신용평가 결과를 받아본 일반인 관점. 명료성·이해 가능성·실제 행동 가능성을 중시한다.
- **Regulator** (규제기관): 차별 회피와 규제 준수의 관점. 보호 속성 마스킹·공정성·감사 적합성을 중시한다.

각 페르소나는 다음 3가지 척도(1~5점)로 평가한다.

- **trustworthiness**: 평가 결과를 얼마나 신뢰할 수 있다고 느끼는가
- **clarity**: 설명을 얼마나 명료하게 이해할 수 있는가
- **actionability**: 실제로 행동(개선·재신청·검토 등)할 수 있는 정보를 제공하는가

총 4 modes × 2 LLM target × 15 instances × 3 personas × 3 metrics = 540 평가 시도가 이루어지며, JSON 파싱 실패와 재시도 후 실제 유효 평가는 약 276건에 도달하였다.

페르소나 평가용 LLM judge는 안정성을 우선하여 Anthropic Claude Sonnet 4.5 단일 모델을 활용한다. 이는 cross-judge의 self-bias를 일부 감수하면서, 평가 결과의 일관성을 우선한 절충 결정이다.

## 5.6 공정성 평가

본 연구는 보호 속성에 대한 공정성을 다음 세 가지 지표로 평가한다.

### 5.6.1 Demographic Parity (DP) 차이

Demographic Parity는 보호 속성 그룹 간 *양성 예측률* 의 차이로 정의된다.

> **수식 5-1**: DP_diff = | P(ŷ=1 | s=0) − P(ŷ=1 | s=1) |

본 지표는 그룹 간 모델 출력의 분포 차이를 직접 측정하지만, 양성 클래스의 정확도 차이를 반영하지 않는다는 한계가 있다.

### 5.6.2 Equal Opportunity (EO) 차이

Equal Opportunity는 양성 클래스에 한정한 *진양성률(TPR)* 의 그룹 간 차이로 정의된다.

> **수식 5-2**: EO_diff = | TPR(s=0) − TPR(s=1) |

이는 *부도 차주에게 정확히 부도를 예측하는* 능력의 그룹 간 차이를 반영하며, DP 대비 모델 정확도까지 고려한 공정성 지표이다.

### 5.6.3 Disparate Impact ratio (DI ratio)

Disparate Impact ratio는 EEOC의 4/5 규칙 [19]에 직접 대응하는 지표이다.

> **수식 5-3**: DI_ratio = min[ P(ŷ=1|s=0), P(ŷ=1|s=1) ] / max[ P(ŷ=1|s=0), P(ŷ=1|s=1) ]

DI ratio가 0.8 이상이면 4/5 규칙을 *통과* 한 것으로 판단하며, 그 미만이면 *불리한 영향(adverse impact)* 으로 분류된다. 본 연구의 공정성 보정(Reweighing 등)은 모델 성능 손실을 최소화하면서 DI ratio를 0.8 이상으로 만드는 것을 목표로 한다.

## 5.7 평가 프레임워크 통합

표 5-1은 본 연구의 4-tier 평가 프레임워크를 통합 정리한 것이다.

**표 5-1. 본 연구의 4-tier 평가 프레임워크**

| Tier | 도구 | 주요 지표 | 측정 대상 | 강점 | 한계 |
|---|---|---|---|---|---|
| 1 | 룰 기반 | feat/val/sign_match_rate, halluc_rate | 표면 토큰 매칭 | 빠름, 결정적 | 표현 다양성 미반영 |
| 2 | NLI (mDeBERTa) | entailment_rate, contradiction_rate | 의미적 함의 | 표현 다양성 반영 | 모델 의존성 |
| 3 | G-Eval (Cross) | factual/completeness/sensitive/style (1~5) | LLM judge 다차원 | 다차원 평가 | self-bias 위험 |
| 4 | Persona | trust/clarity/actionability (1~5) × 3 페르소나 | 이해관계자 관점 | 응용 적합성 | LLM proxy 한계 |

본 4-tier 체계는 동일한 LLM 출력을 *4가지 다른 관점에서 동시 측정* 하여, 단일 지표로는 잡히지 않는 trade-off를 정량적으로 드러내는 데 목적이 있다. 다음 제6장에서는 이 평가 프레임워크를 본 연구의 4-mode 컨텍스트(no_shap / generic_rag / shaponly / fusion)에 적용한 모든 실험 결과를 통합 보고한다.
