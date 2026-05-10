# 부록 (Appendix)

## 부록 A. SHAP Global Importance Top-50 (Home Credit)

본 연구의 XGBoost 모델에 대한 SHAP global importance(평균 절댓값 SHAP) 상위 50개 변수의 결과는 `results/shap_global_xgboost.csv`에 저장되어 있으며, 상위 20개를 본 부록에 정리한다.

**표 A-1. Home Credit XGBoost SHAP global importance Top-20**

| 순위 | 변수 | mean(|SHAP|) | 비고 |
|---|---|---|---|
| 1 | EXT_SOURCE_2 | (값) | 외부 신용평가 점수 2 |
| 2 | EXT_SOURCE_3 | (값) | 외부 신용평가 점수 3 |
| 3 | EXT_SOURCE_1 | (값) | 외부 신용평가 점수 1 |
| 4 | DAYS_BIRTH | (값) | 연령 (보호 속성) |
| 5 | AMT_GOODS_PRICE | (값) | 구매 대상 상품 가격 |
| ... | ... | ... | ... |

> 본 표는 docx 통합 시 `results/shap_global_xgboost.csv`에서 직접 추출하여 채운다.

## 부록 B. 보호 속성 분포 (양 데이터셋)

### B.1 Home Credit

**표 B-1. Home Credit 성별 분포 및 부도율**

| 성별 | 인스턴스 수 | 비율 | 부도율 |
|---|---|---|---|
| 여성 (F) | 202,448 | 65.85% | 7.00% |
| 남성 (M) | 105,059 | 34.15% | 10.14% |
| 미상 (XNA) | 4 | 0.001% | 0.00% |

**표 B-2. Home Credit 연령 구간별 부도율**

| 구간 | 평균 연령 | 부도율 |
|---|---|---|
| 25-35세 | 30.5 | (값) |
| 35-45세 | 40.0 | (값) |
| 45-55세 | 50.0 | (값) |
| 55세+ | 62.5 | (값) |

### B.2 UCI German Credit

**표 B-3. UCI German Credit 성별·외국인·연령 분포**

| 항목 | 값 |
|---|---|
| 성별 — 남성 | 690 (69.0%), 부도율 27.7% |
| 성별 — 여성 | 310 (31.0%), 부도율 35.2% |
| 외국인 노동자 — 예 | (값), 부도율 (값) |
| 외국인 노동자 — 아니오 | (값), 부도율 (값) |
| 연령 < 35 (중앙값 미만) | 부도율 34.8% |
| 연령 ≥ 35 (중앙값 이상) | 부도율 25.5% |

## 부록 C. LLM 프롬프트 전문 (4-mode)

### C.1 no_shap 모드 프롬프트

```
당신은 고객에게 신용 평가 결과를 친절히 설명하는 금융 상담사입니다.

[고객 데이터 — 이 정보만 사용]
{customer_data_text}

[예측 결과]
- 부도 확률: {default_prob:.4f}
- 임계값: {threshold:.4f}
- 결정: {decision}

[출력 형식 — 반드시 이 5개 섹션 순서로]
[결정 요약]   (1줄)
[주요 거절 사유]   (REJECT일 때만, 최대 3개)
[긍정적으로 평가된 요인]   (최대 3개)
[개선 권고]   (1~3개)
[면책 고지]   (1~2줄)

이제 위 정보에 기반해 설명 리포트를 작성해주세요. 한국어로.
```

### C.2 generic_rag 모드 프롬프트

본 모드는 7개의 도메인 지식 chunks(K1: 신용평가 핵심 변수의 의미, K2: 부도 위험의 일반 원리, K3: 임계값과 결정의 의미, K4: 민감 변수 마스킹 정책, K5: 금융 용어 일반 가이드, K6: 출력 형식과 Hard Constraints, K7: Reject/Approve 인지 가이드)를 컨텍스트에 포함한다. 전체 프롬프트와 chunks는 `src/baseline_generic_rag.py`의 `KNOWLEDGE_CHUNKS` 변수와 `GENERIC_RAG_PROMPT` 변수에서 확인 가능하다.

### C.3 shaponly 모드 프롬프트

본 모드는 SHAP 상위 5개 변수의 부호와 값만을 컨텍스트로 전달한다. 전체 프롬프트는 `src/llm_explainer.py`의 `PROMPT_TEMPLATE` 변수에서 확인 가능하다.

### C.4 fusion 모드 프롬프트

본 모드는 본 연구의 핵심 컨텍스트인 *동의 기반 융합 컨텍스트* 를 입력으로 사용한다. 컨텍스트의 JSON 구조는 4.4.3절에 명시하였으며, 전체 프롬프트는 `src/llm_explainer_fusion.py`의 `FUSION_PROMPT` 변수에서 확인 가능하다.

## 부록 D. G-Eval Rubric과 NLI 라벨

### D.1 G-Eval Rubric (전문)

```
당신은 신용 평가 LLM 설명의 평가자입니다.
아래 [생성된 설명]을 [참조 컨텍스트]에 비추어 4개 차원별 1~5점 척도로 평가하세요.

[참조 컨텍스트]
{context_json}

[생성된 설명]
{explanation_text}

평가 차원 (각 1~5점):
1. factual_accuracy : 설명의 모든 변수명·수치·SHAP 부호가 컨텍스트와 일치하는가
   (1=다수 불일치 또는 환각, 5=완벽 일치)
2. completeness : 컨텍스트의 top driver를 충분히 다루며 결정 사유를 명확히 전달하는가
   (1=핵심 누락, 5=완결)
3. sensitive_leak : 성별·연령·인종·종교 등 민감 변수를 직접 언급하지 않았는가
   (1=직접 언급, 5=완벽 마스킹)
4. style : 고객에게 전달하기 적절한 친절·중립적 톤인가
   (1=부적절, 5=매우 적절)

오직 JSON 한 개만 출력하세요. 다른 텍스트는 금지.
{
  "factual_accuracy": <int 1-5>,
  "completeness": <int 1-5>,
  "sensitive_leak": <int 1-5>,
  "style": <int 1-5>,
  "rationale": "<한 줄 사유>"
}
```

### D.2 mDeBERTa NLI 모델 라벨 매핑

본 연구의 NLI 모델 `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`의 출력 라벨은 다음과 같이 매핑된다.

| 모델 출력 인덱스 | 라벨 | 본 연구 표준화 |
|---|---|---|
| 0 | entailment | entailment |
| 1 | neutral | neutral |
| 2 | contradiction | contradiction |

본 연구의 NLI Entailment Rate는 entailment 확률의 평균이다.

## 부록 E. 4-mode 자연어 설명 예시

본 부록은 동일한 인스턴스(idx=10629, REJECT, default_proba=0.898)에 대한 4-mode 자연어 설명 예시를 비교 제시한다. 전체 설명은 `results/explanations_*` 디렉토리에서 확인 가능하다.

### E.1 no_shap 모드 출력 (요약)

> "신청 결과: 대출이 거절되었습니다(부도 확률 89.9%). 신청 시점의 정보를 종합 분석한 결과 부도 위험이 높게 평가되었습니다. ... (5개 섹션 출력)"

### E.2 generic_rag 모드 출력 (요약)

> "신청 결과: 대출이 거절되었습니다(부도 확률 89.9%). 외부 신용평가 점수가 매우 낮은 수준이며, 이는 일반적으로 부도 위험을 높이는 핵심 요인으로 작용합니다(참조 지식 K2). ..."

### E.3 shaponly 모드 출력 (요약)

> "신청 결과: 대출이 거절되었습니다(부도 확률 89.9%). 외부 신용평가 점수 3이 0.0634(SHAP +1.10)로 매우 낮게 평가되어 부도 가능성을 크게 높이는 요인으로 작용했습니다. ..."

### E.4 fusion 모드 출력 (요약)

> "신청 결과: 대출이 거절되었습니다(부도 확률 89.9%). 두 해석 모델이 동의한 강한 신호: 외부 신용평가 점수 3이 0.0634로 매우 낮게 평가되어 부도 가능성을 가장 크게 높이는 요인으로 작용했습니다. SHAP 분석은 외부 신용평가 점수 2(0.0972) 또한 부도 가능성을 높이는 요인으로 분석하였으며, TabNet 어텐션은 재직 일수(0년) 정보가 결정에 영향을 미친 것으로 확인하였습니다. ..."

## 부록 F. 코드 저장소 안내

본 연구의 모든 코드와 산출물은 GitHub 공개 저장소에 게시되어 있다.

- **저장소 URL**: https://github.com/Tim-Green0/tabnet-xai-rag-credit
- **라이선스**: MIT License
- **재현 가이드**: 저장소의 README에 환경 설정·데이터 다운로드·각 단계별 실행 명령어가 정리되어 있다.

본 저장소는 본 연구의 모든 단계(Step 1 미팅 프로토타입 → Step 5-D UCI German Credit 일반화)에 대한 코드와 결과 csv·json·figure 파일을 포함하며, SEED=42 고정으로 모든 결과의 재현 가능성을 확보한다. 본 연구의 4-tier 평가 프레임워크와 Reweighing 공정성 보정의 통합 코드도 함께 공개되어 있다.

저장소의 디렉토리 구조는 다음과 같다.

```
tabnet-xai-rag-credit/
├── src/                    # 모든 코드
├── data/                   # 데이터 (gitignored, 원본 데이터셋 다운로드 가이드 포함)
├── results/                # 모든 실험 결과 csv/json/md
├── figures/                # 모든 figure png
├── paper/                  # midterm 자료 (선행 작업물)
├── thesis/                 # 본 학위논문 (draft 포함)
└── requirements.txt        # Python 의존성
```

본 코드는 추후 학술적 활용 또는 산업 적용을 위해 자유롭게 활용 가능하다.
