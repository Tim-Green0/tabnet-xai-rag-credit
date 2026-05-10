# 초록 / Abstract

> 학위논문지침 ⑺: "본문이 국문일 경우에는 외국어로 된 초록을 앞세우고, 본문이 외국어인 경우에는 국문초록을 앞세운다."
> 본 논문은 한국어 본문이므로 **영문 Abstract** 가 앞에 배치된다.

---

## English Abstract

**Title**: Explainable Credit Scoring and Customer-Tailored Reports via Tabular-Specialized Deep Learning (TabNet) and LLM-based XAI-RAG

**Subtitle**: An Application of Attention-SHAP Fusion Context and a Multi-Tier Evaluation Framework

**By Hyuntaek Oh**
**Department of Data Science · AI**
**Under the supervision of Professor Unsang Park**

Machine learning credit-scoring models often face a dual demand of high predictive accuracy and transparent justification for downstream regulatory and customer-facing applications. While SHAP and attention-based explanations have emerged separately, their fusion in a single context for downstream large language model (LLM) based natural-language reports remains under-explored. This thesis proposes an *agreement-aware fusion context* that combines local SHAP values from XGBoost with TabNet attention masks at the instance level, classifying drivers into three groups — *agreed*, *shap_only*, and *attention_only* — and feeding the structured JSON to a retrieval-augmented generation (RAG) pipeline based on Anthropic Claude Sonnet 4.5 and Google Gemini 2.5 Flash.

We evaluate the proposed system on Kaggle Home Credit Default Risk (307K samples, 214 features) and validate generalization on UCI German Credit (1K samples, 63 features). A four-tier evaluation framework — rule-based faithfulness and hallucination, mDeBERTa-NLI entailment for semantic faithfulness, cross-judge G-Eval (Claude × Gemini) over four dimensions, and a three-persona (Credit Expert / Customer / Regulator) LLM-proxy pilot — together with Reweighing-based fairness mitigation, demonstrates the following findings.

First, the fusion mode achieves the highest NLI entailment in both datasets (0.625 / 0.711), with a consistent monotonic increase across the four modes (no_shap < generic_rag < shaponly < fusion). Second, hallucination is eliminated in all modes via hard constraints, but only fusion achieves fact-grounded faithfulness in semantic and value-citation dimensions. Third, Reweighing-based fairness mitigation passes the 4/5 rule on all four protected-attribute combinations with negligible AUROC change (loss within 0.004). Fourth, mode selection follows an *application × evaluation-dimension × data-complexity* trade-off — fusion fits audit/regulation in complex-domain settings while generic_rag fits customer-facing in simpler domains.

This thesis contributes (a) a novel agreement-aware fusion context schema for SHAP × attention integration, (b) a four-tier evaluation framework combining rule-based, NLI-based, cross-judge LLM, and persona-based measures, (c) generalization evidence on a second dataset with consistent Spearman ρ ≈ 0.11 and monotonic NLI patterns, (d) a Reweighing-based fairness solution preserving model performance, and (e) an honest reporting of customer-clarity weakness and data-complexity dependence. All code is released as an open-source repository at `Tim-Green0/tabnet-xai-rag-credit` on GitHub for reproducibility.

**Keywords**: Explainable Artificial Intelligence (XAI), SHAP, TabNet attention, Credit Scoring, Large Language Model (LLM), Retrieval-Augmented Generation (RAG), Fairness Mitigation, Cross-Judge Evaluation, Natural Language Inference (NLI), Reweighing

---

## 국문 초록

**논문 제목**: 정형 데이터 특화 딥러닝(TabNet)과 거대언어모델(LLM) 기반 XAI-RAG를 활용한 설명 가능한 신용 평가 및 사용자 맞춤형 리포트 생성

**부제목**: 어텐션-SHAP 융합 컨텍스트와 다중 평가 프레임워크 적용

**저자**: 오현택
**전공**: 데이터사이언스·인공지능
**지도교수**: 박운상

본 연구는 정형 데이터 특화 딥러닝 모델인 TabNet의 인스턴스 어텐션과 트리 부스팅 모델인 XGBoost의 SHAP 지역 기여도를 *동의 기반(agreement-aware)* 으로 인스턴스 수준에서 융합한 신규 컨텍스트를 설계하고, 이를 거대언어모델(LLM)의 검색 증강 생성(Retrieval-Augmented Generation, RAG) 파이프라인에 입력하여 신용평가 결과의 자연어 설명 리포트를 자동 생성하는 시스템을 제안한다. 융합 컨텍스트는 SHAP 상위 변수와 어텐션 상위 변수를 *동의 그룹(agreed)·SHAP 단독(shap_only)·어텐션 단독(attention_only)* 의 세 그룹으로 분해한 JSON 형식으로 구성되며, Anthropic Claude Sonnet 4.5와 Google Gemini 2.5 Flash 두 LLM에 동시 입력되어 한국어 자연어 설명을 생성한다.

본 연구는 Kaggle Home Credit Default Risk(307,511명, 214 변수)를 주 데이터셋으로 활용하고, UCI German Credit(1,000명, 63 변수)에 동일 파이프라인을 이식하여 일반화 가능성을 검증한다. 평가 체계는 룰 기반 충실성·환각률 측정, mDeBERTa 다국어 자연어 추론(NLI) 모델 기반 함의도, Anthropic과 Google 두 모델의 교차 LLM G-Eval (factual·completeness·sensitive·style 4차원), 그리고 신용 전문가·고객·규제기관 3-페르소나 LLM 대리 평가의 4-tier 구조로 설계되며, Kamiran-Calders Reweighing 기반의 공정성 보정 또한 함께 적용한다.

주요 결과는 다음과 같다. 첫째, 융합 컨텍스트는 두 데이터셋 모두에서 NLI 함의도 1위(0.625 / 0.711)를 달성하였으며 *no_shap < generic_rag < shaponly < fusion* 의 단조 증가 패턴이 일관되게 관찰되었다. 둘째, hard constraint 조건에서 모든 모드의 환각률은 0%이지만 사실 기반 충실성과 값 정확 인용은 융합 모드에서만 일관되게 우월하였다. 셋째, Reweighing 공정성 보정은 보호 속성 4가지 조합 모두에서 4/5 규칙을 통과하면서 AUROC 손실은 0.004 이내에 머물렀다. 넷째, 페르소나 평가에서 사실성 1위는 fusion이지만 고객 명료성 1위는 generic_rag이며(2.67 vs 4.93), G-Eval 충실성은 데이터 복잡도에 따라 우월 모드가 정반대로 갈리는 *응용 시나리오 × 평가 차원 × 데이터 복잡도* 의 3차원 trade-off가 정량 입증되었다.

본 연구는 (1) 동의 기반 융합 컨텍스트의 신규 설계, (2) 4-tier 평가 프레임워크 구축, (3) 환각 차단과 fact-grounded 충실성의 분리 입증, (4) 두 데이터셋 일반화 검증, (5) 무손실 공정성 보정의 다섯 가지 학술적 기여를 제시하며, 모든 코드는 GitHub 공개 저장소(`Tim-Green0/tabnet-xai-rag-credit`)에 공개하여 재현 가능성을 확보한다.

**중심어**: 설명 가능한 인공지능, SHAP, TabNet 어텐션, 신용 평가, 거대언어모델, 검색 증강 생성, 공정성 보정, 교차 LLM 평가, 자연어 추론, Reweighing
