# 제 7 장. 논의

본 장은 제6장의 분석 결과가 갖는 학술적·산업적 의미를 정리하고, 본 연구의 한계와 향후 연구 방향을 *honest reporting* 의 관점에서 명시적으로 논의한다.

## 7.1 본 연구의 핵심 발견

### 7.1.1 Fusion 메커니즘의 일관 우월성

본 연구가 제안한 *동의 기반 융합 컨텍스트* 는 NLI 함의도(entailment rate) 측면에서 두 데이터셋 모두에서 일관되게 1위를 차지하였다. Home Credit에서 0.625, UCI German Credit에서 0.711의 entailment를 보이며, 두 번째인 shaponly 모드 대비 약 0.10~0.15의 큰 차이를 보였다. 또한 4-mode 단조 증가 패턴(no_shap < generic_rag < shaponly < fusion)이 두 데이터셋에서 동일하게 관찰되어, 본 연구의 메커니즘이 *데이터셋 특수 패턴* 이 아닌 *신용평가 도메인의 일반 패턴* 에 기반함을 강력히 시사한다.

이 발견은 RQ1 ("동의 기반 융합 컨텍스트가 LLM 자연어 설명의 사실 기반 충실성을 향상시키는가?")에 대한 직접적인 긍정 답변이며, RQ4 ("본 연구의 fusion 메커니즘은 다른 데이터셋에서도 일관된 패턴으로 작동하는가?")에 대한 일반화 검증의 결과이기도 하다.

### 7.1.2 환각 차단과 fact-grounding의 분리

본 연구의 가장 의미 있는 학술적 발견 중 하나는 *환각 차단* 과 *fact-grounded 충실성* 이 별개의 차원이라는 점을 정량적으로 입증한 것이다. 4-mode 모두에서 환각률 0%가 달성되었으나, NLI Entailment Rate와 Value Match Rate는 모드 간에 큰 차이를 보였다. 즉 *환각 차단은 hard constraint만으로 충분조건* 이지만, *fact-grounded 의미적 충실성은 융합 컨텍스트가 필수* 이다.

이는 본 연구의 메시지를 정교화하는 핵심 근거이다. 만약 본 연구가 *"환각률 0%"* 만을 강조하였다면, 이는 hard constraint 정책의 결과로 trivial하게 반박 가능하다. 그러나 *환각률 0%는 baseline에서도 충족되며, fusion의 진정한 차별성은 fact-grounded 충실성에 있다* 는 메시지는 본 연구의 학술적 기여를 명확히 한다.

### 7.1.3 데이터 복잡도에 따른 모드 선택 trade-off

UCI German Credit의 G-Eval Completeness 결과는 본 연구의 메시지에 *데이터 복잡도* 차원을 추가하는 중요한 발견이다. Home Credit (214 features)에서는 fusion이 Completeness 1위 (4.82)이지만, UCI German Credit (63 features)에서는 generic_rag가 1위 (4.58)이며 fusion은 4위 (3.77)에 그친다.

이 차이는 *fusion의 추가 정보(SHAP 부호, agreement 라벨)가 복잡한 도메인에서는 충실성 향상에 기여하지만, 단순 도메인에서는 LLM judge가 over-claim으로 인지* 하여 점수가 낮아질 수 있음을 시사한다. 즉 본 연구의 메커니즘 효과는 데이터의 복잡도와 LLM judge의 평가 기준에 따라 *조건부* 일 수 있다.

### 7.1.4 사실성과 친화성의 trade-off

Step 5-C에서 발견된 페르소나 평가 결과는 본 연구의 가장 *honest* 한 발견이다. 사실성(NLI/G-Eval)에서 1위인 fusion이 *Customer clarity* 차원에서는 2.67로 *최하위* 를 차지하며, 일반 도메인 지식만으로 구성된 generic_rag가 4.93으로 1위를 차지한다.

이 trade-off는 본 연구가 fusion 메커니즘을 *모든 차원에서 우월한 단일 해결책* 으로 주장하지 않게 하는 honest 근거이다. 대신 본 연구는 *응용 시나리오에 따른 mode 선택 가이드라인* 을 제시한다.

## 7.2 응용 시사점

### 7.2.1 응용 시나리오별 모드 선택 가이드

본 연구의 결과는 신용평가 자연어 설명 시스템의 응용 시나리오별로 다음과 같은 모드 선택을 제시한다.

- **Audit / 규제 준수 / 내부 감사**: **fusion 모드** 권장. NLI 사실성 1위, G-Eval Completeness 1위(복잡한 도메인), Credit Expert·Regulator 페르소나의 trustworthiness 1위. 두 해석 모델의 동의 정보가 감사 추적성을 강화한다.

- **고객 응대 / Customer-facing**: **generic_rag 모드** 권장. Customer clarity 1위(4.93), 표현이 친근하고 도메인 지식이 자연스럽게 설명된다. 단 인스턴스 수준의 정확성이 fusion 대비 떨어질 수 있음을 인지하고, 중요 결정의 *인용* 보다는 *안내* 측면에서 활용해야 한다.

- **소수 정형 데이터셋 (≤ 100 features)**: 데이터 복잡도가 낮은 경우 generic_rag와 fusion의 차이가 작을 수 있음. 학술적 신뢰성이 핵심이라면 fusion, 운영 단순성이 핵심이라면 generic_rag.

### 7.2.2 hard constraint의 산업 적용

본 연구의 환각률 0% 달성은 hard constraint 정책의 산업 적용 가능성을 직접적으로 시사한다. 5가지 항목의 단순한 자연어 제약문(컨텍스트 외부 정보 생성 금지, 의료/법률 자문 금지, 민감 변수 직접 언급 금지, 컨텍스트 driver 우선 선택, SHAP 부호 정확 반영)만으로 환각이 안정적으로 차단되며, 이는 LLM 기반 신용평가 자동화 시스템의 산업 적용에 중요한 근거가 된다.

### 7.2.3 공정성 보정의 무손실 적용

Reweighing의 4/4 통과 결과는 공정성 보정이 모델 성능 손실 없이 가능함을 입증한다. 특히 aux 데이터셋(보조 테이블 추가)에서 GENDER에 대한 Reweighing은 AUROC를 오히려 +0.003 향상시키면서도 Disparate Impact ratio를 0.643에서 0.867로 끌어올렸다. 이는 신용평가 산업의 공정성 규제 준수에 직접적인 시사점을 제공한다.

## 7.3 본 연구의 한계 (Honest Reporting)

본 연구는 제한된 자원과 시간 범위 내에서 수행된 석사학위 연구로, 다음과 같은 명시적 한계를 갖는다.

### 7.3.1 표본 크기의 한계

본 연구의 4-mode 자연어 설명 평가는 동일한 30개 인스턴스에 대해 4 modes × 2 LLM = 240건의 호출로 수행되었다. 페르소나 평가에서는 15 instances × 4 modes × 2 LLM × 3 personas = 360 평가 시도로 확장되었으나, 통계적 검정의 power 측면에서 mode 간 차이의 *통계적 유의성* 을 엄밀히 확보하기에는 부족할 수 있다. 본 연구는 이 한계를 명시하며, 표본 30 → 100으로의 확장을 향후 연구 방향으로 제시한다.

### 7.3.2 정식 IRB 인간평가의 부재

본 연구는 정식 인간 평가(human evaluation)를 IRB(Institutional Review Board) 승인 부담과 시간 제약으로 *LLM 페르소나 대리(proxy) 평가* 로 대체하였다. LLM이 페르소나의 관점을 정확히 시뮬레이션할 수 있는지에 대한 자체적 검증은 본 연구의 범위를 벗어나며, 페르소나 평가 결과는 *예비 신호* 의 위상에서 해석되어야 한다. 정식 IRB 승인을 받은 다수 평가자 대상 인간 평가는 향후 연구의 핵심 과제이다.

### 7.3.3 Hard Constraint 의존성

본 연구의 환각률 0% 결과는 hard constraint 정책의 결과이며, 이를 제거할 경우 환각이 발생할 수 있음을 Step 1의 counterfactual 실험에서 확인하였다(Gemini의 경우 환각률 45.5%). 따라서 본 연구의 메시지는 *hard constraint와 fusion 컨텍스트의 결합* 이 환각 차단과 fact-grounding을 동시에 달성한다는 *조건부 메시지* 이다. Hard constraint 없이 환각을 차단하는 일반적 LLM 메커니즘은 본 연구의 범위를 벗어난다.

### 7.3.4 Customer Clarity 약점

본 연구의 fusion 메커니즘은 Customer 페르소나 clarity에서 2.67로 *최하위* 를 차지한다. 이는 SHAP 부호와 agreement 라벨이 일반 고객에게는 이해 장벽이 됨을 시사한다. 본 한계는 향후 *Customer-friendly 표현 정제* 연구를 통해 부분적으로 보완할 수 있을 것으로 예상된다(7.4.4절 참조).

### 7.3.5 한국어 humanize 매핑 의존성

본 연구는 두 데이터셋 모두에서 영문 변수명을 한국어 라벨로 *수동 매핑* 하여 사용자 친화적 표현을 구현하였다(예: `EXT_SOURCE_3` → "외부 신용평가 점수 3"; `checking_status_no_checking` → "체크 계좌 없음"). 자동 한국어 변환 framework가 부재하므로, 새 도메인이나 새 데이터셋에 적용 시 매핑 작업이 추가로 필요하다. 다국어 LLM의 추론 능력을 활용한 자동 변환 모듈은 향후 연구 과제이다.

### 7.3.6 평가 도구 자체의 한계

본 연구의 평가 도구들 또한 각자의 한계를 갖는다. 룰 기반 평가는 표현 다양성을 충분히 잡지 못하며, NLI 모델은 mDeBERTa-v3 단일 모델에 의존한다. G-Eval 또한 LLM judge의 self-bias 위험을 cross-judge로만 부분 완화한 상태이다. 다중 NLI 모델 비교, 다중 LLM judge ensemble 등은 본 연구의 평가 신뢰성을 더 강화할 수 있는 방향이다.

## 7.4 향후 연구 방향

본 연구의 한계와 발견을 바탕으로 다음 연구 방향을 제시한다.

### 7.4.1 정식 IRB 인간 평가

본 연구의 가장 시급한 향후 과제는 정식 IRB 승인을 받은 *다수 평가자 대상 인간 평가* 이다. Credit Expert(은행 신용 심사 담당자), Customer(일반 신용 신청자), Regulator(금융감독원/금융위 담당자) 각 5~10명 규모의 평가자를 모집하여 5점 척도 평가를 수행하고, Cohen's κ 또는 Krippendorff's α 등 평가자 간 일치도를 함께 보고하는 것이 권장된다. 이는 약점 #2 (LLM 평가 객관성)의 완전 해소에 직접 기여할 것이다.

### 7.4.2 추가 데이터셋 일반화

본 연구는 Home Credit과 UCI German Credit 두 데이터셋에서의 일반화를 입증하였으나, 더 다양한 환경에서의 검증이 가능하다. 후보 데이터셋으로는 다음과 같은 것들이 있다.

- **UCI Australian Credit** (690 × 14): 중간 복잡도, 호주 신용평가 도메인
- **Lending Club** (~1.3M × 150): 미국 P2P 대출 도메인, 대규모
- **Korea Credit Bureau 합성 데이터**: 한국 신용평가 도메인 (구할 수 있는 경우)

각 데이터셋에서 본 연구의 메커니즘 일반화를 추가 검증하면, *데이터 복잡도 × 도메인 × 부도율* 의 다차원 환경에서 mode 선택 가이드를 더 견고히 할 수 있다.

### 7.4.3 3-way Ablation (Attention 단독 검증)

본 연구는 *shaponly* 와 *fusion* 의 비교를 수행하였으나, *attention_only* 단독 모드는 별도로 검증하지 않았다. *no_shap / attention_only / shaponly / fusion* 의 4-mode를 추가 비교하면, *fusion 우월성이 attention의 추가적 기여인지 단순 union 효과인지* 를 isolate할 수 있다. 이는 본 연구 메커니즘의 정확한 분해 분석으로 향후 연구의 핵심 과제이다.

### 7.4.4 Customer-friendly 표현 정제

본 연구의 fusion 메커니즘이 Customer clarity 약점(2.67)을 갖는다는 발견을 바탕으로, *prompt 엔지니어링을 통한 표현 정제* 연구가 가능하다. 구체적으로 다음과 같은 방향이 있다.

- SHAP 부호의 자연어화: "+/-" 대신 "부도 가능성을 높이는/낮추는"
- Agreement 라벨의 직관 표현화: "agreed" 대신 "두 분석 방법이 모두 본"
- 정성 표현의 도입: "매우 큰 영향(SHAP 0.95)" 등

다만 이러한 표현 변경은 NLI 사실성과의 trade-off를 야기할 수 있으므로, *사실성 손실 최소화 + 친화성 일부 향상* 의 하이브리드 접근이 필요하다.

### 7.4.5 평가 도구의 다중화

본 연구의 평가 신뢰성을 강화하기 위해 다음과 같은 다중화가 가능하다.

- **NLI 모델 ensemble**: mDeBERTa-v3 외에 KLUE-NLI, KoBERT-NLI 등 한국어 전용 NLI 모델 추가
- **LLM judge ensemble**: Anthropic + Google + OpenAI GPT-5 등 3개 이상 judge로 G-Eval 평가
- **Embedding-based sensitive leak**: G-Eval에 더해 임베딩 cosine 거리로 민감 정보 노출 추가 측정

### 7.4.6 보조 테이블의 활용 확장

본 연구의 Step 3-B에서는 bureau와 previous_application 두 테이블만 활용하여 AUROC +2.22% 향상을 확인하였다. Home Credit Default Risk 데이터셋에는 POS_CASH_balance, credit_card_balance, installments_payments, bureau_balance 등 추가 보조 테이블 4개가 더 존재하며, 이들의 활용은 SHAP 분석에서 *EXT_SOURCE 응축 가설* 의 직접 검증과 함께 본 연구의 예측 성능을 더 향상시킬 가능성이 있다.

## 7.5 종합 정리

본 연구의 학술적 메시지는 *"fusion이 모든 환경·모든 차원에서 1위"* 가 아니라, *"응용 시나리오 × 평가 차원 × 데이터 복잡도의 3차원 trade-off"* 의 정량 입증이다. 이러한 honest reporting은 본 연구가 학술 논문으로서 완결성을 갖추기 위한 핵심 자세이며, 또한 산업 적용 시점에 시스템 운영자가 응용 시나리오에 맞는 mode를 선택할 수 있는 근거로도 작용한다. 다음 제8장에서는 본 연구의 학술적·산업적 기여를 종합 결론으로 정리한다.
