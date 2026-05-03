# LLM 호출 옵션 정리

## 전제: Claude Max 플랜은 직접 사용 불가

Claude Max ($100/$200/월)는:
- ✅ claude.ai 웹/데스크톱 앱 사용량 ↑
- ✅ Claude Code CLI 사용량 ↑
- ❌ **Anthropic API에 자동으로 크레딧이 붙지 않음** (별도 청구)
- ❌ Python `anthropic` SDK로 호출하려면 별도 API 키 필요

본 프로젝트는 SHAP→JSON→LLM 호출을 **자동화된 파이프라인**으로 평가해야 하므로, 코드에서 직접 API를 때리는 형태가 표준. Claude Max 구독을 유지한 상태에서 추가 옵션이 두 개 있어.

---

## 옵션 1. Anthropic API 키 별도 발급 ⭐ 권장

1. https://console.anthropic.com 접속 → Claude Max와 같은 이메일로 로그인 가능
2. **"Get API Keys"** → **"Create Key"**
3. **"Plans & Billing"** → 소액 충전 ($5~10이면 미팅용으로 충분)
   - 본 프로젝트 미팅용 사용량 추정: 5~10명 샘플 × 5번 변형 = 50회 호출, 토큰당 input ~500 + output ~300 → 약 $0.5~1.5 수준 (Claude Sonnet 기준)
4. 코드:
   ```python
   from anthropic import Anthropic
   client = Anthropic(api_key="sk-ant-...")  # 환경변수 ANTHROPIC_API_KEY 권장
   resp = client.messages.create(
       model="claude-sonnet-4-5",
       max_tokens=1000,
       messages=[{"role": "user", "content": prompt}],
   )
   ```

**장점:** 본 논문의 LLM 평가 챕터에서 Claude Sonnet (또는 Opus) 사용했다고 일관되게 쓸 수 있음. 미팅에서 "Claude API 사용"이 명료한 메시지.
**단점:** 소액이지만 별도 결제가 필요.

---

## 옵션 2. 무료/저비용 대체 API

미팅용으로 비용 0원이 필요하면:

| 제공자 | 모델 | 무료 여부 | 비고 |
|---|---|---|---|
| Google AI Studio | Gemini 2.0 Flash | 무료 tier (RPM 제한) | https://aistudio.google.com — API 키 즉시 발급 |
| Groq | Llama 3.3 70B, Llama 3.1 8B | 무료 tier (RPD 제한) | https://console.groq.com — 빠른 추론 속도 |
| OpenRouter | 다수 모델 | 무료 모델 일부 + 유료 | https://openrouter.ai |

가장 간단한 무료 경로: **Google AI Studio → Gemini 2.0 Flash**. API 키 1분이면 발급, 신용카드 불필요.

```python
# pip install google-genai
from google import genai
client = genai.Client(api_key="...")
resp = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
)
```

**장점:** 비용 0, 즉시 사용.
**단점:** 논문에서는 "본 연구 LLM은 Gemini로 진행"이라고 별도 명시. RPM 제한이 있어 batch 평가 시 느림(10 RPM 무료).

---

## 옵션 3. Claude Code CLI를 subprocess로 호출 (비권장)

기술적으로 가능하지만:
- 출력이 stdout 텍스트 파싱이라 불안정
- 평가 자동화에 적합하지 않음
- 논문 방법론 기술이 어색

이 옵션은 비권장.

---

## 권장 결정 흐름

- **돈 좀 써도 됨 + 깔끔한 논문 메시지 원함** → 옵션 1 (Anthropic API, $5~10)
- **0원 필수** → 옵션 2 (Gemini 2.0 Flash 무료)
- **혼합**: Day 6 LLM 평가만 옵션 2로 빠르게 끝내고, 논문 본 작업(미팅 후) 때 옵션 1로 전환

---

## 사용자에게 묻는 질문

1. 옵션 1, 2, 1+2(혼합) 중 어느 쪽?
2. 옵션 1이면 API 키를 발급해서 알려줄 수 있어?
   - 키는 환경변수 `ANTHROPIC_API_KEY` 또는 `D:\paper\.env`에 저장 권장. 절대 git에 커밋 금지.
3. 옵션 2면 Gemini API 키 발급 후 동일하게 환경변수 또는 .env로.
