"""Step 3-C-1-c / Step 3-C-2: Agreement-aware fusion 컨텍스트 → LLM 호출.

기존 src.llm_explainer (SHAP-only 컨텍스트 처리)의 _call_llm/make_client를
재사용하고, **프롬프트만 fusion 전용으로 새로 정의**.

차이점:
- 컨텍스트에 agreed_drivers / shap_only_drivers / attention_only_drivers 3그룹 존재
- 시스템 프롬프트에 두 해석 모델(SHAP, TabNet attention)의 의미와
  agreement-label 의미를 명시 → LLM이 그룹별 가중치를 인지
- few-shot 예시도 융합 컨텍스트 포맷

산출:
    results/explanations_fusion_{provider}_{n}/{idx}_{tag}.json
    results/explanations_fusion_{provider}_{n}/_index.json

사용:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.llm_explainer_fusion \
        --provider anthropic --n-samples 30
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List

from src.llm_explainer import _call_llm, make_client
from src.utils import RESULTS_DIR, SEED, set_seed

# ─────────────────────────────────────────────────────────────
# Fusion-aware Few-shot
# ─────────────────────────────────────────────────────────────
FUSION_FEW_SHOT = """
[예시: REJECT, fusion 컨텍스트]
입력 컨텍스트:
{
  "decision": "REJECT",
  "default_probability": 0.78,
  "model_predict": "XGBoost",
  "model_explain": ["SHAP_xgb_local", "TabNet_attention_local"],
  "agreed_drivers": [
    {"feature":"외부 신용평가 점수 3","value":"0.10","shap":0.95,"attention":0.21,
     "sign_for_default":"+","group":"agreed","rank":1},
    {"feature":"외부 신용평가 점수 2","value":"0.15","shap":0.62,"attention":0.18,
     "sign_for_default":"+","group":"agreed","rank":2}
  ],
  "shap_only_drivers": [
    {"feature":"재직 일수","value":"9년 재직","shap":-0.12,
     "sign_for_default":"-","group":"shap_only","rank":1}
  ],
  "attention_only_drivers": [
    {"feature":"신청 대출 유형: 리볼빙(한도 대출)","value":"예","attention":0.14,
     "group":"attention_only","rank":1}
  ]
}
출력 예시:
[결정 요약]
신청 결과: 대출 신청이 거절되었습니다(예측 부도 확률 78%).

[주요 거절 사유 — 두 해석 모델이 동의한 강한 신호]
1. 외부 신용평가 점수 3이 0.10으로 매우 낮게 평가되었습니다(두 모델 동의).
2. 외부 신용평가 점수 2도 0.15로 낮은 편입니다(두 모델 동의).

[보완적으로 고려된 요인]
- TabNet 어텐션은 신청 대출 유형(리볼빙)도 결정에 영향을 준 것으로 봤습니다.
- SHAP 관점에서 9년의 재직 기간은 부도 가능성을 낮추는 긍정 요인이었습니다.

[개선 권고]
- 외부 신용평가 점수 향상을 위한 신용 관리(연체 해소, 사용률 조정)를 권장합니다.
- 일정 기간 후 재신청을 검토하실 수 있습니다.

[면책 고지]
본 안내는 두 해석 모델(SHAP, TabNet 어텐션)의 결과를 자연어로 정리한 것이며, 최종 의사결정은 담당자의 검토를 거칩니다.
""".strip()


# ─────────────────────────────────────────────────────────────
# Fusion-aware Prompt
# ─────────────────────────────────────────────────────────────
FUSION_PROMPT_TEMPLATE = """당신은 고객에게 신용 평가 결과를 친절히 설명하는 금융 상담사입니다.

본 시스템은 **두 가지 해석 모델**의 결과를 융합한 컨텍스트를 제공합니다:
- **SHAP** (XGBoost 분류기 기반): 각 변수의 부도 확률에 대한 양/음의 기여도
- **TabNet attention** (정형 데이터 딥러닝 어텐션): 모델이 인스턴스마다 집중한 변수 (sparse, 부호 없음)

[그룹 라벨의 의미]
- **agreed_drivers**: 두 모델이 모두 중요시한 변수. 가장 신뢰할 만한 강한 신호.
- **shap_only_drivers**: SHAP top-k에만 있는 변수. 부호(sign_for_default)와 기여도 정보 보존.
- **attention_only_drivers**: TabNet 어텐션 top-k에만 있는 변수. 방향성(부호)은 알 수 없으나 모델이 본 변수.

[Hard Constraints]
- 아래 [컨텍스트] JSON에 명시된 변수, 값, SHAP 부호, attention 정보만 사용하세요.
- 컨텍스트에 없는 변수, 수치, 추론은 절대 생성하지 마세요.
- 의료적·법률적 자문이나 단정적 미래 예측을 하지 마세요.
- 민감 변수(성별, 연령, 인종, 종교, 출신 지역)를 직접 언급하지 마세요.
- agreed_drivers를 거절/승인의 주요 사유로 우선 사용하세요. shap_only/attention_only는 보완으로 표현하세요.
- SHAP 부호: "+" 면 부도 확률↑(거절 측), "-" 면 부도 확률↓(승인 측).
- attention_only에는 부호가 없으므로 "결정에 영향을 준 변수"로만 표현하고 방향성을 추측하지 마세요.

[출력 형식 — 반드시 이 5개 섹션 순서로]
[결정 요약]   (1줄)
[주요 거절/승인 사유 — 두 해석 모델이 동의한 강한 신호]   (agreed_drivers 기반, 최대 3개)
[보완적으로 고려된 요인]   (shap_only / attention_only 기반, 각 출처 명시, 합계 최대 4개)
[개선 권고]   (1~3개)
[면책 고지]   (1~2줄, 두 해석 모델을 사용했음을 명시)

[모범 예시]
{few_shot}

[컨텍스트 — 이 사실만 사용]
{context_json}

이제 위 컨텍스트에 대한 자연어 설명 리포트를 위 출력 형식 그대로 작성해주세요. 한국어로.
"""


def generate_one_fusion(client_tuple, context: dict, model: str) -> dict:
    ctx_for_prompt = {k: v for k, v in context.items() if not k.startswith("_meta")}
    prompt = FUSION_PROMPT_TEMPLATE.format(
        few_shot=FUSION_FEW_SHOT,
        context_json=json.dumps(ctx_for_prompt, ensure_ascii=False, indent=2),
    )
    t0 = time.time()
    text, usage = _call_llm(client_tuple, prompt, model=model)
    elapsed = time.time() - t0
    return {
        "sample_idx": context.get("sample_idx"),
        "decision": context.get("decision"),
        "true_label": context.get("_meta_true_label"),
        "provider": client_tuple[0],
        "model": model,
        "elapsed_sec": round(elapsed, 2),
        "explanation": text,
        "usage_metadata": usage,
        "context_sent": ctx_for_prompt,
        "context_type": "fusion",
        "n_agreed": context.get("n_agreed"),
        "n_shap_only": context.get("n_shap_only"),
        "n_attention_only": context.get("n_attention_only"),
    }


def run_fusion(provider: str = "anthropic", model: str | None = None,
                n_samples: int = 30, sleep_sec: float = 4.0,
                contexts_dir: Path | None = None,
                output_dir: Path | None = None,
                seed: int = SEED) -> List[Path]:
    set_seed(seed)
    contexts_dir = contexts_dir or (RESULTS_DIR / "contexts_fusion_100")
    if model is None:
        model = {"anthropic": "claude-sonnet-4-5",
                  "gemini": "gemini-2.5-flash"}[provider]
    output_dir = output_dir or (RESULTS_DIR /
        f"explanations_fusion_{provider}_{n_samples}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in contexts_dir.glob("*.json")
                     if p.name != "_index.json"])
    rnd = random.Random(seed)
    chosen = sorted(rnd.sample(files, min(n_samples, len(files))))
    print(f"[provider] {provider} / model={model}")
    print(f"[contexts] from {contexts_dir}, total {len(files)}, picked {len(chosen)}")
    print(f"[output] {output_dir}")

    client_tuple = make_client(provider=provider)
    out_paths = []
    for i, ctx_path in enumerate(chosen, start=1):
        with open(ctx_path, "r", encoding="utf-8") as f:
            ctx = json.load(f)
        print(f"[{i}/{len(chosen)}] {ctx_path.stem}  decision={ctx['decision']}  "
              f"prob={ctx['default_probability']:.3f}  "
              f"agreed={ctx.get('n_agreed')}, shap_only={ctx.get('n_shap_only')}, "
              f"att_only={ctx.get('n_attention_only')}")
        try:
            result = generate_one_fusion(client_tuple, ctx, model=model)
        except Exception as e:
            print(f"  ERROR: {e}")
            print("  60초 대기 후 재시도...")
            time.sleep(60)
            result = generate_one_fusion(client_tuple, ctx, model=model)

        out = output_dir / f"{ctx_path.stem}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        out_paths.append(out)
        print(f"  elapsed={result['elapsed_sec']}s, "
              f"tokens={result['usage_metadata'].get('total_token_count', '?')}")
        if i < len(chosen):
            time.sleep(sleep_sec)

    index = {
        "n_explanations": len(out_paths),
        "provider": provider,
        "model": model,
        "context_type": "fusion",
        "files": [p.name for p in out_paths],
        "selected_idx": [int(p.stem.split("_")[0]) for p in out_paths],
    }
    with open(output_dir / "_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return out_paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "gemini"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--sleep", type=float, default=4.0)
    ap.add_argument("--contexts-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    contexts_dir = Path(args.contexts_dir) if args.contexts_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    paths = run_fusion(provider=args.provider, model=args.model,
                        n_samples=args.n_samples, sleep_sec=args.sleep,
                        contexts_dir=contexts_dir, output_dir=output_dir)
    print(f"\n[OK] {len(paths)}개 fusion 설명 생성 완료")
