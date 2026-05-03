"""Day 6 — XAI-RAG LLM 설명 생성 (Gemini 2.5 Flash).

워크플로:
  1) results/contexts/*.json (Day 6 context_builder 산출물) 로딩
  2) 프롬프트 템플릿(Role + Hard Constraints + Context + Output Schema + Few-shot)에 주입
  3) Gemini API 호출 → 자연어 설명 생성
  4) 결과 + 메타데이터를 results/explanations/*.json 으로 저장

프롬프트 정책 (계획서 3.7):
  - Role: 친절한 금융 상담사
  - Hard Constraints:
    * 컨텍스트에 없는 변수·수치 절대 생성 금지
    * 의료/법률 자문, 단정적 미래 예측 금지
    * 민감 변수(성별, 연령, 인종, 종교) 직접 언급 금지
  - Output Schema: 결정 → 거절 사유 Top 3 → 긍정 요인 → 개선 권고 → 면책 고지

사용:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.llm_explainer \
        [--model gemini-2.5-flash] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from src.utils import RESULTS_DIR

load_dotenv("D:/paper/.env", override=True)

CONTEXTS_DIR = RESULTS_DIR / "contexts"
# 기본 출력 디렉토리 — provider별 분리 가능
EXPLANATIONS_DIR_DEFAULT = RESULTS_DIR / "explanations"
EXPLANATIONS_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)

PROVIDER_DEFAULTS = {
    "gemini":   {"model": "gemini-2.5-flash",       "output_subdir": "explanations"},
    "anthropic": {"model": "claude-sonnet-4-5",      "output_subdir": "explanations_anthropic"},
}


# ─────────────────────────────────────────────────────────────
# Few-shot 예시 (LLM 출력 톤·구조 가이드)
# ─────────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = """
[예시 1: REJECT]
입력 컨텍스트:
{
  "decision": "REJECT",
  "default_probability": 0.78,
  "threshold": 0.5,
  "top_drivers_for_default": [
    {"feature":"외부 신용평가 점수 3","value":"0.10","shap":0.95,"rank":1},
    {"feature":"외부 신용평가 점수 2","value":"0.15","shap":0.62,"rank":2}
  ],
  "top_drivers_against_default": [
    {"feature":"재직 일수","value":"9년 재직","shap":-0.12,"rank":1}
  ]
}
출력 예시:
[결정 요약]
신청 결과: 대출 신청이 거절되었습니다(예측 부도 확률 78%).

[주요 거절 사유]
1. 외부 신용평가 점수 3이 0.10으로 매우 낮게 평가되었습니다.
2. 외부 신용평가 점수 2도 0.15로 낮은 편입니다.

[긍정적으로 평가된 요인]
- 9년의 안정적인 재직 기간은 긍정적으로 반영되었습니다.

[개선 권고]
- 외부 신용평가 점수 향상을 위한 신용 관리(연체 해소, 사용률 조정)를 권장합니다.
- 일정 기간 후 재신청을 검토하실 수 있습니다.

[면책 고지]
본 안내는 모델의 자동화된 평가 결과를 자연어로 정리한 것이며, 최종 의사결정은 담당자의 검토를 거칩니다. 미래 신청 결과를 보장하지 않습니다.
""".strip()


PROMPT_TEMPLATE = """당신은 고객에게 신용 평가 결과를 친절히 설명하는 금융 상담사입니다.

[Hard Constraints]
- 아래 [컨텍스트] JSON에 명시된 변수, 값, SHAP 부호만 사용하여 설명을 작성하세요.
- 컨텍스트에 없는 변수, 수치, 추론은 절대 생성하지 마세요.
- 의료적·법률적 자문이나 단정적 미래 예측을 하지 마세요.
- 민감 변수(성별, 연령, 인종, 종교, 출신 지역)를 직접 언급하지 마세요.
- 거절 사유는 컨텍스트의 top_drivers_for_default에서, 긍정 요인은 top_drivers_against_default에서만 선택하세요.
- SHAP 부호를 정확히 반영하세요: 양수=부도 가능성↑, 음수=부도 가능성↓.

[출력 형식 — 반드시 이 5개 섹션 순서로]
[결정 요약]   (1줄)
[주요 거절 사유]   (REJECT일 때만, 최대 3개)
[긍정적으로 평가된 요인]   (top_drivers_against_default 기반, 최대 3개)
[개선 권고]   (1~3개)
[면책 고지]   (1~2줄)

[모범 예시]
{few_shot}

[컨텍스트 — 이 사실만 사용]
{context_json}

이제 위 컨텍스트에 대한 자연어 설명 리포트를 위 출력 형식 그대로 작성해주세요. 한국어로.
"""


# ─────────────────────────────────────────────────────────────
# Gemini 클라이언트
# ─────────────────────────────────────────────────────────────
def make_client(provider: str = "gemini"):
    import os
    if provider == "gemini":
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY가 .env에 없습니다")
        return ("gemini", genai.Client(api_key=api_key))
    elif provider == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY가 .env에 없습니다")
        return ("anthropic", anthropic.Anthropic(api_key=api_key))
    else:
        raise ValueError(f"unknown provider: {provider}")


def _call_llm(client_tuple, prompt: str, model: str,
                max_tokens: int = 1500) -> Tuple[str, Dict]:
    """provider 추상화. (text, usage_dict) 반환."""
    provider, client = client_tuple
    if provider == "gemini":
        resp = client.models.generate_content(model=model, contents=prompt)
        text = resp.text
        usage = {
            "prompt_token_count": getattr(resp.usage_metadata, "prompt_token_count", None),
            "candidates_token_count": getattr(resp.usage_metadata, "candidates_token_count", None),
            "total_token_count": getattr(resp.usage_metadata, "total_token_count", None),
        } if getattr(resp, "usage_metadata", None) else {}
    elif provider == "anthropic":
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "total_token_count": resp.usage.input_tokens + resp.usage.output_tokens,
        }
    else:
        raise ValueError(f"unknown provider: {provider}")
    return text, usage


def generate_one(client_tuple, context: Dict, model: str) -> Dict:
    """단일 컨텍스트 → 자연어 설명. provider 자동 추상화."""
    ctx_for_prompt = {k: v for k, v in context.items() if not k.startswith("_meta")}
    prompt = PROMPT_TEMPLATE.format(
        few_shot=FEW_SHOT_EXAMPLES,
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
    }


# ─────────────────────────────────────────────────────────────
# 일괄 처리
# ─────────────────────────────────────────────────────────────
def run_all(provider: str = "gemini", model: str | None = None,
             sleep_sec: float = 4.0, dry_run: bool = False,
             output_dir: Path | None = None) -> List[Path]:
    if model is None:
        model = PROVIDER_DEFAULTS[provider]["model"]
    if output_dir is None:
        output_dir = RESULTS_DIR / PROVIDER_DEFAULTS[provider]["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in CONTEXTS_DIR.glob("*.json")
                     if p.name != "_index.json"])
    if dry_run:
        files = files[:1]
        print(f"[dry-run] 첫 1개만 처리: {files[0].name}")

    print(f"[provider] {provider} / model={model} / output={output_dir}")
    client_tuple = make_client(provider=provider)
    out_paths = []
    for i, ctx_path in enumerate(files, start=1):
        with open(ctx_path, "r", encoding="utf-8") as f:
            ctx = json.load(f)
        print(f"[{i}/{len(files)}] {ctx_path.stem}  decision={ctx['decision']}  "
              f"true_label={ctx.get('_meta_true_label')}  prob={ctx['default_probability']:.3f}")
        try:
            result = generate_one(client_tuple, ctx, model=model)
        except Exception as e:
            print(f"  ERROR: {e}")
            print("  60초 대기 후 재시도...")
            time.sleep(60)
            result = generate_one(client_tuple, ctx, model=model)

        out = output_dir / f"{ctx_path.stem}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        out_paths.append(out)
        print(f"  elapsed={result['elapsed_sec']}s, "
              f"tokens={result['usage_metadata'].get('total_token_count', '?')}")
        if i < len(files):
            time.sleep(sleep_sec)

    index = {
        "n_explanations": len(out_paths),
        "provider": provider,
        "model": model,
        "files": [p.name for p in out_paths],
    }
    with open(output_dir / "_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    return out_paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini",
                    choices=list(PROVIDER_DEFAULTS.keys()))
    ap.add_argument("--model", default=None,
                    help="provider별 기본 모델 사용. 명시 시 override.")
    ap.add_argument("--output-dir", default=None,
                    help="기본은 results/explanations(_anthropic).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=4.0,
                    help="호출 간 대기 시간(초). free tier RPM 회피.")
    args = ap.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    paths = run_all(provider=args.provider, model=args.model,
                     sleep_sec=args.sleep, dry_run=args.dry_run,
                     output_dir=output_dir)
    print(f"\n[OK] {len(paths)}개 설명 생성 완료 → {paths[0].parent if paths else '?'}")
