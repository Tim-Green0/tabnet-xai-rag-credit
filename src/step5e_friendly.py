"""Step 5-E: Customer-friendly fusion explanation.

동기 (Step 5-C+5-D 일관 약점):
  - Step 5-C Customer persona clarity: fusion 2.67 (vs generic_rag 4.93)
  - Step 5-D G-Eval Factual: fusion 3.97/2.97 (vs generic_rag 5.0/5.0)
  → 두 차원 모두 fusion의 SHAP 부호 raw + agreement 라벨이 over-explain으로 평가됨.

목표:
  Fusion의 사실성(NLI 0.71) 우월성은 유지하면서, customer clarity + LLM judge
  factual을 개선. 같은 fusion context를 그대로 사용하되 prompt만 친근하게 변경.

새 mode: `fusion_friendly`
  - SHAP 부호 자연어화: "+/-" → "부도 가능성을 높이는/낮추는"
  - Agreement 라벨 직관 표현: "agreed_drivers" → "두 분석 방법이 모두 동의한 강한 신호"
  - 정성 표현 추가: "매우 낮은 0.10" 식
  - attention_only는 "추가 참고 정보"로 부드럽게

대상:
  - Home Credit: Step 3-C-1 fusion 30 인스턴스 (results/contexts_fusion_100/ 중 30)
  - German Credit: Step 5-D fusion 30 인스턴스 (results/contexts_german_fusion_30/)

산출:
  results/explanations_friendly_{home,german}_{anthropic,gemini}_30/
  results/step5e_friendly_eval.csv (NLI + value_match + G-Eval)
  results/step5e_summary.md
  figures/42_friendly_vs_fusion.png

실행:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.step5e_friendly \\
        --dataset home --provider anthropic --n-samples 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from src.llm_explainer import _call_llm, make_client, PROVIDER_DEFAULTS
from src.utils import PROJECT_ROOT, RESULTS_DIR, SEED, set_seed


# ─────────────────────────────────────────────────────────────
# Friendly prompt (mode-agnostic, fusion context 그대로 사용)
# ─────────────────────────────────────────────────────────────
FRIENDLY_PROMPT = """당신은 고객에게 신용 평가 결과를 **친근하고 명료하게** 설명하는 금융 상담사입니다.

본 시스템은 XGBoost(SHAP)와 TabNet(어텐션) 두 가지 분석 방법을 함께 사용합니다.
[컨텍스트]에는 두 방법의 결과가 다음 세 그룹으로 분류되어 있습니다:
- **agreed_drivers**: 두 방법 모두 강하게 주목한 신호 (가장 신뢰할 수 있음)
- **shap_only_drivers**: SHAP 분석에서 추가 주목한 신호
- **attention_only_drivers**: TabNet 어텐션이 추가 주목한 신호 (참고 정보)

[Hard Constraints — 반드시 준수]
- 변수명·수치·sign_for_default(부호)만 [컨텍스트] JSON에서 인용. 컨텍스트에 없는 변수, 수치 절대 생성 금지
- 의료·법률 자문, 단정적 미래 예측, 특정 금융 상품 추천 금지
- 민감 변수(나이·성별·외국인 여부 등 보호 속성) 직접 언급 금지

[★ Customer-friendly 표현 규칙 — 이 규칙이 본 모드의 핵심]
1. **부호 자연어화**: sign_for_default가 '+'면 "**부도 가능성을 높이는** 요인",
                      '-'이면 "**부도 가능성을 낮추는** 요인"으로 표현하라.
                      "+/-" 또는 "양수/음수" 같은 raw 표현 사용 금지.
2. **Agreement 자연어화**: agreed_drivers는 "**두 분석 방법(SHAP과 TabNet 어텐션)이
                            모두 동의한** 강한 신호"로 도입.
                            shap_only는 "**SHAP 분석이 추가로 주목한**" 신호로,
                            attention_only는 "**TabNet 어텐션이 추가로 참고한**" 정보로 표현.
                            "agreed", "shap_only" 등 라벨 raw 노출 금지.
3. **정성 표현 추가**: 수치는 정확히 인용하되, 매우 큰/작은 값은 한국어 정성 표현 추가.
                        예) shap=0.95 → "매우 큰 영향(SHAP 0.95)",
                            value=0.10 → "매우 낮은 0.10".
4. **친근한 톤**: 고객에게 직접 말하듯 작성. "고객님께서는", "안내드립니다" 등.
                   전문 용어는 풀어서 설명.
5. **숫자 강박 회피**: SHAP 점수 자체를 강조하지 말고, 그 의미("얼마나 큰 영향인지")를 설명.

[출력 형식 — 반드시 이 5개 섹션 순서로]
[결정 요약]   (1줄)
[주요 거절 사유]   (REJECT일 때만, 최대 3개, agreed → shap_only 우선순위로)
[긍정적으로 평가된 요인]   (sign='-' 인 항목 위주, 최대 3개)
[개선 권고]   (1~3개)
[면책 고지]   (1~2줄)

[컨텍스트 — 이 사실만 사용]
{context_json}

이제 위 컨텍스트를 customer-friendly 표현 규칙에 따라 자연어로 풀어내, 5개 섹션 그대로 한국어로 작성하세요.
"""


# ─────────────────────────────────────────────────────────────
# 데이터셋별 fusion context 디렉토리 매핑
# ─────────────────────────────────────────────────────────────
DATASETS = {
    "home": {
        "contexts_dir": RESULTS_DIR / "contexts_fusion_100",
        "selected_idx_source": "explanations_fusion_anthropic_30",  # 동일 30 idx
    },
    "german": {
        "contexts_dir": RESULTS_DIR / "contexts_german_fusion_30",
        "selected_idx_source": None,  # 해당 디렉토리 자체가 30개
    },
}


def load_fusion_contexts(dataset: str, n_samples: int = 30) -> List[Dict]:
    """fusion 30 인스턴스 컨텍스트 로딩."""
    cfg = DATASETS[dataset]
    cdir = cfg["contexts_dir"]
    if not cdir.exists():
        raise FileNotFoundError(f"{cdir} 없음. Step 3-C-1 또는 5-D 먼저 실행.")

    if cfg["selected_idx_source"]:
        # Home Credit: explanations_fusion_anthropic_30 의 idx 따라
        idx_dir = RESULTS_DIR / cfg["selected_idx_source"]
        with open(idx_dir / "_index.json", "r", encoding="utf-8") as f:
            selected = set(json.load(f).get("selected_idx", []))
        files = sorted([p for p in cdir.glob("*.json")
                         if p.name != "_index.json"
                         and int(p.stem.split("_")[0]) in selected])[:n_samples]
    else:
        # German: 디렉토리의 모든 파일
        files = sorted([p for p in cdir.glob("*.json")
                         if p.name != "_index.json"])[:n_samples]

    contexts = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            ctx = json.load(f)
        ctx["_source_file"] = fp.name
        contexts.append(ctx)
    return contexts


def make_friendly_prompt(ctx: Dict) -> str:
    ctx_clean = {k: v for k, v in ctx.items() if not k.startswith("_")}
    return FRIENDLY_PROMPT.format(
        context_json=json.dumps(ctx_clean, ensure_ascii=False, indent=2)
    )


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main(dataset: str, provider: str, n_samples: int = 30,
         sleep_sec: float = 3.0) -> None:
    set_seed(SEED)
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")

    print(f"[1/2] {dataset} fusion contexts 로딩")
    contexts = load_fusion_contexts(dataset, n_samples)
    print(f"  {len(contexts)}개 컨텍스트 준비")

    output_dir = RESULTS_DIR / f"explanations_friendly_{dataset}_{provider}_{n_samples}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/2] LLM 호출 (provider={provider}, mode=fusion_friendly)")
    model = PROVIDER_DEFAULTS[provider]["model"]
    client_tuple = make_client(provider)
    print(f"  model={model}, output={output_dir}")

    out_paths = []
    for i, ctx in enumerate(contexts, start=1):
        sample_idx = ctx.get("sample_idx")
        decision = ctx.get("decision")
        prompt = make_friendly_prompt(ctx)
        print(f"[{i}/{len(contexts)}] idx={sample_idx}  decision={decision}  prob={ctx.get('default_probability', 0):.3f}")
        for retry in range(3):
            try:
                t0 = time.time()
                text, usage = _call_llm(client_tuple, prompt, model=model)
                elapsed = time.time() - t0
                break
            except Exception as e:
                wait = 30 * (retry + 1)
                print(f"  ERROR: {str(e)[:120]}, {wait}s 대기 후 재시도 {retry+1}/3")
                time.sleep(wait)
        else:
            print(f"  3회 실패 — skip")
            continue

        result = {
            "sample_idx": sample_idx,
            "decision": decision,
            "true_label": ctx.get("_meta_true_label"),
            "provider": provider, "model": model, "mode": "fusion_friendly",
            "dataset": dataset,
            "elapsed_sec": round(elapsed, 2),
            "explanation": text,
            "usage_metadata": usage,
            "context_sent": {k: v for k, v in ctx.items() if not k.startswith("_")},
            "context_type": f"{dataset}_fusion_friendly",
        }
        # 파일명: idx_tag.json (tag는 source file에서 추출)
        src = ctx.get("_source_file", f"{sample_idx}.json")
        out_name = src  # 동일 이름으로 저장
        out = output_dir / out_name
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        out_paths.append(out)
        tot = usage.get("total_token_count", "?")
        print(f"  elapsed={elapsed:.1f}s, tokens={tot}")
        if i < len(contexts):
            time.sleep(sleep_sec)

    with open(output_dir / "_index.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_explanations": len(out_paths),
            "provider": provider, "model": model, "mode": "fusion_friendly",
            "dataset": dataset,
            "selected_idx": [int(p.stem.split("_")[0]) for p in out_paths],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[완료] {len(out_paths)}개 friendly 설명 → {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["home", "german"])
    ap.add_argument("--provider", required=True, choices=["anthropic", "gemini"])
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--sleep", type=float, default=3.0)
    args = ap.parse_args()
    main(dataset=args.dataset, provider=args.provider,
         n_samples=args.n_samples, sleep_sec=args.sleep)
