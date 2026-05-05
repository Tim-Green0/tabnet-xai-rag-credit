"""Step 2-A Phase 7 — Robustness 평가 (계획서 3.8.5).

목적:
  동일 컨텍스트에 대해 프롬프트를 미세 변형했을 때 LLM 출력이 얼마나
  일관되게 유지되는가를 측정. 일관성이 높을수록 본 시스템은 안정적.

변형 종류 (3가지):
  A. role_swap     — Role 문장 변형 ("당신은 금융 상담사" → "당신은 신용 평가 전문가")
  B. example_swap  — Few-shot 예시 위치 셔플 (예시를 컨텍스트 앞/뒤로 이동)
  C. driver_shuffle — 컨텍스트의 driver 순서를 SHAP 절대값 동률 내에서 셔플
                       (rank 정보는 유지)

프로토콜:
  - 100명 중 20명 무작위 선택
  - 각 인스턴스마다 원본 + 3개 변형 = 4번 호출
  - 4쌍 (orig vs A/B/C) cosine 유사도 + ROUGE-L 측정
  - 변형 간 일관성: cosine 평균 ≥ 0.85 목표

산출:
  - results/explanations_robustness_{provider}/{sid}_{variant}.json
  - results/robustness_eval.csv
  - figures/25_robustness.png
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from src.llm_explainer import (
    FEW_SHOT_EXAMPLES,
    PROMPT_TEMPLATE,
    PROVIDER_DEFAULTS,
    _call_llm,
    make_client,
)
from src.text_similarity import rouge_l, similarity_pairs
from src.utils import FIGURES_DIR, RESULTS_DIR, savefig

load_dotenv("D:/paper/.env", override=True)
sns.set_theme(style="whitegrid", context="notebook")


CONTEXTS100_DIR = RESULTS_DIR / "contexts_100"

ROLE_VARIANTS = {
    "orig": "당신은 고객에게 신용 평가 결과를 친절히 설명하는 금융 상담사입니다.",
    "role_swap": "당신은 신용 평가 결과를 고객 관점에서 풀어 설명하는 신용 평가 전문가입니다.",
}


# ─────────────────────────────────────────────────────────────
# 변형
# ─────────────────────────────────────────────────────────────
def variant_role_swap(prompt: str) -> str:
    return prompt.replace(ROLE_VARIANTS["orig"], ROLE_VARIANTS["role_swap"], 1)


def variant_example_swap(prompt: str, ctx: dict) -> str:
    """[모범 예시] 블록을 [컨텍스트] 블록 뒤로 이동."""
    parts = prompt.split("[모범 예시]")
    if len(parts) != 2:
        return prompt
    head = parts[0]
    rest = parts[1]
    # rest = few_shot 본문 + "[컨텍스트 — 이 사실만 사용]" + ctx + "이제..."
    if "[컨텍스트 — 이 사실만 사용]" not in rest:
        return prompt
    fs_body, ctx_part = rest.split("[컨텍스트 — 이 사실만 사용]", 1)
    new_prompt = (head
                    + "[컨텍스트 — 이 사실만 사용]" + ctx_part
                    + "\n[모범 예시]" + fs_body)
    return new_prompt


def variant_driver_shuffle(ctx: dict, seed: int) -> dict:
    """drivers 순서를 SHAP 절대값 동률 내에서 약하게 셔플 (rank는 그대로 유지).

    완전한 random shuffle보다 동률 그룹 내 섞기로 SHAP 의미 보존.
    여기서는 단순히 같은 그룹(for/against) 내에서 위치만 약간 swap.
    """
    rng = random.Random(seed)
    new_ctx = deepcopy(ctx)
    for key in ("top_drivers_for_default", "top_drivers_against_default"):
        ds = new_ctx.get(key, [])
        if len(ds) >= 2:
            # rank 메타만 유지하고 list 순서만 변경
            ds_copy = ds[:]
            rng.shuffle(ds_copy)
            new_ctx[key] = ds_copy
    return new_ctx


def build_prompt(ctx: dict, variant: str, seed: int = 42) -> str:
    """변형 프롬프트 빌드."""
    ctx_for_prompt = {k: v for k, v in ctx.items() if not k.startswith("_meta")}
    if variant == "driver_shuffle":
        ctx_for_prompt = variant_driver_shuffle(ctx_for_prompt, seed)

    prompt = PROMPT_TEMPLATE.format(
        few_shot=FEW_SHOT_EXAMPLES,
        context_json=json.dumps(ctx_for_prompt, ensure_ascii=False, indent=2),
    )
    if variant == "role_swap":
        prompt = variant_role_swap(prompt)
    elif variant == "example_swap":
        prompt = variant_example_swap(prompt, ctx_for_prompt)
    return prompt


def main(provider: str = "anthropic", n_samples: int = 20,
          sleep_sec: float = 1.5, seed: int = 42) -> None:
    random.seed(seed)

    print(f"[1/4] 컨텍스트 로딩 ({CONTEXTS100_DIR})")
    ctx_files = sorted([p for p in CONTEXTS100_DIR.glob("*.json")
                          if p.name != "_index.json"])
    chosen = random.sample(ctx_files, min(n_samples, len(ctx_files)))
    print(f"     무작위 {len(chosen)}개 인스턴스 선택")

    print(f"[2/4] 4가지 프롬프트로 LLM 호출 (provider={provider})")
    client_tuple = make_client(provider)
    model = PROVIDER_DEFAULTS[provider]["model"]
    out_dir = RESULTS_DIR / f"explanations_robustness_{provider}"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ["orig", "role_swap", "example_swap", "driver_shuffle"]
    rows = []
    pair_buf = []  # (orig_text, var_text, sample_id, variant) 임시
    for i, fp in enumerate(chosen, start=1):
        sid = fp.stem
        ctx = json.loads(fp.read_text(encoding="utf-8"))
        outputs = {}
        for v in variants:
            prompt = build_prompt(ctx, v, seed=seed)
            try:
                t0 = time.time()
                text, usage = _call_llm(client_tuple, prompt, model=model)
                elapsed = time.time() - t0
            except Exception as e:
                print(f"  [{i}/{len(chosen)}] {sid}/{v} ERROR: {e}")
                time.sleep(60)
                text, usage = _call_llm(client_tuple, prompt, model=model)
                elapsed = -1
            outputs[v] = {"text": text, "elapsed_sec": round(elapsed, 2),
                            "usage": usage}
            time.sleep(sleep_sec)

        # 저장
        with open(out_dir / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump({"sample_id": sid, "model": model,
                         "outputs": outputs}, f, indent=2,
                        ensure_ascii=False)

        # ROUGE-L 즉시 계산
        orig_text = outputs["orig"]["text"]
        for v in ["role_swap", "example_swap", "driver_shuffle"]:
            var_text = outputs[v]["text"]
            rl = rouge_l(orig_text, var_text)
            rows.append({"sample_id": sid, "variant": v,
                            "rouge_l": rl, "cosine_sim": np.nan,
                            "orig_elapsed": outputs["orig"]["elapsed_sec"],
                            "var_elapsed": outputs[v]["elapsed_sec"]})
            pair_buf.append((orig_text, var_text))

        print(f"  [{i}/{len(chosen)}] {sid}  ROUGE-L "
                f"role={rows[-3]['rouge_l']:.2f}  "
                f"ex={rows[-2]['rouge_l']:.2f}  "
                f"shuf={rows[-1]['rouge_l']:.2f}")

    print(f"\n[3/4] 임베딩 cosine 일괄 계산 ({len(pair_buf)} pairs)")
    cosines = similarity_pairs(pair_buf)
    for k, c in enumerate(cosines):
        rows[k]["cosine_sim"] = c

    print(f"[4/4] 결과 저장 + 시각화")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"robustness_eval_{provider}.csv", index=False)

    print("\n[Robustness 결과 — variant 별]")
    print(df.groupby("variant")[["cosine_sim", "rouge_l"]].agg(
        ["mean", "std"]).round(4).to_string())

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=df, x="variant", y="cosine_sim", ax=axes[0],
                  palette="Set2")
    sns.stripplot(data=df, x="variant", y="cosine_sim", ax=axes[0],
                    color="black", size=3, alpha=0.6)
    axes[0].set_title("Cosine similarity (orig vs variant)")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.85, color="red", linestyle="--", alpha=0.5,
                     label="target ≥ 0.85")
    axes[0].legend()

    sns.boxplot(data=df, x="variant", y="rouge_l", ax=axes[1],
                  palette="Set2")
    sns.stripplot(data=df, x="variant", y="rouge_l", ax=axes[1],
                    color="black", size=3, alpha=0.6)
    axes[1].set_title("ROUGE-L F1")
    axes[1].set_ylim(0, 1.05)

    plt.suptitle(f"Robustness Test — {provider}, "
                  f"{df['sample_id'].nunique()} samples × 3 variants")
    fname = f"25_robustness_{provider}"
    savefig(fig, fname)

    print(f"\n[OK] results/robustness_eval_{provider}.csv")
    print(f"     figures/{fname}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic",
                    choices=["gemini", "anthropic"])
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=1.5)
    args = ap.parse_args()
    main(provider=args.provider, n_samples=args.n_samples,
          sleep_sec=args.sleep)
