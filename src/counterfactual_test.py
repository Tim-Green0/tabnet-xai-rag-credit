"""Step 2-A Phase 6 — Counterfactual Test 정량화.

목적 (계획서 3.8.1.d):
  SHAP 컨텍스트의 핵심 변수를 제거(masking)한 새 컨텍스트를 LLM에 주면,
  설명이 얼마나 달라지는가? 변화가 클수록 LLM이 SHAP에 진짜로 의존
  한다는 증거 — Faithfulness의 강한 검증.

설계:
  - 100명 중 30명 무작위 추출
  - 각 인스턴스에서 top_drivers_for_default rank 1 변수를 컨텍스트에서 제거
  - 동일 LLM에 변경된 컨텍스트로 재호출
  - 원본 출력과 변경 출력의 cosine similarity 측정 (다국어 임베딩)
  - 추가: ROUGE-L 보조 지표

해석:
  - cosine 1.0에 가까울수록: LLM이 컨텍스트 무시 (나쁨)
  - cosine 0.0~0.5: 컨텍스트에 따라 출력이 명확히 변함 (좋음)
  - 본 연구 가설: cosine 평균 ≤ 0.85, top driver 변경 시 출력 명확히 달라짐

산출:
  - results/explanations_counterfactual_{provider}/   (변경 컨텍스트 출력)
  - results/counterfactual_eval.csv
  - figures/24_counterfactual.png

실행:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.counterfactual_test
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from src.llm_explainer import generate_one, make_client, PROVIDER_DEFAULTS
from src.text_similarity import similarity_pairs, rouge_l
from src.utils import FIGURES_DIR, RESULTS_DIR, savefig

load_dotenv("D:/paper/.env", override=True)
sns.set_theme(style="whitegrid", context="notebook")


CONTEXTS100_DIR = RESULTS_DIR / "contexts_100"


def remove_top_driver(ctx: Dict) -> Tuple[Dict, str | None]:
    """top_drivers_for_default rank 1을 제거한 컨텍스트 반환."""
    ctx_cf = json.loads(json.dumps(ctx))  # deep copy
    drivers = ctx_cf.get("top_drivers_for_default", [])
    if not drivers:
        return ctx_cf, None
    removed = drivers[0]["feature"]
    ctx_cf["top_drivers_for_default"] = drivers[1:]
    # rank 재부여
    for k, d in enumerate(ctx_cf["top_drivers_for_default"], start=1):
        d["rank"] = k
    ctx_cf["_meta_counterfactual"] = {"removed_top_driver": removed}
    return ctx_cf, removed


def load_original(provider: str, sample_id: str) -> Dict | None:
    # explanations_{provider}_100 (Gemini는 explanations_gemini_100, Claude는 explanations_anthropic_100)
    fp = RESULTS_DIR / f"explanations_{provider}_100" / f"{sample_id}.json"
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    # 호환: 옛날 경로
    if provider == "anthropic":
        legacy = RESULTS_DIR / "explanations_anthropic_100" / f"{sample_id}.json"
        if legacy.exists():
            return json.loads(legacy.read_text(encoding="utf-8"))
    return None


def main(provider: str = "anthropic", n_samples: int = 30,
          sleep_sec: float = 1.5, seed: int = 42) -> None:
    random.seed(seed)

    print(f"[1/4] 컨텍스트 100개 로딩 ({CONTEXTS100_DIR})")
    ctx_files = sorted([p for p in CONTEXTS100_DIR.glob("*.json")
                          if p.name != "_index.json"])
    print(f"     총 {len(ctx_files)}개")

    chosen = random.sample(ctx_files, min(n_samples, len(ctx_files)))
    print(f"[2/4] 무작위 {len(chosen)}개 인스턴스 선택, top driver 제거")

    out_dir = RESULTS_DIR / f"explanations_counterfactual_{provider}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[3/4] LLM 재호출 (provider={provider})")
    client_tuple = make_client(provider)
    model = PROVIDER_DEFAULTS[provider]["model"]

    rows = []
    pairs = []  # (orig_text, cf_text) 모음 — 일괄 임베딩
    for i, fp in enumerate(chosen, start=1):
        sid = fp.stem
        ctx_orig = json.loads(fp.read_text(encoding="utf-8"))

        # 원본 출력 로딩 (이미 생성된 100건에서)
        orig = load_original(provider, sid)
        if orig is None or "explanation" not in orig:
            print(f"  [{i}/{len(chosen)}] {sid} — 원본 출력 없음. skip")
            continue
        orig_text = orig["explanation"]

        # CF 컨텍스트 생성
        ctx_cf, removed = remove_top_driver(ctx_orig)
        if removed is None:
            continue

        # CF 호출
        try:
            t0 = time.time()
            cf_res = generate_one(client_tuple, ctx_cf, model=model)
        except Exception as e:
            print(f"  [{i}/{len(chosen)}] {sid} ERROR: {e}")
            time.sleep(60)
            cf_res = generate_one(client_tuple, ctx_cf, model=model)

        cf_text = cf_res["explanation"]
        elapsed = cf_res["elapsed_sec"]

        # 저장
        with open(out_dir / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump({
                "sample_id": sid,
                "removed_top_driver": removed,
                "original_explanation": orig_text,
                "counterfactual_explanation": cf_text,
                "model": model,
                "elapsed_sec": elapsed,
            }, f, indent=2, ensure_ascii=False)

        # ROUGE-L 즉시 계산
        rl = rouge_l(orig_text, cf_text)
        rows.append({"sample_id": sid, "removed_top_driver": removed,
                       "elapsed_sec": elapsed,
                       "rouge_l": rl,
                       "cosine_sim": np.nan})  # 일괄 채움
        pairs.append((orig_text, cf_text))

        print(f"  [{i}/{len(chosen)}] {sid}  removed={removed[:30]}  "
              f"elapsed={elapsed}s, ROUGE-L={rl:.3f}")
        time.sleep(sleep_sec)

    print(f"\n[4/4] 임베딩 cosine 일괄 계산")
    cosines = similarity_pairs(pairs)
    for k, c in enumerate(cosines):
        rows[k]["cosine_sim"] = c

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"counterfactual_eval_{provider}.csv", index=False)

    print("\n[Counterfactual 결과]")
    print(f"  cosine_sim   mean={df['cosine_sim'].mean():.4f}  std={df['cosine_sim'].std():.4f}")
    print(f"  ROUGE-L      mean={df['rouge_l'].mean():.4f}  std={df['rouge_l'].std():.4f}")
    print(f"  → 낮을수록 LLM이 컨텍스트에 강하게 의존 (Faithfulness 강함)")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df["cosine_sim"], bins=15, ax=axes[0], color="#4C72B0",
                  edgecolor="black")
    axes[0].axvline(df["cosine_sim"].mean(), color="red", linestyle="--",
                     label=f"mean={df['cosine_sim'].mean():.3f}")
    axes[0].set_xlabel("Cosine similarity (original vs counterfactual)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Counterfactual cosine — {provider}\n"
                        f"(lower = LLM follows context more)")
    axes[0].legend()

    sns.histplot(df["rouge_l"], bins=15, ax=axes[1], color="#DD8452",
                  edgecolor="black")
    axes[1].axvline(df["rouge_l"].mean(), color="red", linestyle="--",
                     label=f"mean={df['rouge_l'].mean():.3f}")
    axes[1].set_xlabel("ROUGE-L F1 (lexical overlap)")
    axes[1].set_title(f"Counterfactual ROUGE-L — {provider}")
    axes[1].legend()

    plt.suptitle(f"Counterfactual Test ({len(df)} samples) — "
                  f"top driver removed → LLM re-call")
    fname = f"24_counterfactual_{provider}"
    savefig(fig, fname)
    print(f"\n[OK] {RESULTS_DIR / f'counterfactual_eval_{provider}.csv'}")
    print(f"     figures/{fname}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic",
                    choices=["gemini", "anthropic"])
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--sleep", type=float, default=1.5)
    args = ap.parse_args()
    main(provider=args.provider, n_samples=args.n_samples,
          sleep_sec=args.sleep)
