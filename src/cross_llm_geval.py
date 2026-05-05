"""Step 2-A Phase 5 — Cross-LLM G-Eval (self-bias 우회).

기존 평가는 Gemini가 자기 출력을 평가했음 (self-judge → self-bias 위험).
이번에는 LLM-A가 LLM-B의 출력을 평가:
  - Claude judge → Gemini 100건 평가
  - Gemini judge → Claude 100건 평가

산출:
  - results/eval_geval_cross_{judge}_judges_{target}.json
  - results/explanation_eval_summary_cross.json
  - figures/26_cross_llm_geval.png
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from src.eval_explanation import G_EVAL_RUBRIC
from src.llm_explainer import PROVIDER_DEFAULTS, _call_llm, make_client
from src.utils import FIGURES_DIR, RESULTS_DIR, savefig

load_dotenv("D:/paper/.env", override=True)
sns.set_theme(style="whitegrid", context="notebook")


def parse_score_json(text: str) -> Dict:
    """LLM 응답 텍스트에서 JSON 추출."""
    m = re.search(r"\{[^{}]*\"factual_accuracy\".*?\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"raw": text, "parse_error": True}
    return {"raw": text, "parse_error": True}


def judge_one(client_tuple, model: str, explanation: str, context: dict) -> Dict:
    ctx_for_prompt = {k: v for k, v in context.items() if not k.startswith("_meta")}
    prompt = G_EVAL_RUBRIC.format(
        context_json=json.dumps(ctx_for_prompt, ensure_ascii=False, indent=2),
        explanation_text=explanation,
    )
    text, usage = _call_llm(client_tuple, prompt, model=model)
    return {"raw": text, "parsed": parse_score_json(text), "usage": usage}


def evaluate_cross(judge_provider: str, target_provider: str,
                     n_samples: int = 30, sleep_sec: float = 5.0) -> pd.DataFrame:
    """judge_provider가 target_provider의 explanations_*_100을 평가."""
    target_dir = RESULTS_DIR / f"explanations_{target_provider}_100"
    files = sorted([p for p in target_dir.glob("*.json")
                     if p.name != "_index.json"])[:n_samples]

    judge_client = make_client(judge_provider)
    judge_model = PROVIDER_DEFAULTS[judge_provider]["model"]

    print(f"[judge={judge_provider}] target={target_provider}, n={len(files)}")
    rows = []
    raw_dump = {}
    for i, fp in enumerate(files, start=1):
        ex = json.loads(fp.read_text(encoding="utf-8"))
        sid = fp.stem
        try:
            t0 = time.time()
            judged = judge_one(judge_client, judge_model,
                                  ex["explanation"], ex["context_sent"])
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  [{i}/{len(files)}] {sid} ERROR: {str(e)[:120]}")
            time.sleep(60)
            try:
                judged = judge_one(judge_client, judge_model,
                                      ex["explanation"], ex["context_sent"])
            except Exception:
                rows.append({"sample_id": sid,
                                "judge": judge_provider,
                                "target": target_provider,
                                "error": str(e)[:120]})
                time.sleep(sleep_sec)
                continue
            elapsed = -1
        raw_dump[sid] = {**judged, "elapsed_sec": round(elapsed, 2)}
        p = judged["parsed"]
        if not p.get("parse_error"):
            rows.append({"sample_id": sid,
                            "judge": judge_provider,
                            "target": target_provider,
                            "factual_accuracy": p.get("factual_accuracy"),
                            "completeness": p.get("completeness"),
                            "sensitive_leak": p.get("sensitive_leak"),
                            "style": p.get("style")})
            print(f"  [{i}/{len(files)}] {sid}  fact={p.get('factual_accuracy')} "
                    f"comp={p.get('completeness')} sens={p.get('sensitive_leak')} "
                    f"style={p.get('style')}  elapsed={elapsed:.1f}s")
        else:
            rows.append({"sample_id": sid, "judge": judge_provider,
                            "target": target_provider,
                            "parse_error": True})
            print(f"  [{i}/{len(files)}] {sid} parse_error")
        time.sleep(sleep_sec)

    # raw 저장
    raw_path = RESULTS_DIR / f"eval_geval_cross_{judge_provider}_judges_{target_provider}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_dump, f, indent=2, ensure_ascii=False)
    return pd.DataFrame(rows)


def main(n_samples: int = 30, sleep_sec: float = 5.0,
         pairs: list[tuple[str, str]] | None = None) -> None:
    if pairs is None:
        pairs = [("anthropic", "gemini"), ("gemini", "anthropic")]

    all_rows = []
    for judge, target in pairs:
        try:
            df = evaluate_cross(judge, target, n_samples=n_samples,
                                  sleep_sec=sleep_sec)
            df.to_csv(RESULTS_DIR /
                          f"explanation_eval_cross_{judge}_judges_{target}.csv",
                      index=False)
            all_rows.append(df)
        except Exception as e:
            print(f"[FAIL] judge={judge} target={target}: {e}")
            continue

    if not all_rows:
        print("[ERROR] no successful judges")
        return

    full = pd.concat(all_rows, ignore_index=True)

    # 요약 — judge × target 별 4 dimension mean
    summary = (full.groupby(["judge", "target"])
                  [["factual_accuracy", "completeness",
                     "sensitive_leak", "style"]]
                  .agg(["mean", "std", "count"]).round(4))
    print("\n[Cross-LLM G-Eval 요약]")
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "explanation_eval_summary_cross.csv")

    # 시각화 — Gemini self vs Claude→Gemini, Claude self(없음, 룰만) vs Gemini→Claude
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metrics = ["factual_accuracy", "completeness", "sensitive_leak", "style"]
    for ax, m in zip(axes, metrics):
        sns.boxplot(data=full, x="target", y=m, hue="judge", ax=ax,
                      palette="Set2")
        sns.stripplot(data=full, x="target", y=m, hue="judge", ax=ax,
                        color="black", size=3, alpha=0.5, dodge=True,
                        legend=False)
        ax.set_title(m)
        ax.set_ylim(0, 5.5)
    plt.suptitle("Cross-LLM G-Eval — judge × target")
    savefig(fig, "26_cross_llm_geval")
    print(f"[OK] figures/26_cross_llm_geval.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=30,
                    help="평가할 샘플 수 (free tier RPD 고려해 기본 30)")
    ap.add_argument("--sleep", type=float, default=5.0)
    args = ap.parse_args()
    main(n_samples=args.n_samples, sleep_sec=args.sleep)
