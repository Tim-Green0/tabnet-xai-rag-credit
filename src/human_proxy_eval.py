"""Step 5-C: Pilot human-proxy evaluation via 3 LLM personas.

정식 IRB 인간평가의 pilot 대안 — LLM persona를 사람 대리(proxy)로 사용해
3가지 stakeholder 관점에서 plausibility를 정량 측정.

Personas:
    1. Credit Expert (10년 경력 신용 분석가) — 전문가 관점 신뢰성
    2. Customer (대출 신청자 본인) — 일반 고객 납득도
    3. Regulator (금융감독원 평가자) — 규제 요건 만족도

각 persona가 5점 척도 + rationale로 평가:
    - trustworthiness (신뢰할 만한가)
    - clarity (이해 가능한가)
    - actionability (행동 지침이 명확한가)

평가 대상:
    4 modes × 2 LLM target × 15 instances × 3 personas = 360 평가
    Judge: Claude (Anthropic, Step 3-C-2-f에서 안정성 검증됨)

산출:
    results/human_proxy_eval.csv
    results/human_proxy_summary.csv
    figures/36_human_proxy_personas.png

사용:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.human_proxy_eval
        [--n-samples 15] [--judge anthropic]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.cross_llm_geval import judge_one as _judge_one_cross
from src.llm_explainer import PROVIDER_DEFAULTS, make_client as _make_provider_client
from src.utils import FIGURES_DIR, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")


# ─────────────────────────────────────────────────────────────
# Persona 정의 + 프롬프트
# ─────────────────────────────────────────────────────────────
PERSONAS = {
    "credit_expert": {
        "title": "신용 전문가",
        "description": (
            "당신은 10년 경력의 신용 분석 전문가입니다. 은행 신용평가팀에서 "
            "수많은 대출 심사 결과를 검토해 왔습니다. 다음 자연어 설명이 "
            "전문가 관점에서 **얼마나 신뢰할 만한가**를 평가합니다."
        ),
        "rubric": (
            "- trustworthiness: 결정 근거가 변수와 값에 정확히 기반하는가? 추측이나 "
            "환각 없이 사실만 인용하는가? "
            "- clarity: 결정 과정의 논리가 명료한가? "
            "- actionability: 신용평가 분야의 일반 원리에 부합하며 다음 의사결정에 활용 "
            "가능한 정보를 담고 있는가?"
        ),
    },
    "customer": {
        "title": "일반 고객 (대출 신청자)",
        "description": (
            "당신은 이 대출을 신청한 본인입니다. 금융 지식이 평균 수준이고, 자신의 "
            "신청 결과가 왜 그렇게 나왔는지 이해하고 싶습니다. 다음 자연어 설명을 "
            "받았을 때 **얼마나 납득되고 도움이 되는가**를 평가합니다."
        ),
        "rubric": (
            "- trustworthiness: 설명이 거짓 정보 없이 내 데이터에 근거해 보이는가? "
            "- clarity: 전문 용어 없이도 이해 가능한가? 어렵지 않은가? "
            "- actionability: 다음에 무엇을 해야 할지 구체적 지침이 있는가?"
        ),
    },
    "regulator": {
        "title": "금융감독원 평가자",
        "description": (
            "당신은 금융감독원의 신용평가 시스템 검사관입니다. 규제(개인정보 보호, "
            "차별 금지, 설명 의무)를 준수하는지 검토합니다. 다음 자연어 설명이 "
            "**규제 요건을 만족하는가**를 평가합니다."
        ),
        "rubric": (
            "- trustworthiness: 사실에 근거하며 환각·허위 정보가 없는가? "
            "- clarity: 결정 사유가 추적 가능한가? "
            "- actionability: 보호 변수(성별·연령 등)를 직접 언급하지 않으면서도 "
            "차별 없는 의사결정을 보장하는 형태인가?"
        ),
    },
}


PERSONA_PROMPT_TEMPLATE = """당신은 다음 페르소나로 자연어 설명을 평가합니다:

[페르소나]
{persona_title} — {persona_description}

[평가 기준 (1~5 척도)]
{rubric}

[참조 컨텍스트 (LLM에 주어진 입력)]
{context_json}

[평가할 자연어 설명]
{explanation_text}

당신의 페르소나로서 이 설명을 평가해주세요. 오직 JSON 한 개만 출력하세요. 다른 텍스트 금지.
{{
  "trustworthiness": <int 1-5>,
  "clarity": <int 1-5>,
  "actionability": <int 1-5>,
  "rationale": "<페르소나 관점에서 한두 줄 사유>"
}}
"""


def parse_score_json(text: str) -> Dict:
    m = re.search(r"\{[^{}]*\"trustworthiness\".*?\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"raw": text, "parse_error": True}
    return {"raw": text, "parse_error": True}


def evaluate_persona(judge_client, judge_model: str,
                       persona_key: str, context: Dict,
                       explanation: str) -> Dict:
    persona = PERSONAS[persona_key]
    ctx_for_prompt = {k: v for k, v in context.items() if not k.startswith("_meta")}
    prompt = PERSONA_PROMPT_TEMPLATE.format(
        persona_title=persona["title"],
        persona_description=persona["description"],
        rubric=persona["rubric"],
        context_json=json.dumps(ctx_for_prompt, ensure_ascii=False, indent=2),
        explanation_text=explanation,
    )
    from src.llm_explainer import _call_llm
    text, usage = _call_llm(judge_client, prompt, model=judge_model)
    return {"raw": text, "parsed": parse_score_json(text), "usage": usage}


# ─────────────────────────────────────────────────────────────
# 메인 평가 루프
# ─────────────────────────────────────────────────────────────
def evaluate_directory(directory: Path, mode: str, target_provider: str,
                        judge_client, judge_model: str,
                        target_idx: List[int],
                        sleep_sec: float = 4.0) -> List[Dict]:
    rows = []
    files = sorted([p for p in directory.glob("*.json")
                     if p.name != "_index.json"])
    target_set = set(target_idx)
    files = [p for p in files if int(p.stem.split("_")[0]) in target_set]
    print(f"  [{mode}/{target_provider}] {len(files)}개 × 3 personas = {len(files)*3} 평가")

    for i, exp_path in enumerate(files, start=1):
        with open(exp_path, "r", encoding="utf-8") as f:
            exp = json.load(f)
        ctx = exp["context_sent"]
        text = exp["explanation"]
        sample_id = exp_path.stem

        for persona_key in PERSONAS:
            for attempt in range(4):
                try:
                    t0 = time.time()
                    res = evaluate_persona(judge_client, judge_model,
                                              persona_key, ctx, text)
                    elapsed = time.time() - t0
                    p = res["parsed"]
                    if p.get("parse_error"):
                        row_extra = {"persona_parse_error": True}
                    else:
                        row_extra = {
                            "trustworthiness": p.get("trustworthiness"),
                            "clarity": p.get("clarity"),
                            "actionability": p.get("actionability"),
                            "rationale": str(p.get("rationale", ""))[:200],
                        }
                    rows.append({
                        "sample_id": sample_id, "mode": mode,
                        "target_provider": target_provider,
                        "persona": persona_key,
                        "decision": exp.get("decision"),
                        "true_label": exp.get("true_label"),
                        "elapsed_sec": round(elapsed, 1),
                        **row_extra,
                    })
                    break
                except Exception as e:
                    err = str(e)
                    is_503 = "503" in err or "UNAVAILABLE" in err or "overloaded" in err.lower()
                    backoff = 30 * (2 ** attempt) if is_503 else 10
                    print(f"    ERROR (attempt {attempt+1}/4) [{persona_key}]: {err[:80]}")
                    if attempt < 3:
                        time.sleep(backoff)
                    else:
                        rows.append({
                            "sample_id": sample_id, "mode": mode,
                            "target_provider": target_provider,
                            "persona": persona_key,
                            "error": err[:200],
                        })
            time.sleep(sleep_sec)
        if i % 3 == 0:
            running = pd.DataFrame(rows)
            cur = running[(running["mode"] == mode) &
                            (running["target_provider"] == target_provider)]
            for col in ["trustworthiness", "clarity", "actionability"]:
                if col in cur.columns:
                    print(f"    [{i}/{len(files)}] running mean {col}={cur[col].dropna().mean():.2f}")
                    break
    return rows


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    metrics = ["trustworthiness", "clarity", "actionability"]
    for (mode, target_provider, persona), grp in df.groupby(
            ["mode", "target_provider", "persona"]):
        for col in metrics:
            if col in grp.columns:
                vals = grp[col].dropna()
                if len(vals) > 0:
                    out.append({
                        "mode": mode, "target_provider": target_provider,
                        "persona": persona, "metric": col,
                        "mean": float(vals.mean()),
                        "std": float(vals.std()) if len(vals) > 1 else 0.0,
                        "n": int(len(vals)),
                    })
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────
def plot_persona_results(summary: pd.DataFrame,
                            out_name: str = "36_human_proxy_personas") -> Path:
    """3 metrics × 3 personas 그리드."""
    metrics = ["trustworthiness", "clarity", "actionability"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    palette = {"no_shap": "#C44E52", "generic_rag": "#55A868",
                "shaponly": "#A0A0A0", "fusion": "#DD8452"}
    persona_titles = {k: PERSONAS[k]["title"] for k in PERSONAS}

    for r_idx, m in enumerate(metrics):
        for c_idx, persona in enumerate(["credit_expert", "customer", "regulator"]):
            ax = axes[r_idx, c_idx]
            sub = summary[(summary["metric"] == m) &
                            (summary["persona"] == persona)]
            if len(sub) == 0:
                ax.set_visible(False); continue
            sns.barplot(data=sub, x="target_provider", y="mean", hue="mode",
                         hue_order=["no_shap", "generic_rag", "shaponly", "fusion"],
                         palette=palette, ax=ax, errorbar=None)
            for _, rrow in sub.iterrows():
                providers = list(sub["target_provider"].unique())
                if rrow["target_provider"] not in providers:
                    continue
                xi = providers.index(rrow["target_provider"])
                offsets = {"no_shap": -0.3, "generic_rag": -0.1,
                            "shaponly": 0.1, "fusion": 0.3}
                xshift = offsets.get(rrow["mode"], 0)
                ax.errorbar(xi + xshift, rrow["mean"], yerr=rrow["std"],
                              fmt="none", ecolor="black", capsize=3)
            ax.set_title(f"{persona_titles[persona]} — {m}")
            ax.set_ylim(0, 5.5)
            ax.set_ylabel(m)
            ax.set_xlabel("Target LLM")
            if not (r_idx == 0 and c_idx == 0):
                if ax.legend_:
                    ax.legend_.remove()
    plt.suptitle("Pilot human-proxy evaluation — 3 personas × 3 metrics × 4 modes")
    return savefig(fig, out_name)


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main(n_samples: int = 15, judge: str = "anthropic",
         sleep_sec: float = 4.0) -> None:
    set_seed(SEED)

    # target idx — fusion 결과 30개 중 random 15
    fusion_idx_path = RESULTS_DIR / f"explanations_fusion_anthropic_30" / "_index.json"
    with open(fusion_idx_path, "r", encoding="utf-8") as f:
        full_idx = json.load(f)["selected_idx"]
    rnd = random.Random(SEED)
    target_idx = sorted(rnd.sample(full_idx, min(n_samples, len(full_idx))))
    print(f"[match] target_idx {len(target_idx)}개 (random subset of 30)")

    # judge client
    judge_client = _make_provider_client(judge)
    judge_model = PROVIDER_DEFAULTS[judge]["model"]
    print(f"[judge] {judge} / {judge_model}")

    # plan: 4 modes × 2 LLM target = 8 그룹
    plan = [
        ("no_shap", "anthropic", RESULTS_DIR / "explanations_baseline_noshap_anthropic"),
        ("no_shap", "gemini", RESULTS_DIR / "explanations_baseline_noshap_gemini"),
        ("generic_rag", "anthropic", RESULTS_DIR / "explanations_generic_rag_anthropic_30"),
        ("generic_rag", "gemini", RESULTS_DIR / "explanations_generic_rag_gemini_30"),
        ("shaponly", "anthropic", RESULTS_DIR / "explanations_anthropic_100"),
        ("shaponly", "gemini", RESULTS_DIR / "explanations_gemini_100"),
        ("fusion", "anthropic", RESULTS_DIR / "explanations_fusion_anthropic_30"),
        ("fusion", "gemini", RESULTS_DIR / "explanations_fusion_gemini_30"),
    ]

    rows = []
    for mode, target_provider, directory in plan:
        if not directory.exists():
            print(f"[skip] {directory} 없음")
            continue
        rows.extend(evaluate_directory(
            directory, mode, target_provider,
            judge_client, judge_model, target_idx,
            sleep_sec=sleep_sec,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "human_proxy_eval.csv", index=False)
    print(f"\n[save] human_proxy_eval.csv ({len(df)} rows)")

    summary = aggregate(df)
    summary.to_csv(RESULTS_DIR / "human_proxy_summary.csv", index=False)
    print(f"[save] human_proxy_summary.csv")

    # mode별 핵심 print
    print("\n[summary] 평균 (mode × persona, target=anthropic만):")
    sub = summary[summary["target_provider"] == "anthropic"]
    pivot = sub.pivot_table(
        index=["persona", "metric"], columns="mode", values="mean").round(2)
    print(pivot.to_string())

    plot_persona_results(summary)
    print(f"[save] figures/36_human_proxy_personas.png")
    print("\n[OK] human-proxy 평가 완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=15)
    ap.add_argument("--judge", default="anthropic",
                    choices=["anthropic", "gemini"])
    ap.add_argument("--sleep", type=float, default=4.0)
    args = ap.parse_args()
    main(n_samples=args.n_samples, judge=args.judge, sleep_sec=args.sleep)
