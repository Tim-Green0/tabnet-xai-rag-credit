"""미팅 데모 — 1명 샘플 end-to-end 시연.

전 파이프라인을 한 번에 보여주는 단일 스크립트:
    정형 데이터 → XGBoost 예측 → SHAP → JSON 컨텍스트 → Gemini & Claude 자연어 설명

실행:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.demo [--sample-idx N]

산출물:
    - results/demo_walkthrough.md   (사람이 읽는 요약)
    - results/demo_walkthrough.json (모든 중간 산출물 raw)
    - figures/22_demo_walkthrough.png (4-panel: 입력 / SHAP / decision / metrics)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from dotenv import load_dotenv
from pytorch_tabnet.tab_model import TabNetClassifier

from src.context_builder import build_context, save_context
from src.llm_explainer import generate_one as llm_generate, make_client
from src.metrics import find_threshold_youden
from src.shap_analysis import _XgbNativeExplainer
from src.utils import FIGURES_DIR, RESULTS_DIR, savefig, set_seed

sns.set_theme(style="whitegrid", context="notebook")
load_dotenv("D:/paper/.env", override=True)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = RESULTS_DIR / "baseline_models"
TARGET_COL = "TARGET"


def load_test_data():
    test = pd.read_parquet(PROCESSED_DIR / "test_unscaled.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val_unscaled.parquet")
    pa = pd.read_parquet(PROCESSED_DIR / "test_protected_attrs.parquet")
    y_test = test[TARGET_COL].astype(int).values
    X_test = test.drop(columns=[TARGET_COL])
    y_val = val[TARGET_COL].astype(int).values
    X_val = val.drop(columns=[TARGET_COL])
    return X_test, y_test, X_val, y_val, pa


def banner(text: str, char: str = "─") -> None:
    line = char * 70
    print(f"\n{line}\n  {text}\n{line}")


def render_walkthrough_md(walk: dict) -> str:
    """결과 dict → 사람이 읽는 markdown."""
    md = []
    s = walk["sample"]
    md.append(f"# Demo Walkthrough — Sample idx={s['idx']}")
    md.append("")
    md.append(f"- **실제 정답**: {'부도(1)' if s['true_label'] == 1 else '정상(0)'}")
    md.append(f"- **연령**: {s['age']:.1f}세")
    md.append(f"- **성별**: {s['gender']}")
    md.append("")

    md.append("## 1. 핵심 정형 변수 (입력 일부)")
    md.append("| 변수 | 값 |")
    md.append("|---|---|")
    for k, v in s["display_features"].items():
        md.append(f"| {k} | {v} |")
    md.append("")

    md.append("## 2. XGBoost 예측")
    p = walk["prediction"]
    md.append(f"- **부도 확률 P(default)**: **{p['default_proba']:.4f}**")
    md.append(f"- **임계치 (Youden's J on val)**: {p['threshold']:.4f}")
    md.append(f"- **결정**: **{p['decision']}**")
    md.append("")

    md.append("## 3. SHAP Local — Top 5 거절 측 요인")
    md.append("| rank | feature | value | SHAP |")
    md.append("|---|---|---|---|")
    for d in walk["context"]["top_drivers_for_default"]:
        md.append(f"| {d['rank']} | {d['feature']} | {d['value']} | {d['shap']:+.4f} |")
    md.append("")

    md.append("## 4. SHAP Local — Top 4 승인 측 요인")
    md.append("| rank | feature | value | SHAP |")
    md.append("|---|---|---|---|")
    for d in walk["context"]["top_drivers_against_default"]:
        md.append(f"| {d['rank']} | {d['feature']} | {d['value']} | {d['shap']:+.4f} |")
    md.append("")

    md.append("## 5. JSON 컨텍스트 (LLM 입력) — 일부")
    md.append("```json")
    ctx_short = {
        "decision": walk["context"]["decision"],
        "default_probability": walk["context"]["default_probability"],
        "threshold": walk["context"]["threshold"],
        "top_drivers_for_default": [
            {"feature": d["feature"], "value": d["value"], "shap": d["shap"]}
            for d in walk["context"]["top_drivers_for_default"]
        ],
        "masked_sensitive_features": walk["context"]["masked_sensitive_features"],
        "model": walk["context"]["model"],
        "explanation_policy": walk["context"]["explanation_policy"],
    }
    md.append(json.dumps(ctx_short, indent=2, ensure_ascii=False))
    md.append("```")
    md.append("")

    for prov_key, prov_label in [("gemini", "Gemini 2.5 Flash"),
                                    ("anthropic", "Claude Sonnet 4.5")]:
        if prov_key in walk["llm_outputs"]:
            ll = walk["llm_outputs"][prov_key]
            md.append(f"## 6.{prov_key[:1].upper()} {prov_label} 자연어 설명 ({ll['elapsed_sec']:.1f}s, "
                       f"{ll.get('total_tokens', '?')} tokens)")
            md.append("")
            md.append(ll["explanation"])
            md.append("")

    return "\n".join(md)


def plot_walkthrough(walk: dict, out_name: str = "22_demo_walkthrough") -> Path:
    """4-panel: 입력 일부 / SHAP / decision / 정답 비교."""
    s = walk["sample"]
    p = walk["prediction"]
    ctx = walk["context"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # (1) SHAP local — Top 10 (positive + negative 합쳐 절대값 순)
    drivers = (ctx["top_drivers_for_default"]
                + ctx["top_drivers_against_default"])
    drivers_sorted = sorted(drivers, key=lambda d: -abs(d["shap"]))[:10]
    drivers_sorted = drivers_sorted[::-1]  # bottom = largest
    names = [d["feature"][:30] for d in drivers_sorted]
    vals = [d["shap"] for d in drivers_sorted]
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in vals]
    axes[0].barh(names, vals, color=colors)
    for i, v in enumerate(vals):
        axes[0].text(v, i, f"  {v:+.3f}", va="center", fontsize=8)
    axes[0].axvline(0, color="black", lw=0.5)
    axes[0].set_xlabel("SHAP value (positive: default↑, negative: default↓)")
    axes[0].set_title(f"SHAP Local — Top 10 (sample idx={s['idx']})")

    # (2) decision summary
    axes[1].axis("off")
    decision_color = "#C44E52" if p["decision"] == "REJECT" else "#55A868"
    text_lines = [
        f"Instance ID: {s['idx']}",
        f"True label : {'default (1)' if s['true_label'] == 1 else 'non-default (0)'}",
        f"Age: {s['age']:.0f} y  /  Gender: {s['gender']}",
        "",
        f"P(default): {p['default_proba']:.4f}",
        f"threshold : {p['threshold']:.4f}",
        f"Decision  : {p['decision']}",
        "",
        "Gemini latency: " + (f"{walk['llm_outputs']['gemini']['elapsed_sec']:.1f}s"
                                if "gemini" in walk["llm_outputs"] else "n/a"),
        "Claude latency: " + (f"{walk['llm_outputs']['anthropic']['elapsed_sec']:.1f}s"
                                if "anthropic" in walk["llm_outputs"] else "n/a"),
    ]
    axes[1].text(0.05, 0.95, "\n".join(text_lines), transform=axes[1].transAxes,
                  va="top", fontsize=12, family="monospace",
                  bbox=dict(boxstyle="round", facecolor="#f4f4f4",
                              edgecolor=decision_color, lw=2))
    axes[1].set_title("Decision Summary", fontsize=12)

    plt.suptitle(f"Demo End-to-End — XGBoost → SHAP → XAI-RAG → LLM",
                  fontsize=13)
    return savefig(fig, out_name)


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main(sample_idx: int | None = None,
          providers: list = None) -> None:
    if providers is None:
        providers = ["gemini", "anthropic"]
    set_seed(42)

    banner("0. 데이터·모델 로딩")
    X_test, y_test, X_val, y_val, pa = load_test_data()
    xgb_model = joblib.load(MODELS_DIR / "xgboost.pkl")

    # threshold from validation
    val_score = xgb_model.predict_proba(X_val)[:, 1]
    thr, _ = find_threshold_youden(y_val, val_score)
    print(f"  validation threshold (Youden's J) = {thr:.4f}")

    # sample 선택
    if sample_idx is None:
        # high-confidence 거절 중에서 미리 검토되지 않은 인덱스 선택
        proba_test = xgb_model.predict_proba(X_test)[:, 1]
        # idx=54529는 이미 본 적 있음. 다른 high-conf reject 1개:
        candidates = np.argsort(-proba_test)
        for c in candidates:
            if c != 54529:
                sample_idx = int(c)
                break
    print(f"  sample_idx = {sample_idx}")

    banner("1. 인스턴스 정보")
    X_one = X_test.iloc[[sample_idx]]
    y_one = int(y_test[sample_idx])
    pa_one = pa.iloc[sample_idx] if sample_idx < len(pa) else None
    age_val = float(pa_one["AGE"]) if pa_one is not None else float("nan")
    gender_val = str(pa_one["CODE_GENDER"]) if pa_one is not None else "?"
    print(f"  실제 정답 = {y_one} ({'부도' if y_one == 1 else '정상'})")
    print(f"  연령 = {age_val:.1f}세, 성별 = {gender_val}")

    # 입력 핵심 변수 일부
    display_keys = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                     "AMT_GOODS_PRICE", "EXT_SOURCE_1", "EXT_SOURCE_2",
                     "EXT_SOURCE_3", "DAYS_EMPLOYED", "CNT_FAM_MEMBERS"]
    display = {}
    for k in display_keys:
        if k in X_one.columns:
            v = float(X_one[k].iloc[0])
            if k.startswith("DAYS_"):
                display[k] = f"{v:.0f} (≈{-v/365.25:.1f}년)"
            elif k.startswith("AMT_"):
                display[k] = f"{v:,.0f}"
            else:
                display[k] = f"{v:.4f}"
    print("  주요 입력 변수:")
    for k, v in display.items():
        print(f"    {k:25s} = {v}")

    banner("2. XGBoost 예측")
    proba = float(xgb_model.predict_proba(X_one)[0, 1])
    decision = "REJECT" if proba >= thr else "APPROVE"
    print(f"  P(default) = {proba:.4f}")
    print(f"  threshold  = {thr:.4f}")
    print(f"  decision   = {decision}")

    banner("3. SHAP Local")
    expl = _XgbNativeExplainer(xgb_model)
    sv = expl.shap_values(X_one)[0]
    contrib = pd.Series(sv, index=X_test.columns)
    top_pos = contrib.sort_values(ascending=False).head(5)
    top_neg = contrib.sort_values().head(5)
    print(f"  Top 5 거절 측 요인:")
    for n, v in top_pos.items():
        print(f"    {n:30s} shap={v:+.4f}, value={float(X_one[n].iloc[0])}")
    print(f"  Top 5 승인 측 요인:")
    for n, v in top_neg.items():
        print(f"    {n:30s} shap={v:+.4f}, value={float(X_one[n].iloc[0])}")

    banner("4. JSON 컨텍스트 빌드")
    feature_values = {c: float(X_one[c].iloc[0]) for c in X_test.columns}
    shap_values = {c: float(v) for c, v in zip(X_test.columns, sv)}
    ctx = build_context(
        sample_idx=sample_idx,
        y_score=proba,
        threshold=thr,
        feature_values=feature_values,
        shap_values=shap_values,
        top_k=5,
        model_name="XGBoost",
        true_label=y_one,
    )
    save_context(ctx, tag="demo")
    print(f"  컨텍스트 저장: results/contexts/{sample_idx}_demo.json")
    print(f"  마스킹된 민감 변수: {ctx['masked_sensitive_features']}")
    print(f"  포함된 driver 수: {len(ctx['top_drivers_for_default'])} for + "
          f"{len(ctx['top_drivers_against_default'])} against")

    banner("5. LLM 자연어 설명 — 두 LLM 비교")
    walk = {
        "sample": {
            "idx": sample_idx,
            "true_label": y_one,
            "age": age_val,
            "gender": gender_val,
            "display_features": display,
        },
        "prediction": {
            "default_proba": proba,
            "threshold": thr,
            "decision": decision,
        },
        "context": ctx,
        "llm_outputs": {},
    }

    for prov in providers:
        try:
            print(f"  [{prov}] 호출 중...")
            client_tuple = make_client(prov)
            from src.llm_explainer import PROVIDER_DEFAULTS
            model = PROVIDER_DEFAULTS[prov]["model"]
            res = llm_generate(client_tuple, ctx, model=model)
            walk["llm_outputs"][prov] = {
                "model": res["model"],
                "elapsed_sec": res["elapsed_sec"],
                "explanation": res["explanation"],
                "total_tokens": res["usage_metadata"].get("total_token_count"),
            }
            print(f"     elapsed={res['elapsed_sec']}s, "
                  f"tokens={res['usage_metadata'].get('total_token_count')}")
            time.sleep(2)
        except Exception as e:
            print(f"  [{prov}] 실패: {e}")
            walk["llm_outputs"][prov] = {"error": str(e)[:200]}

    banner("6. 산출물 저장")
    with open(RESULTS_DIR / "demo_walkthrough.json", "w", encoding="utf-8") as f:
        json.dump(walk, f, indent=2, ensure_ascii=False)
    md_path = RESULTS_DIR / "demo_walkthrough.md"
    md_path.write_text(render_walkthrough_md(walk), encoding="utf-8")
    fig_path = plot_walkthrough(walk)
    print(f"  - results/demo_walkthrough.json")
    print(f"  - results/demo_walkthrough.md")
    print(f"  - {fig_path}")

    banner("[OK] Demo 완료 — 미팅 시 이 파일들 참조")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-idx", type=int, default=None,
                    help="test set 인덱스. 미지정 시 두 번째 high-conf reject")
    ap.add_argument("--providers", nargs="+",
                    default=["gemini", "anthropic"],
                    choices=["gemini", "anthropic"])
    args = ap.parse_args()
    main(sample_idx=args.sample_idx, providers=args.providers)
