"""Step 3-C-3: Fusion vs SHAP-only 비교 평가.

기존 src.eval_explanation의 룰(faithfulness, hallucination, G-Eval)을 거의 그대로
재사용하되, fusion 컨텍스트의 driver 구조에 맞춰 extract_drivers_from_context만 교체.

비교 대상 (같은 30개 idx):
    - SHAP-only: results/explanations_anthropic_100/, results/explanations_gemini_100/
                  (Step 2-A 산출물 중 30개 부분집합)
    - Fusion   : results/explanations_fusion_anthropic_30/,
                  results/explanations_fusion_gemini_30/

산출:
    results/fusion_eval.csv        — 샘플별 row (mode=shaponly|fusion, provider, 메트릭)
    results/fusion_vs_shaponly.csv — 평균 비교 표
    figures/30_fusion_vs_shaponly.png

사용:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.eval_fusion \
        [--skip-geval]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eval_explanation import (
    faithfulness_per_driver as _faithfulness_per_driver_orig,
    faithfulness_score,
    hallucination_rate,
    load_all_feature_names,
)
from src.cross_llm_geval import judge_one as _judge_one_cross
from src.llm_explainer import PROVIDER_DEFAULTS, make_client as _make_provider_client
from src.utils import FIGURES_DIR, RESULTS_DIR, savefig

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")


def faithfulness_per_driver(text: str, drivers: List[Dict]) -> List[Dict]:
    """fusion-aware wrapper: attention_only 그룹은 sign 평가 없음.

    드라이버에 group=='attention_only'인 경우, sign_in을 항상 None으로 처리해서
    sign_match_rate에 반영되지 않도록 한다 (부호 없는 신호이므로).
    """
    results = _faithfulness_per_driver_orig(text, drivers)
    # 매핑: drivers와 results가 같은 길이/순서라고 가정
    for r, d in zip(results, drivers):
        if d.get("group") == "attention_only":
            r["sign_in"] = None
            # all_match도 sign 제외하고 재정의
            r["all_match"] = bool(r["feat_in"] and r["val_in"])
    return results


def _english_prefixes(feature_name: str) -> List[str]:
    """one-hot/value-suffixed 컬럼명에서 영문 prefix 후보들을 추출.

    예: 'FONDKAPREMONT_MODE_org spec account'
         → ['FONDKAPREMONT_MODE_ORG', 'FONDKAPREMONT_MODE', 'FONDKAPREMONT']
    `\\b[A-Z][A-Z0-9_]{2,}\\b` regex로 추출되는 토큰을 prefix로 매칭하기 위함.
    """
    # 공백 이전까지의 영문/숫자/_ 부분만 사용
    head = feature_name.split(" ")[0]
    parts = head.split("_")
    out = []
    for i in range(len(parts), 0, -1):
        piece = "_".join(parts[:i])
        # 토큰이 [A-Z][A-Z0-9_]{2,} 와 매칭되려면 길이 ≥ 3 + 첫 글자 영문 대문자
        if len(piece) >= 3 and piece[0].isalpha() and piece.isupper():
            out.append(piece)
    return out


def extract_drivers_fusion(ctx: Dict) -> List[Dict]:
    """fusion 컨텍스트의 모든 driver를 단일 리스트로 통합.

    각 entry에 group 라벨 ('agreed', 'shap_only', 'attention_only') 보존.
    SHAP-only entry에는 shap값 + sign_for_default 있음.
    Attention-only entry에는 shap=None.
    Faithfulness 평가 시 attention_only는 sign_in 점검을 None으로 설정.
    """
    out = []
    for d in ctx.get("agreed_drivers", []):
        e = dict(d)
        # eval_explanation의 faithfulness_per_driver 호환을 위해 shap 키 보장
        if "shap" not in e:
            e["shap"] = 0.0
        out.append(e)
    for d in ctx.get("shap_only_drivers", []):
        e = dict(d)
        if "shap" not in e:
            e["shap"] = 0.0
        out.append(e)
    for d in ctx.get("attention_only_drivers", []):
        e = dict(d)
        # attention_only는 SHAP 부호 없음 → 0으로 두면 faithfulness sign_in이 negative path로 평가됨
        # 이걸 "평가에서 제외"하려면 더미 값 + 후처리에서 제외하는 방식이 정확.
        # 여기서는 shap=0.0으로 두고, faithfulness rollup 시 group으로 필터링한다.
        if "shap" not in e:
            e["shap"] = 0.0
        out.append(e)
    return out


def hallucination_rate_fusion(text: str, ctx: Dict, all_features: set) -> Dict:
    """기존 hallucination_rate를 fusion 컨텍스트에 맞게 in_context_features 재정의.

    추가 패치: one-hot 컬럼명의 prefix 매칭 허용.
    예: 'FONDKAPREMONT_MODE_org spec account'가 컨텍스트에 있고 LLM이
        'FONDKAPREMONT_MODE'만 인용한 경우, 영문 토큰 추출 regex가 공백 전까지만
        잡으므로 prefix가 outside_dataset로 잘못 분류됨. prefix 일치 시 컨텍스트
        내로 인정.
    """
    drivers = extract_drivers_fusion(ctx)
    in_context_features = {d["feature_raw"] for d in drivers}

    # ★ prefix 확장: 컨텍스트 변수의 영문 prefix들을 추가
    in_context_prefixes = set()
    for f in in_context_features:
        # one-hot/SUFFIXED 변수 prefix 추출 (예: FONDKAPREMONT_MODE_org... → FONDKAPREMONT_MODE)
        # 규칙: 마지막 영문/공백 segment 제거 + 뒤에서부터 _ 단위로 자르며 영문만 prefix 후보
        for prefix in _english_prefixes(f):
            in_context_prefixes.add(prefix)
    # dataset 내 변수의 prefix도 동일 처리 (raw_in_dataset 후보 정의용)
    dataset_prefixes = set()
    for f in all_features:
        for prefix in _english_prefixes(f):
            dataset_prefixes.add(prefix)

    raw_candidates = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text))
    common_excludes = {"SHAP", "REJECT", "APPROVE", "TARGET", "JSON", "API",
                        "AUC", "AUROC", "AUPRC", "OK", "AI", "LLM", "TABNET"}
    raw_candidates -= common_excludes

    # 정확 매칭 + prefix 매칭 둘 다 허용
    raw_in_ctx = (raw_candidates & in_context_features) | (raw_candidates & in_context_prefixes)
    raw_in_dataset = (raw_candidates & all_features) | (raw_candidates & dataset_prefixes)
    raw_outside = raw_candidates - raw_in_dataset
    raw_inside_dataset_outside_ctx = raw_in_dataset - raw_in_ctx

    return {
        "n_raw_candidates": len(raw_candidates),
        "n_raw_in_context": len(raw_in_ctx),
        "n_raw_in_dataset_only": len(raw_inside_dataset_outside_ctx),
        "n_raw_outside_dataset": len(raw_outside),
        "raw_outside_dataset_list": sorted(list(raw_outside)),
        "raw_in_dataset_outside_context_list": sorted(list(raw_inside_dataset_outside_ctx)),
        "hallucination_rate_strict": (
            len(raw_outside) / max(len(raw_candidates), 1)
        ),
        "hallucination_rate_broad": (
            (len(raw_outside) + len(raw_inside_dataset_outside_ctx))
            / max(len(raw_candidates), 1)
        ),
    }


def evaluate_directory(directory: Path, mode: str, provider: str,
                        all_features: set, geval_client=None,
                        geval_model: str = None,
                        geval_sleep: float = 5.0,
                        target_idx: List[int] = None) -> List[Dict]:
    """단일 디렉토리(SHAP-only 또는 Fusion) 평가.

    Parameters
    ----------
    mode : "shaponly" | "fusion"
    target_idx : 평가할 sample idx 리스트. None이면 디렉토리의 모든 파일.
    """
    rows = []
    files = sorted([p for p in directory.glob("*.json")
                     if p.name != "_index.json"])
    if target_idx is not None:
        target_set = set(target_idx)
        files = [p for p in files
                  if int(p.stem.split("_")[0]) in target_set]
    print(f"  [{mode}/{provider}] {len(files)}개 평가")

    for i, exp_path in enumerate(files, start=1):
        with open(exp_path, "r", encoding="utf-8") as f:
            exp = json.load(f)

        ctx = exp["context_sent"]
        text = exp["explanation"]
        sample_id = exp_path.stem

        # mode별 driver 추출 + halluc
        if mode == "fusion":
            drivers = extract_drivers_fusion(ctx)
            h_scores = hallucination_rate_fusion(text, ctx, all_features)
        else:  # shaponly
            from src.eval_explanation import (
                extract_drivers_from_context as _ext, hallucination_rate as _hl)
            drivers = _ext(ctx)
            h_scores = _hl(text, ctx, all_features)

        per_driver = faithfulness_per_driver(text, drivers)
        f_scores = faithfulness_score(per_driver)

        row = {
            "sample_id": sample_id,
            "mode": mode,
            "provider": provider,
            "decision": exp.get("decision"),
            "true_label": exp.get("true_label"),
            "default_proba": ctx.get("default_probability"),
            "n_drivers": len(drivers),
            **f_scores,
            "halluc_rate_strict": h_scores["hallucination_rate_strict"],
            "halluc_rate_broad": h_scores["hallucination_rate_broad"],
            "n_raw_candidates": h_scores["n_raw_candidates"],
            "n_raw_outside_dataset": h_scores["n_raw_outside_dataset"],
        }

        # G-Eval (with 503 retry; judge = anthropic | gemini)
        if geval_client is not None:
            ge = None
            last_err = None
            for attempt in range(4):
                try:
                    t0 = time.time()
                    ge = _judge_one_cross(geval_client, geval_model,
                                            text, ctx)
                    row["geval_elapsed_sec"] = round(time.time() - t0, 2)
                    break
                except Exception as e:
                    last_err = str(e)
                    is_503 = "503" in last_err or "UNAVAILABLE" in last_err or "overloaded" in last_err.lower()
                    backoff = 30 * (2 ** attempt) if is_503 else 10
                    print(f"  G-Eval ERROR (attempt {attempt+1}/4): {last_err[:100]}")
                    if attempt < 3:
                        print(f"    → {backoff}s 대기 후 재시도")
                        time.sleep(backoff)
            if ge is not None:
                p = ge["parsed"]
                if not p.get("parse_error"):
                    row.update({
                        "geval_factual_accuracy": p.get("factual_accuracy"),
                        "geval_completeness": p.get("completeness"),
                        "geval_sensitive_leak": p.get("sensitive_leak"),
                        "geval_style": p.get("style"),
                    })
            else:
                row["geval_error"] = (last_err or "unknown")[:200]
            if i < len(files):
                time.sleep(geval_sleep)

        rows.append(row)

    return rows


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """mode × provider 별 평균 ± std."""
    metric_cols = [
        "feat_match_rate", "val_match_rate", "sign_match_rate", "full_match_rate",
        "halluc_rate_strict", "halluc_rate_broad",
        "geval_factual_accuracy", "geval_completeness",
        "geval_sensitive_leak", "geval_style",
    ]
    out = []
    for (mode, provider), grp in df.groupby(["mode", "provider"]):
        for col in metric_cols:
            if col in grp.columns:
                vals = grp[col].dropna()
                if len(vals) > 0:
                    out.append({
                        "mode": mode, "provider": provider, "metric": col,
                        "mean": float(vals.mean()),
                        "std": float(vals.std()) if len(vals) > 1 else 0.0,
                        "n": int(len(vals)),
                    })
    return pd.DataFrame(out)


def plot_comparison(summary: pd.DataFrame,
                     out_name: str = "30_fusion_vs_shaponly") -> Path:
    """주요 4 메트릭에 대해 mode × provider 막대그래프."""
    metrics = [
        ("halluc_rate_strict", "Halluc Rate (strict, ↓)"),
        ("full_match_rate", "Faithfulness (full match, ↑)"),
        ("geval_factual_accuracy", "G-Eval Factual (1-5, ↑)"),
        ("geval_completeness", "G-Eval Completeness (1-5, ↑)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (m, label) in zip(axes, metrics):
        sub = summary[summary["metric"] == m]
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        sns.barplot(data=sub, x="provider", y="mean", hue="mode", ax=ax,
                     errorbar=None, palette={"shaponly": "#A0A0A0", "fusion": "#DD8452"})
        # error bars
        for i, prov in enumerate(sub["provider"].unique()):
            for j, mode in enumerate(["shaponly", "fusion"]):
                row = sub[(sub["provider"] == prov) & (sub["mode"] == mode)]
                if len(row) > 0:
                    x = i + (-0.2 if mode == "shaponly" else 0.2)
                    ax.errorbar(x, row.iloc[0]["mean"], yerr=row.iloc[0]["std"],
                                  fmt="none", ecolor="black", capsize=4)
        ax.set_title(label)
        ax.set_ylabel(label.split(" (")[0])
        ax.set_xlabel("LLM provider")
        if m != "halluc_rate_strict":
            ax.legend_.remove() if ax.legend_ else None

    plt.suptitle("Fusion (SHAP+TabNet attention) vs SHAP-only — n=30 each")
    return savefig(fig, out_name)


def main(skip_geval: bool = False, geval_sleep: float = 5.0,
         n_samples: int = 30, judge: str = "anthropic") -> None:
    all_features = load_all_feature_names()

    # 같은 30개 idx 파악 (fusion 결과의 _index.json에서)
    fusion_a_index = RESULTS_DIR / f"explanations_fusion_anthropic_{n_samples}" / "_index.json"
    fusion_g_index = RESULTS_DIR / f"explanations_fusion_gemini_{n_samples}" / "_index.json"
    target_idx = None
    for ip in [fusion_a_index, fusion_g_index]:
        if ip.exists():
            with open(ip, "r", encoding="utf-8") as f:
                idx_obj = json.load(f)
            target_idx = idx_obj.get("selected_idx")
            print(f"[match] target_idx 파악 ({len(target_idx)}개) from {ip.name}")
            break
    if target_idx is None:
        raise RuntimeError("fusion 결과 디렉토리의 _index.json 미발견 — fusion LLM 호출 먼저 완료해야 함")

    # G-Eval client + model (judge LLM은 anthropic 또는 gemini 선택 가능)
    if skip_geval:
        client = None
        judge_model = None
    else:
        client = _make_provider_client(judge)
        judge_model = PROVIDER_DEFAULTS[judge]["model"]
        print(f"[judge] {judge} / model={judge_model}")

    rows = []
    plan = [
        ("shaponly", "anthropic", RESULTS_DIR / "explanations_anthropic_100"),
        ("shaponly", "gemini", RESULTS_DIR / "explanations_gemini_100"),
        ("fusion", "anthropic", RESULTS_DIR / f"explanations_fusion_anthropic_{n_samples}"),
        ("fusion", "gemini", RESULTS_DIR / f"explanations_fusion_gemini_{n_samples}"),
    ]
    for mode, provider, directory in plan:
        if not directory.exists():
            print(f"[skip] {directory} 없음")
            continue
        rows.extend(evaluate_directory(
            directory, mode, provider, all_features,
            geval_client=client, geval_model=judge_model,
            geval_sleep=geval_sleep,
            target_idx=target_idx,
        ))

    df = pd.DataFrame(rows)
    df["judge"] = judge if not skip_geval else "(skipped)"
    df.to_csv(RESULTS_DIR / "fusion_eval.csv", index=False)
    print(f"\n[save] results/fusion_eval.csv ({len(df)} rows, judge={df['judge'].iloc[0] if len(df) else '-'})")

    summary = aggregate(df)
    summary.to_csv(RESULTS_DIR / "fusion_vs_shaponly.csv", index=False)
    print(f"[save] results/fusion_vs_shaponly.csv")

    # 주요 차이 print
    print("\n[delta] fusion - shaponly (provider별):")
    pivot = summary.pivot_table(index=["provider", "metric"], columns="mode",
                                  values="mean").reset_index()
    if "fusion" in pivot.columns and "shaponly" in pivot.columns:
        pivot["delta"] = pivot["fusion"] - pivot["shaponly"]
        print(pivot.to_string(index=False))

    plot_comparison(summary)
    print(f"[save] figures/30_fusion_vs_shaponly.png")
    print("\n[OK] fusion 평가 완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-geval", action="store_true")
    ap.add_argument("--geval-sleep", type=float, default=5.0)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--judge", default="anthropic",
                    choices=["anthropic", "gemini"],
                    help="G-Eval judge LLM (Gemini 503 시 anthropic 권장)")
    args = ap.parse_args()
    main(skip_geval=args.skip_geval, geval_sleep=args.geval_sleep,
         n_samples=args.n_samples, judge=args.judge)
