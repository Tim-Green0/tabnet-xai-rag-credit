"""Step 3-C-1-b: Agreement-aware fusion context builder.

XGBoost SHAP local + TabNet local attention → 단일 융합 JSON 컨텍스트.

세 그룹으로 분류해 LLM에 전달:
    1. agreed_drivers      — SHAP top-k(both signs) ∩ Attention top-k
                              "두 해석 모델이 동의한 강한 신호"
    2. shap_only_drivers   — SHAP top-k - Attention top-k
                              "SHAP 관점의 보완 신호 (방향 정보 + 기여도 부호)"
    3. attention_only_drivers — Attention top-k - SHAP top-k
                              "TabNet 어텐션 관점의 보완 신호 (sparse 해석)"

기존 src.context_builder의 humanize_feature/value, SENSITIVE_FEATURES 재사용.

산출:
    results/contexts_fusion_100/{idx}_{tag}.json
    results/contexts_fusion_100/_index.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.context_builder import (
    SENSITIVE_FEATURES,
    humanize_feature,
    humanize_value,
)
from src.utils import RESULTS_DIR


def _shap_topk_both(top_pos: List[Dict], top_neg: List[Dict]) -> Dict[str, Dict]:
    """SHAP positive/negative drivers를 feature 이름 키 dict로 합치기.

    각 entry에 sign_for_default ('+', '-') 표시.
    """
    out = {}
    for d in top_pos:
        out[d["feature"]] = {
            "value_raw": d["value"],
            "shap": float(d["shap"]),
            "sign_for_default": "+",  # P(default) ↑
        }
    for d in top_neg:
        out[d["feature"]] = {
            "value_raw": d["value"],
            "shap": float(d["shap"]),
            "sign_for_default": "-",  # P(default) ↓
        }
    return out


def _attention_topk(top_att: List[Dict]) -> Dict[str, Dict]:
    """Attention top-k 리스트를 feature 이름 키 dict로."""
    out = {}
    for d in top_att:
        out[d["feature"]] = {
            "value_raw": d["value"],
            "attention": float(d["attention_score"]),
        }
    return out


def _make_driver_entry(feature: str, value_raw, shap: Optional[float],
                        attention: Optional[float],
                        sign_for_default: Optional[str],
                        rank: int, group: str) -> Dict:
    entry = {
        "feature": humanize_feature(feature),
        "feature_raw": feature,
        "value": humanize_value(feature, value_raw),
        "value_raw": value_raw,
        "rank": rank,
        "group": group,
    }
    if shap is not None:
        entry["shap"] = round(shap, 4)
    if attention is not None:
        entry["attention"] = round(attention, 6)
    if sign_for_default is not None:
        entry["sign_for_default"] = sign_for_default
    return entry


def build_fusion_context(
    shap_example: Dict,
    attention_example: Dict,
    threshold: float,
    fairness_flags: Optional[List[str]] = None,
    mask_sensitive: bool = True,
) -> Dict:
    """단일 인스턴스의 SHAP + Attention 정보를 받아 융합 컨텍스트 JSON 생성."""
    assert shap_example["idx"] == attention_example["idx"], "idx mismatch"

    sample_idx = shap_example["idx"]
    y_score = float(shap_example["predicted_proba"])
    decision = "REJECT" if y_score >= threshold else "APPROVE"

    shap_dict = _shap_topk_both(
        shap_example["top_5_positive_drivers"],
        shap_example["top_5_negative_drivers"],
    )
    att_dict = _attention_topk(attention_example["top_k_attention"])

    # 민감 변수 마스킹
    if mask_sensitive:
        masked = sorted([f for f in (set(shap_dict) | set(att_dict))
                         if f in SENSITIVE_FEATURES])
        for f in masked:
            shap_dict.pop(f, None)
            att_dict.pop(f, None)
    else:
        masked = []

    shap_set = set(shap_dict.keys())
    att_set = set(att_dict.keys())
    agreed_set = shap_set & att_set
    shap_only_set = shap_set - att_set
    att_only_set = att_set - shap_set

    # ── 1) agreed (강한 신호) — SHAP |값| 기준 정렬, 부호와 attention 모두 보존
    agreed = []
    for rank, f in enumerate(sorted(agreed_set,
                                     key=lambda x: -abs(shap_dict[x]["shap"])),
                              start=1):
        s = shap_dict[f]
        a = att_dict[f]
        agreed.append(_make_driver_entry(
            f, s["value_raw"], s["shap"], a["attention"],
            s["sign_for_default"], rank, group="agreed",
        ))

    # ── 2) shap_only (보완 신호 1) — SHAP |값| 정렬, 부호 보존
    shap_only = []
    for rank, f in enumerate(sorted(shap_only_set,
                                     key=lambda x: -abs(shap_dict[x]["shap"])),
                              start=1):
        s = shap_dict[f]
        shap_only.append(_make_driver_entry(
            f, s["value_raw"], s["shap"], None,
            s["sign_for_default"], rank, group="shap_only",
        ))

    # ── 3) attention_only (보완 신호 2) — attention 정렬, 부호 없음
    att_only = []
    for rank, f in enumerate(sorted(att_only_set,
                                     key=lambda x: -att_dict[x]["attention"]),
                              start=1):
        a = att_dict[f]
        att_only.append(_make_driver_entry(
            f, a["value_raw"], None, a["attention"],
            None, rank, group="attention_only",
        ))

    ctx = {
        "sample_idx": sample_idx,
        "decision": decision,
        "default_probability": round(y_score, 4),
        "threshold": round(threshold, 4),
        # 융합 핵심 ↓
        "agreed_drivers": agreed,
        "shap_only_drivers": shap_only,
        "attention_only_drivers": att_only,
        "fairness_flags": fairness_flags or [],
        "model_predict": "XGBoost",
        "model_explain": ["SHAP_xgb_local", "TabNet_attention_local"],
        "explanation_policy": "fact_only_with_agreement_labels",
        "masked_sensitive_features": masked,
        # 통계
        "n_agreed": len(agreed),
        "n_shap_only": len(shap_only),
        "n_attention_only": len(att_only),
    }
    if "true_label" in shap_example:
        ctx["_meta_true_label"] = int(shap_example["true_label"])
    return ctx


def main(shap_path: Path, att_path: Path, out_dir: Path,
         threshold: float = 0.476) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] SHAP examples: {shap_path}")
    with open(shap_path, "r", encoding="utf-8") as f:
        shap_examples = json.load(f)
    print(f"[load] Attention examples: {att_path}")
    with open(att_path, "r", encoding="utf-8") as f:
        att_examples = json.load(f)

    # idx 정렬해서 1:1 매칭 보장
    shap_examples = {ex["idx"]: ex for ex in shap_examples}
    att_examples = {ex["idx"]: ex for ex in att_examples}
    common_idx = sorted(set(shap_examples.keys()) & set(att_examples.keys()))
    print(f"[match] {len(common_idx)} common instances")

    paths = []
    agree_stats = []
    for idx in common_idx:
        ctx = build_fusion_context(
            shap_examples[idx], att_examples[idx], threshold=threshold,
        )
        out = out_dir / f"{idx}_{ctx.get('decision', '').lower()}_{shap_examples[idx]['tag']}.json"
        # 더 단순한 파일명: idx_tag.json
        out = out_dir / f"{idx}_{shap_examples[idx]['tag']}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(ctx, f, indent=2, ensure_ascii=False)
        paths.append(out)
        agree_stats.append({
            "idx": idx,
            "n_agreed": ctx["n_agreed"],
            "n_shap_only": ctx["n_shap_only"],
            "n_attention_only": ctx["n_attention_only"],
        })

    # 인덱스 + 통계 저장
    df_stats = pd.DataFrame(agree_stats)
    summary = {
        "n_contexts": len(paths),
        "shap_source": str(shap_path),
        "attention_source": str(att_path),
        "threshold": threshold,
        "agreement_stats": {
            "n_agreed_mean": round(float(df_stats["n_agreed"].mean()), 2),
            "n_shap_only_mean": round(float(df_stats["n_shap_only"].mean()), 2),
            "n_attention_only_mean": round(float(df_stats["n_attention_only"].mean()), 2),
            "n_agreed_dist": df_stats["n_agreed"].value_counts().sort_index().to_dict(),
        },
        "files": [p.name for p in paths],
    }
    with open(out_dir / "_index.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] {len(paths)}개 fusion 컨텍스트 → {out_dir}")
    print(f"     agreement 평균: {summary['agreement_stats']['n_agreed_mean']} 개 / "
          f"shap_only {summary['agreement_stats']['n_shap_only_mean']} / "
          f"att_only {summary['agreement_stats']['n_attention_only_mean']}")
    print(f"     n_agreed 분포: {summary['agreement_stats']['n_agreed_dist']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap",
                    default=str(RESULTS_DIR / "shap_local_examples_100.json"))
    ap.add_argument("--attention",
                    default=str(RESULTS_DIR / "tabnet_local_attention_100.json"))
    ap.add_argument("--out-dir",
                    default=str(RESULTS_DIR / "contexts_fusion_100"))
    ap.add_argument("--threshold", type=float, default=0.476,
                    help="XGBoost test threshold (Day 2)")
    args = ap.parse_args()
    main(Path(args.shap), Path(args.attention), Path(args.out_dir),
         threshold=args.threshold)
