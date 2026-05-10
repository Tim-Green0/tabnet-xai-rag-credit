"""Step 5-A: Fairness-aware Learning (Reweighing + Fairlearn ExponentiatedGradient).

Day 5의 진단(8/8 4/5 rule 위반)에 대응하는 mitigation 단계.
보호속성 ablation 비교(Day 5)에서 한 단계 더 나아가 정식 fairness-aware 학습 비교.

방법론:
    1. Reweighing (Kamiran & Calders 2012) — sample_weight 조정
       - w(s, y) = P(S=s) * P(Y=y) / P(S=s, Y=y)
    2. Fairlearn ExponentiatedGradient — Reduction-based, DP / EO constraint
       - Agarwal et al. 2018, AISTATS
    3. Baseline (no mitigation) — 비교 기준

데이터:
    - baseline (214 features) — Step 1 데이터, Day 5와 직접 비교
    - aux (1161 features) — Step 3-B 데이터, Reweighing만 (Fairlearn은 시간 부담)

보호 속성:
    - GENDER: CODE_GENDER_F/M (binary)
    - AGE: DAYS_BIRTH < median (binary, 50대 기준)

평가 모델:
    - XGBoost (Step 1~3 메인)

산출:
    results/fairness_mitigation_v2.csv  — 모델 × 데이터 × 방법 × 보호속성 × 메트릭
    figures/33_fairness_tradeoff.png    — AUROC vs DI scatter
    figures/34_mitigation_bars.png      — 방법별 DI/DP/AUROC 비교

사용:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.fairness_mitigation
        [--data baseline|aux|both] [--methods reweighing fairlearn_dp fairlearn_eo]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from fairlearn.reductions import (
    DemographicParity, EqualizedOdds, ExponentiatedGradient
)

from src.fairness import fairness_metrics
from src.metrics import compute_metrics, find_threshold_youden
from src.utils import FIGURES_DIR, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

PROCESSED_DIR = Path("data/processed")
TARGET_COL = "TARGET"
GENDER_COLS = ["CODE_GENDER_F", "CODE_GENDER_M"]
AGE_COL = "DAYS_BIRTH"


# ─────────────────────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────────────────────
def load_data(suffix: str = ""):
    """suffix='' (baseline 214) 또는 '_aux' (1161 features) 데이터 로드."""
    train = pd.read_parquet(PROCESSED_DIR / f"train_unscaled{suffix}.parquet")
    val = pd.read_parquet(PROCESSED_DIR / f"val_unscaled{suffix}.parquet")
    test = pd.read_parquet(PROCESSED_DIR / f"test_unscaled{suffix}.parquet")
    y_tr = train[TARGET_COL].astype(int).values
    y_val = val[TARGET_COL].astype(int).values
    y_te = test[TARGET_COL].astype(int).values
    X_tr = train.drop(columns=[TARGET_COL])
    X_val = val.drop(columns=[TARGET_COL])
    X_te = test.drop(columns=[TARGET_COL])
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def get_sensitive_binary(X: pd.DataFrame, attr: str) -> np.ndarray:
    """보호 속성을 binary 벡터로 변환.

    GENDER: CODE_GENDER_F=1 → "F", else → "M"  (XNA는 minority라 "M"으로 통합)
    AGE: DAYS_BIRTH < median(train) → "young", else → "old"
    """
    if attr == "GENDER":
        if "CODE_GENDER_F" in X.columns:
            return np.where(X["CODE_GENDER_F"].values >= 0.5, "F", "M")
        else:
            raise KeyError("CODE_GENDER_F not in features")
    elif attr == "AGE":
        if AGE_COL not in X.columns:
            raise KeyError(f"{AGE_COL} not in features")
        # DAYS_BIRTH는 음수, 절대값이 클수록 나이 많음. median으로 binary
        median_birth = float(np.median(X[AGE_COL].values))
        return np.where(X[AGE_COL].values < median_birth, "old", "young")
    else:
        raise ValueError(f"unknown attr: {attr}")


# ─────────────────────────────────────────────────────────────
# Reweighing (Kamiran & Calders 2012)
# ─────────────────────────────────────────────────────────────
def compute_reweighing(y: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
    """w(s, y) = P(S=s) * P(Y=y) / P(S=s, Y=y).

    공정한 분포에서 기대되는 P(S=s) * P(Y=y)와 실제 P(S=s, Y=y)의 비율로
    각 sample에 가중치 부여 → 학습 시 보호 속성과 target의 의존성 약화.
    """
    n = len(y)
    weights = np.ones(n, dtype=np.float64)
    for s in np.unique(sensitive):
        for label in [0, 1]:
            mask = (sensitive == s) & (y == label)
            n_s = (sensitive == s).sum()
            n_y = (y == label).sum()
            n_sy = mask.sum()
            if n_sy == 0:
                continue
            weights[mask] = (n_s / n) * (n_y / n) / (n_sy / n)
    return weights


# ─────────────────────────────────────────────────────────────
# Mitigation 학습 함수
# ─────────────────────────────────────────────────────────────
def _xgb_estimator(pos_weight=None) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_weight if pos_weight else 1.0,
        random_state=SEED, eval_metric="auc",
        n_jobs=-1, tree_method="hist", verbosity=0,
    )


def train_baseline(X_tr, y_tr, X_val, y_val):
    """기존 XGBoost (no mitigation, scale_pos_weight 적용)."""
    pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_w, random_state=SEED,
        eval_metric="auc", early_stopping_rounds=30,
        n_jobs=-1, tree_method="hist", verbosity=0,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def train_reweighing(X_tr, y_tr, X_val, y_val, sensitive_tr):
    """Reweighing — sample_weight를 Kamiran-Calders 공식으로 부여 후 학습."""
    weights = compute_reweighing(y_tr, sensitive_tr)
    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        random_state=SEED, eval_metric="auc",
        early_stopping_rounds=30, n_jobs=-1,
        tree_method="hist", verbosity=0,
    )
    # sample_weight과 scale_pos_weight 둘 다 쓸 수 있으나 충돌 가능 → sample_weight만
    clf.fit(X_tr, y_tr, sample_weight=weights,
            eval_set=[(X_val, y_val)], verbose=False)
    return clf, weights


def train_fairlearn(X_tr, y_tr, sensitive_tr, constraint_name: str = "DP",
                     n_estimators: int = 200, max_iter: int = 30):
    """Fairlearn ExponentiatedGradient — Reduction-based fairness training.

    ExpGrad는 내부적으로 estimator를 max_iter회 재학습하므로 시간 부담.
    n_estimators를 줄여서 inner XGBoost를 가볍게.
    """
    if constraint_name == "DP":
        constraint = DemographicParity()
    elif constraint_name == "EO":
        constraint = EqualizedOdds()
    else:
        raise ValueError(f"unknown constraint: {constraint_name}")

    pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    estimator = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=5, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_w, random_state=SEED,
        eval_metric="auc", n_jobs=-1, tree_method="hist", verbosity=0,
    )
    eg = ExponentiatedGradient(estimator=estimator, constraints=constraint,
                                  max_iter=max_iter)
    # Fairlearn은 sensitive_features를 그대로 받음 (string 가능)
    eg.fit(X_tr, y_tr, sensitive_features=sensitive_tr)
    return eg


# ─────────────────────────────────────────────────────────────
# Inference + threshold
# ─────────────────────────────────────────────────────────────
def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """ExpGrad의 predict는 binary, predict_proba는 randomized predictor라 deterministic 아님.
    여기서는 _pmf_predict로 weighted prediction을 받아 score로 사용.
    """
    if hasattr(model, "_pmf_predict"):
        # Fairlearn ExpGrad
        try:
            scores = model._pmf_predict(X)
            # _pmf_predict는 (n, 2) — column 1이 P(Y=1)
            if scores.ndim == 2 and scores.shape[1] == 2:
                return scores[:, 1]
            else:
                return scores
        except Exception:
            return model.predict(X).astype(float)
    return model.predict_proba(X)[:, 1]


def evaluate(model, X_te, y_te, sensitive_te,
              X_val=None, y_val=None) -> Dict:
    """val에서 threshold 결정 (Youden) → test에서 fairness + 성능 측정."""
    if X_val is not None and y_val is not None:
        val_s = predict_proba(model, X_val)
        thr, _ = find_threshold_youden(y_val, val_s)
    else:
        thr = 0.5
    test_s = predict_proba(model, X_te)
    test_pred = (test_s >= thr).astype(int)

    # 성능
    perf = compute_metrics(y_te, test_s, thr)
    # 공정성
    fair = fairness_metrics(y_te, test_pred, sensitive_te)

    return {
        "threshold": float(thr),
        "auroc": perf["auroc"],
        "auprc": perf["auprc"],
        "ks": perf["ks"],
        "f1": perf["f1"],
        "demographic_parity_diff": fair["demographic_parity_diff"],
        "equal_opportunity_diff": fair["equal_opportunity_diff"],
        "equalized_odds_diff": fair["equalized_odds_diff"],
        "disparate_impact_ratio": fair["disparate_impact_ratio"],
        "passes_4_5_rule": fair["passes_4_5_rule"],
        "selection_rate": fair["selection_rate"],
        "n_per_group": fair["n_per_group"],
    }


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────
def plot_tradeoff(df: pd.DataFrame,
                    out_name: str = "33_fairness_tradeoff") -> Path:
    """AUROC vs DI scatter. method/data/attr별 색상."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, attr in zip(axes, ["GENDER", "AGE"]):
        sub = df[df["protected_attr"] == attr]
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        markers = {"baseline": "o", "reweighing": "s",
                   "fairlearn_dp": "^", "fairlearn_eo": "D"}
        palette = {"baseline_data": "#4C72B0", "aux_data": "#DD8452"}
        for _, r in sub.iterrows():
            ax.scatter(r["auroc"], r["disparate_impact_ratio"],
                        marker=markers.get(r["method"], "o"),
                        color=palette.get(r["data"], "#666666"),
                        s=180, edgecolors="black", linewidth=1,
                        label=f"{r['method']} / {r['data']}")
        ax.axhline(0.8, color="red", linestyle="--", alpha=0.5,
                    label="4/5 rule = 0.8")
        ax.set_xlabel("Test AUROC (↑)")
        ax.set_ylabel("Disparate Impact ratio (↑, ≥0.8 = pass)")
        ax.set_title(f"{attr}")
        # 중복 legend 제거
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique_handles = []
        unique_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        ax.legend(unique_handles, unique_labels, fontsize=8, loc="best")
    plt.suptitle("Fairness-aware mitigation — AUROC vs Disparate Impact (test)")
    return savefig(fig, out_name)


def plot_metric_bars(df: pd.DataFrame,
                      out_name: str = "34_mitigation_bars") -> Path:
    """방법별 DP/DI/AUROC 막대 그래프 (보호속성 × 데이터)."""
    metrics = [("disparate_impact_ratio", "Disparate Impact (↑)", 0.8),
                ("demographic_parity_diff", "Demographic Parity Diff (↓)", 0.1),
                ("auroc", "Test AUROC (↑)", None)]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(14, 12))
    for r_idx, (m, label, threshold) in enumerate(metrics):
        for c_idx, attr in enumerate(["GENDER", "AGE"]):
            ax = axes[r_idx, c_idx]
            sub = df[df["protected_attr"] == attr]
            if len(sub) == 0:
                ax.set_visible(False)
                continue
            sns.barplot(data=sub, x="method", y=m, hue="data",
                         ax=ax, palette={"baseline_data": "#4C72B0",
                                          "aux_data": "#DD8452"})
            if threshold is not None:
                ax.axhline(threshold, color="red", linestyle="--", alpha=0.5)
            ax.set_title(f"{attr}: {label}")
            ax.tick_params(axis="x", rotation=15)
            if r_idx > 0 or c_idx > 0:
                if ax.legend_:
                    ax.legend_.remove()
    plt.suptitle("Mitigation methods × dataset × protected attribute")
    return savefig(fig, out_name)


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def run_one_setting(data_label: str, suffix: str, attr: str,
                     methods: List[str]) -> List[Dict]:
    """단일 (data, attr) 조합에 대해 N 방법 학습 + 평가."""
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_data(suffix)
    sensitive_tr = get_sensitive_binary(X_tr, attr)
    sensitive_te = get_sensitive_binary(X_te, attr)
    print(f"\n[setting] data={data_label} ({X_tr.shape[1]} feat), attr={attr}, "
          f"groups={dict(zip(*np.unique(sensitive_tr, return_counts=True)))}")

    rows = []
    if "baseline" in methods:
        print(f"  [baseline]")
        t0 = time.time()
        clf = train_baseline(X_tr, y_tr, X_val, y_val)
        m = evaluate(clf, X_te, y_te, sensitive_te, X_val, y_val)
        m.update({"data": data_label, "method": "baseline",
                   "protected_attr": attr,
                   "elapsed_sec": round(time.time() - t0, 1)})
        rows.append(m)
        print(f"    AUROC={m['auroc']:.4f}, DI={m['disparate_impact_ratio']:.3f}, "
              f"DP={m['demographic_parity_diff']:.3f}, pass={m['passes_4_5_rule']}")

    if "reweighing" in methods:
        print(f"  [reweighing]")
        t0 = time.time()
        clf, _ = train_reweighing(X_tr, y_tr, X_val, y_val, sensitive_tr)
        m = evaluate(clf, X_te, y_te, sensitive_te, X_val, y_val)
        m.update({"data": data_label, "method": "reweighing",
                   "protected_attr": attr,
                   "elapsed_sec": round(time.time() - t0, 1)})
        rows.append(m)
        print(f"    AUROC={m['auroc']:.4f}, DI={m['disparate_impact_ratio']:.3f}, "
              f"DP={m['demographic_parity_diff']:.3f}, pass={m['passes_4_5_rule']}")

    if "fairlearn_dp" in methods:
        print(f"  [fairlearn DP]")
        t0 = time.time()
        eg = train_fairlearn(X_tr, y_tr, sensitive_tr, "DP")
        m = evaluate(eg, X_te, y_te, sensitive_te, X_val, y_val)
        m.update({"data": data_label, "method": "fairlearn_dp",
                   "protected_attr": attr,
                   "elapsed_sec": round(time.time() - t0, 1)})
        rows.append(m)
        print(f"    AUROC={m['auroc']:.4f}, DI={m['disparate_impact_ratio']:.3f}, "
              f"DP={m['demographic_parity_diff']:.3f}, pass={m['passes_4_5_rule']}")

    if "fairlearn_eo" in methods:
        print(f"  [fairlearn EO]")
        t0 = time.time()
        eg = train_fairlearn(X_tr, y_tr, sensitive_tr, "EO")
        m = evaluate(eg, X_te, y_te, sensitive_te, X_val, y_val)
        m.update({"data": data_label, "method": "fairlearn_eo",
                   "protected_attr": attr,
                   "elapsed_sec": round(time.time() - t0, 1)})
        rows.append(m)
        print(f"    AUROC={m['auroc']:.4f}, DI={m['disparate_impact_ratio']:.3f}, "
              f"DP={m['demographic_parity_diff']:.3f}, pass={m['passes_4_5_rule']}")

    return rows


def main(data_choice: str = "both",
         methods: List[str] = None,
         skip_fairlearn_aux: bool = True) -> None:
    set_seed(SEED)
    if methods is None:
        methods = ["baseline", "reweighing", "fairlearn_dp", "fairlearn_eo"]

    plan = []
    if data_choice in ("baseline", "both"):
        plan.append(("baseline_data", ""))
    if data_choice in ("aux", "both"):
        plan.append(("aux_data", "_aux"))

    all_rows = []
    for data_label, suffix in plan:
        for attr in ["GENDER", "AGE"]:
            # aux 데이터에서 fairlearn은 너무 오래 걸려서 옵션으로 skip
            cur_methods = methods.copy()
            if data_label == "aux_data" and skip_fairlearn_aux:
                cur_methods = [m for m in cur_methods if not m.startswith("fairlearn")]
                print(f"\n[note] aux_data + fairlearn_* 스킵 (시간 부담). "
                      f"baseline + reweighing만.")
            rows = run_one_setting(data_label, suffix, attr, cur_methods)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / "fairness_mitigation_v2.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv} ({len(df)} rows)")
    print(df[["data", "protected_attr", "method", "auroc",
                "disparate_impact_ratio", "demographic_parity_diff",
                "passes_4_5_rule"]].to_string(index=False))

    # 시각화
    plot_tradeoff(df)
    plot_metric_bars(df)
    print(f"[save] figures/33_fairness_tradeoff.png, figures/34_mitigation_bars.png")
    print("\n[OK] fairness mitigation 완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="both",
                    choices=["baseline", "aux", "both"])
    ap.add_argument("--methods", nargs="+", default=None,
                    help="baseline reweighing fairlearn_dp fairlearn_eo")
    ap.add_argument("--include-fairlearn-aux", action="store_true",
                    help="aux 데이터에도 fairlearn 적용 (시간 매우 오래)")
    args = ap.parse_args()
    main(data_choice=args.data,
          methods=args.methods,
          skip_fairlearn_aux=not args.include_fairlearn_aux)
