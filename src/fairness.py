"""Day 5 — 공정성 진단 + 변수 ablation 기반 mitigation.

지표 (계획서 3.5.2):
  - Demographic Parity (DP):  ΔP(Ŷ=1)  by group
  - Equal Opportunity (EO):   ΔTPR     by group  (실제 양성에서)
  - Equalized Odds (EOdds):   max(ΔTPR, ΔFPR)
  - Disparate Impact (DI):    min/max P(Ŷ=1)  → 0.8 이상이면 4/5 rule 통과

보호 속성:
  - GENDER:  CODE_GENDER (F, M)
  - AGE:     DAYS_BIRTH로 50세 ± 이분
                (또는 25세 미만 vs 65세 이상 같은 두 그룹 비교)

Mitigation:
  - Ablation: 보호 속성 컬럼(CODE_GENDER_*, DAYS_BIRTH 등) 제거 후 재학습 → 비교

산출물:
  - results/fairness_metrics.csv      : 모델 × 보호속성 × 지표 (baseline)
  - results/fairness_mitigation.csv   : baseline vs ablated 비교
  - figures/18_fairness_metrics.png
  - figures/19_fairness_mitigation.png

실행:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.fairness
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression

from src.metrics import compute_metrics, find_threshold_youden
from src.utils import FIGURES_DIR, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = RESULTS_DIR / "baseline_models"
ABLATION_DIR = RESULTS_DIR / "fairness_models"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL = "TARGET"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 보호 속성 컬럼 (전처리된 feature 기준)
GENDER_COLS = ["CODE_GENDER_F", "CODE_GENDER_M", "CODE_GENDER_XNA"]
AGE_COL = "DAYS_BIRTH"


# ─────────────────────────────────────────────────────────────
# 데이터 + 보호속성 로딩
# ─────────────────────────────────────────────────────────────
def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = [re.sub(r'[",:{}\[\]\s]+', "_", c).strip("_") for c in df.columns]
    out = df.copy()
    out.columns = new_cols
    return out


def load_processed():
    splits = {}
    for tag in ("scaled", "unscaled"):
        d = {}
        for split in ("train", "val", "test"):
            df = pd.read_parquet(PROCESSED_DIR / f"{split}_{tag}.parquet")
            d[f"X_{split}"] = df.drop(columns=[TARGET_COL])
            d[f"y_{split}"] = df[TARGET_COL].astype(int).values
        splits[tag] = d
    # protected attrs (Day 1 preprocess에서 저장)
    pa = pd.read_parquet(PROCESSED_DIR / "test_protected_attrs.parquet")
    return splits, pa


# ─────────────────────────────────────────────────────────────
# 공정성 지표
# ─────────────────────────────────────────────────────────────
def fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       sensitive: np.ndarray) -> Dict[str, float]:
    """4종 지표를 dict로 반환. sensitive는 binary or categorical."""
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return {"error": f"sensitive has <2 groups: {groups}"}

    # Selection rate (P(Y=1|A=a))
    sel_rate = {}
    tpr = {}
    fpr = {}
    for a in groups:
        mask_a = sensitive == a
        sel_rate[str(a)] = float(y_pred[mask_a].mean())
        # TPR
        mask_pos = mask_a & (y_true == 1)
        tpr[str(a)] = float(y_pred[mask_pos].mean()) if mask_pos.sum() > 0 else float("nan")
        # FPR
        mask_neg = mask_a & (y_true == 0)
        fpr[str(a)] = float(y_pred[mask_neg].mean()) if mask_neg.sum() > 0 else float("nan")

    sr_vals = list(sel_rate.values())
    tpr_vals = [v for v in tpr.values() if not np.isnan(v)]
    fpr_vals = [v for v in fpr.values() if not np.isnan(v)]

    dp = float(max(sr_vals) - min(sr_vals))
    eo = float(max(tpr_vals) - min(tpr_vals)) if len(tpr_vals) >= 2 else float("nan")
    fpr_diff = float(max(fpr_vals) - min(fpr_vals)) if len(fpr_vals) >= 2 else float("nan")
    eodds = float(max(eo, fpr_diff)) if not (np.isnan(eo) or np.isnan(fpr_diff)) else float("nan")
    di = float(min(sr_vals) / max(sr_vals)) if max(sr_vals) > 0 else float("nan")

    return {
        "groups": [str(g) for g in groups],
        "n_per_group": {str(g): int((sensitive == g).sum()) for g in groups},
        "selection_rate": sel_rate,
        "tpr": tpr,
        "fpr": fpr,
        "demographic_parity_diff": dp,
        "equal_opportunity_diff": eo,
        "fpr_diff": fpr_diff,
        "equalized_odds_diff": eodds,
        "disparate_impact_ratio": di,
        "passes_4_5_rule": bool(di >= 0.8),
    }


# ─────────────────────────────────────────────────────────────
# 모델 inference 헬퍼
# ─────────────────────────────────────────────────────────────
def predict_proba_logistic(model, X):
    return model.predict_proba(X)[:, 1]


def predict_proba_xgb(model, X):
    return model.predict_proba(X)[:, 1]


def predict_proba_lgb(model, X):
    return model.predict_proba(_sanitize(X))[:, 1]


def predict_proba_tabnet(model, X):
    return model.predict_proba(X.values.astype(np.float32))[:, 1]


# ─────────────────────────────────────────────────────────────
# Ablation training (보호 속성 제거 후 재학습)
# ─────────────────────────────────────────────────────────────
def get_protected_cols_to_drop(feature_cols: list) -> list:
    cols = []
    for c in feature_cols:
        if c in GENDER_COLS or c == AGE_COL:
            cols.append(c)
    return cols


def train_xgb_ablation(X_tr, y_tr, X_val, y_val, drop_cols):
    Xt, Xv = X_tr.drop(columns=drop_cols), X_val.drop(columns=drop_cols)
    pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_w, random_state=SEED,
        eval_metric="auc", early_stopping_rounds=30,
        n_jobs=-1, tree_method="hist",
    )
    clf.fit(Xt, y_tr, eval_set=[(Xv, y_val)], verbose=False)
    return clf, drop_cols


def train_tabnet_ablation(X_tr, y_tr, X_val, y_val, drop_cols, params: dict):
    """보호 속성 제거 후 best params로 TabNet 재학습."""
    Xt = X_tr.drop(columns=drop_cols).values.astype(np.float32)
    Xv = X_val.drop(columns=drop_cols).values.astype(np.float32)
    clf = TabNetClassifier(
        n_d=params["n_d"], n_a=params["n_a"], n_steps=params["n_steps"],
        gamma=params["gamma"], lambda_sparse=params["lambda_sparse"],
        n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=params["lr"], weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.7),
        mask_type=params["mask_type"],
        seed=SEED, verbose=0, device_name=DEVICE,
    )
    clf.fit(Xt, y_tr, eval_set=[(Xv, y_val)],
            eval_name=["val"], eval_metric=["auc"],
            max_epochs=60, patience=12, batch_size=1024,
            virtual_batch_size=128, num_workers=0,
            drop_last=False, weights=1)
    return clf, drop_cols


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────
def plot_fairness_metrics(rows: list, out_name: str = "18_fairness_metrics") -> Path:
    df = pd.DataFrame(rows)
    metrics = ["demographic_parity_diff", "equal_opportunity_diff",
                "equalized_odds_diff", "disparate_impact_ratio"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, m in zip(axes, metrics):
        sns.barplot(data=df, x="model", y=m, hue="protected_attr",
                    ax=ax, palette="Set2")
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=15)
        if m == "disparate_impact_ratio":
            ax.axhline(0.8, color="red", linestyle="--", alpha=0.5,
                        label="4/5 rule = 0.8")
            ax.legend(fontsize=8)
        else:
            # 일반적으로 0에 가까울수록 공정 — 임계 0.1
            ax.axhline(0.1, color="orange", linestyle=":", alpha=0.5,
                        label="caution = 0.1")
            ax.legend(fontsize=8)

    plt.suptitle("공정성 지표 — baseline 모델 비교 (test set)")
    return savefig(fig, out_name)


def plot_mitigation(df: pd.DataFrame, out_name: str = "19_fairness_mitigation") -> Path:
    """baseline vs ablated 비교 — 성능과 공정성의 trade-off."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1) 성능 비교
    sns.barplot(data=df, x="model_label", y="auroc",
                hue="condition", ax=axes[0], palette="Set1")
    axes[0].set_title("Test AUROC: baseline vs ablated")
    axes[0].set_ylim(df["auroc"].min() - 0.01, df["auroc"].max() + 0.005)
    axes[0].tick_params(axis="x", rotation=15)

    # 2) 공정성 비교 (DP)
    sns.barplot(data=df, x="model_label", y="demographic_parity_diff",
                hue="condition", ax=axes[1], palette="Set1")
    axes[1].set_title("Demographic Parity Diff: baseline vs ablated")
    axes[1].axhline(0.1, color="orange", linestyle=":", alpha=0.6)
    axes[1].tick_params(axis="x", rotation=15)

    plt.suptitle("공정성 mitigation — 보호 속성 제거 후 재학습 효과")
    return savefig(fig, out_name)


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def reconstruct_sensitive(pa: pd.DataFrame, attr: str) -> np.ndarray:
    """test_protected_attrs.parquet에서 binary sensitive vector 재구성."""
    if attr == "GENDER":
        return pa["CODE_GENDER"].values  # 'F', 'M', 'XNA'
    elif attr == "AGE":
        # 50세 기준 이분
        return np.where(pa["AGE"].values >= 50, "older", "younger")
    else:
        raise ValueError(f"unknown attr {attr}")


def threshold_predictions(y_score: np.ndarray, threshold: float) -> np.ndarray:
    return (y_score >= threshold).astype(int)


def main(skip_mitigation: bool = False) -> None:
    set_seed(SEED)
    print(f"[device] {DEVICE}")

    print("[1/4] 데이터 + 모델 로딩")
    splits, pa = load_processed()
    X_train_sc = splits["scaled"]["X_train"]
    X_val_sc = splits["scaled"]["X_val"]
    X_test_sc = splits["scaled"]["X_test"]
    y_train = splits["scaled"]["y_train"]
    y_val = splits["scaled"]["y_val"]
    y_test = splits["scaled"]["y_test"]
    X_train_un = splits["unscaled"]["X_train"]
    X_val_un = splits["unscaled"]["X_val"]
    X_test_un = splits["unscaled"]["X_test"]

    log_model = joblib.load(MODELS_DIR / "logistic.pkl")
    xgb_model = joblib.load(MODELS_DIR / "xgboost.pkl")
    lgb_model = joblib.load(MODELS_DIR / "lightgbm.pkl")
    tabnet_model = TabNetClassifier()
    tabnet_model.load_model(str(MODELS_DIR / "tabnet_best.zip"))

    print("[2/4] 베이스라인 공정성 지표 계산")
    # 모델별 score 계산
    scores = {
        "Logistic": predict_proba_logistic(log_model, X_test_sc),
        "XGBoost": predict_proba_xgb(xgb_model, X_test_un),
        "LightGBM": predict_proba_lgb(lgb_model, X_test_un),
        "TabNet": predict_proba_tabnet(tabnet_model, X_test_sc),
    }
    # 각 모델별 threshold (validation에서 결정)
    val_scores = {
        "Logistic": predict_proba_logistic(log_model, X_val_sc),
        "XGBoost": predict_proba_xgb(xgb_model, X_val_un),
        "LightGBM": predict_proba_lgb(lgb_model, X_val_un),
        "TabNet": predict_proba_tabnet(tabnet_model, X_val_sc),
    }
    thresholds = {m: find_threshold_youden(y_val, vs)[0] for m, vs in val_scores.items()}
    print(f"  thresholds: {thresholds}")

    rows_baseline = []
    for model_name, score in scores.items():
        y_pred = threshold_predictions(score, thresholds[model_name])
        for attr in ["GENDER", "AGE"]:
            sens = reconstruct_sensitive(pa, attr)
            # XNA group 매우 적어서 (4명) 제외
            mask = sens != "XNA" if attr == "GENDER" else np.ones(len(sens), dtype=bool)
            fm = fairness_metrics(y_test[mask], y_pred[mask], sens[mask])
            rows_baseline.append({
                "model": model_name,
                "protected_attr": attr,
                "threshold": thresholds[model_name],
                **fm,
            })

    base_df = pd.DataFrame(rows_baseline)
    # JSON serializable 형태로 저장
    base_df_flat = base_df.copy()
    for col in ["groups", "n_per_group", "selection_rate", "tpr", "fpr"]:
        base_df_flat[col] = base_df_flat[col].apply(lambda v: json.dumps(v, ensure_ascii=False))
    base_df_flat.to_csv(RESULTS_DIR / "fairness_metrics.csv", index=False)
    plot_fairness_metrics(rows_baseline)

    # 표 출력
    print("\n[베이스라인 공정성 지표]")
    summary_cols = ["model", "protected_attr", "demographic_parity_diff",
                     "equal_opportunity_diff", "equalized_odds_diff",
                     "disparate_impact_ratio", "passes_4_5_rule"]
    print(base_df[summary_cols].round(4).to_string(index=False))

    # ── 3) Mitigation: 보호 속성 ablation 후 재학습 ──
    if skip_mitigation:
        print("\n[skip] mitigation 단계 건너뜀")
        return

    print("\n[3/4] 보호 속성 ablation 재학습 (XGBoost + TabNet만)")
    drop_un = get_protected_cols_to_drop(X_train_un.columns.tolist())
    drop_sc = get_protected_cols_to_drop(X_train_sc.columns.tolist())
    print(f"  drop (unscaled, for XGB): {drop_un}")
    print(f"  drop (scaled, for TabNet): {drop_sc}")

    # XGBoost ablated
    print("  [XGB ablated 학습]")
    t0 = time.time()
    xgb_abl, _ = train_xgb_ablation(X_train_un, y_train, X_val_un, y_val, drop_un)
    print(f"     elapsed={time.time()-t0:.1f}s")
    joblib.dump(xgb_abl, ABLATION_DIR / "xgboost_ablated.pkl")
    score_xgb_abl_val = xgb_abl.predict_proba(X_val_un.drop(columns=drop_un))[:, 1]
    score_xgb_abl_test = xgb_abl.predict_proba(X_test_un.drop(columns=drop_un))[:, 1]
    thr_xgb_abl, _ = find_threshold_youden(y_val, score_xgb_abl_val)

    # TabNet ablated
    print("  [TabNet ablated 학습 — best params 사용]")
    # Day 3 best params 그대로 사용
    tabnet_best_params = {
        "n_d": 16, "n_a": 16, "n_steps": 3,
        "gamma": 1.1559945203362028,
        "lambda_sparse": 1.3066739238053272e-05,
        "lr": 0.03674059202635224,
        "mask_type": "entmax",
    }
    t0 = time.time()
    tab_abl, _ = train_tabnet_ablation(X_train_sc, y_train, X_val_sc, y_val,
                                          drop_sc, tabnet_best_params)
    print(f"     elapsed={time.time()-t0:.1f}s")
    tab_abl.save_model(str(ABLATION_DIR / "tabnet_ablated"))
    score_tab_abl_val = tab_abl.predict_proba(
        X_val_sc.drop(columns=drop_sc).values.astype(np.float32))[:, 1]
    score_tab_abl_test = tab_abl.predict_proba(
        X_test_sc.drop(columns=drop_sc).values.astype(np.float32))[:, 1]
    thr_tab_abl, _ = find_threshold_youden(y_val, score_tab_abl_val)

    print("\n[4/4] mitigation 비교")

    rows_mit = []
    # 비교 대상: XGBoost, TabNet × {baseline, ablated} × {GENDER, AGE}
    pairs = [
        ("XGBoost", scores["XGBoost"], thresholds["XGBoost"], "baseline"),
        ("XGBoost", score_xgb_abl_test, thr_xgb_abl, "ablated"),
        ("TabNet", scores["TabNet"], thresholds["TabNet"], "baseline"),
        ("TabNet", score_tab_abl_test, thr_tab_abl, "ablated"),
    ]
    for model_name, score, thr, condition in pairs:
        y_pred = threshold_predictions(score, thr)
        # 성능 (mitigated 데이터에 동일 y_test)
        m_perf = compute_metrics(y_test, score, threshold=thr)
        for attr in ["GENDER", "AGE"]:
            sens = reconstruct_sensitive(pa, attr)
            mask = sens != "XNA" if attr == "GENDER" else np.ones(len(sens), dtype=bool)
            fm = fairness_metrics(y_test[mask], y_pred[mask], sens[mask])
            rows_mit.append({
                "model_label": f"{model_name}",
                "model": model_name,
                "condition": condition,
                "protected_attr": attr,
                "threshold": thr,
                "auroc": m_perf["auroc"],
                "auprc": m_perf["auprc"],
                "ks": m_perf["ks"],
                "f1": m_perf["f1"],
                **{k: v for k, v in fm.items()
                    if k in {"demographic_parity_diff", "equal_opportunity_diff",
                              "equalized_odds_diff", "disparate_impact_ratio",
                              "passes_4_5_rule"}},
            })

    mit_df = pd.DataFrame(rows_mit)
    mit_df.to_csv(RESULTS_DIR / "fairness_mitigation.csv", index=False)
    print("\n[mitigation 결과 비교]")
    print(mit_df.round(4).to_string(index=False))

    plot_mitigation(mit_df)

    # 한 줄 요약
    print("\n[요약]")
    for (m, attr), grp in mit_df.groupby(["model", "protected_attr"]):
        b = grp[grp["condition"] == "baseline"].iloc[0]
        a = grp[grp["condition"] == "ablated"].iloc[0]
        print(f"  {m} × {attr}: AUROC {b['auroc']:.4f} → {a['auroc']:.4f} "
              f"(Δ={a['auroc']-b['auroc']:+.4f}), DP {b['demographic_parity_diff']:.4f} "
              f"→ {a['demographic_parity_diff']:.4f}")

    print("\n[OK] Day 5 공정성 완료")
    print("     - results/fairness_metrics.csv")
    print("     - results/fairness_mitigation.csv")
    print("     - figures/18, 19")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mitigation", action="store_true")
    args = ap.parse_args()
    main(skip_mitigation=args.skip_mitigation)
