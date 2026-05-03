"""Day 4 — SHAP 분석 + 어텐션-SHAP 일관성.

대상 모델:
  - XGBoost: TreeExplainer (수천 샘플 빠름)
  - TabNet:  KernelExplainer로 200 샘플 (GradientExplainer가 pytorch-tabnet wrapper와
            호환 불완전한 경우가 있어 안정적인 KernelExplainer 사용)
  - Logistic: LinearExplainer (선형 가중치 기반, 빠름)

산출물:
  - results/shap_global_{xgboost,tabnet,logistic}.csv  : 변수별 mean(|SHAP|)
  - results/attention_vs_shap.csv      : Spearman ρ, Top-K Overlap
  - results/shap_local_examples.json   : 거절 5명 + 정상 5명 샘플의 local SHAP
                                          (Day 6 LLM 컨텍스트 입력용)
  - figures/14_shap_global_xgb.png
  - figures/15_shap_global_tabnet.png
  - figures/16_attention_vs_shap_scatter.png
  - figures/17_shap_local_waterfall.png

실행:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.shap_analysis [--n-test N]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from scipy.stats import spearmanr

from src.utils import FIGURES_DIR, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = RESULTS_DIR / "baseline_models"
TARGET_COL = "TARGET"


# ─────────────────────────────────────────────────────────────
# 유틸
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
    return splits


# ─────────────────────────────────────────────────────────────
# SHAP — 모델별
# ─────────────────────────────────────────────────────────────
class _XgbNativeExplainer:
    """SHAP 0.49 + XGBoost 3.x 호환성 버그 우회.

    XGBoost 자체의 pred_contribs API로 SHAP value 계산. 결과는 동일.
    """
    def __init__(self, model):
        self._booster = model.get_booster()
        # 한 번 prediction 돌려서 expected_value 계산 (1행이면 충분)
        self.expected_value: float = 0.0

    def shap_values(self, X: pd.DataFrame):
        dm = xgb.DMatrix(X, feature_names=list(X.columns))
        contribs = self._booster.predict(dm, pred_contribs=True)
        # contribs shape: (n, n_features+1). 마지막 컬럼은 bias(=base_score+intercept)
        if not hasattr(self, "_base_set") or not self._base_set:
            self.expected_value = float(contribs[:, -1].mean())
            self._base_set = True
        return contribs[:, :-1]


def shap_xgboost(model, X_test: pd.DataFrame, n_test: int):
    print(f"[SHAP/XGB] xgboost native pred_contribs (SHAP호환 우회), n_test={n_test}")
    t0 = time.time()
    expl = _XgbNativeExplainer(model)
    Xs = X_test.iloc[:n_test]
    sv = expl.shap_values(Xs)
    el = time.time() - t0
    print(f"  elapsed={el:.1f}s, shape={sv.shape}, base={expl.expected_value:.4f}")
    return sv, Xs, expl


def shap_logistic(model, X_train: pd.DataFrame, X_test: pd.DataFrame, n_test: int):
    print(f"[SHAP/Logistic] LinearExplainer, n_test={n_test}")
    t0 = time.time()
    expl = shap.LinearExplainer(model, X_train.sample(min(2000, len(X_train)),
                                                        random_state=SEED))
    Xs = X_test.iloc[:n_test]
    sv = expl.shap_values(Xs)
    el = time.time() - t0
    print(f"  elapsed={el:.1f}s, shape={sv.shape}")
    return sv, Xs, expl


def shap_tabnet(model: TabNetClassifier, X_train: pd.DataFrame,
                  X_test: pd.DataFrame, n_test: int = 100,
                  n_background: int = 50, nsamples: int = 100):
    print(f"[SHAP/TabNet] KernelExplainer (background={n_background}, "
          f"n_test={n_test}, nsamples={nsamples})")
    t0 = time.time()

    bg = X_train.sample(n_background, random_state=SEED).values.astype(np.float32)

    def predict_fn(X: np.ndarray):
        if X.ndim == 1:
            X = X[None, :]
        return model.predict_proba(X.astype(np.float32))[:, 1]

    expl = shap.KernelExplainer(predict_fn, bg)
    Xs = X_test.iloc[:n_test]
    sv = expl.shap_values(Xs.values.astype(np.float32),
                            nsamples=nsamples, silent=True)
    el = time.time() - t0
    print(f"  elapsed={el:.1f}s, shape={sv.shape}")
    return sv, Xs, expl


# ─────────────────────────────────────────────────────────────
# 분석
# ─────────────────────────────────────────────────────────────
def global_importance(sv: np.ndarray, feature_names: list) -> pd.Series:
    """mean(|SHAP|) per feature. SHAP global importance의 표준 정의."""
    return pd.Series(np.abs(sv).mean(axis=0), index=feature_names)


def attention_vs_shap(att: pd.Series, shap_imp: pd.Series, top_k: int = 20) -> dict:
    """어텐션 importance와 SHAP global의 일관성.

    - Spearman 순위 상관 (전체 + top 50)
    - Top-K overlap (Jaccard or count)
    """
    common = att.index.intersection(shap_imp.index)
    a = att.loc[common]
    s = shap_imp.loc[common]

    rho_all, p_all = spearmanr(a.values, s.values)
    # top 50만으로 별도 계산
    top50_union = a.nlargest(50).index.union(s.nlargest(50).index)
    if len(top50_union) >= 5:
        rho_top, p_top = spearmanr(a.loc[top50_union].values, s.loc[top50_union].values)
    else:
        rho_top, p_top = np.nan, np.nan

    a_top = set(a.nlargest(top_k).index.tolist())
    s_top = set(s.nlargest(top_k).index.tolist())
    overlap = len(a_top & s_top)
    union = a_top | s_top
    jaccard = overlap / len(union) if union else 0

    return {
        "n_common_features": int(len(common)),
        "spearman_rho_all": float(rho_all),
        "spearman_p_all": float(p_all),
        "spearman_rho_top50": float(rho_top) if not np.isnan(rho_top) else None,
        f"top{top_k}_overlap_count": int(overlap),
        f"top{top_k}_jaccard": float(jaccard),
        f"top{top_k}_attention": list(a_top),
        f"top{top_k}_shap": list(s_top),
        f"top{top_k}_intersection": list(a_top & s_top),
    }


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────
def plot_shap_global(imp: pd.Series, model_name: str, top: int = 20,
                       out_name: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    top_imp = imp.sort_values(ascending=False).head(top)[::-1]
    ax.barh(top_imp.index, top_imp.values, color="#4C72B0")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(f"{model_name} — SHAP Global Importance Top {top}")
    for i, v in enumerate(top_imp.values):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
    return savefig(fig, out_name or f"shap_global_{model_name.lower()}")


def plot_attention_vs_shap_scatter(att: pd.Series, shap_imp: pd.Series,
                                      stats: dict,
                                      out_name: str = "16_attention_vs_shap_scatter") -> Path:
    common = att.index.intersection(shap_imp.index)
    a = att.loc[common].rename("attention")
    s = shap_imp.loc[common].rename("shap_global")
    df = pd.concat([a, s], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # 전체
    axes[0].scatter(df["attention"], df["shap_global"], alpha=0.4, s=20, color="#8172B2")
    axes[0].set_xlabel("TabNet attention importance")
    axes[0].set_ylabel("SHAP global importance (mean |SHAP|)")
    axes[0].set_title(
        f"전체 변수 ({len(df)}개)\nSpearman ρ = {stats['spearman_rho_all']:.3f} (p={stats['spearman_p_all']:.3g})"
    )
    # Top 20 강조
    a_top = set(a.nlargest(20).index)
    s_top = set(s.nlargest(20).index)
    union_top = list(a_top | s_top)
    sub = df.loc[union_top]
    colors = ["#C44E52" if c in a_top & s_top else "#4C72B0" if c in a_top else "#DD8452"
               for c in sub.index]
    axes[1].scatter(sub["attention"], sub["shap_global"], c=colors, s=80, alpha=0.8,
                    edgecolor="black", linewidth=0.5)
    for c in sub.index:
        axes[1].annotate(c, (sub.loc[c, "attention"], sub.loc[c, "shap_global"]),
                         fontsize=7, alpha=0.8)
    axes[1].set_xlabel("TabNet attention importance")
    axes[1].set_ylabel("SHAP global importance")
    axes[1].set_title(
        f"Top-20 합집합 ({len(union_top)}개)\n"
        f"교집합 {stats['top20_overlap_count']}개 (Jaccard {stats['top20_jaccard']:.2f})\n"
        f"빨강=교집합, 파랑=어텐션 only, 주황=SHAP only"
    )
    plt.suptitle("어텐션 vs SHAP 일관성 (TabNet)")
    return savefig(fig, out_name)


def plot_local_waterfall(sv_local: np.ndarray, X_local: pd.DataFrame,
                            base_value: float, idx: int, label: int,
                            top: int = 10) -> Path:
    """단일 인스턴스의 SHAP local explanation을 막대그래프로."""
    contrib = pd.Series(sv_local, index=X_local.columns)
    top_idx = contrib.abs().sort_values(ascending=False).head(top).index
    sub = contrib.loc[top_idx][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in sub.values]
    ax.barh(sub.index, sub.values, color=colors)
    for i, (n, v) in enumerate(sub.items()):
        ax.text(v, i, f" {v:+.3f}  (값={X_local.loc[X_local.index[0], n]:.3f})",
                va="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("SHAP contribution to log-odds (양: 부도 ↑, 음: 부도 ↓)")
    ax.set_title(f"Local SHAP — sample idx={idx}, true label={label}, base={base_value:.3f}")
    return savefig(fig, f"17_local_waterfall_idx{idx}")


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main(n_test_xgb: int = 5000, n_test_tabnet: int = 200) -> None:
    set_seed(SEED)
    print("[1/6] 데이터/모델 로딩")
    splits = load_processed()
    X_train_sc = splits["scaled"]["X_train"]
    X_test_sc = splits["scaled"]["X_test"]
    y_test = splits["scaled"]["y_test"]
    X_train_un = splits["unscaled"]["X_train"]
    X_test_un = splits["unscaled"]["X_test"]

    # 모델 로딩
    log_model = joblib.load(MODELS_DIR / "logistic.pkl")
    xgb_model = joblib.load(MODELS_DIR / "xgboost.pkl")
    tabnet_model = TabNetClassifier()
    tabnet_model.load_model(str(MODELS_DIR / "tabnet_best.zip"))

    # ── 2) XGBoost SHAP ──
    print("\n[2/6] XGBoost SHAP")
    sv_xgb, Xs_xgb, expl_xgb = shap_xgboost(xgb_model, X_test_un, n_test=n_test_xgb)
    imp_xgb = global_importance(sv_xgb, X_test_un.columns.tolist())
    imp_xgb.sort_values(ascending=False).to_csv(
        RESULTS_DIR / "shap_global_xgboost.csv", header=["mean_abs_shap"])
    plot_shap_global(imp_xgb, "XGBoost", out_name="14_shap_global_xgboost")

    # ── 3) Logistic SHAP ──
    print("\n[3/6] Logistic SHAP")
    sv_log, Xs_log, expl_log = shap_logistic(log_model, X_train_sc, X_test_sc,
                                                n_test=n_test_xgb)
    imp_log = global_importance(sv_log, X_test_sc.columns.tolist())
    imp_log.sort_values(ascending=False).to_csv(
        RESULTS_DIR / "shap_global_logistic.csv", header=["mean_abs_shap"])
    plot_shap_global(imp_log, "Logistic", out_name="14b_shap_global_logistic")

    # ── 4) TabNet SHAP (KernelExplainer, 느림) ──
    print("\n[4/6] TabNet SHAP")
    sv_tab, Xs_tab, expl_tab = shap_tabnet(tabnet_model, X_train_sc, X_test_sc,
                                             n_test=n_test_tabnet,
                                             n_background=50, nsamples=100)
    imp_tab = global_importance(sv_tab, X_test_sc.columns.tolist())
    imp_tab.sort_values(ascending=False).to_csv(
        RESULTS_DIR / "shap_global_tabnet.csv", header=["mean_abs_shap"])
    plot_shap_global(imp_tab, "TabNet", out_name="15_shap_global_tabnet")

    # ── 5) 어텐션 vs SHAP 일관성 ──
    print("\n[5/6] 어텐션 vs SHAP 일관성 분석")
    att_imp = pd.read_csv(RESULTS_DIR / "tabnet_attention_importance.csv",
                           index_col=0)["importance"]
    stats = attention_vs_shap(att_imp, imp_tab, top_k=20)
    print(f"  Spearman ρ (전체) = {stats['spearman_rho_all']:.4f} (p={stats['spearman_p_all']:.3g})")
    print(f"  Top-20 overlap = {stats['top20_overlap_count']}/20 "
          f"(Jaccard {stats['top20_jaccard']:.3f})")
    print(f"  Top-20 intersection: {sorted(stats['top20_intersection'])[:10]}{'...' if len(stats['top20_intersection']) > 10 else ''}")

    with open(RESULTS_DIR / "attention_vs_shap.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 보기 좋은 csv도 따로
    cmp_df = pd.DataFrame({"attention": att_imp, "shap_global": imp_tab}).fillna(0)
    cmp_df["att_rank"] = cmp_df["attention"].rank(ascending=False).astype(int)
    cmp_df["shap_rank"] = cmp_df["shap_global"].rank(ascending=False).astype(int)
    cmp_df["rank_diff"] = (cmp_df["att_rank"] - cmp_df["shap_rank"]).abs()
    cmp_df.sort_values("attention", ascending=False).to_csv(
        RESULTS_DIR / "attention_vs_shap.csv")

    plot_attention_vs_shap_scatter(att_imp, imp_tab, stats)

    # ── 6) Local SHAP — Day 6 LLM 입력용 5+5 샘플 ──
    print("\n[6/6] Local SHAP 샘플 추출 (XGBoost 기준, 거절 5 + 정상 5)")
    proba = xgb_model.predict_proba(X_test_un)[:, 1]
    # high-confidence 거절 5명, high-confidence 정상 5명
    rng = np.random.RandomState(SEED)
    reject_idx_pool = np.argsort(-proba)[:200]   # 부도 확률 top 200
    accept_idx_pool = np.argsort(proba)[:200]    # 부도 확률 bottom 200
    reject_idx = rng.choice(reject_idx_pool, 5, replace=False)
    accept_idx = rng.choice(accept_idx_pool, 5, replace=False)

    local_examples = []
    for tag, indices in [("reject", reject_idx), ("accept", accept_idx)]:
        for i in indices:
            X_one = X_test_un.iloc[[i]]
            sv_one = expl_xgb.shap_values(X_one)[0]
            base = float(expl_xgb.expected_value)
            contrib = pd.Series(sv_one, index=X_test_un.columns)
            top_pos = contrib.sort_values(ascending=False).head(5)
            top_neg = contrib.sort_values().head(5)
            local_examples.append({
                "idx": int(i),
                "tag": tag,
                "true_label": int(y_test[i]),
                "predicted_proba": float(proba[i]),
                "shap_base_value": base,
                "shap_sum": float(sv_one.sum()),
                "top_5_positive_drivers": [
                    {"feature": k, "shap": float(v),
                     "value": float(X_one[k].iloc[0]) if pd.api.types.is_numeric_dtype(X_one[k]) else str(X_one[k].iloc[0])}
                    for k, v in top_pos.items()
                ],
                "top_5_negative_drivers": [
                    {"feature": k, "shap": float(v),
                     "value": float(X_one[k].iloc[0]) if pd.api.types.is_numeric_dtype(X_one[k]) else str(X_one[k].iloc[0])}
                    for k, v in top_neg.items()
                ],
            })
            # 1개만 워터폴 시각화 — 첫 reject
            if tag == "reject" and i == reject_idx[0]:
                plot_local_waterfall(sv_one, X_one, base, int(i), int(y_test[i]))

    with open(RESULTS_DIR / "shap_local_examples.json", "w", encoding="utf-8") as f:
        json.dump(local_examples, f, indent=2, ensure_ascii=False)
    print(f"  {len(local_examples)}개 샘플 저장")

    print("\n[OK] Day 4 SHAP 완료")
    print("     - results/shap_global_{xgboost,logistic,tabnet}.csv")
    print("     - results/attention_vs_shap.{csv,json}")
    print("     - results/shap_local_examples.json")
    print("     - figures/14, 14b, 15, 16, 17")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test-xgb", type=int, default=5000,
                    help="XGB/Logistic SHAP 평가 샘플 수")
    ap.add_argument("--n-test-tabnet", type=int, default=200,
                    help="TabNet KernelExplainer 평가 샘플 수 (느려서 작게)")
    args = ap.parse_args()
    main(n_test_xgb=args.n_test_xgb, n_test_tabnet=args.n_test_tabnet)
