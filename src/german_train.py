"""UCI German Credit — XGBoost CV + TabNet 학습 + SHAP + TabNet attention (Step 5-D Day 1~2).

Home Credit 메커니즘 (XGBoost+SHAP + TabNet attention) 그대로 이식.
1000 샘플이라 단일 파일로 통합 (Home Credit은 모듈 분리).

Steps (cli `--step` 으로 분기):
  cv        : 5-fold CV (Logistic + XGBoost + LightGBM + TabNet)
  tabnet    : TabNet 단발 학습 → baseline_models/german_tabnet.zip
  xgb       : XGBoost 단발 학습 → baseline_models/german_xgb.pkl (SHAP용)
  shap      : XGBoost SHAP global + local (n=30 instances)
  attention : TabNet attention local (동일 30 instances)
  all       : 위 전체 순차 실행

산출물:
  results/baseline_models/german_xgb.pkl, german_tabnet.zip
  results/german_cv_metrics.csv, german_cv_summary.csv
  results/german_shap_global.csv
  results/german_shap_local.json (n=30)
  results/german_tabnet_attention.json (n=30)
  figures/38_german_cv.png, 39_german_shap_global.png

실행: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.german_train --step all
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Callable

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.metrics import compute_metrics, find_threshold_youden, metrics_table_row
from src.utils import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

GERMAN_DIR = PROJECT_ROOT / "data" / "german_credit"
PROCESSED_DIR = GERMAN_DIR / "processed"
MODELS_DIR = RESULTS_DIR / "baseline_models"
TARGET_COL = "TARGET"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────────────────────
def load_processed(scaled: bool = True):
    tag = "scaled" if scaled else "unscaled"
    train = pd.read_parquet(PROCESSED_DIR / f"train_{tag}.parquet")
    val = pd.read_parquet(PROCESSED_DIR / f"val_{tag}.parquet")
    test = pd.read_parquet(PROCESSED_DIR / f"test_{tag}.parquet")
    return train, val, test


def load_for_cv(scaled: bool = True):
    train, val, test = load_processed(scaled)
    df_all = pd.concat([train, val], ignore_index=True)
    y_all = df_all[TARGET_COL].astype(int).values
    X_all = df_all.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL].astype(int).values
    X_test = test.drop(columns=[TARGET_COL])
    return X_all, y_all, X_test, y_test


# ─────────────────────────────────────────────────────────────
# 모델별 fold fit (cv_eval.py 패턴)
# ─────────────────────────────────────────────────────────────
def fit_logistic(X_tr, y_tr, X_val, y_val, X_test):
    t0 = time.time()
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                              solver="lbfgs", n_jobs=-1, random_state=SEED)
    clf.fit(X_tr, y_tr)
    el = time.time() - t0
    return clf, clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1], el


def fit_xgboost(X_tr, y_tr, X_val, y_val, X_test):
    t0 = time.time()
    pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             scale_pos_weight=pos_w, random_state=SEED,
                             eval_metric="auc", early_stopping_rounds=30,
                             n_jobs=-1, tree_method="hist")
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    el = time.time() - t0
    return clf, clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1], el


def fit_lightgbm(X_tr, y_tr, X_val, y_val, X_test):
    t0 = time.time()
    clf = lgb.LGBMClassifier(n_estimators=300, max_depth=-1, num_leaves=31,
                              learning_rate=0.05, subsample=0.9,
                              colsample_bytree=0.9, class_weight="balanced",
                              random_state=SEED, n_jobs=-1, verbose=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(0)])
    el = time.time() - t0
    return clf, clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1], el


# 1000 샘플용 TabNet hyperparams (작은 batch)
TABNET_PARAMS = {
    "n_d": 8, "n_a": 8, "n_steps": 3, "gamma": 1.3,
    "lambda_sparse": 1e-4, "lr": 0.02, "mask_type": "entmax",
}


def fit_tabnet(X_tr, y_tr, X_val, y_val, X_test):
    t0 = time.time()
    clf = TabNetClassifier(
        n_d=TABNET_PARAMS["n_d"], n_a=TABNET_PARAMS["n_a"],
        n_steps=TABNET_PARAMS["n_steps"], gamma=TABNET_PARAMS["gamma"],
        lambda_sparse=TABNET_PARAMS["lambda_sparse"],
        n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=TABNET_PARAMS["lr"], weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.7),
        mask_type=TABNET_PARAMS["mask_type"],
        seed=SEED, verbose=0, device_name=DEVICE,
    )
    clf.fit(X_tr.values.astype(np.float32), y_tr,
            eval_set=[(X_val.values.astype(np.float32), y_val)],
            eval_name=["val"], eval_metric=["auc"],
            max_epochs=80, patience=15, batch_size=128, virtual_batch_size=32,
            num_workers=0, drop_last=False, weights=1)
    el = time.time() - t0
    val_s = clf.predict_proba(X_val.values.astype(np.float32))[:, 1]
    test_s = clf.predict_proba(X_test.values.astype(np.float32))[:, 1]
    return clf, val_s, test_s, el


# ─────────────────────────────────────────────────────────────
# CV 루프
# ─────────────────────────────────────────────────────────────
def cv_evaluate(model_name, X_all, y_all, X_test, y_test, fit_fn,
                n_splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    print(f"\n=== CV: {model_name} ===")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all)):
        X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all.iloc[va_idx], y_all[va_idx]
        clf, val_s, test_s, t = fit_fn(X_tr, y_tr, X_va, y_va, X_test)
        thr, _ = find_threshold_youden(y_va, val_s)
        m_val = compute_metrics(y_va, val_s, thr)
        m_test = compute_metrics(y_test, test_s, thr)
        rows.append(metrics_table_row(model_name, f"fold{fold}_val", t, m_val))
        rows.append(metrics_table_row(model_name, f"fold{fold}_test", t, m_test))
        print(f"  fold{fold}: time={t:.1f}s, val_AUROC={m_val['auroc']:.4f}, "
              f"test_AUROC={m_test['auroc']:.4f}, thr={thr:.3f}")
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split_kind), grp in df.assign(
            split_kind=df["split"].str.replace(r"fold\d+_", "", regex=True)
        ).groupby(["model", "split_kind"]):
        for col in ["auroc", "auprc", "ks", "f1", "precision", "recall", "accuracy", "time_sec"]:
            if col in grp.columns:
                rows.append({"model": model, "split": split_kind, "metric": col,
                              "mean": float(grp[col].mean()),
                              "std": float(grp[col].std()),
                              "n": int(len(grp))})
    return pd.DataFrame(rows)


def plot_cv(summary, name="38_german_cv"):
    test_summary = summary[summary["split"] == "test"]
    metrics = ["auroc", "auprc", "ks", "f1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    for ax, m in zip(axes, metrics):
        sub = test_summary[test_summary["metric"] == m].sort_values("mean", ascending=False)
        if len(sub) == 0:
            continue
        x = np.arange(len(sub))
        ax.bar(x, sub["mean"], yerr=sub["std"], capsize=6,
               color=["#DD8452", "#4C72B0", "#55A868", "#8172B2"][:len(sub)])
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"].tolist(), rotation=15, ha="right")
        ax.set_ylabel(m.upper())
        ax.set_title(f"{m.upper()} (test, mean ± std, 5-fold)")
        for i, (mn, sd) in enumerate(zip(sub["mean"].values, sub["std"].values)):
            ax.text(i, mn + sd + 0.005, f"{mn:.4f}\n±{sd:.4f}",
                    ha="center", va="bottom", fontsize=8)
    plt.suptitle("UCI German Credit — 5-fold CV (test set)")
    return savefig(fig, name)


# ─────────────────────────────────────────────────────────────
# Step: cv
# ─────────────────────────────────────────────────────────────
def step_cv():
    set_seed(SEED)
    print(f"[device] {DEVICE}")
    X_sc, y_sc, X_test_sc, y_test_sc = load_for_cv(scaled=True)
    X_un, y_un, X_test_un, y_test_un = load_for_cv(scaled=False)
    print(f"trainval={X_sc.shape}, test={X_test_sc.shape}")

    all_rows = []
    rows = cv_evaluate("Logistic", X_sc, y_sc, X_test_sc, y_test_sc, fit_logistic)
    all_rows.append(rows)
    rows = cv_evaluate("XGBoost", X_un, y_un, X_test_un, y_test_un, fit_xgboost)
    all_rows.append(rows)
    rows = cv_evaluate("LightGBM", X_un, y_un, X_test_un, y_test_un, fit_lightgbm)
    all_rows.append(rows)
    rows = cv_evaluate("TabNet", X_sc, y_sc, X_test_sc, y_test_sc, fit_tabnet)
    all_rows.append(rows)

    raw = pd.concat(all_rows, ignore_index=True)
    raw.to_csv(RESULTS_DIR / "german_cv_metrics.csv", index=False)
    summary = aggregate(raw)
    summary.to_csv(RESULTS_DIR / "german_cv_summary.csv", index=False)

    print("\n[CV summary — test set]")
    test_only = summary[summary["split"] == "test"].pivot_table(
        index="model", columns="metric", values=["mean", "std"]
    )
    print(test_only.round(4).to_string())

    plot_cv(summary)
    print("\n[OK] German Credit 5-fold CV 완료")


# ─────────────────────────────────────────────────────────────
# Step: tabnet (단발 학습 → SHAP/attention용)
# ─────────────────────────────────────────────────────────────
def step_tabnet():
    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = load_processed(scaled=True)
    y_tr, y_val, y_te = train[TARGET_COL].values, val[TARGET_COL].values, test[TARGET_COL].values
    X_tr = train.drop(columns=[TARGET_COL])
    X_val = val.drop(columns=[TARGET_COL])
    X_te = test.drop(columns=[TARGET_COL])

    print(f"[TabNet] train={X_tr.shape}")
    clf, val_s, test_s, t = fit_tabnet(X_tr, y_tr, X_val, y_val, X_te)
    thr, _ = find_threshold_youden(y_val, val_s)
    m = compute_metrics(y_te, test_s, thr)
    print(f"  test AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}, KS={m['ks']:.4f}, F1={m['f1']:.4f}")

    fp = MODELS_DIR / "german_tabnet"
    clf.save_model(str(fp))
    print(f"  [saved] {fp}.zip ({t:.1f}s 학습)")


# ─────────────────────────────────────────────────────────────
# Step: xgb (단발 학습 → SHAP용)
# ─────────────────────────────────────────────────────────────
def step_xgb():
    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = load_processed(scaled=False)
    y_tr, y_val, y_te = train[TARGET_COL].values, val[TARGET_COL].values, test[TARGET_COL].values
    X_tr = train.drop(columns=[TARGET_COL])
    X_val = val.drop(columns=[TARGET_COL])
    X_te = test.drop(columns=[TARGET_COL])

    print(f"[XGBoost] train={X_tr.shape}")
    clf, val_s, test_s, t = fit_xgboost(X_tr, y_tr, X_val, y_val, X_te)
    thr, _ = find_threshold_youden(y_val, val_s)
    m = compute_metrics(y_te, test_s, thr)
    print(f"  test AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}, KS={m['ks']:.4f}, F1={m['f1']:.4f}")

    fp = MODELS_DIR / "german_xgb.pkl"
    joblib.dump({"model": clf, "threshold": thr}, fp)
    print(f"  [saved] {fp} ({t:.1f}s 학습)")


# ─────────────────────────────────────────────────────────────
# SHAP (XGBoost native pred_contribs로 0.49 호환성 우회)
# ─────────────────────────────────────────────────────────────
class _XgbNativeExplainer:
    def __init__(self, model):
        self._booster = model.get_booster()
        self.expected_value: float = 0.0
        self._base_set = False

    def shap_values(self, X: pd.DataFrame):
        dm = xgb.DMatrix(X, feature_names=list(X.columns))
        contribs = self._booster.predict(dm, pred_contribs=True)
        if not self._base_set:
            self.expected_value = float(contribs[:, -1].mean())
            self._base_set = True
        return contribs[:, :-1]


def step_shap(n_local: int = 30):
    set_seed(SEED)
    train, val, test = load_processed(scaled=False)
    y_te = test[TARGET_COL].values
    X_te = test.drop(columns=[TARGET_COL])

    fp = MODELS_DIR / "german_xgb.pkl"
    bundle = joblib.load(fp)
    clf, thr = bundle["model"], bundle["threshold"]
    print(f"[SHAP/XGB] loaded {fp}, threshold={thr:.3f}")

    expl = _XgbNativeExplainer(clf)

    # Global: test 전체 (200)
    sv_all = expl.shap_values(X_te)
    print(f"  global SHAP shape={sv_all.shape}, base={expl.expected_value:.4f}")
    mean_abs = np.abs(sv_all).mean(axis=0)
    global_df = pd.DataFrame({
        "feature": list(X_te.columns),
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    global_df.to_csv(RESULTS_DIR / "german_shap_global.csv", index=False)
    print(f"  [saved] results/german_shap_global.csv (top5: {list(global_df.head(5)['feature'])})")

    # Global figure
    top20 = global_df.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color="#DD8452")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("XGBoost SHAP global importance (top 20) — German Credit")
    plt.tight_layout()
    savefig(fig, "39_german_shap_global")

    # Local: 30 instances (15 reject + 15 accept by predicted prob)
    proba = clf.predict_proba(X_te.values)[:, 1]
    rank = np.argsort(proba)
    n_each = n_local // 2
    accept_idx = rank[:n_each].tolist()  # 낮은 prob = 'good' 예측 = 정상
    reject_idx = rank[-n_each:][::-1].tolist()  # 높은 prob = 'bad' 예측 = 거절

    examples = []
    for tag, idxs in [("reject", reject_idx), ("accept", accept_idx)]:
        for i in idxs:
            sv = sv_all[i]  # (n_features,)
            top_idx = np.argsort(np.abs(sv))[::-1][:5]
            top_features = []
            for rank_, fi in enumerate(top_idx, start=1):
                top_features.append({
                    "feature": list(X_te.columns)[fi],
                    "shap_value": round(float(sv[fi]), 6),
                    "value": float(X_te.iloc[i, fi]),
                    "rank": rank_,
                })
            examples.append({
                "idx": int(i),
                "tag": tag,
                "true_label": int(y_te[i]),
                "predicted_proba": round(float(proba[i]), 6),
                "predicted_label": int(proba[i] >= thr),
                "expected_value": expl.expected_value,
                "top_k_shap": top_features,
            })

    out_path = RESULTS_DIR / "german_shap_local.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"  [saved] {out_path} (n={len(examples)}, {n_each} reject + {n_each} accept)")


# ─────────────────────────────────────────────────────────────
# TabNet attention local (local examples idx 일치)
# ─────────────────────────────────────────────────────────────
def step_attention():
    set_seed(SEED)
    train, val, test = load_processed(scaled=True)
    X_te = test.drop(columns=[TARGET_COL])

    # Local examples 로딩 (idx 동기화)
    with open(RESULTS_DIR / "german_shap_local.json", "r", encoding="utf-8") as f:
        examples = json.load(f)
    print(f"[Attention] {len(examples)} examples 로드")

    fp = MODELS_DIR / "german_tabnet.zip"
    clf = TabNetClassifier()
    clf.load_model(str(fp))
    print(f"  [load] {fp}")

    target_idx = [ex["idx"] for ex in examples]
    X_sub = X_te.iloc[target_idx].copy()
    feature_names = list(X_sub.columns)

    M_explain, masks = clf.explain(X_sub.values.astype(np.float32))
    M_explain = np.asarray(M_explain)
    print(f"  M_explain shape={M_explain.shape}")
    print(f"  stats: min={M_explain.min():.4f}, max={M_explain.max():.4f}, mean={M_explain.mean():.4f}")

    output = []
    for i, ex in enumerate(examples):
        att = M_explain[i]
        top_idx = np.argsort(att)[::-1][:5]
        top_features = []
        for rank_, fi in enumerate(top_idx, start=1):
            fname = feature_names[fi]
            top_features.append({
                "feature": fname,
                "attention_score": round(float(att[fi]), 6),
                "value": float(X_sub.iloc[i][fname]),
                "rank": rank_,
            })
        output.append({
            "idx": ex["idx"],
            "tag": ex["tag"],
            "true_label": ex["true_label"],
            "predicted_proba_xgb": ex["predicted_proba"],
            "n_features_active": int((att > 0).sum()),
            "top_k_attention": top_features,
        })

    out_path = RESULTS_DIR / "german_tabnet_attention.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    n_active_mean = float(np.mean([o["n_features_active"] for o in output]))
    print(f"  [saved] {out_path}, 인스턴스당 활성 feature 평균={n_active_mean:.1f}")


# ─────────────────────────────────────────────────────────────
# Attention vs SHAP 일관성 (Spearman ρ + Top-K overlap)
# ─────────────────────────────────────────────────────────────
def step_consistency():
    set_seed(SEED)
    # Global SHAP 기준 비교: TabNet attention global (test 200 평균) vs SHAP global
    train, val, test = load_processed(scaled=True)
    X_te_sc = test.drop(columns=[TARGET_COL])
    train_un, _, test_un = load_processed(scaled=False)
    X_te_un = test_un.drop(columns=[TARGET_COL])

    # SHAP global 로드
    shap_df = pd.read_csv(RESULTS_DIR / "german_shap_global.csv")
    shap_df = shap_df.set_index("feature")["mean_abs_shap"]

    # TabNet attention global (test set 전체 평균)
    fp = MODELS_DIR / "german_tabnet.zip"
    clf = TabNetClassifier()
    clf.load_model(str(fp))
    M_all, _ = clf.explain(X_te_sc.values.astype(np.float32))
    M_all = np.asarray(M_all)
    att_global = pd.Series(M_all.mean(axis=0), index=X_te_sc.columns)

    # 컬럼 정렬 (한 컬럼이 두 셋 다 있어야 함; one-hot 차이 등은 join)
    common = list(set(shap_df.index) & set(att_global.index))
    s = shap_df.loc[common]
    a = att_global.loc[common]
    rho_full, p = spearmanr(s.values, a.values)

    # Top-K overlap
    top_overlap = {}
    for k in [10, 20, 50]:
        if k > len(common):
            continue
        top_s = set(s.sort_values(ascending=False).head(k).index)
        top_a = set(a.sort_values(ascending=False).head(k).index)
        top_overlap[f"top{k}_overlap"] = len(top_s & top_a) / k

    summary = {
        "n_features_common": len(common),
        "spearman_rho_full": float(rho_full),
        "spearman_p": float(p),
        **top_overlap,
    }
    with open(RESULTS_DIR / "german_attention_vs_shap.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[consistency] ρ={rho_full:.3f} (p={p:.3g}), top10_overlap={top_overlap.get('top10_overlap'):.2f}, "
          f"top20_overlap={top_overlap.get('top20_overlap'):.2f}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main(step: str = "all") -> None:
    set_seed(SEED)
    if step in ("cv", "all"):
        step_cv()
    if step in ("xgb", "all"):
        step_xgb()
    if step in ("tabnet", "all"):
        step_tabnet()
    if step in ("shap", "all"):
        step_shap()
    if step in ("attention", "all"):
        step_attention()
    if step in ("consistency", "all"):
        step_consistency()
    print("\n[OK] German Credit train pipeline 완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["cv", "xgb", "tabnet", "shap", "attention",
                                        "consistency", "all"], default="all")
    args = ap.parse_args()
    main(step=args.step)
