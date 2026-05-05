"""Step 3-B-6: aux feature 추가 후 XGBoost SHAP 재실행 + top-20 변화 분석.

Step 1의 SHAP global (results/shap_global_xgboost.csv) 와 비교하여
보조 테이블에서 어떤 feature가 새로 top-k에 진입했는지 정량 분석.

산출:
    results/shap_global_xgboost_aux.csv
    results/shap_top20_diff.csv          — baseline top20 vs aux top20 비교
    figures/25_shap_global_xgb_aux.png   — aux 모델 SHAP top 20
    figures/26_shap_top20_overlap.png    — overlap diagram
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from src.metrics import find_threshold_youden
from src.shap_analysis import _XgbNativeExplainer, global_importance, plot_shap_global
from src.utils import FIGURES_DIR, RESULTS_DIR, SEED, savefig, set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = RESULTS_DIR / "baseline_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL = "TARGET"


def load_unscaled_aux():
    train = pd.read_parquet(PROCESSED_DIR / "train_unscaled_aux.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val_unscaled_aux.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test_unscaled_aux.parquet")
    return train, val, test


def train_xgb_aux(train: pd.DataFrame, val: pd.DataFrame):
    y_tr = train[TARGET_COL].astype(int).values
    X_tr = train.drop(columns=[TARGET_COL])
    y_va = val[TARGET_COL].astype(int).values
    X_va = val.drop(columns=[TARGET_COL])

    pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"[train] aux XGBoost: train={X_tr.shape}, val={X_va.shape}, pos_w={pos_w:.3f}")
    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_w, random_state=SEED,
        eval_metric="auc", early_stopping_rounds=30,
        n_jobs=-1, tree_method="hist",
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    el = time.time() - t0
    print(f"[train] elapsed={el:.1f}s, best_iteration={clf.best_iteration}")
    return clf, X_tr, X_va, y_tr, y_va


def compare_top20(aux_imp: pd.Series, baseline_path: Path = None) -> pd.DataFrame:
    baseline_path = baseline_path or (RESULTS_DIR / "shap_global_xgboost.csv")
    if not baseline_path.exists():
        print(f"[warn] baseline {baseline_path} 없음 - 비교 스킵")
        return pd.DataFrame()

    base = pd.read_csv(baseline_path)
    # 컬럼명 추정 (Step 1 shap_global_xgboost.csv 형식)
    if "feature" in base.columns:
        feat_col = "feature"
    else:
        feat_col = base.columns[0]
    if "importance" in base.columns:
        imp_col = "importance"
    elif "mean_abs_shap" in base.columns:
        imp_col = "mean_abs_shap"
    else:
        imp_col = base.columns[1]

    base_top = base.sort_values(imp_col, ascending=False).head(20)
    base_top_features = set(base_top[feat_col].tolist())
    aux_top_features = set(aux_imp.nlargest(20).index.tolist())

    rows = []
    for rank, (f, v) in enumerate(aux_imp.nlargest(20).items(), 1):
        is_new = f not in base_top_features
        is_aux = (
            f.startswith("BUREAU_") or f.startswith("PREV_")
            or f.endswith("_MISSING_FLAG") and (
                "BUREAU_" in f or "PREV_" in f
            )
        )
        rows.append({
            "rank_aux": rank,
            "feature": f,
            "shap_aux": float(v),
            "in_baseline_top20": not is_new,
            "is_aux_table_feature": is_aux,
        })
    return pd.DataFrame(rows)


def plot_top20_overlap(aux_imp: pd.Series, baseline_path: Path = None,
                        out_name: str = "29_shap_top20_overlap") -> Path:
    baseline_path = baseline_path or (RESULTS_DIR / "shap_global_xgboost.csv")
    base = pd.read_csv(baseline_path)
    feat_col = "feature" if "feature" in base.columns else base.columns[0]
    imp_col = "importance" if "importance" in base.columns else (
        "mean_abs_shap" if "mean_abs_shap" in base.columns else base.columns[1])
    base_top = base.sort_values(imp_col, ascending=False).head(20)
    aux_top = aux_imp.nlargest(20)

    base_set = set(base_top[feat_col].tolist())
    aux_set = set(aux_top.index.tolist())
    common = base_set & aux_set
    only_base = base_set - aux_set
    only_aux = aux_set - base_set

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    # 1) baseline top 20
    ax = axes[0]
    base_sorted = base_top.set_index(feat_col)[imp_col][::-1]
    colors = ["#55A868" if f in common else "#A0A0A0" for f in base_sorted.index]
    ax.barh(base_sorted.index, base_sorted.values, color=colors)
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title(f"Baseline (Step 1) top 20\n  Green = also in aux top 20 ({len(common)})")
    ax.tick_params(axis="y", labelsize=9)

    # 2) aux top 20
    ax = axes[1]
    aux_sorted = aux_top[::-1]
    def _color(f):
        if f in common:
            return "#55A868"
        if f.startswith("BUREAU_") or f.startswith("PREV_"):
            return "#DD8452"
        return "#A0A0A0"
    colors = [_color(f) for f in aux_sorted.index]
    ax.barh(aux_sorted.index, aux_sorted.values, color=colors)
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title(f"Aux model (Step 3-B) top 20\n"
                  f"  Green = overlap, Orange = new aux-table feature")
    ax.tick_params(axis="y", labelsize=9)

    plt.suptitle(f"SHAP top-20 change: baseline vs aux model "
                  f"(overlap={len(common)}, only-baseline={len(only_base)}, only-aux={len(only_aux)})")
    return savefig(fig, out_name)


def main(n_test: int = 5000) -> None:
    set_seed(SEED)
    print("[1/4] aux 데이터 로드 (unscaled)")
    train, val, test = load_unscaled_aux()
    print(f"     train={train.shape}, val={val.shape}, test={test.shape}")

    print("\n[2/4] aux XGBoost 학습")
    clf, X_tr, X_va, y_tr, y_va = train_xgb_aux(train, val)
    # 모델 저장
    out_model = MODELS_DIR / "xgboost_aux.pkl"
    joblib.dump(clf, out_model)
    print(f"     saved: {out_model}")

    # test 메트릭 (val에서 threshold)
    val_s = clf.predict_proba(X_va)[:, 1]
    thr, _ = find_threshold_youden(y_va, val_s)
    X_te = test.drop(columns=[TARGET_COL])
    y_te = test[TARGET_COL].astype(int).values
    test_s = clf.predict_proba(X_te)[:, 1]
    from src.metrics import compute_metrics
    m_test = compute_metrics(y_te, test_s, thr)
    print(f"     test AUROC={m_test['auroc']:.4f}, AUPRC={m_test['auprc']:.4f}, "
          f"KS={m_test['ks']:.4f}, F1={m_test['f1']:.4f}")

    print(f"\n[3/4] SHAP 계산 (n_test={n_test})")
    expl = _XgbNativeExplainer(clf)
    Xs = X_te.iloc[:n_test]
    t0 = time.time()
    sv = expl.shap_values(Xs)
    print(f"     elapsed={time.time()-t0:.1f}s, shape={sv.shape}")

    aux_imp = global_importance(sv, list(X_te.columns)).sort_values(ascending=False)
    aux_imp.to_csv(RESULTS_DIR / "shap_global_xgboost_aux.csv",
                    header=["mean_abs_shap"])
    print(f"     saved: results/shap_global_xgboost_aux.csv")

    print("\n[4/4] top 20 비교 + figure")
    diff = compare_top20(aux_imp)
    if not diff.empty:
        diff.to_csv(RESULTS_DIR / "shap_top20_diff.csv", index=False)
        n_aux_in_top20 = int(diff["is_aux_table_feature"].sum())
        n_new = int((~diff["in_baseline_top20"]).sum())
        print(f"     aux-table feature가 top 20에 {n_aux_in_top20}개 진입")
        print(f"     baseline top 20 대비 새로 진입한 feature {n_new}개")

    plot_shap_global(aux_imp, "XGBoost_aux", top=20, out_name="28_shap_global_xgb_aux")
    plot_top20_overlap(aux_imp)
    print(f"     figures saved.")
    print("\n[OK] aux SHAP 분석 완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=5000, help="SHAP 계산 샘플 수")
    args = ap.parse_args()
    main(n_test=args.n_test)
