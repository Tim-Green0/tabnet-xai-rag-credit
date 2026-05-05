"""Step 2-A Phase 1 — SHAP local 샘플 100명으로 확장.

기존: 5 reject + 5 accept = 10
신규: 50 reject + 50 accept = 100

source: XGBoost test set의 high-confidence 예측 인스턴스
- reject 후보: 부도 확률 상위 200명 → 무작위 50명
- accept 후보: 부도 확률 하위 200명 → 무작위 50명

산출:
  - results/shap_local_examples_100.json
  - results/contexts_100/{idx}_{tag}.json (100개)

실행:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.expand_samples
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.context_builder import build_context, save_context
from src.metrics import find_threshold_youden
from src.shap_analysis import _XgbNativeExplainer
from src.utils import RESULTS_DIR, SEED, set_seed

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = RESULTS_DIR / "baseline_models"
TARGET_COL = "TARGET"

CONTEXTS100_DIR = RESULTS_DIR / "contexts_100"
CONTEXTS100_DIR.mkdir(parents=True, exist_ok=True)


def main(n_per_class: int = 50, pool_size: int = 200) -> None:
    set_seed(SEED)
    print(f"[1/4] 데이터·모델 로딩")
    test = pd.read_parquet(PROCESSED_DIR / "test_unscaled.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val_unscaled.parquet")
    y_test = test[TARGET_COL].astype(int).values
    X_test = test.drop(columns=[TARGET_COL])
    y_val = val[TARGET_COL].astype(int).values
    X_val = val.drop(columns=[TARGET_COL])
    xgb_model = joblib.load(MODELS_DIR / "xgboost.pkl")
    val_score = xgb_model.predict_proba(X_val)[:, 1]
    thr, _ = find_threshold_youden(y_val, val_score)
    print(f"     threshold = {thr:.4f}")

    print(f"[2/4] high-conf reject {n_per_class} + high-conf accept "
          f"{n_per_class} 추출")
    proba_test = xgb_model.predict_proba(X_test)[:, 1]
    rng = np.random.RandomState(SEED)
    reject_pool = np.argsort(-proba_test)[:pool_size]
    accept_pool = np.argsort(proba_test)[:pool_size]
    reject_idx = rng.choice(reject_pool, n_per_class, replace=False)
    accept_idx = rng.choice(accept_pool, n_per_class, replace=False)
    indices = list(reject_idx) + list(accept_idx)
    tags = ["reject"] * n_per_class + ["accept"] * n_per_class

    print(f"[3/4] SHAP local 계산 ({len(indices)}명)")
    expl = _XgbNativeExplainer(xgb_model)
    X_subset = X_test.iloc[indices]
    sv_all = expl.shap_values(X_subset)  # (100, n_features)
    print(f"     sv shape = {sv_all.shape}, base = {expl.expected_value:.4f}")

    print(f"[4/4] context 빌드 + 저장 ({len(indices)}개)")
    feature_cols = X_test.columns.tolist()
    local_examples = []
    paths = []
    for k, (i, tag) in enumerate(zip(indices, tags)):
        sv = sv_all[k]
        X_one = X_test.iloc[[i]]
        feature_values = {c: float(X_one[c].iloc[0]) for c in feature_cols}
        shap_values = {c: float(v) for c, v in zip(feature_cols, sv)}
        proba = float(proba_test[i])

        ctx = build_context(
            sample_idx=int(i), y_score=proba, threshold=thr,
            feature_values=feature_values, shap_values=shap_values,
            top_k=5, model_name="XGBoost", true_label=int(y_test[i]),
        )
        # contexts_100 폴더에 저장
        fn = CONTEXTS100_DIR / f"{int(i)}_{tag}.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(ctx, f, indent=2, ensure_ascii=False)
        paths.append(fn)

        # local examples 형식 (Day 4 호환)
        contrib = pd.Series(sv, index=feature_cols)
        top_pos = contrib.sort_values(ascending=False).head(5)
        top_neg = contrib.sort_values().head(5)
        local_examples.append({
            "idx": int(i),
            "tag": tag,
            "true_label": int(y_test[i]),
            "predicted_proba": proba,
            "shap_base_value": expl.expected_value,
            "shap_sum": float(sv.sum()),
            "top_5_positive_drivers": [
                {"feature": k_, "shap": float(v),
                 "value": float(X_one[k_].iloc[0])
                 if pd.api.types.is_numeric_dtype(X_one[k_])
                 else str(X_one[k_].iloc[0])}
                for k_, v in top_pos.items()
            ],
            "top_5_negative_drivers": [
                {"feature": k_, "shap": float(v),
                 "value": float(X_one[k_].iloc[0])
                 if pd.api.types.is_numeric_dtype(X_one[k_])
                 else str(X_one[k_].iloc[0])}
                for k_, v in top_neg.items()
            ],
        })

    out_examples = RESULTS_DIR / "shap_local_examples_100.json"
    with open(out_examples, "w", encoding="utf-8") as f:
        json.dump(local_examples, f, indent=2, ensure_ascii=False)

    # 인덱스 파일
    idx_file = CONTEXTS100_DIR / "_index.json"
    with open(idx_file, "w", encoding="utf-8") as f:
        json.dump({
            "n_contexts": len(paths),
            "n_reject": n_per_class,
            "n_accept": n_per_class,
            "threshold": thr,
            "model": "XGBoost",
            "files": [p.name for p in paths],
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] {len(paths)}개 컨텍스트 + {out_examples}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--pool-size", type=int, default=200)
    args = ap.parse_args()
    main(n_per_class=args.n_per_class, pool_size=args.pool_size)
