"""Step 3-B-4: main + aux features 통합 전처리.

기존 src.preprocess는 application_train.csv 만 사용. 이 모듈은 거기에
src.aux_features 가 만든 집계 변수를 SK_ID_CURR 기준 left merge 한 뒤
같은 전처리기(HomeCreditPreprocessor)로 처리한다.

산출:
    data/processed/{train,val,test}_{scaled,unscaled}_aux.parquet
    data/processed/test_protected_attrs_aux.parquet
    results/preprocessor_aux.pkl
    results/preprocess_aux_summary.json

설계 결정:
- aux feature는 결측 많을 수 있음 (예: bureau 미커버 14.3%) → A1 정책에 따라
  결측 50%+ 컬럼 자동으로 _MISSING_FLAG 생성됨 (기존 전처리기 그대로 사용).
- bool/float/int 다 numeric으로 분류 → median impute + clip 적용.
- ID 컬럼(SK_ID_CURR)은 split_data가 자동 분리.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader import load_application_train
from src.preprocess import HomeCreditPreprocessor, split_data, ID_COL
from src.utils import RESULTS_DIR, SEED, set_seed

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def merge_aux(main_df: pd.DataFrame, aux_path: Path) -> pd.DataFrame:
    """main + aux features를 SK_ID_CURR 기준 left merge."""
    print(f"     aux 로드: {aux_path}")
    aux = pd.read_parquet(aux_path)
    print(f"     aux shape: {aux.shape}, unique IDs: {aux['SK_ID_CURR'].nunique():,}")
    before = main_df.shape
    merged = main_df.merge(aux, on="SK_ID_CURR", how="left")
    print(f"     main {before} + aux({aux.shape[1] - 1} new) -> merged {merged.shape}")
    return merged


def main(nrows: Optional[int] = None,
         aux_path: Optional[Path] = None) -> None:
    set_seed(SEED)
    aux_path = aux_path or (PROCESSED_DIR / "aux_features.parquet")
    if not aux_path.exists():
        raise FileNotFoundError(f"aux features not found: {aux_path}. Run src.aux_features first.")

    print(f"[1/5] application_train 로드 (nrows={nrows or 'all'})")
    df = load_application_train(nrows=nrows)
    print(f"     shape={df.shape}")

    print(f"[2/5] aux features merge")
    df = merge_aux(df, aux_path)

    print(f"[3/5] 분할 60/20/20 stratified")
    X_tr, X_val, X_te, y_tr, y_val, y_te, ids = split_data(df, seed=SEED)
    print(f"     train={X_tr.shape}, val={X_val.shape}, test={X_te.shape}")
    print(f"     pos_rate train={y_tr.mean():.4f}, val={y_val.mean():.4f}, test={y_te.mean():.4f}")

    print(f"[4/5] 전처리기 fit on train")
    pre = HomeCreditPreprocessor()
    pre.fit(X_tr, y_tr)
    print(f"     n_flag_cols={len(pre.flag_cols_)}, n_onehot_cats={len(pre.onehot_cols_)}, "
          f"n_target_encode={len(pre.target_encode_cols_)}")
    print(f"     n_features_out={len(pre.feature_names_)}")

    print(f"[5/5] transform + 저장")
    X_tr_p = pre.transform(X_tr, scale=True)
    X_val_p = pre.transform(X_val, scale=True)
    X_te_p = pre.transform(X_te, scale=True)
    X_tr_u = pre.transform(X_tr, scale=False)
    X_val_u = pre.transform(X_val, scale=False)
    X_te_u = pre.transform(X_te, scale=False)
    print(f"     scaled shapes: {X_tr_p.shape}, {X_val_p.shape}, {X_te_p.shape}")
    print(f"     NaN check: train={int(X_tr_p.isna().sum().sum())}, "
          f"val={int(X_val_p.isna().sum().sum())}, test={int(X_te_p.isna().sum().sum())}")

    X_tr_p.assign(TARGET=y_tr.values).to_parquet(PROCESSED_DIR / "train_scaled_aux.parquet")
    X_val_p.assign(TARGET=y_val.values).to_parquet(PROCESSED_DIR / "val_scaled_aux.parquet")
    X_te_p.assign(TARGET=y_te.values).to_parquet(PROCESSED_DIR / "test_scaled_aux.parquet")
    X_tr_u.assign(TARGET=y_tr.values).to_parquet(PROCESSED_DIR / "train_unscaled_aux.parquet")
    X_val_u.assign(TARGET=y_val.values).to_parquet(PROCESSED_DIR / "val_unscaled_aux.parquet")
    X_te_u.assign(TARGET=y_te.values).to_parquet(PROCESSED_DIR / "test_unscaled_aux.parquet")

    # 보호 속성 보존 (test set)
    age = (-X_te["DAYS_BIRTH"] / 365.25).clip(0, 100)
    fairness_df = pd.DataFrame({
        "SK_ID_CURR": ids["test"].values,
        "CODE_GENDER": X_te["CODE_GENDER"].values,
        "AGE": age.values,
        "TARGET": y_te.values,
    })
    fairness_df.to_parquet(PROCESSED_DIR / "test_protected_attrs_aux.parquet")

    pre.save(RESULTS_DIR / "preprocessor_aux.pkl")

    summary = {
        "config": {
            "seed": SEED,
            "aux_path": str(aux_path),
            "missing_flag_threshold": pre.missing_flag_threshold,
            "onehot_cardinality_max": pre.onehot_cardinality_max,
            "clip_quantiles": list(pre.clip_quantiles),
        },
        "shapes": {
            "input_train": list(X_tr.shape),
            "output_train": list(X_tr_p.shape),
        },
        "n_flag_cols": len(pre.flag_cols_),
        "n_onehot_input_cols": len(pre.onehot_cols_),
        "n_target_encoded_cols": len(pre.target_encode_cols_),
        "n_features_out": len(pre.feature_names_),
        "class_rates": {
            "train": float(y_tr.mean()),
            "val": float(y_val.mean()),
            "test": float(y_te.mean()),
        },
    }
    with open(RESULTS_DIR / "preprocess_aux_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] aux 통합 전처리 완료 — n_features_out={len(pre.feature_names_)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None, help="dry-run: 일부 행만 처리")
    ap.add_argument("--aux", type=str, default=None, help="aux features parquet 경로")
    args = ap.parse_args()
    main(nrows=args.nrows, aux_path=Path(args.aux) if args.aux else None)
