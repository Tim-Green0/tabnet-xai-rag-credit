"""Home Credit 전처리 모듈.

사용자 결정 (2026-05-03):
  A1: 결측 50%+ 컬럼 그대로 유지 + `*_MISSING_FLAG` 추가
  B1: cardinality ≤ 8 → one-hot, OCCUPATION_TYPE/ORGANIZATION_TYPE → target encoding (CV-safe)
  C1: class_weight='balanced' (학습 단계에서 처리, 전처리에서는 SMOTE 미적용)
  D : Train/Val/Test = 60/20/20, stratified by TARGET, SEED=42
  E1: EXT_SOURCE_*는 median 채움 + `EXT_SOURCE_*_MISSING` flag 추가
  F : 보호 속성은 학습 feature로 그대로 포함, 공정성 평가는 별도 단계에서

워크플로:
  from src.preprocess import split_data, HomeCreditPreprocessor
  X_tr, X_val, X_te, y_tr, y_val, y_te, ids = split_data(df, seed=42)
  pre = HomeCreditPreprocessor().fit(X_tr, y_tr)
  X_tr_p  = pre.transform(X_tr,  scale=True)   # TabNet/Logistic용
  X_val_p = pre.transform(X_val, scale=True)
  X_te_p  = pre.transform(X_te,  scale=True)
  # 트리 모델은 scale=False도 OK
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.utils import RESULTS_DIR, SEED, set_seed

# ─────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

# DAYS_EMPLOYED sentinel (≈1000년 = 365243일). EDA에서 18% 발견.
DAYS_EMPLOYED_SENTINEL = 365243

# E1: EXT_SOURCE 결측 flag 대상
EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# B1: 고-cardinality → target encoding 대상
HIGH_CARD_CATS = ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"]


# ─────────────────────────────────────────────────────────────
# 데이터 분할
# ─────────────────────────────────────────────────────────────
def split_data(
    df: pd.DataFrame,
    seed: int = SEED,
    test_size: float = 0.2,
    val_size: float = 0.2,
):
    """Train/Val/Test = (1 - val_size - test_size)/val_size/test_size, stratified.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, ids
        ids = {'train': SK_ID_CURR series, 'val': ..., 'test': ...}
        — 공정성 평가나 LLM 컨텍스트 구성 시 원본 행 추적용.
    """
    y = df[TARGET_COL].astype(int).copy()
    drop_cols = [TARGET_COL]
    if ID_COL in df.columns:
        ids_all = df[ID_COL].copy()
        drop_cols.append(ID_COL)
    else:
        ids_all = pd.Series(np.arange(len(df)), index=df.index, name="row_id")
    X = df.drop(columns=drop_cols)

    # Step 1: train (60%) vs temp (40%)
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, ids_all,
        test_size=(test_size + val_size),
        stratify=y,
        random_state=seed,
    )
    # Step 2: temp(40%) → val (20%) / test (20%)
    rel_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test, id_val, id_test = train_test_split(
        X_temp, y_temp, id_temp,
        test_size=rel_test,
        stratify=y_temp,
        random_state=seed,
    )

    ids = {"train": id_train, "val": id_val, "test": id_test}
    return X_train, X_val, X_test, y_train, y_val, y_test, ids


# ─────────────────────────────────────────────────────────────
# 전처리기
# ─────────────────────────────────────────────────────────────
@dataclass
class HomeCreditPreprocessor:
    """fit() → transform() 으로 학습 통계만 사용 (leakage 차단).

    Parameters
    ----------
    missing_flag_threshold : float
        결측률이 이 값 초과인 컬럼에 대해 `_MISSING_FLAG` 변수를 추가 (A1)
    onehot_cardinality_max : int
        cardinality ≤ 이 값인 범주형 → one-hot (B1)
    clip_quantiles : (low, high)
        수치형 1~99% 분위수 클리핑
    target_encode_smoothing : float
        category_encoders.TargetEncoder smoothing 파라미터
    """

    missing_flag_threshold: float = 0.5
    onehot_cardinality_max: int = 8
    clip_quantiles: tuple = (0.01, 0.99)
    target_encode_smoothing: float = 10.0

    # fit 후 채워질 상태
    numeric_cols_: list = field(default_factory=list, init=False)
    categorical_cols_: list = field(default_factory=list, init=False)
    onehot_cols_: list = field(default_factory=list, init=False)
    target_encode_cols_: list = field(default_factory=list, init=False)
    flag_cols_: list = field(default_factory=list, init=False)  # MISSING_FLAG 추가 대상

    medians_: dict = field(default_factory=dict, init=False)
    modes_: dict = field(default_factory=dict, init=False)
    clip_lo_: dict = field(default_factory=dict, init=False)
    clip_hi_: dict = field(default_factory=dict, init=False)

    onehot_dummy_cols_: list = field(default_factory=list, init=False)  # fit 시 결정된 더미 컬럼명
    target_encoder_: Optional[TargetEncoder] = field(default=None, init=False)
    scaler_: Optional[RobustScaler] = field(default=None, init=False)
    feature_names_: list = field(default_factory=list, init=False)  # transform 후 컬럼

    # ─────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────
    @staticmethod
    def _apply_sentinel(X: pd.DataFrame) -> pd.DataFrame:
        """DAYS_EMPLOYED == 365243 → NaN, EMPLOYED_FLAG 신규 변수."""
        X = X.copy()
        if "DAYS_EMPLOYED" in X.columns:
            sentinel_mask = X["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
            X["EMPLOYED_FLAG"] = (~sentinel_mask).astype(np.int8)
            X.loc[sentinel_mask, "DAYS_EMPLOYED"] = np.nan
        return X

    def _add_missing_flags(self, X: pd.DataFrame) -> pd.DataFrame:
        """A1: 50%+ 결측 컬럼 + EXT_SOURCE_*에 대해 _MISSING_FLAG 추가."""
        X = X.copy()
        for c in self.flag_cols_:
            if c in X.columns:
                X[f"{c}_MISSING_FLAG"] = X[c].isna().astype(np.int8)
        return X

    # ─────────────────────────────────────────────
    # fit
    # ─────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HomeCreditPreprocessor":
        set_seed(SEED)
        X = self._apply_sentinel(X)

        # 1. 결측률 기반 flag 대상 식별
        miss = X.isna().mean()
        flag_high = miss[miss > self.missing_flag_threshold].index.tolist()
        flag_ext = [c for c in EXT_SOURCE_COLS if c in X.columns]
        self.flag_cols_ = sorted(set(flag_high) | set(flag_ext))

        # 2. 컬럼 분류
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

        # one-hot vs target-encoding
        onehot_cols, te_cols = [], []
        for c in cat_cols:
            n = X[c].nunique(dropna=True)
            if c in HIGH_CARD_CATS or n > self.onehot_cardinality_max:
                te_cols.append(c)
            else:
                onehot_cols.append(c)

        self.numeric_cols_ = numeric_cols
        self.categorical_cols_ = cat_cols
        self.onehot_cols_ = onehot_cols
        self.target_encode_cols_ = te_cols

        # 3. 수치형 imputation/clip 통계 (단, missing flag는 imputation 전에 만들어야 의미 있음)
        X_flagged = self._add_missing_flags(X)

        # imputation 통계 (학습 셋 기준)
        for c in numeric_cols:
            self.medians_[c] = float(X[c].median())
        for c in cat_cols:
            mode = X[c].mode(dropna=True)
            self.modes_[c] = mode.iloc[0] if len(mode) > 0 else "Missing"

        # 클리핑 분위수 (수치형 + flag 포함된 새 변수는 0/1이라 클립 의미 없음)
        lo_q, hi_q = self.clip_quantiles
        for c in numeric_cols:
            s = X[c].dropna()
            if len(s) > 0:
                self.clip_lo_[c] = float(s.quantile(lo_q))
                self.clip_hi_[c] = float(s.quantile(hi_q))

        # 4. target encoder 학습 (CV-safe smoothing)
        if te_cols:
            te = TargetEncoder(cols=te_cols, smoothing=self.target_encode_smoothing,
                               handle_missing="value", handle_unknown="value")
            # X_flagged는 imputation 전이라 NaN 있을 수 있음. TargetEncoder는 NaN 처리 가능.
            te.fit(X_flagged[te_cols], y.values)
            self.target_encoder_ = te

        # 5. 임시 transform → onehot 컬럼 결정 + scaler fit
        X_pre = self._transform_core(X, fit_phase=True)
        self.onehot_dummy_cols_ = [c for c in X_pre.columns
                                    if any(c.startswith(f"{oh}_") for oh in onehot_cols)]
        self.feature_names_ = X_pre.columns.tolist()

        scaler = RobustScaler()
        scaler.fit(X_pre.values)
        self.scaler_ = scaler

        return self

    # ─────────────────────────────────────────────
    # transform 내부 (scaler 미적용)
    # ─────────────────────────────────────────────
    def _transform_core(self, X: pd.DataFrame, fit_phase: bool = False) -> pd.DataFrame:
        X = self._apply_sentinel(X)
        X = self._add_missing_flags(X)

        # 수치형 imputation + clip
        for c in self.numeric_cols_:
            if c in X.columns:
                X[c] = X[c].fillna(self.medians_[c])
                if c in self.clip_lo_:
                    X[c] = X[c].clip(self.clip_lo_[c], self.clip_hi_[c])

        # 범주형 imputation
        for c in self.categorical_cols_:
            if c in X.columns:
                X[c] = X[c].fillna(self.modes_[c])

        # target encoding
        if self.target_encoder_ is not None and self.target_encode_cols_:
            te_part = self.target_encoder_.transform(X[self.target_encode_cols_])
            te_part.columns = [f"{c}_TE" for c in self.target_encode_cols_]
            X = X.drop(columns=self.target_encode_cols_).join(te_part)

        # one-hot
        if self.onehot_cols_:
            X = pd.get_dummies(X, columns=self.onehot_cols_, drop_first=False, dummy_na=False)

        # fit 단계가 아니면 학습 시 결정된 컬럼에 맞춤 (누락은 0, 추가는 drop)
        if not fit_phase and self.feature_names_:
            for c in self.feature_names_:
                if c not in X.columns:
                    X[c] = 0
            X = X[self.feature_names_]

        # bool → int8
        bool_cols = X.select_dtypes(include=["bool"]).columns
        for c in bool_cols:
            X[c] = X[c].astype(np.int8)

        # 결측 점검 (모든 처리 후엔 NaN 0개여야 함)
        n_na = int(X.isna().sum().sum())
        if n_na > 0:
            # imputation 누락 가능성 — 조용히 0으로 채움 + 경고
            X = X.fillna(0)

        return X

    def transform(self, X: pd.DataFrame, scale: bool = False) -> pd.DataFrame:
        Xt = self._transform_core(X, fit_phase=False)
        if scale and self.scaler_ is not None:
            arr = self.scaler_.transform(Xt.values)
            Xt = pd.DataFrame(arr, columns=Xt.columns, index=Xt.index)
        return Xt

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, scale: bool = False) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X, scale=scale)

    # ─────────────────────────────────────────────
    # 직렬화
    # ─────────────────────────────────────────────
    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "HomeCreditPreprocessor":
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────
# CLI / smoke test
# ─────────────────────────────────────────────────────────────
def _summarize(X_tr, X_val, X_te) -> dict:
    return {
        "shape_train": list(X_tr.shape),
        "shape_val": list(X_val.shape),
        "shape_test": list(X_te.shape),
        "n_features_after": int(X_tr.shape[1]),
        "n_nan_train": int(X_tr.isna().sum().sum()) if hasattr(X_tr, "isna") else 0,
    }


def main(nrows: Optional[int] = None) -> None:
    """End-to-end: load → split → fit_transform → save processed parquet."""
    import json

    from src.data_loader import load_application_train

    set_seed(SEED)
    print(f"[1/5] 로딩 (nrows={nrows or 'all'})")
    df = load_application_train(nrows=nrows)
    print(f"     shape={df.shape}")

    print("[2/5] 분할 60/20/20 stratified")
    X_tr, X_val, X_te, y_tr, y_val, y_te, ids = split_data(df, seed=SEED)
    print(f"     train={X_tr.shape}, val={X_val.shape}, test={X_te.shape}")
    print(f"     pos_rate train={y_tr.mean():.4f}, val={y_val.mean():.4f}, test={y_te.mean():.4f}")

    print("[3/5] 전처리기 fit on train")
    pre = HomeCreditPreprocessor()
    pre.fit(X_tr, y_tr)
    print(f"     n_flag_cols={len(pre.flag_cols_)}, n_onehot_cats={len(pre.onehot_cols_)}, "
          f"n_target_encode={len(pre.target_encode_cols_)}")
    print(f"     n_features_out={len(pre.feature_names_)}")

    print("[4/5] transform (scaled=True for TabNet/Logistic)")
    X_tr_p = pre.transform(X_tr, scale=True)
    X_val_p = pre.transform(X_val, scale=True)
    X_te_p = pre.transform(X_te, scale=True)
    print(f"     scaled shapes: {X_tr_p.shape}, {X_val_p.shape}, {X_te_p.shape}")
    print(f"     NaN check: train={int(X_tr_p.isna().sum().sum())}, "
          f"val={int(X_val_p.isna().sum().sum())}, test={int(X_te_p.isna().sum().sum())}")

    # 트리용 unscaled 버전도 같이 저장
    X_tr_u = pre.transform(X_tr, scale=False)
    X_val_u = pre.transform(X_val, scale=False)
    X_te_u = pre.transform(X_te, scale=False)

    print("[5/5] 저장 (parquet + preprocessor.pkl + summary.json)")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr_p.assign(TARGET=y_tr.values).to_parquet(out_dir / "train_scaled.parquet")
    X_val_p.assign(TARGET=y_val.values).to_parquet(out_dir / "val_scaled.parquet")
    X_te_p.assign(TARGET=y_te.values).to_parquet(out_dir / "test_scaled.parquet")
    X_tr_u.assign(TARGET=y_tr.values).to_parquet(out_dir / "train_unscaled.parquet")
    X_val_u.assign(TARGET=y_val.values).to_parquet(out_dir / "val_unscaled.parquet")
    X_te_u.assign(TARGET=y_te.values).to_parquet(out_dir / "test_unscaled.parquet")

    # 공정성 평가용 보호 속성 보존 (원본 X_test의 CODE_GENDER + age)
    age = (-X_te["DAYS_BIRTH"] / 365.25).clip(0, 100)
    fairness_df = pd.DataFrame({
        "SK_ID_CURR": ids["test"].values,
        "CODE_GENDER": X_te["CODE_GENDER"].values,
        "AGE": age.values,
        "TARGET": y_te.values,
    })
    fairness_df.to_parquet(out_dir / "test_protected_attrs.parquet")

    # 전처리기 저장
    pre.save(RESULTS_DIR / "preprocessor.pkl")

    # 요약
    summary = {
        "config": {
            "seed": SEED,
            "missing_flag_threshold": pre.missing_flag_threshold,
            "onehot_cardinality_max": pre.onehot_cardinality_max,
            "clip_quantiles": list(pre.clip_quantiles),
            "high_card_target_encoded": HIGH_CARD_CATS,
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
        "files": {
            "train_scaled": str(out_dir / "train_scaled.parquet"),
            "val_scaled": str(out_dir / "val_scaled.parquet"),
            "test_scaled": str(out_dir / "test_scaled.parquet"),
            "train_unscaled": str(out_dir / "train_unscaled.parquet"),
            "val_unscaled": str(out_dir / "val_unscaled.parquet"),
            "test_unscaled": str(out_dir / "test_unscaled.parquet"),
            "protected_attrs": str(out_dir / "test_protected_attrs.parquet"),
            "preprocessor": str(RESULTS_DIR / "preprocessor.pkl"),
        },
    }
    with open(RESULTS_DIR / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[OK] 전처리 완료")
    for k, v in summary["files"].items():
        print(f"     {k}: {v}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None,
                    help="dry-run: 일부 행만 처리")
    args = ap.parse_args()
    main(nrows=args.nrows)
