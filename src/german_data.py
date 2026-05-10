"""UCI German Credit 데이터 로딩 + EDA + 전처리 (Step 5-D 일반화 검증).

본 모듈은 메인 메커니즘 (XGBoost+SHAP + TabNet attention + LLM RAG fusion)을
다른 데이터셋에 이식해 일반화를 입증하기 위한 첫 단계 (Day 1).

데이터: sklearn.datasets.fetch_openml('credit-g', version=1)
  - 1000 samples × 20 features (+ target 'class')
  - 13 categorical, 7 numerical
  - target: 'good' (700) → 0, 'bad' (300) → 1 (Home Credit과 일치: 1=default)
  - 보호 속성: age (numerical), personal_status (sex 결합)

처리 정책 (Home Credit과 일관):
  - 60/20/20 stratified split, SEED=42
  - categorical → one-hot (모두 cardinality ≤ ~10)
  - numerical → median imputation (실제 결측 없음) + RobustScaler
  - 보호 속성: AGE, GENDER (personal_status에서 분해) — 학습 feature로 포함

산출물:
  data/german_credit/raw.parquet
  data/german_credit/processed/{train,val,test}_{scaled,unscaled}.parquet
  data/german_credit/processed/test_protected_attrs.parquet
  results/german_eda.json
  results/german_preprocess_summary.json
  figures/37_german_eda.png

실행: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.german_data [--step {load,eda,preprocess,all}]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.utils import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR, SEED, savefig, set_seed

sns.set_theme(style="whitegrid", context="notebook")

GERMAN_DIR = PROJECT_ROOT / "data" / "german_credit"
PROCESSED_DIR = GERMAN_DIR / "processed"
RAW_PATH = GERMAN_DIR / "raw.parquet"
TARGET_COL = "TARGET"


# ─────────────────────────────────────────────────────────────
# Step 1: 로딩
# ─────────────────────────────────────────────────────────────
def load_raw(force_download: bool = False) -> pd.DataFrame:
    """sklearn에서 받아 raw.parquet으로 캐시. target은 'good'→0, 'bad'→1."""
    if RAW_PATH.exists() and not force_download:
        return pd.read_parquet(RAW_PATH)

    GERMAN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_openml] credit-g v1 다운로드 중...")
    ds = fetch_openml(name="credit-g", version=1, as_frame=True)
    df = ds.frame.copy()

    # target 정수화: bad(default)=1, good=0 → Home Credit과 일치
    df[TARGET_COL] = (df["class"] == "bad").astype(np.int8)
    df = df.drop(columns=["class"])

    # category dtype → object (parquet 호환 + 후속 처리 단순화)
    cat_cols = df.select_dtypes(include=["category"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype(str)

    # 보호 속성: GENDER (personal_status 분해)
    # 카테고리 값: 'male single', 'male div/sep', 'male mar/wid', 'female div/dep/mar'
    gender_map = {
        "male single": "M", "male div/sep": "M", "male mar/wid": "M",
        "female div/dep/mar": "F",
    }
    df["GENDER"] = df["personal_status"].map(gender_map).fillna("Unknown")

    df.to_parquet(RAW_PATH)
    print(f"[saved] {RAW_PATH}, shape={df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# Step 2: EDA
# ─────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> dict:
    print("\n=== EDA ===")
    summary = {
        "shape": list(df.shape),
        "target_pos_rate": float(df[TARGET_COL].mean()),
        "target_counts": df[TARGET_COL].value_counts().to_dict(),
        "n_numeric": int(df.select_dtypes(include=[np.number]).shape[1]),
        "n_categorical": int(df.select_dtypes(include=["object"]).shape[1]),
        "missing_count": int(df.isna().sum().sum()),
    }
    summary["target_counts"] = {int(k): int(v) for k, v in summary["target_counts"].items()}
    print(f"  shape={summary['shape']}, pos_rate={summary['target_pos_rate']:.3f}")
    print(f"  n_numeric={summary['n_numeric']}, n_categorical={summary['n_categorical']}")
    print(f"  missing={summary['missing_count']}")

    # 숫자형 통계
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET_COL]
    summary["numeric_stats"] = df[num_cols].describe().round(2).to_dict()

    # 카테고리 cardinality
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    summary["cat_cardinality"] = {c: int(df[c].nunique()) for c in cat_cols}
    print(f"  cat cardinality: {summary['cat_cardinality']}")

    # 보호 속성 분포
    summary["gender_dist"] = df["GENDER"].value_counts().to_dict()
    summary["age_stats"] = {
        "mean": float(df["age"].mean()), "std": float(df["age"].std()),
        "min": int(df["age"].min()), "max": int(df["age"].max()),
    }
    print(f"  GENDER: {summary['gender_dist']}, AGE mean={summary['age_stats']['mean']:.1f}")

    # default rate by group (4/5 rule 사전 진단)
    summary["default_by_gender"] = {
        g: float(df.loc[df["GENDER"] == g, TARGET_COL].mean())
        for g in df["GENDER"].unique()
    }
    age_old = df["age"] >= df["age"].median()
    summary["default_by_age"] = {
        "young (< median)": float(df.loc[~age_old, TARGET_COL].mean()),
        "old (>= median)": float(df.loc[age_old, TARGET_COL].mean()),
    }
    print(f"  default by gender: {summary['default_by_gender']}")
    print(f"  default by age:    {summary['default_by_age']}")

    # ── EDA figure ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    # 1) target dist
    ax = axes[0, 0]
    counts = df[TARGET_COL].value_counts().sort_index()
    ax.bar(["good (0)", "bad (1)"], counts.values,
           color=["#4C72B0", "#DD8452"])
    ax.set_title(f"Target distribution (pos_rate={summary['target_pos_rate']:.1%})")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, str(int(v)), ha="center")

    # 2) age hist
    ax = axes[0, 1]
    for tg, color in [(0, "#4C72B0"), (1, "#DD8452")]:
        ax.hist(df.loc[df[TARGET_COL] == tg, "age"], bins=30, alpha=0.6,
                label=f"target={tg}", color=color)
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Age by target")
    ax.legend()

    # 3) credit_amount hist (log scale)
    ax = axes[0, 2]
    for tg, color in [(0, "#4C72B0"), (1, "#DD8452")]:
        ax.hist(df.loc[df[TARGET_COL] == tg, "credit_amount"], bins=30,
                alpha=0.6, label=f"target={tg}", color=color)
    ax.set_xlabel("Credit amount")
    ax.set_ylabel("Count")
    ax.set_title("Credit amount by target")
    ax.legend()

    # 4) duration hist
    ax = axes[1, 0]
    for tg, color in [(0, "#4C72B0"), (1, "#DD8452")]:
        ax.hist(df.loc[df[TARGET_COL] == tg, "duration"], bins=20,
                alpha=0.6, label=f"target={tg}", color=color)
    ax.set_xlabel("Duration (months)")
    ax.set_title("Duration by target")
    ax.legend()

    # 5) default by gender
    ax = axes[1, 1]
    g_order = sorted(summary["default_by_gender"].keys())
    rates = [summary["default_by_gender"][g] for g in g_order]
    ax.bar(g_order, rates, color=["#DD8452", "#4C72B0", "#55A868"][:len(g_order)])
    ax.axhline(summary["target_pos_rate"], ls="--", color="grey", label="overall")
    ax.set_ylabel("Default rate")
    ax.set_title("Default rate by GENDER")
    ax.legend()
    for i, r in enumerate(rates):
        ax.text(i, r + 0.005, f"{r:.3f}", ha="center")

    # 6) credit history vs target
    ax = axes[1, 2]
    ch = pd.crosstab(df["credit_history"], df[TARGET_COL], normalize="index")
    ch[1].sort_values().plot(kind="barh", ax=ax, color="#DD8452")
    ax.axvline(summary["target_pos_rate"], ls="--", color="grey", label="overall")
    ax.set_xlabel("Default rate")
    ax.set_title("Default rate by credit_history")
    ax.legend()

    plt.suptitle("UCI German Credit — EDA", fontsize=14, y=1.00)
    plt.tight_layout()
    savefig(fig, "37_german_eda")
    print(f"  [fig] figures/37_german_eda.png")
    return summary


# ─────────────────────────────────────────────────────────────
# Step 3: 전처리
# ─────────────────────────────────────────────────────────────
@dataclass
class GermanCreditPreprocessor:
    """Home Credit 전처리 정책 단순화 (1000 샘플, 결측 0개).

    - one-hot for all categoricals (cardinality ≤ ~10)
    - RobustScaler for numerical (scaled 버전)
    - 클립은 생략 (소규모 데이터 + outlier도 학습에 의미 있을 수 있음)
    """
    numeric_cols_: list = field(default_factory=list, init=False)
    categorical_cols_: list = field(default_factory=list, init=False)
    medians_: dict = field(default_factory=dict, init=False)
    modes_: dict = field(default_factory=dict, init=False)
    scaler_: Optional[RobustScaler] = field(default=None, init=False)
    feature_names_: list = field(default_factory=list, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GermanCreditPreprocessor":
        set_seed(SEED)
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(include=["object"]).columns.tolist()

        for c in self.numeric_cols_:
            self.medians_[c] = float(X[c].median())
        for c in self.categorical_cols_:
            mode = X[c].mode(dropna=True)
            self.modes_[c] = mode.iloc[0] if len(mode) > 0 else "Missing"

        X_pre = self._transform_core(X, fit_phase=True)
        self.feature_names_ = X_pre.columns.tolist()
        scaler = RobustScaler()
        scaler.fit(X_pre.values)
        self.scaler_ = scaler
        return self

    @staticmethod
    def _sanitize_columns(X: pd.DataFrame) -> pd.DataFrame:
        """XGBoost/LightGBM 호환 위해 < > [ ] 등 특수문자 치환."""
        import re as _re
        new_cols = [_re.sub(r'[<>\[\],:{}\s]+', "_", c).strip("_") for c in X.columns]
        out = X.copy()
        out.columns = new_cols
        return out

    def _transform_core(self, X: pd.DataFrame, fit_phase: bool = False) -> pd.DataFrame:
        X = X.copy()
        for c in self.numeric_cols_:
            if c in X.columns:
                X[c] = X[c].fillna(self.medians_[c])
        for c in self.categorical_cols_:
            if c in X.columns:
                X[c] = X[c].fillna(self.modes_[c])

        if self.categorical_cols_:
            X = pd.get_dummies(X, columns=self.categorical_cols_, drop_first=False, dummy_na=False)

        # 특수문자 치환 (<, >, [, ] 등 → _)
        X = self._sanitize_columns(X)

        if not fit_phase and self.feature_names_:
            for c in self.feature_names_:
                if c not in X.columns:
                    X[c] = 0
            X = X[self.feature_names_]

        bool_cols = X.select_dtypes(include=["bool"]).columns
        for c in bool_cols:
            X[c] = X[c].astype(np.int8)
        if X.isna().sum().sum() > 0:
            X = X.fillna(0)
        return X

    def transform(self, X: pd.DataFrame, scale: bool = False) -> pd.DataFrame:
        Xt = self._transform_core(X, fit_phase=False)
        if scale and self.scaler_ is not None:
            arr = self.scaler_.transform(Xt.values)
            Xt = pd.DataFrame(arr, columns=Xt.columns, index=Xt.index)
        return Xt

    def save(self, path: Path) -> None:
        joblib.dump(self, path)


def split_data(df: pd.DataFrame, seed: int = SEED):
    """60/20/20 stratified split. ID는 row_id."""
    y = df[TARGET_COL].astype(int).copy()
    drop_cols = [TARGET_COL]
    X = df.drop(columns=drop_cols)
    ids_all = pd.Series(np.arange(len(df)), index=df.index, name="row_id")

    X_tr, X_temp, y_tr, y_temp, id_tr, id_temp = train_test_split(
        X, y, ids_all, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_te, y_val, y_te, id_val, id_te = train_test_split(
        X_temp, y_temp, id_temp, test_size=0.5, stratify=y_temp, random_state=seed)

    ids = {"train": id_tr, "val": id_val, "test": id_te}
    return X_tr, X_val, X_te, y_tr, y_val, y_te, ids


def run_preprocess(df: pd.DataFrame) -> dict:
    print("\n=== 전처리 ===")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_tr, X_val, X_te, y_tr, y_val, y_te, ids = split_data(df)
    print(f"  shapes: train={X_tr.shape}, val={X_val.shape}, test={X_te.shape}")
    print(f"  pos_rate: train={y_tr.mean():.4f}, val={y_val.mean():.4f}, test={y_te.mean():.4f}")

    pre = GermanCreditPreprocessor().fit(X_tr, y_tr)
    print(f"  n_features_out={len(pre.feature_names_)}")

    X_tr_s = pre.transform(X_tr, scale=True)
    X_val_s = pre.transform(X_val, scale=True)
    X_te_s = pre.transform(X_te, scale=True)

    X_tr_u = pre.transform(X_tr, scale=False)
    X_val_u = pre.transform(X_val, scale=False)
    X_te_u = pre.transform(X_te, scale=False)

    X_tr_s.assign(TARGET=y_tr.values).to_parquet(PROCESSED_DIR / "train_scaled.parquet")
    X_val_s.assign(TARGET=y_val.values).to_parquet(PROCESSED_DIR / "val_scaled.parquet")
    X_te_s.assign(TARGET=y_te.values).to_parquet(PROCESSED_DIR / "test_scaled.parquet")
    X_tr_u.assign(TARGET=y_tr.values).to_parquet(PROCESSED_DIR / "train_unscaled.parquet")
    X_val_u.assign(TARGET=y_val.values).to_parquet(PROCESSED_DIR / "val_unscaled.parquet")
    X_te_u.assign(TARGET=y_te.values).to_parquet(PROCESSED_DIR / "test_unscaled.parquet")

    # 보호 속성 (test set만)
    fairness_df = pd.DataFrame({
        "row_id": ids["test"].values,
        "GENDER": X_te["GENDER"].values,
        "AGE": X_te["age"].values,
        "TARGET": y_te.values,
    })
    fairness_df.to_parquet(PROCESSED_DIR / "test_protected_attrs.parquet")

    # 원본 컬럼 (LLM context용 - test set의 unscaled raw)
    raw_test = X_te.copy()
    raw_test["row_id"] = ids["test"].values
    raw_test["TARGET"] = y_te.values
    raw_test.to_parquet(PROCESSED_DIR / "test_raw.parquet")

    pre.save(RESULTS_DIR / "german_preprocessor.pkl")

    summary = {
        "config": {
            "seed": SEED,
            "split": "60/20/20 stratified",
        },
        "shapes": {
            "input_train": list(X_tr.shape),
            "output_train": list(X_tr_s.shape),
            "n_features_out": len(pre.feature_names_),
        },
        "n_numeric_in": len(pre.numeric_cols_),
        "n_categorical_in": len(pre.categorical_cols_),
        "class_rates": {
            "train": float(y_tr.mean()),
            "val": float(y_val.mean()),
            "test": float(y_te.mean()),
        },
    }
    with open(RESULTS_DIR / "german_preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [saved] data/german_credit/processed/*.parquet")
    print(f"  [saved] results/german_preprocess_summary.json")
    return summary


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main(step: str = "all") -> None:
    set_seed(SEED)

    if step in ("load", "all"):
        df = load_raw()
        print(f"[load] shape={df.shape}, pos_rate={df[TARGET_COL].mean():.3f}")
    else:
        df = pd.read_parquet(RAW_PATH)

    if step in ("eda", "all"):
        eda_summary = run_eda(df)
        with open(RESULTS_DIR / "german_eda.json", "w", encoding="utf-8") as f:
            json.dump(eda_summary, f, indent=2, ensure_ascii=False, default=str)

    if step in ("preprocess", "all"):
        run_preprocess(df)

    print("\n[OK] Step 5-D Day 1 (data) 완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["load", "eda", "preprocess", "all"], default="all")
    args = ap.parse_args()
    main(step=args.step)
