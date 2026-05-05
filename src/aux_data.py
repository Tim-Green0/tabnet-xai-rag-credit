"""Home Credit 보조 테이블 로더 + dtype 최적화.

Step 3-B에서 사용. bureau, bureau_balance, previous_application 등 6개 보조 테이블을
메모리 효율적으로 로드한다.

핵심 함수:
    optimize_dtypes(df)        : 모든 수치형을 가능한 한 작은 타입으로 다운캐스트
    load_aux_table(name, ...)  : 단일 보조 테이블 로딩 + 다운캐스트
    iter_aux_tables(names, ...): 메모리 절약형 제너레이터 (한 번에 한 테이블만)

dtype 다운캐스트 규칙 (안전 우선):
    - float64 → float32 (정밀도 7~9자리, 신용 데이터에 충분)
    - int64   → int32/int16/int8 (값 범위에 따라)
    - object  → category (uniqueness < 50% 일 때만; 메모리 절약 효과 큼)

Why: 6개 보조 테이블 합치면 raw csv 기준 ~2.6GB → DataFrame 로드 시 8~10GB까지 부풀음.
     downcast로 1/3 수준까지 줄여서 16GB RAM에서도 작업 가능.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from src.utils import DATA_DIR

# ─────────────────────────────────────────────────────────────
# 보조 테이블 메타 (key = 키 컬럼, parent = 어느 테이블에 join될지)
# ─────────────────────────────────────────────────────────────
AUX_TABLES = {
    "bureau": {
        "file": "bureau.csv",
        "keys": ["SK_ID_CURR", "SK_ID_BUREAU"],
        "parent": "application",  # 직접 SK_ID_CURR로 main에 join
    },
    "bureau_balance": {
        "file": "bureau_balance.csv",
        "keys": ["SK_ID_BUREAU"],
        "parent": "bureau",  # bureau 통해 간접 join
    },
    "previous_application": {
        "file": "previous_application.csv",
        "keys": ["SK_ID_CURR", "SK_ID_PREV"],
        "parent": "application",
    },
    "POS_CASH_balance": {
        "file": "POS_CASH_balance.csv",
        "keys": ["SK_ID_CURR", "SK_ID_PREV"],
        "parent": "previous_application",
    },
    "credit_card_balance": {
        "file": "credit_card_balance.csv",
        "keys": ["SK_ID_CURR", "SK_ID_PREV"],
        "parent": "previous_application",
    },
    "installments_payments": {
        "file": "installments_payments.csv",
        "keys": ["SK_ID_CURR", "SK_ID_PREV"],
        "parent": "previous_application",
    },
}


# ─────────────────────────────────────────────────────────────
# Dtype 최적화
# ─────────────────────────────────────────────────────────────
def optimize_dtypes(
    df: pd.DataFrame,
    use_category: bool = True,
    category_threshold: float = 0.5,
    verbose: bool = False,
) -> pd.DataFrame:
    """수치형/object 컬럼을 더 작은 타입으로 다운캐스트.

    Parameters
    ----------
    use_category
        True이면 object 컬럼 중 unique/total < category_threshold 인 컬럼을 category로 변환.
    verbose
        다운캐스트 전후 메모리 사용량 출력.
    """
    mem_before = df.memory_usage(deep=True).sum() / (1024**2)

    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if pd.isna(c_min) or pd.isna(c_max):
                continue
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = df[col].astype(np.float32)
        elif use_category and col_type == "object":
            n_unique = df[col].nunique(dropna=True)
            n_total = len(df[col])
            if n_total > 0 and (n_unique / n_total) < category_threshold:
                df[col] = df[col].astype("category")

    mem_after = df.memory_usage(deep=True).sum() / (1024**2)
    if verbose:
        reduction = (mem_before - mem_after) / mem_before * 100
        print(f"     mem: {mem_before:.1f}MB -> {mem_after:.1f}MB (-{reduction:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────
# 단일 테이블 로더
# ─────────────────────────────────────────────────────────────
def load_aux_table(
    name: str,
    nrows: Optional[int] = None,
    optimize: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """보조 테이블 한 개를 로드하고 dtype 최적화 적용.

    Parameters
    ----------
    name
        AUX_TABLES 딕셔너리의 키 (예: "bureau").
    nrows
        dry-run 용 일부만 로드.
    optimize
        True이면 optimize_dtypes 적용.
    """
    if name not in AUX_TABLES:
        raise KeyError(f"Unknown aux table: {name}. Available: {list(AUX_TABLES)}")

    fp = DATA_DIR / AUX_TABLES[name]["file"]
    if verbose:
        print(f"[load] {name} from {fp.name}")
    df = pd.read_csv(fp, nrows=nrows)
    if verbose:
        print(f"     shape={df.shape}")
    if optimize:
        df = optimize_dtypes(df, verbose=verbose)
    return df


def iter_aux_tables(
    names: Optional[list] = None,
    nrows: Optional[int] = None,
    optimize: bool = True,
    verbose: bool = True,
) -> Iterator[tuple]:
    """메모리 절약형 제너레이터: 한 번에 한 테이블만 yield.

    호출자는 yield 받은 후 사용 끝나면 del + gc 권장.
    """
    if names is None:
        names = list(AUX_TABLES.keys())
    for name in names:
        df = load_aux_table(name, nrows=nrows, optimize=optimize, verbose=verbose)
        yield name, df
        del df
        gc.collect()


# ─────────────────────────────────────────────────────────────
# 메타 정보
# ─────────────────────────────────────────────────────────────
def summarize_table(df: pd.DataFrame, name: str) -> dict:
    """단일 테이블 요약 — EDA 보고용."""
    miss = df.isna().mean()
    return {
        "name": name,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024**2)), 2),
        "n_unique_sk_id_curr": int(df["SK_ID_CURR"].nunique()) if "SK_ID_CURR" in df.columns else None,
        "n_dtypes": {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()},
        "n_cols_high_missing": int((miss > 0.5).sum()),
        "n_cols_any_missing": int((miss > 0).sum()),
        "max_missing_rate": round(float(miss.max()), 4),
    }


if __name__ == "__main__":
    # smoke test (1000행만)
    print("[smoke] loading bureau (nrows=1000)")
    bu = load_aux_table("bureau", nrows=1000, verbose=True)
    print(summarize_table(bu, "bureau"))
    print(bu.dtypes.value_counts().to_dict())
