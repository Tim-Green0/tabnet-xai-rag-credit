"""Home Credit Default Risk 데이터 로더.

본 프로젝트는 메인 테이블(application_train.csv)만 사용한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils import DATA_DIR

APPLICATION_TRAIN: Path = DATA_DIR / "application_train.csv"
COLUMNS_DESC: Path = DATA_DIR / "HomeCredit_columns_description.csv"


def load_application_train(
    path: Optional[Path] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """메인 테이블 로드.

    Parameters
    ----------
    path : 기본값은 data/home_credit/application_train.csv
    nrows : dry-run 용으로 일부만 읽고 싶을 때 (예: 1000)
    """
    fp = path if path is not None else APPLICATION_TRAIN
    df = pd.read_csv(fp, nrows=nrows)
    return df


def load_columns_description() -> pd.DataFrame:
    """변수 설명 테이블 (Kaggle에서 함께 제공)."""
    return pd.read_csv(COLUMNS_DESC, encoding="latin-1")


def basic_info(df: pd.DataFrame) -> dict:
    """간단한 메타 정보 반환."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "n_target_pos": int(df["TARGET"].sum()) if "TARGET" in df.columns else None,
        "target_rate": float(df["TARGET"].mean()) if "TARGET" in df.columns else None,
    }


if __name__ == "__main__":
    # smoke test
    df = load_application_train(nrows=1000)
    print(df.shape, df["TARGET"].mean())
    print(basic_info(df))
