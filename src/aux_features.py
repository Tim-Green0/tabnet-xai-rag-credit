"""Step 3-B-2/3: 보조 테이블 → SK_ID_CURR 단위 집계 feature.

세 테이블 (bureau, bureau_balance, previous_application)을 main 테이블의 키
(SK_ID_CURR) 단위로 집계해서 feature matrix를 만든다.

전략:
    1. bureau_balance → SK_ID_BUREAU 단위 집계 (STATUS 비율, MONTHS_BALANCE 통계)
    2. bureau에 위 결과 merge → SK_ID_CURR 단위로 다시 집계
       - 전체 / Active / Closed 분리 집계로 "현재 부담 vs 과거 이력" 구분
    3. previous_application → SK_ID_CURR 단위 집계
       - 전체 / Approved / Refused 분리 집계
    4. (1)+(2)+(3) 결과를 SK_ID_CURR 키로 outer merge → main에 left join 가능

산출:
    data/processed/aux_features.parquet  — index=SK_ID_CURR, 컬럼=집계 features
    results/aux_features_summary.json    — feature 개수 / 결측률 / 분포 요약

사용 예:
    python -m src.aux_features              # 풀 데이터
    python -m src.aux_features --dry-run    # 1만행만
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.aux_data import load_aux_table, optimize_dtypes
from src.utils import RESULTS_DIR, set_seed, SEED

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────
def _agg_numeric(df: pd.DataFrame, group_col: str, num_cols: list,
                 prefix: str, aggs: list = None) -> pd.DataFrame:
    """수치형 컬럼들을 group_col 기준으로 다중 집계."""
    if aggs is None:
        aggs = ["count", "mean", "sum", "min", "max", "std"]
    g = df.groupby(group_col, observed=True)[num_cols].agg(aggs)
    # 컬럼명 평탄화: ('AMT_CREDIT', 'mean') → 'BUREAU_AMT_CREDIT_mean'
    g.columns = [f"{prefix}_{c}_{a}" for c, a in g.columns]
    return g


def _agg_categorical(df: pd.DataFrame, group_col: str, cat_cols: list,
                     prefix: str) -> pd.DataFrame:
    """범주형 → one-hot 후 group 평균(=비율) + sum(=개수)."""
    if not cat_cols:
        return pd.DataFrame()
    # one-hot
    dummies = pd.get_dummies(df[cat_cols], dummy_na=False, prefix=cat_cols, prefix_sep="_")
    dummies = dummies.astype(np.int8)
    dummies[group_col] = df[group_col].values
    g_mean = dummies.groupby(group_col, observed=True).mean()
    g_mean.columns = [f"{prefix}_{c}_mean" for c in g_mean.columns]
    return g_mean


# ─────────────────────────────────────────────────────────────
# bureau_balance → SK_ID_BUREAU 집계
# ─────────────────────────────────────────────────────────────
def aggregate_bureau_balance(bb: pd.DataFrame) -> pd.DataFrame:
    """bureau_balance를 SK_ID_BUREAU 단위로 집계.

    컬럼: SK_ID_BUREAU, MONTHS_BALANCE (음수, 현재로부터 몇 달 전), STATUS (C/X/0/1/2/3/4/5)
    STATUS 의미:
        C = 마감, X = 알 수 없음, 0~5 = 연체 정도 (5 = 가장 심각)
    """
    print(f"     [bureau_balance] 집계 시작: {bb.shape}")
    # MONTHS_BALANCE 수치 집계
    num_g = bb.groupby("SK_ID_BUREAU", observed=True).agg(
        BB_MONTHS_BALANCE_min=("MONTHS_BALANCE", "min"),
        BB_MONTHS_BALANCE_max=("MONTHS_BALANCE", "max"),
        BB_MONTHS_BALANCE_size=("MONTHS_BALANCE", "size"),
    )

    # STATUS one-hot → 비율 + count
    status_dummies = pd.get_dummies(bb["STATUS"], prefix="BB_STATUS", dummy_na=False).astype(np.int8)
    status_dummies["SK_ID_BUREAU"] = bb["SK_ID_BUREAU"].values
    cat_g = status_dummies.groupby("SK_ID_BUREAU", observed=True).mean()
    cat_g.columns = [f"{c}_mean" for c in cat_g.columns]

    out = num_g.join(cat_g, how="left")
    print(f"     [bureau_balance] 결과: {out.shape}")
    return out


# ─────────────────────────────────────────────────────────────
# bureau → SK_ID_CURR 집계 (with bureau_balance merge)
# ─────────────────────────────────────────────────────────────
def aggregate_bureau(bureau: pd.DataFrame, bb_agg: pd.DataFrame) -> pd.DataFrame:
    """bureau + bureau_balance 집계 결과를 SK_ID_CURR 단위로.

    전체 집계 + CREDIT_ACTIVE='Active' 부분 집계 + 'Closed' 부분 집계.
    """
    print(f"     [bureau] bureau_balance와 merge: bureau={bureau.shape}")
    # bureau에 bureau_balance 집계 결과 merge
    bureau = bureau.merge(bb_agg, how="left", on="SK_ID_BUREAU")
    print(f"     [bureau] merge 후: {bureau.shape}")

    # 수치/범주 분리
    num_cols = [c for c in bureau.columns
                if c not in ("SK_ID_CURR", "SK_ID_BUREAU")
                and pd.api.types.is_numeric_dtype(bureau[c])]
    cat_cols = [c for c in bureau.columns
                if bureau[c].dtype.name == "category"
                and c not in ("SK_ID_CURR", "SK_ID_BUREAU")]

    print(f"     [bureau] 전체 집계 (num={len(num_cols)}, cat={len(cat_cols)})")
    full_num = _agg_numeric(bureau, "SK_ID_CURR", num_cols, "BUREAU")
    full_cat = _agg_categorical(bureau, "SK_ID_CURR", cat_cols, "BUREAU")
    full = full_num.join(full_cat, how="left") if not full_cat.empty else full_num

    # Active 부분
    if "CREDIT_ACTIVE" in bureau.columns:
        active_mask = bureau["CREDIT_ACTIVE"].astype(str) == "Active"
        active = bureau[active_mask]
        if len(active) > 0:
            print(f"     [bureau] Active 부분 집계: {active.shape}")
            active_num = _agg_numeric(active, "SK_ID_CURR", num_cols, "BUREAU_ACTIVE",
                                       aggs=["count", "mean", "sum", "max"])
            full = full.join(active_num, how="left")
            del active, active_num

        # Closed 부분
        closed_mask = bureau["CREDIT_ACTIVE"].astype(str) == "Closed"
        closed = bureau[closed_mask]
        if len(closed) > 0:
            print(f"     [bureau] Closed 부분 집계: {closed.shape}")
            closed_num = _agg_numeric(closed, "SK_ID_CURR", num_cols, "BUREAU_CLOSED",
                                       aggs=["count", "mean", "sum", "max"])
            full = full.join(closed_num, how="left")
            del closed, closed_num

    # COUNT 변수 추가 (전체 bureau 행 수 = 타사 대출 건수)
    full["BUREAU_COUNT"] = bureau.groupby("SK_ID_CURR", observed=True).size()

    print(f"     [bureau] SK_ID_CURR 단위 결과: {full.shape}")
    return full


# ─────────────────────────────────────────────────────────────
# previous_application → SK_ID_CURR 집계
# ─────────────────────────────────────────────────────────────
def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """previous_application을 SK_ID_CURR 단위로.

    전체 + Approved + Refused 분리 집계. NAME_CONTRACT_STATUS 기준.
    """
    print(f"     [prev] 집계 시작: {prev.shape}")
    num_cols = [c for c in prev.columns
                if c not in ("SK_ID_CURR", "SK_ID_PREV")
                and pd.api.types.is_numeric_dtype(prev[c])]
    cat_cols = [c for c in prev.columns
                if prev[c].dtype.name == "category"
                and c not in ("SK_ID_CURR", "SK_ID_PREV")]

    print(f"     [prev] 전체 집계 (num={len(num_cols)}, cat={len(cat_cols)})")
    full_num = _agg_numeric(prev, "SK_ID_CURR", num_cols, "PREV")
    full_cat = _agg_categorical(prev, "SK_ID_CURR", cat_cols, "PREV")
    full = full_num.join(full_cat, how="left") if not full_cat.empty else full_num

    # Approved / Refused 분리
    if "NAME_CONTRACT_STATUS" in prev.columns:
        for status, label in [("Approved", "PREV_APPROVED"), ("Refused", "PREV_REFUSED")]:
            mask = prev["NAME_CONTRACT_STATUS"].astype(str) == status
            sub = prev[mask]
            if len(sub) > 0:
                print(f"     [prev] {status} 부분 집계: {sub.shape}")
                sub_num = _agg_numeric(sub, "SK_ID_CURR", num_cols, label,
                                        aggs=["count", "mean", "sum", "max"])
                full = full.join(sub_num, how="left")
                del sub, sub_num

    full["PREV_COUNT"] = prev.groupby("SK_ID_CURR", observed=True).size()

    print(f"     [prev] SK_ID_CURR 단위 결과: {full.shape}")
    return full


# ─────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────
def main(dry_run: bool = False) -> None:
    set_seed(SEED)
    nrows = 10000 if dry_run else None
    suffix = "_dryrun" if dry_run else ""
    print(f"[1/4] bureau_balance 로드 + 집계 (nrows={nrows or 'all'})")
    bb = load_aux_table("bureau_balance", nrows=nrows, verbose=True)
    bb_agg = aggregate_bureau_balance(bb)
    bb_agg = optimize_dtypes(bb_agg.reset_index(), verbose=True).set_index("SK_ID_BUREAU")
    del bb
    gc.collect()

    print(f"\n[2/4] bureau 로드 + bureau_balance merge + 집계")
    bureau = load_aux_table("bureau", nrows=nrows, verbose=True)
    bureau_agg = aggregate_bureau(bureau, bb_agg)
    bureau_agg = optimize_dtypes(bureau_agg.reset_index(), verbose=True).set_index("SK_ID_CURR")
    del bureau, bb_agg
    gc.collect()

    print(f"\n[3/4] previous_application 로드 + 집계")
    prev = load_aux_table("previous_application", nrows=nrows, verbose=True)
    prev_agg = aggregate_previous_application(prev)
    prev_agg = optimize_dtypes(prev_agg.reset_index(), verbose=True).set_index("SK_ID_CURR")
    del prev
    gc.collect()

    print(f"\n[4/4] 통합 + 저장")
    # bureau_agg와 prev_agg를 SK_ID_CURR outer merge
    aux = bureau_agg.join(prev_agg, how="outer")
    aux = optimize_dtypes(aux.reset_index(), verbose=True).set_index("SK_ID_CURR")
    print(f"     최종 aux features: {aux.shape}")
    print(f"     unique SK_ID_CURR (aux): {aux.index.nunique()}")

    # 저장
    out_pq = PROCESSED_DIR / f"aux_features{suffix}.parquet"
    aux.reset_index().to_parquet(out_pq, index=False)
    print(f"     {out_pq}")

    # 요약
    miss = aux.isna().mean()
    summary = {
        "n_rows": int(aux.shape[0]),
        "n_features": int(aux.shape[1]),
        "memory_mb": round(float(aux.memory_usage(deep=True).sum() / (1024**2)), 2),
        "missing_quartiles": {
            "q25": round(float(miss.quantile(0.25)), 4),
            "q50": round(float(miss.quantile(0.50)), 4),
            "q75": round(float(miss.quantile(0.75)), 4),
            "max": round(float(miss.max()), 4),
        },
        "n_features_high_missing": int((miss > 0.5).sum()),
        "feature_groups": {
            "BUREAU": int(sum(1 for c in aux.columns if c.startswith("BUREAU_") and not c.startswith("BUREAU_ACTIVE") and not c.startswith("BUREAU_CLOSED"))),
            "BUREAU_ACTIVE": int(sum(1 for c in aux.columns if c.startswith("BUREAU_ACTIVE_"))),
            "BUREAU_CLOSED": int(sum(1 for c in aux.columns if c.startswith("BUREAU_CLOSED_"))),
            "PREV": int(sum(1 for c in aux.columns if c.startswith("PREV_") and not c.startswith("PREV_APPROVED") and not c.startswith("PREV_REFUSED"))),
            "PREV_APPROVED": int(sum(1 for c in aux.columns if c.startswith("PREV_APPROVED_"))),
            "PREV_REFUSED": int(sum(1 for c in aux.columns if c.startswith("PREV_REFUSED_"))),
        },
    }
    out_json = RESULTS_DIR / f"aux_features_summary{suffix}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"     {out_json}")
    print(f"\n[OK] aux feature engineering 완료 — n_features={aux.shape[1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="1만행만 처리")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
