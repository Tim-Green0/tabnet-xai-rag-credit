"""Step 3-B-1: 보조 테이블 EDA.

대상 테이블 (중간 옵션, 임팩트 큰 3개):
    1. bureau              — 외부 신용기관(타사) 과거 대출 이력
    2. bureau_balance      — bureau의 월별 상환 추이
    3. previous_application — Home Credit 자체에서의 이전 대출 신청 이력

산출:
    results/aux_eda_summary.json   — 각 테이블의 메타 (행/열/메모리/결측/SK_ID 분포)
    results/aux_eda.md              — 사람이 읽는 보고서
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd

from src.aux_data import iter_aux_tables, summarize_table
from src.data_loader import load_application_train
from src.utils import RESULTS_DIR, set_seed, SEED

TARGET_TABLES = ["bureau", "bureau_balance", "previous_application"]


def main() -> None:
    set_seed(SEED)
    print(f"[1/3] application_train SK_ID_CURR 집합 로드 (전체 행)")
    main_df = load_application_train()
    main_ids = set(main_df["SK_ID_CURR"].astype(int).tolist())
    n_main = len(main_ids)
    print(f"     application_train: {main_df.shape}, unique SK_ID_CURR={n_main}")
    del main_df
    gc.collect()

    print(f"[2/3] 보조 테이블 EDA (총 {len(TARGET_TABLES)}개)")
    summaries = {}
    bureau_ids_for_balance = None  # bureau_balance 커버리지 계산용

    for name, df in iter_aux_tables(TARGET_TABLES, optimize=True, verbose=True):
        s = summarize_table(df, name)
        # main 대비 SK_ID_CURR 커버리지
        if "SK_ID_CURR" in df.columns:
            covered = len(set(df["SK_ID_CURR"].astype(int).unique()) & main_ids)
            s["main_id_coverage_pct"] = round(covered / n_main * 100, 2)
            s["n_sk_id_in_main"] = covered
        else:
            s["main_id_coverage_pct"] = None
            s["n_sk_id_in_main"] = None

        # bureau는 SK_ID_BUREAU도 필요 (bureau_balance join 키)
        if name == "bureau":
            bureau_ids_for_balance = set(df["SK_ID_BUREAU"].astype(int).unique())
            s["n_unique_sk_id_bureau"] = len(bureau_ids_for_balance)

        # bureau_balance는 SK_ID_BUREAU만 있음 → bureau의 SK_ID_BUREAU와 매칭
        if name == "bureau_balance" and bureau_ids_for_balance is not None:
            bb_ids = set(df["SK_ID_BUREAU"].astype(int).unique())
            covered_b = len(bb_ids & bureau_ids_for_balance)
            s["bureau_id_coverage_pct"] = round(covered_b / len(bureau_ids_for_balance) * 100, 2)
            s["n_unique_sk_id_bureau_in_balance"] = len(bb_ids)

        # 컬럼별 결측률 분포 (간략)
        miss = df.isna().mean()
        s["missing_quartiles"] = {
            "q25": round(float(miss.quantile(0.25)), 4),
            "q50": round(float(miss.quantile(0.50)), 4),
            "q75": round(float(miss.quantile(0.75)), 4),
            "max": round(float(miss.max()), 4),
        }

        # 컬럼 목록 (+ dtype) — 미래 feature engineering 시 참조
        s["columns"] = [
            {"name": c, "dtype": str(df[c].dtype), "missing_rate": round(float(miss[c]), 4)}
            for c in df.columns
        ]

        summaries[name] = s
        print(f"     [{name}] rows={s['n_rows']:,}, cols={s['n_cols']}, mem={s['memory_mb']}MB, "
              f"main_cov={s.get('main_id_coverage_pct')}%")
        # df는 yield 후 generator에서 del 됨

    print(f"[3/3] 결과 저장")
    out_json = RESULTS_DIR / "aux_eda_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"     {out_json}")

    # 사람이 읽는 보고서
    md_lines = ["# 보조 테이블 EDA (Step 3-B-1)\n"]
    md_lines.append(f"메인 테이블 application_train: **{n_main:,}명** unique SK_ID_CURR\n")
    md_lines.append("## 요약 표\n")
    md_lines.append("| 테이블 | 행 | 열 | 메모리 | unique SK_ID_CURR | main 커버리지 | 결측률 q50 / max |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, s in summaries.items():
        cov = s.get("main_id_coverage_pct")
        cov_str = f"{cov}%" if cov is not None else "-"
        n_id = s.get("n_unique_sk_id_curr")
        n_id_str = f"{n_id:,}" if n_id is not None else "-"
        mq = s["missing_quartiles"]
        md_lines.append(
            f"| `{name}` | {s['n_rows']:,} | {s['n_cols']} | {s['memory_mb']}MB | "
            f"{n_id_str} | {cov_str} | {mq['q50']} / {mq['max']} |"
        )

    md_lines.append("\n## 테이블별 노트\n")
    for name, s in summaries.items():
        md_lines.append(f"### `{name}`")
        md_lines.append(f"- shape: {s['n_rows']:,} × {s['n_cols']}")
        md_lines.append(f"- 메모리 (downcast 후): {s['memory_mb']} MB")
        if s.get("main_id_coverage_pct") is not None:
            md_lines.append(f"- main 커버리지: **{s['main_id_coverage_pct']}%** "
                            f"({s['n_sk_id_in_main']:,} / {n_main:,})")
        if name == "bureau":
            md_lines.append(f"- unique SK_ID_BUREAU: {s.get('n_unique_sk_id_bureau'):,}")
        if name == "bureau_balance":
            md_lines.append(f"- bureau 대비 SK_ID_BUREAU 커버리지: "
                            f"**{s.get('bureau_id_coverage_pct')}%**")
        md_lines.append(f"- dtype 분포: {s['n_dtypes']}")
        md_lines.append(f"- 결측 분위수 q25/q50/q75/max: "
                        f"{s['missing_quartiles']['q25']} / "
                        f"{s['missing_quartiles']['q50']} / "
                        f"{s['missing_quartiles']['q75']} / "
                        f"{s['missing_quartiles']['max']}")
        # 결측 50%+ 컬럼 명시
        high_miss = [c for c in s["columns"] if c["missing_rate"] > 0.5]
        if high_miss:
            md_lines.append(f"- 결측 50%+ 컬럼 ({len(high_miss)}개): "
                            f"{', '.join(c['name'] for c in high_miss)}")
        md_lines.append("")

    out_md = RESULTS_DIR / "aux_eda.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"     {out_md}")
    print("[OK] 보조 테이블 EDA 완료")


if __name__ == "__main__":
    main()
