"""Step 3-C-1-a: TabNet local attention 추출 (100 instances).

Step 1의 SHAP local examples 100과 동일한 인스턴스에 대해 TabNet의
instance-level attention mask (M_explain) 추출한다.

TabNet.explain(X) 의 반환:
    - M_explain: (n_samples, n_features) — n_steps의 합산 attention. 0~1 범위.
    - masks  : dict[step_idx -> (n_samples, n_features)] — step별 raw mask
SHAP의 |값|에 대응하는 instance-level importance로 사용 가능.

산출:
    results/tabnet_local_attention_100.json
        [{idx, tag, top_k_attention: [{feature, attention_score, value, rank}]}, ...]

사용:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.tabnet_attention_local
        --local-examples results/shap_local_examples_100.json
        --top-k 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from src.utils import RESULTS_DIR, SEED, set_seed

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = RESULTS_DIR / "baseline_models"
TARGET_COL = "TARGET"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_tabnet(model_name: str = "tabnet_best") -> TabNetClassifier:
    """저장된 TabNet 모델 로드.

    pytorch-tabnet의 save/load는 .zip 파일에 저장. load_model은 빈 클래스에 호출.
    """
    clf = TabNetClassifier()
    fp = MODELS_DIR / f"{model_name}.zip"
    if not fp.exists():
        raise FileNotFoundError(f"TabNet model not found: {fp}")
    clf.load_model(str(fp))
    print(f"[load] TabNet from {fp}")
    return clf


def load_test_set():
    test = pd.read_parquet(PROCESSED_DIR / "test_scaled.parquet")
    y = test[TARGET_COL].astype(int).values
    X = test.drop(columns=[TARGET_COL])
    return X, y


def select_instances_by_idx(X_test: pd.DataFrame, target_indices: List[int]) -> pd.DataFrame:
    """test_scaled.parquet의 row index와 SHAP local examples의 idx가 같은
    test set 내 ordinal index를 사용한다. shap_analysis.py의 컨벤션.
    """
    return X_test.iloc[target_indices].copy()


def main(local_examples_path: Path, top_k: int = 5,
         out_path: Path = None) -> None:
    set_seed(SEED)
    out_path = out_path or (RESULTS_DIR / "tabnet_local_attention_100.json")

    print(f"[1/4] SHAP local examples 로드: {local_examples_path}")
    with open(local_examples_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    print(f"     {len(examples)}개 인스턴스")

    print(f"[2/4] test set + TabNet 모델 로드")
    X_test, y_test = load_test_set()
    print(f"     X_test={X_test.shape}, n_features={X_test.shape[1]}")
    clf = load_tabnet("tabnet_best")

    # 인스턴스 idx 리스트
    target_idx = [ex["idx"] for ex in examples]
    X_sub = select_instances_by_idx(X_test, target_idx)
    print(f"     선택된 X_sub={X_sub.shape}")
    feature_names = list(X_sub.columns)

    print(f"[3/4] TabNet attention 추출 (.explain)")
    # explain returns (M_explain, masks)
    M_explain, masks = clf.explain(X_sub.values.astype(np.float32))
    M_explain = np.asarray(M_explain)  # (n, n_features)
    print(f"     M_explain shape={M_explain.shape}")
    print(f"     M_explain stats: min={M_explain.min():.4f}, max={M_explain.max():.4f}, "
          f"mean={M_explain.mean():.4f}")

    # 인스턴스별 top-k feature 추출 (attention 큰 순)
    print(f"[4/4] 인스턴스별 top-{top_k} attention features 추출")
    output = []
    for i, ex in enumerate(examples):
        att = M_explain[i]  # (n_features,)
        # top-k 인덱스
        top_idx = np.argsort(att)[::-1][:top_k]
        top_features = []
        for rank, fi in enumerate(top_idx, start=1):
            fname = feature_names[fi]
            top_features.append({
                "feature": fname,
                "attention_score": round(float(att[fi]), 6),
                "value": float(X_sub.iloc[i][fname]),
                "rank": rank,
            })
        output.append({
            "idx": ex["idx"],
            "tag": ex["tag"],
            "true_label": ex["true_label"],
            "predicted_proba_xgb": ex["predicted_proba"],  # 참조용 (XGB 확률)
            "n_features_active": int((att > 0).sum()),
            "top_k_attention": top_features,
        })

    print(f"\n[save] {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 요약 통계
    n_active_mean = float(np.mean([o["n_features_active"] for o in output]))
    print(f"     인스턴스당 활성 feature 평균 = {n_active_mean:.1f}")
    print(f"[OK] TabNet local attention 추출 완료 ({len(output)}개)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-examples",
                    default=str(RESULTS_DIR / "shap_local_examples_100.json"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    main(local_examples_path=Path(args.local_examples), top_k=args.top_k,
         out_path=Path(args.out) if args.out else None)
