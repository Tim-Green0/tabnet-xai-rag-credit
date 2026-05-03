"""평가 지표 모듈. Day 2 베이스라인부터 Day 3 TabNet까지 공용으로 사용.

- AUROC, AUPRC: 임계치 무관 분류 성능
- KS statistic: 신용평가 표준 — max(TPR - FPR)
- F1, Precision, Recall, Accuracy: 임계치 기반 (디폴트 임계치는 Youden's J)
- find_threshold_youden: KS를 달성하는 threshold (validation 셋에서 결정 후 test 적용 권장)
- find_threshold_max_f1: F1을 최대화하는 threshold (대안)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, y_score) -> float:
    """KS = max_t |TPR(t) - FPR(t)|. 신용평가 도메인 표준."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def find_threshold_youden(y_true, y_score) -> Tuple[float, float]:
    """Youden's J = TPR - FPR을 최대화하는 임계치. KS 정의와 동일."""
    fpr, tpr, thrs = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = int(np.argmax(j))
    # roc_curve의 첫 thr는 +inf 비슷한 값일 수 있으므로 클립
    thr = float(thrs[idx])
    if not np.isfinite(thr):
        thr = 0.5
    return thr, float(j[idx])


def find_threshold_max_f1(y_true, y_score, n: int = 200) -> Tuple[float, float]:
    """F1을 최대화하는 임계치를 grid search로 탐색."""
    thrs = np.linspace(0.01, 0.99, n)
    best_thr, best_f1 = 0.5, -1.0
    for t in thrs:
        pred = (y_score >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = float(f), float(t)
    return best_thr, best_f1


def compute_metrics(
    y_true,
    y_score,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """모든 핵심 지표를 dict로 반환.

    threshold가 None이면 Youden's J로 자동 결정.
    실전에서는 validation에서 threshold를 정한 뒤 test에 적용 (leakage 방지).
    """
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    ks = ks_statistic(y_true, y_score)

    if threshold is None:
        threshold, _ = find_threshold_youden(y_true, y_score)

    pred = (y_score >= threshold).astype(int)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "ks": ks,
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def metrics_table_row(model: str, split: str, time_sec: float,
                       metrics: Dict[str, float]) -> Dict:
    """결과 표(csv용) 한 행."""
    return {"model": model, "split": split, "time_sec": round(time_sec, 2), **metrics}
