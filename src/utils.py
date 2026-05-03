"""공통 유틸리티: SEED, 경로, matplotlib 설정.

모든 스크립트는 from src.utils import ... 형태로 사용.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """numpy, random, (torch가 있으면) torch까지 SEED 고정."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: WPS433

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 결정론 옵션은 학습 속도와 트레이드오프라 기본 OFF
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "home_credit"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
SRC_DIR: Path = PROJECT_ROOT / "src"

for _d in (RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Matplotlib 기본 설정 (Windows 한글 + 비대화형)
# ─────────────────────────────────────────────────────────────
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.figsize"] = (10, 6)


def savefig(fig, name: str) -> Path:
    """figures/<name>.png 로 저장하고 경로 반환."""
    if not name.endswith(".png"):
        name = f"{name}.png"
    out = FIGURES_DIR / name
    fig.savefig(out)
    plt.close(fig)
    return out


__all__ = [
    "SEED",
    "set_seed",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "SRC_DIR",
    "savefig",
]
