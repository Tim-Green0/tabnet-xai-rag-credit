"""그림 4-1 시스템 전체 파이프라인 도식 생성 (matplotlib).

산출: figures/42_thesis_pipeline.png
"""
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # 제목
    ax.text(6.5, 6.6, "Fig 4-1. 4-Stage Pipeline of XAI-RAG Credit Scoring System",
             ha="center", va="center", fontsize=13, weight="bold")

    # Stage 1: 예측
    box1 = mpatches.FancyBboxPatch((0.3, 4), 2.5, 1.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor="#FFE5B4", edgecolor="black", linewidth=1.5)
    ax.add_patch(box1)
    ax.text(1.55, 5.4, "Stage 1: 예측", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(1.55, 4.9, "XGBoost", ha="center", va="center", fontsize=10)
    ax.text(1.55, 4.5, "(부도 확률 산출)", ha="center", va="center", fontsize=9)
    ax.text(1.55, 4.15, "TabNet (어텐션 추출)", ha="center", va="center", fontsize=9)

    # Stage 2: 해석
    box2 = mpatches.FancyBboxPatch((3.5, 4), 2.5, 1.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor="#B4E5FF", edgecolor="black", linewidth=1.5)
    ax.add_patch(box2)
    ax.text(4.75, 5.4, "Stage 2: 해석", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(4.75, 4.9, "TreeSHAP", ha="center", va="center", fontsize=10)
    ax.text(4.75, 4.5, "(SHAP local)", ha="center", va="center", fontsize=9)
    ax.text(4.75, 4.15, "TabNet attention", ha="center", va="center", fontsize=9)

    # Stage 3: 융합 컨텍스트 ★
    box3 = mpatches.FancyBboxPatch((6.7, 3.7), 2.8, 2.0,
                                     boxstyle="round,pad=0.1",
                                     facecolor="#FFB4B4", edgecolor="red", linewidth=2.5)
    ax.add_patch(box3)
    ax.text(8.1, 5.4, "★ Stage 3: 동의 기반 융합", ha="center", va="center",
             fontsize=11, weight="bold", color="darkred")
    ax.text(8.1, 5.0, "Agreement-aware Fusion Context", ha="center", va="center", fontsize=9)
    ax.text(8.1, 4.6, "agreed / shap_only / attention_only", ha="center", va="center", fontsize=9)
    ax.text(8.1, 4.25, "+ 보호 속성 마스킹", ha="center", va="center", fontsize=9)
    ax.text(8.1, 3.9, "(JSON 컨텍스트)", ha="center", va="center", fontsize=9, style="italic")

    # Stage 4: LLM RAG
    box4 = mpatches.FancyBboxPatch((10.2, 4), 2.5, 1.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor="#B4FFB4", edgecolor="black", linewidth=1.5)
    ax.add_patch(box4)
    ax.text(11.45, 5.4, "Stage 4: LLM-RAG", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(11.45, 4.9, "Anthropic Claude", ha="center", va="center", fontsize=9)
    ax.text(11.45, 4.6, "Google Gemini", ha="center", va="center", fontsize=9)
    ax.text(11.45, 4.2, "한국어 자연어 설명", ha="center", va="center", fontsize=9, style="italic")

    # 화살표 (Stage 1 → 2 → 3 → 4)
    for x_start, x_end in [(2.85, 3.45), (6.05, 6.65), (9.55, 10.15)]:
        ax.annotate("", xy=(x_end, 4.8), xytext=(x_start, 4.8),
                     arrowprops=dict(arrowstyle="->", color="black", lw=2))

    # 입력 (왼쪽 아래)
    box_in = mpatches.FancyBboxPatch((0.3, 1.5), 2.5, 1.0,
                                       boxstyle="round,pad=0.1",
                                       facecolor="#F0F0F0", edgecolor="gray", linewidth=1)
    ax.add_patch(box_in)
    ax.text(1.55, 2.1, "정형 데이터 입력", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(1.55, 1.7, "Home Credit / German Credit", ha="center", va="center", fontsize=8)

    # 입력 → Stage 1 화살표
    ax.annotate("", xy=(1.55, 3.95), xytext=(1.55, 2.55),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # 출력 (오른쪽 아래)
    box_out = mpatches.FancyBboxPatch((10.2, 1.5), 2.5, 1.0,
                                        boxstyle="round,pad=0.1",
                                        facecolor="#F0F0F0", edgecolor="gray", linewidth=1)
    ax.add_patch(box_out)
    ax.text(11.45, 2.1, "5-section 자연어 리포트", ha="center", va="center",
             fontsize=10, weight="bold")
    ax.text(11.45, 1.7, "결정·사유·긍정·권고·면책", ha="center", va="center", fontsize=8)

    # Stage 4 → 출력 화살표
    ax.annotate("", xy=(11.45, 2.55), xytext=(11.45, 3.95),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # 평가 박스 (아래)
    box_eval = mpatches.FancyBboxPatch((3.5, 0.3), 6.0, 0.9,
                                         boxstyle="round,pad=0.1",
                                         facecolor="#FFFACD", edgecolor="black", linewidth=1)
    ax.add_patch(box_eval)
    ax.text(6.5, 0.85, "4-tier 평가 프레임워크 (Chapter 5)", ha="center", va="center",
             fontsize=10, weight="bold")
    ax.text(6.5, 0.5, "룰 기반 + NLI(mDeBERTa) + G-Eval(Cross-judge) + Persona Pilot",
             ha="center", va="center", fontsize=8)

    # 출력 → 평가 화살표
    ax.annotate("", xy=(9.5, 0.85), xytext=(10.2, 1.7),
                 arrowprops=dict(arrowstyle="->", color="gray", lw=1, linestyle="dashed"))

    plt.tight_layout()
    out = FIGURES_DIR / "42_thesis_pipeline.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
