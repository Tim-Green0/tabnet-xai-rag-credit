"""친절한 버전의 미팅 보고서 docx.

기존 midterm_report.docx (압축적·결과 위주)와는 별개:
  - 각 섹션마다 "왜 → 어떻게 → 결과 → 의미" 흐름으로 풀어 쓴다
  - 문제의식 / 배경 설명을 많이
  - 비전공자도 따라 읽을 수 있는 톤
  - 실제 인스턴스를 따라가며 설명 (idx 59291)

산출:  paper/midterm_report_friendly.docx

실행:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.gen_friendly_report
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from src.utils import RESULTS_DIR

PAPER_DIR = Path("paper")
PAPER_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("figures")


# ─────────────────────────────────────────────────────────────
# 헬퍼 (gen_report.py와 유사하나 톤·여백 풍부)
# ─────────────────────────────────────────────────────────────
def _set_cell_bg(cell, color: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), color)
    tc_pr.append(shd)


def _set_run_korean_font(run, size: int = 11, bold: bool = False,
                            color: tuple | None = None):
    run.font.name = "Malgun Gothic"
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor(*color)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), "Malgun Gothic")
    rFonts.set(qn("w:ascii"), "Malgun Gothic")
    rFonts.set(qn("w:hAnsi"), "Malgun Gothic")


def H(doc, text: str, level: int = 1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        _set_run_korean_font(run, size=18 - 3 * (level - 1), bold=True)
    return h


def P(doc, text: str, bold: bool = False, size: int = 11,
      color: tuple | None = None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    _set_run_korean_font(run, size=size, bold=bold, color=color)
    p.paragraph_format.space_after = Pt(6)
    return p


def Quote(doc, text: str):
    """인용/강조 박스 — 회색 음영 + 들여쓰기."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.right_indent = Cm(0.8)
    run = p.add_run(text)
    _set_run_korean_font(run, size=10.5, bold=False, color=(0x33, 0x33, 0x33))
    run.italic = True
    # 회색 배경
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F4F4F4")
    pPr.append(shd)
    return p


def Bullet(doc, text: str, level: int = 0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.6 + 0.6 * level)
    run = p.add_run(text)
    _set_run_korean_font(run, size=11)
    return p


def Table(doc, headers: list, rows: list, header_bg: str = "4C72B0"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        _set_run_korean_font(run, size=10, bold=True, color=(0xFF, 0xFF, 0xFF))
        _set_cell_bg(cell, header_bg)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            _set_run_korean_font(run, size=10)
    return table


def Fig(doc, path: Path, caption: str, width_cm: float = 14):
    if not path.exists():
        P(doc, f"[그림 누락: {path}]")
        return
    doc.add_picture(str(path), width=Cm(width_cm))
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cp.add_run(caption)
    _set_run_korean_font(run, size=9, color=(0x55, 0x55, 0x55))
    run.italic = True


def PageBreak(doc):
    doc.add_page_break()


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    doc = Document()

    # ── 표지 ──
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("석사 논문 중간 보고 — 친절판")
    _set_run_korean_font(run, size=22, bold=True)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("\nTabNet과 거대언어모델(LLM) 기반 XAI-RAG를 활용한\n"
                       "설명 가능한 신용 평가 및 자연어 리포트 생성")
    _set_run_korean_font(run, size=14)

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run("\n").font.size = Pt(10)
    for txt in ["A70067 오현택", "지도교수: 박운상", "보고일: 2026-05-10",
                "Step 1 (8일 작업 완료, 미팅 프로토타입)"]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(txt)
        _set_run_korean_font(r, size=11)

    P(doc, "")
    Quote(doc,
        "이 보고서는 발표 자료의 압축판이 아니라, 연구의 전체 흐름을 처음부터 "
        "차근차근 풀어 쓴 \"친절판\"입니다. 같이 읽으면 내가 무엇을 왜 했는지 "
        "어렵지 않게 따라가실 수 있도록 작성했습니다.")

    PageBreak(doc)

    # ── 0. 한 페이지 요약 ──
    H(doc, "0. 한 페이지 요약 (Executive Summary)", level=1)
    P(doc, "왜 이 연구를 시작했는가", bold=True, size=12)
    P(doc, "신용 평가 모델은 점점 더 \"잘 맞히지만 설명하기 어려운\" 블랙박스가 "
            "되어 가고 있습니다. SHAP 같은 사후 해석 기법은 변수 기여도를 "
            "수치로 알려주지만, 그 결과를 일반 고객이나 심사역이 그대로 이해하기는 "
            "여전히 어렵습니다. 한편 LLM은 자연어 변환을 잘하지만 학습되지 않은 "
            "사실을 그럴듯하게 만들어내는 환각(hallucination) 문제가 있어, "
            "신용평가 같이 정확성·신뢰성·법적 책임이 요구되는 도메인에 그대로 "
            "쓸 수 없습니다.", size=11)

    P(doc, "본 연구가 제안하는 것", bold=True, size=12)
    P(doc, "본 연구는 SHAP의 수치 기반 설명을 LLM의 \"검색된 근거(retrieved "
            "evidence)\"로 재정의하는 XAI-RAG 구조를 제안합니다. "
            "LLM은 자유롭게 추론하지 않고, SHAP가 만든 사실 컨텍스트만 자연어로 "
            "변환하는 역할에 한정됩니다. 그 결과 환각이 원천적으로 차단되며, "
            "사용자에게는 친절한 자연어 설명 리포트가 자동 생성됩니다.")

    P(doc, "핵심 발견 (10 샘플 평가 기준)", bold=True, size=12)
    Bullet(doc, "Hallucination Rate가 두 상용 LLM(Gemini 2.5 Flash, Claude "
                "Sonnet 4.5)에서 모두 0.000 — 본 구조는 LLM 종속성 없이 환각을 "
                "차단합니다.")
    Bullet(doc, "동일 데이터를 SHAP 컨텍스트 없이 LLM에 직접 주는 baseline에서 "
                "Claude는 45.5%의 환각을 만들어냈습니다 (DTI/LTV/DSR 같이 데이터에 "
                "없는 일반 금융 용어, 가짜 고객센터 번호 등). 이 둘의 비교가 본 "
                "구조의 효과를 가장 분명하게 보여줍니다.")
    Bullet(doc, "G-Eval 자동 평가에서 정확성·민감변수 마스킹·문체 모두 5/5점 만점.")
    Bullet(doc, "예측 성능은 5-fold Stratified CV에서 XGBoost가 test AUROC "
                "0.7587 ± 0.0008로 가장 안정적이며, TabNet은 0.7518 ± 0.0017로 "
                "약간 부족하지만 본 연구의 가치는 어텐션 기반 내재적 해석성에 "
                "있습니다.")
    Bullet(doc, "8개 공정성 케이스 모두 4/5 rule을 위반했고, 보호 속성 ablation은 "
                "성별 편향에 대해서만 효과적이었습니다 — 연령은 다른 변수에 "
                "간접 인코딩되어 있어 단순 제거로는 줄지 않습니다.")

    PageBreak(doc)

    # ── 1. 연구 배경 ──
    H(doc, "1. 왜 이 연구가 필요한가 — 연구 배경", level=1)

    H(doc, "1.1 자동화된 신용 평가의 두 얼굴", level=2)
    P(doc, "최근 금융기관은 머신러닝과 딥러닝 모델을 여신 심사·리스크 관리·"
            "이상 거래 탐지에 광범위하게 사용합니다. XGBoost, LightGBM, "
            "TabNet 같은 모델은 전통적인 로지스틱 회귀보다 부도 예측 성능이 "
            "훨씬 좋습니다. 그러나 이 모델들은 \"왜 거절되었는가\"를 직관적으로 "
            "설명하기 어렵습니다.")
    P(doc, "이는 단순한 기술적 한계를 넘어, 제도적·윤리적 문제로 이어집니다. "
            "국내 금융소비자보호법 및 신용정보법은 자동화된 신용평가에 대한 "
            "설명 의무와 이의제기 권리를 명시합니다. EU의 일반정보보호규정"
            "(GDPR)도 알고리즘적 의사결정에 대한 \"설명 요구권(Right to "
            "Explanation)\"을 보장합니다.")

    H(doc, "1.2 기존 XAI의 한계 — \"모델은 설명 가능해졌으나 사람에게는 "
              "설명되지 않는다\"", level=2)
    P(doc, "이런 요구에 대응해 SHAP, LIME 같은 사후 해석 기법이 활발히 연구되어 "
            "왔습니다. 그러나 SHAP의 출력은 변수 중요도 그래프, 기여도 수치, "
            "Force Plot 같은 시각화·수치 기반입니다. 데이터 분석 전문가가 "
            "아니라면 직관적으로 해석하기 어렵습니다.")
    Quote(doc, "즉 \"모델은 설명 가능해졌으나 사람에게는 여전히 설명되지 않는\" "
                "의사 설명(pseudo-explanation) 문제가 남아 있습니다.")

    H(doc, "1.3 LLM의 가능성과 위험", level=2)
    P(doc, "GPT-4, Claude, Gemini 같은 최신 LLM은 수치형·구조적 정보를 "
            "자연어로 변환하는 데 뛰어납니다. SHAP 출력값을 자연어로 옮기는 "
            "시도가 활발합니다. 하지만 LLM에는 학습 데이터에 없는 사실도 "
            "그럴듯하게 만들어내는 환각이라는 본질적 약점이 있어, 신용평가 "
            "같이 정확성과 법적 책임이 중요한 도메인에서 그대로 쓰기는 위험합니다.")
    P(doc, "본 연구는 이 두 가지 문제를 동시에 풉니다: SHAP의 \"전문가용\" "
            "출력을 일반 사용자가 이해할 수 있는 자연어로 변환하면서, LLM의 "
            "환각을 원천적으로 차단합니다.")

    PageBreak(doc)

    # ── 2. 핵심 아이디어 ──
    H(doc, "2. 핵심 아이디어 — XAI-RAG가 무엇인가", level=1)
    P(doc, "본 연구가 제안하는 프레임워크는 4단계로 구성됩니다.")

    P(doc, "1단계 — 예측 (Prediction)", bold=True, size=12)
    P(doc, "정형 데이터를 모델(XGBoost 또는 TabNet)에 입력해 부도 확률 P(default)을 "
            "예측합니다. 본 연구는 트리 기반 모델과 정형 딥러닝(TabNet)을 모두 "
            "비교합니다.")

    P(doc, "2단계 — 해석 (Interpretation)", bold=True, size=12)
    P(doc, "학습된 모델에 SHAP을 적용해 개별 예측의 변수 기여도를 산출합니다. "
            "동시에 TabNet의 어텐션 마스크도 추출해 두 해석이 서로 일관되는지 "
            "분석합니다 (RQ2).")

    P(doc, "3단계 — 컨텍스트 구성 (Context Construction)", bold=True, size=12)
    P(doc, "SHAP 결과를 \"사실 단위\"의 구조화된 JSON으로 변환합니다. "
            "이 컨텍스트는 LLM에게 주는 \"검색된 근거\" 역할을 합니다. "
            "민감 변수(성별, 연령)는 마스킹되며, 도메인 용어는 한국어로 풀어 "
            "씁니다.")

    P(doc, "4단계 — 생성 + 평가 (Generation & Evaluation)", bold=True, size=12)
    P(doc, "LLM은 위 컨텍스트만 받아 자연어 설명 리포트를 생성합니다. "
            "그리고 그 리포트의 충실성(Faithfulness)·환각률(Hallucination Rate)·"
            "품질(G-Eval)을 정량 평가합니다.")

    Quote(doc, "핵심 아이디어를 한 줄로: \"LLM이 임의로 거절 사유를 추론하는 "
                "것이 아니라, 모델이 수학적으로 산출한 근거 데이터를 컨텍스트로 "
                "받아 자연어로 변환하는 역할에만 한정된다.\"")

    PageBreak(doc)

    # ── 3. 데이터 + EDA ──
    H(doc, "3. 데이터와 EDA — 무엇을 다루고 무엇을 발견했나", level=1)
    H(doc, "3.1 사용 데이터", level=2)
    P(doc, "Kaggle Home Credit Default Risk 대회의 메인 테이블 application_train.csv를 "
            "사용했습니다. 약 30만 7천 명의 신용 신청 기록이 있고 각 신청마다 122개의 "
            "특성이 있습니다. TARGET=1이 부도, 0이 정상입니다.")
    P(doc, "보조 테이블(과거 대출 이력 등 6개)은 미팅용 프로토타입에서는 제외하고, "
            "본 작업 단계에서 추가하기로 했습니다.")

    H(doc, "3.2 EDA에서 가장 중요했던 발견 5가지", level=2)
    Bullet(doc, "TARGET 분포가 8.07%로 매우 불균형합니다. Accuracy는 의미가 적고, "
                "AUROC·AUPRC·KS 통계량을 주 지표로 써야 합니다.")
    Bullet(doc, "EXT_SOURCE_2 / EXT_SOURCE_3 (외부 신용평가 점수)이 단연 강한 신호입니다. "
                "그 외 변수들의 단순 상관계수는 |ρ|<0.06로 약합니다. 즉 \"단일 변수 "
                "선형 신호\"로는 한계가 있고, 비선형·상호작용을 포착할 수 있는 "
                "트리/딥러닝이 유리합니다.")
    Bullet(doc, "결측이 50%를 넘는 컬럼이 41개 있습니다 (대부분 주거 정보). "
                "그대로 제거하지 않고 *_MISSING_FLAG 변수를 추가해 \"입력 여부 "
                "자체\"가 신호가 되도록 보존했습니다.")
    Bullet(doc, "DAYS_EMPLOYED에 365243(약 1000년)이라는 sentinel 값이 18% 있습니다. "
                "이는 \"무직\" 의미의 특수값으로, 그대로 두면 모델이 거대한 양수로 "
                "오해석합니다. NaN으로 변환하고 EMPLOYED_FLAG를 새로 만들었습니다.")
    Bullet(doc, "공정성 신호가 데이터에 이미 강하게 존재합니다. 남성 부도율 "
                "10.1% vs 여성 7.0% (1.45배), 25세 미만 12.3% vs 65세 이상 3.7% "
                "(3.4배). 이 발견은 5장에서 본격적으로 분석합니다.")

    Fig(doc, FIG_DIR / "03_protected_attrs.png",
        "그림 3-1. 성별·연령에 따른 부도율 차이. 데이터 자체에 구조적 편향이 "
        "이미 존재합니다.")

    PageBreak(doc)

    # ── 4. 모델 ──
    H(doc, "4. 모델 — 왜 베이스라인과 TabNet을 같이 보는가", level=1)
    P(doc, "본 연구는 단순히 \"가장 잘 맞히는 모델 하나\"를 고르는 것이 목적이 "
            "아닙니다. 두 가지가 모두 필요합니다.")
    Bullet(doc, "강한 베이스라인이 있어야 \"TabNet이 정형 데이터에서도 정말로 "
                "경쟁력 있는가\"를 정직하게 평가할 수 있습니다.")
    Bullet(doc, "TabNet은 어텐션 마스크라는 내재적 해석성을 제공합니다. "
                "5장에서 SHAP과의 일관성을 분석하기 위해서는 TabNet이 필요합니다.")

    H(doc, "4.1 5-fold Stratified CV 결과 (test set 기준)", level=2)
    Table(doc,
        headers=["모델", "AUROC", "AUPRC", "KS", "F1", "특징"],
        rows=[
            ["XGBoost", "0.7587 ± 0.0008", "0.2445 ± 0.0011",
             "0.3846 ± 0.0015", "0.2698 ± 0.0047", "5/5 fold 모두 1등"],
            ["LightGBM", "0.7544 ± 0.0009", "0.2402 ± 0.0018",
             "0.3788 ± 0.0028", "0.2584 ± 0.0052", "가장 빠름"],
            ["Logistic", "0.7544 ± 0.0001", "0.2343 ± 0.0006",
             "0.3804 ± 0.0010", "0.2631 ± 0.0087", "선형 모델 최고 안정성"],
            ["TabNet", "0.7518 ± 0.0017", "0.2331 ± 0.0023",
             "0.3749 ± 0.0056", "0.2657 ± 0.0079", "어텐션 해석성 제공"],
        ])

    P(doc, "결과 해석", bold=True, size=12)
    P(doc, "표준편차가 모두 0.001 안팎으로 매우 작습니다. 이는 \"단발 운\"이 "
            "아니라 안정적인 우열 관계라는 뜻입니다. XGBoost가 살짝 앞서지만 "
            "TabNet과의 차이는 0.007 AUROC 정도로 절대적 격차는 작습니다.")
    P(doc, "본 연구의 메시지는 \"XGBoost를 제치고 TabNet이 1등이다\"가 아닙니다. "
            "오히려 \"TabNet이 비슷한 성능을 내면서 어텐션 해석성을 추가로 "
            "제공한다\"입니다. 이 어텐션이 다음 장의 SHAP 일관성 분석으로 "
            "이어집니다.")

    Fig(doc, FIG_DIR / "13_cv_comparison.png",
        "그림 4-1. 5-fold CV 비교. 모든 지표에서 XGBoost가 일관되게 1등이지만 "
        "표준편차가 매우 작아 \"안정적인 격차\"임을 보여줍니다.")

    PageBreak(doc)

    # ── 5. SHAP × Attention ──
    H(doc, "5. SHAP과 어텐션은 같은 것을 말하는가? (RQ2)", level=1)
    P(doc, "TabNet의 어텐션과 사후 SHAP이 \"비슷한 변수를 강조한다\"는 가정은 "
            "기존 연구에서 종종 암묵적으로 받아들여졌지만, 정량적으로 검증된 "
            "적은 드뭅니다. 본 연구는 이를 직접 측정합니다.")

    H(doc, "5.1 측정 방법", level=2)
    Bullet(doc, "TabNet 모델 학습 후 어텐션 기반 변수 중요도(feature_importances_) "
                "추출")
    Bullet(doc, "동일 모델에 KernelExplainer로 SHAP 적용 후 mean(|SHAP|)로 "
                "global importance 산출")
    Bullet(doc, "둘의 Spearman 순위 상관 + Top-K 변수 교집합/Jaccard 측정")

    H(doc, "5.2 결과", level=2)
    Table(doc,
        headers=["지표", "값", "해석"],
        rows=[
            ["Spearman ρ (전체 214 변수)", "0.117 (p=0.089)", "전반적으로는 약한 양의 상관"],
            ["Spearman ρ (Top-50 합집합)", "−0.195", "주요 변수에서는 오히려 음의 상관 — 부분 상보적"],
            ["Top-20 교집합 / Jaccard", "9 / 0.29", "상위 20개 중 9개는 일치"],
        ])

    P(doc, "이 결과의 의미", bold=True, size=12)
    P(doc, "두 방법은 \"가장 핵심적인 변수\"(예: EXT_SOURCE_2/3)에서는 일관되게 "
            "같은 신호를 잡습니다. 그러나 그 다음 순위로 가면 두 방법이 다른 "
            "측면을 강조합니다. 어텐션은 비선형·상호작용으로 활용되는 변수를, "
            "SHAP은 marginal contribution이 명확한 변수를 더 강조합니다.")
    Quote(doc, "결론: 어텐션과 SHAP은 \"같은 답을 다른 단어로 말하는 것\"이 "
                "아닙니다. 두 방법을 함께 사용해야 모델 해석이 입체적으로 "
                "보입니다. 이는 본 연구가 두 정보를 모두 활용하는 정당성을 "
                "제공합니다.")

    Fig(doc, FIG_DIR / "16_attention_vs_shap_scatter.png",
        "그림 5-1. 어텐션 vs SHAP 산점도. 좌측은 전체, 우측은 Top-20 합집합. "
        "교집합(빨강)은 9개, 어텐션 only(파랑) 11개, SHAP only(주황) 11개.")

    PageBreak(doc)

    # ── 6. 공정성 ──
    H(doc, "6. 공정성 진단 — 모델은 누구를 차별하는가?", level=1)
    P(doc, "신용 평가 자동화에서 가장 민감한 부분은 보호 속성(성별, 연령 등)에 "
            "대한 차별 가능성입니다. 본 연구는 4개 모델 모두에 대해 4종 공정성 "
            "지표를 측정했습니다.")

    H(doc, "6.1 공정성 지표 4종 — 무엇을 재는가", level=2)
    Bullet(doc, "Demographic Parity (DP): 두 집단의 양성 예측률 차이 — "
                "\"누가 더 자주 거절되는가?\"")
    Bullet(doc, "Equal Opportunity (EO): 실제 부도자 중 거절률(TPR) 차이 — "
                "\"진짜 위험한 사람을 잡는 정확도가 집단별로 같은가?\"")
    Bullet(doc, "Equalized Odds: TPR과 FPR 모두를 종합")
    Bullet(doc, "Disparate Impact (DI): 최소 양성률/최대 양성률 비율. "
                "0.8 미만이면 미국 4/5 rule 위반.")

    H(doc, "6.2 베이스라인 결과 — 8/8 모두 4/5 rule 위반", level=2)
    P(doc, "4개 모델 × 보호속성 {GENDER, AGE} = 8개 케이스 전부에서 DI < 0.8입니다. "
            "특히 연령(AGE)이 성별(GENDER)보다 편향이 더 큽니다 — DI ≈ 0.50 (AGE) "
            "vs 0.62 (GENDER).")
    P(doc, "이는 \"어떤 모델을 골라도 데이터 자체의 구조적 편향이 그대로 "
            "옮겨진다\"는 뜻입니다. 알고리즘 차원의 문제가 아니라 데이터 차원의 "
            "문제입니다.")

    H(doc, "6.3 단순한 mitigation — 보호 속성 ablation", level=2)
    P(doc, "그렇다면 단순히 보호 속성 컬럼(CODE_GENDER, DAYS_BIRTH)을 빼고 "
            "다시 학습하면 어떨까? 결과는 흥미로웠습니다.")
    Table(doc,
        headers=["케이스", "AUROC 변화", "DP 변화", "DI 변화"],
        rows=[
            ["XGBoost × GENDER", "−0.005", "0.164 → 0.105 (−36%)", "0.622 → 0.718"],
            ["TabNet × GENDER", "−0.010", "0.156 → 0.093 (−40%)", "0.621 → 0.757"],
            ["XGBoost × AGE", "−0.005", "0.197 → 0.179 (−9%)", "0.498 → 0.507"],
            ["TabNet × AGE", "−0.010", "0.185 → 0.188 (+2%)", "0.504 → 0.513"],
        ])
    Bullet(doc, "성별(GENDER) ablation은 효과적입니다. AUROC 1% 미만 손실로 "
                "DP를 30~40% 줄였습니다.")
    Bullet(doc, "연령(AGE) ablation은 거의 효과가 없습니다. 이유는 연령 정보가 "
                "DAYS_EMPLOYED, DAYS_REGISTRATION, OWN_CAR_AGE 같은 다른 변수에 "
                "간접적으로 인코딩되어 있기 때문입니다 — \"proxy variable\" 문제.")
    Bullet(doc, "즉 단순한 컬럼 제거로는 4/5 rule을 통과하지 못합니다. 본격적인 "
                "fairness-aware 학습(Reweighing, Adversarial Debiasing)이 향후 "
                "과제로 자연스럽게 도출됩니다.")

    Fig(doc, FIG_DIR / "19_fairness_mitigation.png",
        "그림 6-1. baseline vs ablated 비교. GENDER ablation은 효과적이지만 "
        "AGE는 거의 변화 없음 — proxy variable 문제.")

    PageBreak(doc)

    # ── 7. XAI-RAG 데모 ──
    H(doc, "7. 실제 인스턴스로 따라가는 XAI-RAG (idx 59291)", level=1)

    walk = json.loads((RESULTS_DIR / "demo_walkthrough.json").read_text(encoding="utf-8"))
    s = walk["sample"]
    p = walk["prediction"]

    P(doc, f"본 장은 한 명의 실제 인스턴스 (idx {s['idx']})를 따라가며 "
            f"전 파이프라인이 어떻게 작동하는지 보여줍니다. 이 인스턴스는 "
            f"True Positive — 모델이 부도라고 예측했고 실제로도 부도였습니다.")

    H(doc, "7.1 입력 데이터", level=2)
    P(doc, f"연령 {s['age']:.0f}세, 성별 {s['gender']}. 주요 정형 변수:")
    Table(doc, ["변수", "값"], list(s["display_features"].items()))

    H(doc, "7.2 1단계 — XGBoost 예측", level=2)
    P(doc, f"모델은 부도 확률 P(default) = {p['default_proba']:.4f}로 예측했고, "
            f"validation에서 결정한 임계치 {p['threshold']:.4f}를 넘었으므로 "
            f"결정은 {p['decision']}입니다.")

    H(doc, "7.3 2단계 — SHAP Local Explanation", level=2)
    P(doc, "SHAP은 이 예측에서 어떤 변수가 얼마나 기여했는지 알려줍니다. "
            "양수 SHAP은 부도 확률을 높이는 방향, 음수는 낮추는 방향입니다.")
    Table(doc,
        ["순위", "변수", "값", "SHAP"],
        [[d["rank"], d["feature"], d["value"], f"{d['shap']:+.4f}"]
          for d in walk["context"]["top_drivers_for_default"]])

    H(doc, "7.4 3단계 — JSON 컨텍스트 빌드", level=2)
    P(doc, "위 SHAP 결과를 LLM이 읽기 쉬운 사실 단위 JSON으로 변환합니다. "
            "민감 변수(성별·연령)는 마스킹되고, 변수명은 한국어로 풀어쓰며, "
            "explanation_policy: \"fact_only\" 정책이 명시됩니다.")

    H(doc, "7.5 4단계 — LLM 자연어 설명 생성", level=2)
    P(doc, "이 컨텍스트와 엄격한 제약 조건이 담긴 프롬프트를 두 LLM(Gemini와 "
            "Claude)에 동시에 보내 각각의 자연어 설명을 받았습니다.")

    for prov_key, prov_label in [("gemini", "Gemini 2.5 Flash"),
                                    ("anthropic", "Claude Sonnet 4.5")]:
        if prov_key in walk["llm_outputs"] and "explanation" in walk["llm_outputs"][prov_key]:
            ll = walk["llm_outputs"][prov_key]
            P(doc, f"[{prov_label}]  처리 시간 {ll['elapsed_sec']:.1f}초, "
                    f"토큰 {ll.get('total_tokens', '?')}",
              bold=True, size=11, color=(0x4C, 0x72, 0xB0))
            for line in ll["explanation"].split("\n"):
                if line.strip():
                    P(doc, line.strip(), size=10)
            P(doc, "")

    Quote(doc, "두 LLM 모두 컨텍스트의 변수명·값·SHAP 부호를 정확히 인용했고, "
                "성별·연령 같은 민감 변수는 한 번도 직접 언급하지 않았습니다. "
                "스타일에서 차이가 있어 — Gemini는 풀 자릿수(0.0633754)를 그대로 "
                "쓰고, Claude는 자연스럽게 반올림(0.0634)합니다.")

    Fig(doc, FIG_DIR / "22_demo_walkthrough.png",
        "그림 7-1. Demo 워크스루 — SHAP Top 10 + Decision Summary.")

    PageBreak(doc)

    # ── 8. 정량 평가 ──
    H(doc, "8. 자연어 설명을 어떻게 평가하는가 (RQ3)", level=1)
    P(doc, "\"좋은 설명\"이라는 개념은 본질적으로 다층적입니다. 본 연구는 "
            "기존 LLM 평가 연구를 따라 4개 차원으로 평가합니다.")

    Bullet(doc, "Faithfulness (충실성): 텍스트의 변수·수치·부호가 컨텍스트와 "
                "정확히 일치하는가?")
    Bullet(doc, "Hallucination Rate (환각률): 컨텍스트에 없는 변수가 텍스트에 "
                "등장한 비율은 얼마인가?")
    Bullet(doc, "G-Eval: 별도의 LLM(여기서는 Gemini)이 정해진 루브릭으로 5점 "
                "척도 평가")
    Bullet(doc, "효율성: 호출당 시간과 토큰")

    H(doc, "8.1 결과 — Hallucination Rate가 양쪽 모두 0", level=2)
    Table(doc,
        ["지표", "Gemini 2.5 Flash", "Claude Sonnet 4.5"],
        [
            ["Hallucination Rate (strict)", "0.000 ± 0.000", "0.000 ± 0.000"],
            ["Hallucination Rate (broad)", "0.000 ± 0.000", "0.000 ± 0.000"],
            ["val_match_rate", "0.811 ± 0.123", "0.901 ± 0.111"],
            ["sign_match_rate", "0.783 ± 0.209", "0.867 ± 0.233"],
            ["G-Eval factual_accuracy", "5.0 / 5", "(skip)"],
            ["G-Eval sensitive_leak", "5.0 / 5", "(skip)"],
            ["G-Eval style", "5.0 / 5", "(skip)"],
            ["elapsed (sec/call)", "12.7 ± 4.5", "8.4 ± 0.8"],
            ["total tokens (per call)", "4,155 ± 834", "2,500 ± 83"],
        ])
    P(doc, "두 LLM 모두에서 환각률이 0이라는 것은 본 연구의 가장 중요한 결과입니다. "
            "이는 SHAP 컨텍스트가 LLM의 자유 추론을 효과적으로 차단함을 보여줍니다.",
      bold=True)
    Fig(doc, FIG_DIR / "21_llm_comparison.png",
        "그림 8-1. Gemini vs Claude — 룰 기반 평가, G-Eval, 효율성 3개 패널.")

    PageBreak(doc)

    # ── 9. Counterfactual Baseline ──
    H(doc, "9. SHAP이 없으면 정말로 환각이 늘어나는가? (RQ3 강한 검증)",
        level=1)
    P(doc, "8장에서 \"본 구조에서 환각률이 0이다\"라고 했지만, 이것이 본 구조 "
            "덕분인지 아니면 단지 LLM이 좋은 모델이라서인지는 별도의 실험이 "
            "필요합니다. 그래서 동일 11개 인스턴스에 대해 \"SHAP 컨텍스트 없이 "
            "raw 데이터만 LLM에 주는\" baseline 실험을 진행했습니다.")

    H(doc, "9.1 결과 — Claude의 환각률 0% → 45.5%", level=2)
    Table(doc,
        ["LLM", "XAI-RAG (SHAP context)", "Baseline (no SHAP)"],
        [
            ["Gemini 2.5 Flash", "0.000", "0.000 (측정 한계 — 9.2절 참조)"],
            ["Claude Sonnet 4.5", "0.000", "0.4545"],
        ])
    P(doc, "Claude의 baseline 환각률 45.5%가 결정적 결과입니다. SHAP 컨텍스트 "
            "없으면 Claude는 학습된 일반 금융 지식을 자유롭게 끌어와 거절 "
            "사유를 만들어냅니다.", bold=True)

    H(doc, "9.2 Claude baseline 환각 사례", level=2)
    Bullet(doc, "DTI(Debt-to-Income), LTV(Loan-to-Value), DSR(Debt Service Ratio) "
                "— Home Credit 데이터에는 없는 일반 금융 비율 약어를 자유 추론")
    Bullet(doc, "\"햇살론, 미소금융\" 같은 한국 특정 정부 지원 금융상품을 "
                "권고에 포함 — 데이터에 없는 외부 지식")
    Bullet(doc, "\"고객센터 ☎1588-XXXX\" 같은 가짜 연락처 생성")
    Bullet(doc, "DEF_30_CNT처럼 변수명을 잘라서 부정확하게 인용")

    H(doc, "9.3 Gemini의 측정 한계 — 환각이 없는 게 아니라 잡히지 않은 것",
        level=2)
    P(doc, "Gemini의 baseline 환각률 0%는 다소 오해의 소지가 있습니다. "
            "Gemini는 영문 변수명을 거의 사용하지 않고 한국어 자연어로 의역하는 "
            "경향이 있어, 영문 정규식 기반 룰로는 환각이 잡히지 않습니다. "
            "내용 단위(claim 단위) 평가를 위한 cross-LLM judge 방식이 향후 "
            "필요합니다.")

    Fig(doc, FIG_DIR / "23_baseline_vs_xairag.png",
        "그림 9-1. XAI-RAG vs baseline (no SHAP) 환각률. Claude에서 결정적 차이 "
        "(0% → 45.5%).")

    PageBreak(doc)

    # ── 10. 종합 정리 ──
    H(doc, "10. 종합 — 본 연구의 가치", level=1)
    P(doc, "본 연구는 정형 데이터 신용 평가에서 다음 세 가지를 동시에 달성했습니다.")
    Bullet(doc, "예측 성능: XGBoost 5-fold AUROC 0.7587 ± 0.0008로 안정적인 "
                "성능 확보. TabNet도 0.7518로 비슷한 수준.")
    Bullet(doc, "해석 가능성: TabNet의 어텐션과 SHAP이 핵심 변수에서 일관되며 "
                "(EXT_SOURCE 등), 미세 영역에서 상보적으로 작동하는 것을 정량 "
                "검증.")
    Bullet(doc, "자연어 설명의 신뢰성: 두 상용 LLM(Gemini, Claude) 모두에서 "
                "환각률 0%. SHAP 없는 baseline에서는 Claude가 45.5%까지 환각 — "
                "본 구조가 환각을 명확히 차단한다는 직접 증거.")
    P(doc, "더불어 8개 공정성 케이스 모두에서 4/5 rule 위반을 발견했고, 단순 "
            "ablation의 효과와 한계(특히 AGE의 proxy variable 문제)를 측정했습니다.")

    H(doc, "11. 한계와 향후 계획 (Future Work)", level=1)
    Bullet(doc, "표본 크기: 평가 샘플이 10명으로 제한됨. 100~500명으로 확장 필요.")
    Bullet(doc, "보조 테이블 미사용: bureau, previous_application 등을 추가하면 "
                "AUROC를 0.78+로 끌어올릴 여지가 있음.")
    Bullet(doc, "G-Eval self-bias: Gemini가 자기 출력을 평가하므로 cross-LLM "
                "judge가 필요.")
    Bullet(doc, "Counterfactual Test 정량화: 본 보고서에선 정성 비교만. "
                "SHAP 변수 ablation 후 출력 변화를 BERTScore 등으로 측정 필요.")
    Bullet(doc, "Robustness 평가: 프롬프트 변형, 컨텍스트 셔플 등.")
    Bullet(doc, "Fairness-aware 학습: Reweighing, Adversarial Debiasing.")
    Bullet(doc, "FT-Transformer 비교 모델 추가.")
    Bullet(doc, "인간 평가 (Plausibility) — 5점 척도 + Cohen's κ.")
    Bullet(doc, "한국어 도메인 특화 금융 LLM 미세조정 (QLoRA).")

    P(doc, "")
    Quote(doc, "Step 1 — 본 보고서 시점까지의 작업은 \"미팅용 작동 프로토타입\"의 "
                "완성입니다. 이후 Step 2~3에서는 위 한계를 하나씩 해소하며 본 "
                "논문으로 확장합니다.")

    out = PAPER_DIR / "midterm_report_friendly.docx"
    doc.save(out)
    print(f"[OK] {out} 저장")


if __name__ == "__main__":
    main()
