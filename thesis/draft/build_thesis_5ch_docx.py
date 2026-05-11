"""학위논문 docx — 5장 구조 (학교 표준, 손지민 2024 패턴) 빌드.

8장 구조와 동일 본문 활용하되 다음과 같이 합쳐서 5장으로 구성:

  제1장 서론                           ← chapter1_introduction.md (그대로)
  제2장 이론적 배경                     ← chapter2_related_work.md (그대로)
  제3장 연구 방법론                     ← chapter3_data + chapter4_methodology + chapter5_evaluation
                                            (3.1 데이터, 3.2 시스템 구조 + XAI + Fusion + LLM-RAG + Reweighing,
                                             3.3 평가 프레임워크)
  제4장 분석 결과                       ← chapter6_results.md (그대로)
  제5장 결론 및 시사점                   ← chapter7_discussion + chapter8_conclusion
                                            (5.1 핵심 발견, 5.2 한계, 5.3 응용·향후, 5.4 결론)

산출:
  thesis/draft/thesis_draft_5ch.docx

실행:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m thesis.draft.build_thesis_5ch_docx
"""
from __future__ import annotations

import re
from pathlib import Path

# 같은 디렉토리의 build_thesis_docx 모듈에서 헬퍼 재사용
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_thesis_docx import (  # type: ignore
    Document, WD_ALIGN_PARAGRAPH, WD_BREAK, Cm, Pt,
    KOREAN_FONT, ENG_FONT, LINE_SPACING,
    TITLE, SUBTITLE, AUTHOR_KR, AUTHOR_EN, ADVISOR_KR, ADVISOR_EN,
    DEPT_KR, DEPT_EN, SCHOOL_KR, SCHOOL_EN,
    SUBMIT_DATE_KR, SUBMIT_DATE_FULL,
    set_run, set_paragraph_format, add_para, add_heading, add_page_break,
    add_toc_field, add_image, add_markdown_table,
    add_cover_page, add_inner_cover, add_submission_page, add_approval_page,
    add_acknowledgement,
)

DRAFT_DIR = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────
# 5장 구조용 chapter renumbering / merging
# ─────────────────────────────────────────────────────────────
def transform_md_for_5ch(md_text: str, mapping: dict) -> str:
    """8장의 chapter 번호를 5장 구조에 맞춰 재매핑.

    mapping: {'1.': '1.', '1.1': '1.1', ...}
    """
    out = md_text
    for old, new in mapping.items():
        out = out.replace(old, new)
    return out


def parse_and_add(doc, md_text: str):
    """build_thesis_docx의 parse_chapter_markdown 재사용."""
    from build_thesis_docx import parse_chapter_markdown  # type: ignore
    parse_chapter_markdown(doc, md_text)


def add_chapter_from_md(doc, md_path: Path, transform: dict | None = None):
    text = md_path.read_text(encoding="utf-8")
    if transform:
        text = transform_md_for_5ch(text, transform)
    parse_and_add(doc, text)
    add_page_break(doc)


def add_merged_chapter(doc, parts: list[tuple[Path, str]], chapter_title: str,
                        intro_text: str = ""):
    """여러 markdown 파일을 합쳐서 하나의 chapter로 추가.

    parts: [(md_path, transform_dict_or_None), ...]
    """
    add_heading(doc, chapter_title, level=1)
    if intro_text:
        add_para(doc, intro_text, size=11, indent=0.5, space_after=12)

    for md_path, prefix_replace in parts:
        text = md_path.read_text(encoding="utf-8")
        # 첫 번째 줄 (# 제목)은 제거 (이미 큰제목 추가했음)
        lines = text.splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
        text = "\n".join(lines).lstrip()
        # prefix_replace 매핑 적용
        if prefix_replace:
            text = transform_md_for_5ch(text, prefix_replace)
        parse_and_add(doc, text)

    add_page_break(doc)


# ─────────────────────────────────────────────────────────────
# 메인 빌드 (5장 구조)
# ─────────────────────────────────────────────────────────────
def main():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(3.0)
    sec.right_margin = Cm(2.5)

    # 별지 양식 (감사의 글 제거 — 사용자 결정)
    add_cover_page(doc)
    add_inner_cover(doc)
    add_submission_page(doc)
    add_approval_page(doc)

    # 목차/표차례/그림차례 — Word 필드
    add_heading(doc, "목   차", level=1)
    add_toc_field(doc)
    add_page_break(doc)

    add_heading(doc, "표 차례", level=1)
    add_toc_field(doc, instr=' TOC \\h \\z \\c "표" ')
    add_page_break(doc)

    add_heading(doc, "그림 차례", level=1)
    add_toc_field(doc, instr=' TOC \\h \\z \\c "그림" ')
    add_page_break(doc)

    # Abstract
    add_chapter_from_md(doc, DRAFT_DIR / "abstract.md")

    # 제1장 서론 (그대로)
    add_chapter_from_md(doc, DRAFT_DIR / "chapter1_introduction.md")

    # 제2장 이론적 배경 (그대로)
    add_chapter_from_md(doc, DRAFT_DIR / "chapter2_related_work.md")

    # 제3장 연구 방법론 (chapter3 + chapter4 + chapter5 합침)
    # 번호 매핑:
    #   chapter3 (제 3 장. 데이터) → 3.1 데이터셋 및 전처리
    #     3.1 → 3.1.1; 3.2 → 3.1.2; 3.3 → 3.1.3
    #   chapter4 (제 4 장. 방법론) → 3.2 시스템 구조 및 모델 설계
    #     4.1 → 3.2.1; 4.2 → 3.2.2; 4.3 → 3.2.3; 4.4 → 3.2.4 (Fusion); 4.5 → 3.2.5; 4.6 → 3.2.6
    #   chapter5 (제 5 장. 평가 프레임워크) → 3.3 평가 프레임워크
    #     5.1 → 3.3.1; 5.2 → 3.3.2; ... 5.7 → 3.3.7

    ch3_map = {
        "## 3.1 ": "### 3.1.1 ",  # 데이터셋 개요 → 3.1.1
        "### 3.1.1": "#### (a)",  # 하위 방지
        "### 3.1.2": "#### (b)",
        "### 3.1.3": "#### (c)",
        "## 3.2 ": "### 3.1.2 ",  # 탐색적 데이터 분석
        "### 3.2.1": "#### (a)",
        "### 3.2.2": "#### (b)",
        "## 3.3 ": "### 3.1.3 ",  # 전처리 정책
        "### 3.3.1": "#### (a)",
        "### 3.3.2": "#### (b)",
        "### 3.3.3": "#### (c)",
        "### 3.3.4": "#### (d)",
        "### 3.3.5": "#### (e)",
    }
    ch4_map = {
        "## 4.1 시스템 전체 구조": "### 3.2.1 시스템 전체 구조",
        "## 4.2 ": "### 3.2.2 ",
        "### 4.2.1": "#### (a)",
        "### 4.2.2": "#### (b)",
        "### 4.2.3": "#### (c)",
        "## 4.3 ": "### 3.2.3 ",
        "### 4.3.1": "#### (a)",
        "### 4.3.2": "#### (b)",
        "### 4.3.3": "#### (c)",
        "## 4.4 ": "### 3.2.4 ",  # ★ Fusion Context
        "### 4.4.1": "#### (a)",
        "### 4.4.2": "#### (b)",
        "### 4.4.3": "#### (c)",
        "## 4.5 ": "### 3.2.5 ",
        "### 4.5.1": "#### (a)",
        "### 4.5.2": "#### (b)",
        "### 4.5.3": "#### (c)",
        "### 4.5.4": "#### (d)",
        "## 4.6 ": "### 3.2.6 ",
        "### 4.6.1": "#### (a)",
        "### 4.6.2": "#### (b)",
    }
    ch5_map = {
        "## 5.1 ": "### 3.3.1 ",
        "## 5.2 ": "### 3.3.2 ",
        "### 5.2.1": "#### (a)",
        "### 5.2.2": "#### (b)",
        "## 5.3 ": "### 3.3.3 ",
        "### 5.3.1": "#### (a)",
        "### 5.3.2": "#### (b)",
        "### 5.3.3": "#### (c)",
        "## 5.4 ": "### 3.3.4 ",
        "### 5.4.1": "#### (a)",
        "### 5.4.2": "#### (b)",
        "## 5.5 ": "### 3.3.5 ",
        "### 5.5.1": "#### (a)",
        "### 5.5.2": "#### (b)",
        "## 5.6 ": "### 3.3.6 ",
        "### 5.6.1": "#### (a)",
        "### 5.6.2": "#### (b)",
        "### 5.6.3": "#### (c)",
        "## 5.7 ": "### 3.3.7 ",
    }

    add_merged_chapter(
        doc,
        [
            (DRAFT_DIR / "chapter3_data.md", {**ch3_map, "## ": "### 3.1 "}),  # not used directly
            (DRAFT_DIR / "chapter4_methodology.md", ch4_map),
            (DRAFT_DIR / "chapter5_evaluation.md", ch5_map),
        ],
        chapter_title="제 3 장. 연구 방법론",
        intro_text="본 장은 본 연구가 활용한 데이터와 전처리 정책(3.1), "
                    "예측·해석·융합·자연어 생성·공정성 보정으로 구성된 시스템 구조와 모델 설계(3.2), "
                    "그리고 4-tier 평가 프레임워크(3.3)를 통합 기술한다.",
    )

    # 제4장 분석 결과 (chapter6 그대로 + 번호 4로 재매핑)
    ch6_map = {
        "# 제 6 장. 분석 결과": "# 제 4 장. 분석 결과",
        "## 6.1 ": "## 4.1 ",
        "### 6.1.1": "### 4.1.1",
        "### 6.1.2": "### 4.1.2",
        "### 6.1.3": "### 4.1.3",
        "## 6.2 ": "## 4.2 ",
        "### 6.2.1": "### 4.2.1",
        "### 6.2.2": "### 4.2.2",
        "## 6.3 ": "## 4.3 ",
        "### 6.3.1": "### 4.3.1",
        "### 6.3.2": "### 4.3.2",
        "### 6.3.3": "### 4.3.3",
        "### 6.3.4": "### 4.3.4",
        "### 6.3.5": "### 4.3.5",
        "### 6.3.6": "### 4.3.6",
        "## 6.4 ": "## 4.4 ",
        "### 6.4.1": "### 4.4.1",
        "## 6.5 ": "## 4.5 ",
        "### 6.5.1": "### 4.5.1",
        "### 6.5.2": "### 4.5.2",
        "### 6.5.3": "### 4.5.3",
        "## 6.6 ": "## 4.6 ",
        "### 6.6.1": "### 4.6.1",
        "### 6.6.2": "### 4.6.2",
        "### 6.6.3": "### 4.6.3",
        "### 6.6.4": "### 4.6.4",
        "## 6.7 ": "## 4.7 ",
        "그림 6-": "그림 4-",
        "표 6-": "표 4-",
    }
    add_chapter_from_md(doc, DRAFT_DIR / "chapter6_results.md", transform=ch6_map)

    # 제5장 결론 및 시사점 (chapter7 + chapter8 합침)
    ch7_map = {
        "## 7.1 ": "### 5.1.1 ",
        "### 7.1.1": "#### (a)",
        "### 7.1.2": "#### (b)",
        "### 7.1.3": "#### (c)",
        "### 7.1.4": "#### (d)",
        "## 7.2 ": "### 5.1.2 ",
        "### 7.2.1": "#### (a)",
        "### 7.2.2": "#### (b)",
        "### 7.2.3": "#### (c)",
        "## 7.3 ": "### 5.1.3 ",
        "### 7.3.1": "#### (a)",
        "### 7.3.2": "#### (b)",
        "### 7.3.3": "#### (c)",
        "### 7.3.4": "#### (d)",
        "### 7.3.5": "#### (e)",
        "### 7.3.6": "#### (f)",
        "## 7.4 ": "### 5.1.4 ",
        "### 7.4.1": "#### (a)",
        "### 7.4.2": "#### (b)",
        "### 7.4.3": "#### (c)",
        "### 7.4.4": "#### (d)",
        "### 7.4.5": "#### (e)",
        "### 7.4.6": "#### (f)",
        "## 7.5 ": "### 5.1.5 ",
    }
    ch8_map = {
        "## 8.1 ": "### 5.2.1 ",
        "## 8.2 ": "### 5.2.2 ",
        "### 8.2.1": "#### (a)",
        "### 8.2.2": "#### (b)",
        "### 8.2.3": "#### (c)",
        "### 8.2.4": "#### (d)",
        "### 8.2.5": "#### (e)",
        "## 8.3 ": "### 5.2.3 ",
        "## 8.4 ": "### 5.2.4 ",
    }
    add_merged_chapter(
        doc,
        [
            (DRAFT_DIR / "chapter7_discussion.md", {**ch7_map, "# 제 7 장. 논의": "## 5.1 논의"}),
            (DRAFT_DIR / "chapter8_conclusion.md", {**ch8_map, "# 제 8 장. 결론 및 시사점": "## 5.2 결론 및 시사점"}),
        ],
        chapter_title="제 5 장. 결론 및 시사점",
        intro_text="본 장은 본 연구의 핵심 발견과 한계를 honest reporting의 관점에서 정리(5.1)하고, "
                    "이를 바탕으로 학술적·산업적 결론과 향후 연구 방향(5.2)을 제시한다.",
    )

    # 참고문헌
    add_chapter_from_md(doc, DRAFT_DIR / "references.md")

    # 부록
    add_chapter_from_md(doc, DRAFT_DIR / "appendix.md")

    out = DRAFT_DIR / "thesis_draft.docx"
    doc.save(str(out))
    print(f"[OK] thesis 5장 docx 빌드 완료: {out}")
    print(f"     파일 크기: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
