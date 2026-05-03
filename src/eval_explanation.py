"""Day 7 — LLM 자연어 설명의 정량 평가.

평가 차원 (계획서 3.8):
  1. Faithfulness Score (3.8.1): 컨텍스트의 변수·값·SHAP 부호가 텍스트에 정확히 인용된 비율
  2. Hallucination Rate (3.8.2): 텍스트에 등장한 변수 중 컨텍스트에 없는 비율
  3. G-Eval (3.8.4): LLM-as-a-Judge — 정확성·완결성·민감변수 노출·문체 5점 평가
  4. (선택) Counterfactual Test (3.8.1.d): SHAP 변수 제거 후 재생성, 설명 변화 측정

산출물:
  - results/explanation_eval.csv          : 샘플 × 차원 별 점수
  - results/explanation_eval_summary.json : 모델별 mean ± std
  - results/eval_geval_raw.json           : G-Eval LLM 응답 raw
  - figures/20_eval_metrics.png

사용:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m src.eval_explanation \
      [--explanations-dir results/explanations] [--skip-geval]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from src.utils import FIGURES_DIR, RESULTS_DIR, savefig

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
load_dotenv("D:/paper/.env")

EXPLANATIONS_DIR_DEFAULT = RESULTS_DIR / "explanations"
CONTEXTS_DIR = RESULTS_DIR / "contexts"

# 본 데이터셋의 모든 변수 raw 이름 (전처리 후 214개)
# 동적으로 build — preprocessor의 feature_names_ 사용
def load_all_feature_names() -> set:
    """train_scaled.parquet에서 컬럼명을 가져옴 (TARGET 제외).

    preprocessor.pkl을 unpickle하는 게 안전하지만 module 경로 issue 우회.
    """
    df = pd.read_parquet(Path("data/processed/train_scaled.parquet"))
    return {c for c in df.columns if c != "TARGET"}


# ─────────────────────────────────────────────────────────────
# 1. Faithfulness Score (룰 기반)
# ─────────────────────────────────────────────────────────────
def extract_drivers_from_context(ctx: Dict) -> List[Dict]:
    """컨텍스트에서 모든 driver 통합."""
    return (ctx.get("top_drivers_for_default", []) +
            ctx.get("top_drivers_against_default", []))


def faithfulness_per_driver(text: str, drivers: List[Dict]) -> List[Dict]:
    """driver별로 (한국어 변수명 또는 raw명) + 값이 텍스트에 등장했는지 점검."""
    results = []
    for d in drivers:
        feat_kr = d["feature"]
        feat_raw = d["feature_raw"]
        # 변수명 매칭: 한국어 풀어쓰기 또는 raw 둘 중 하나라도
        feat_in = (feat_kr in text) or (feat_raw in text)

        # 값 매칭: 보다 견고하게 — 정수, 소수점 표기 모두 시도
        val_str = str(d["value"])
        val_raw = d.get("value_raw")
        val_in = val_str in text
        if not val_in and isinstance(val_raw, (int, float)):
            # 다양한 포맷으로 매칭 시도
            for cand in [
                f"{val_raw:.6g}",
                f"{val_raw:.4f}",
                f"{val_raw:,.0f}",
                f"{int(val_raw):,}",
                f"{val_raw:.7f}",
            ]:
                if cand in text:
                    val_in = True
                    break

        # SHAP 부호 반영 점검: positive shap → "높였습니다" / "부정적인" 등 키워드
        sign = "+" if d["shap"] > 0 else "-"
        # 키워드 휴리스틱
        # positive(=거절 측): "높였", "부정적", "부도 가능성" 등 가까이 있는지
        # negative(=승인 측): "낮추는", "긍정적" 등 가까이 있는지
        # 변수명을 찾고 그 주변 60자에 키워드가 있는지
        sign_in = None
        if feat_in:
            idx = text.find(feat_kr) if feat_kr in text else text.find(feat_raw)
            window = text[max(0, idx - 30): idx + 100]
            pos_words = ["높였", "높아", "부정", "악영향", "거절", "낮은", "낮게"]
            neg_words = ["낮추", "긍정", "안정", "유리", "도움", "높게"]
            if d["shap"] > 0:
                sign_in = any(w in window for w in pos_words)
            else:
                sign_in = any(w in window for w in neg_words)

        results.append({
            "feature_raw": feat_raw,
            "shap": d["shap"],
            "feat_in": feat_in,
            "val_in": val_in,
            "sign_in": sign_in,
            "all_match": bool(feat_in and val_in and (sign_in is None or sign_in)),
        })
    return results


def faithfulness_score(per_driver: List[Dict]) -> Dict[str, float]:
    if not per_driver:
        return {"feat_match_rate": 0.0, "val_match_rate": 0.0,
                "sign_match_rate": 0.0, "full_match_rate": 0.0}
    return {
        "feat_match_rate": float(np.mean([r["feat_in"] for r in per_driver])),
        "val_match_rate": float(np.mean([r["val_in"] for r in per_driver])),
        "sign_match_rate": float(np.mean([r["sign_in"] for r in per_driver if r["sign_in"] is not None])),
        "full_match_rate": float(np.mean([r["all_match"] for r in per_driver])),
    }


# ─────────────────────────────────────────────────────────────
# 2. Hallucination Rate (룰 기반)
# ─────────────────────────────────────────────────────────────
def hallucination_rate(text: str, ctx: Dict, all_features: set) -> Dict:
    """텍스트에 등장한 변수처럼 보이는 토큰 중 컨텍스트에 없는 비율."""
    drivers = extract_drivers_from_context(ctx)
    in_context_features = {d["feature_raw"] for d in drivers}

    # 1) 영문 변수명 (UPPER_SNAKE_CASE) 등장 점검
    raw_candidates = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text))
    # SHAP, REJECT, APPROVE 같은 일반 단어 제외
    common_excludes = {"SHAP", "REJECT", "APPROVE", "TARGET", "JSON", "API",
                        "AUC", "AUROC", "AUPRC", "OK", "AI", "LLM"}
    raw_candidates -= common_excludes

    raw_in_ctx = raw_candidates & in_context_features
    raw_in_dataset = raw_candidates & all_features
    raw_outside = raw_candidates - all_features  # 데이터셋에 없는 완전 가짜
    raw_inside_dataset_outside_ctx = (raw_in_dataset - in_context_features)

    # 2) 한국어 풀어쓰기 변수명 — 컨텍스트의 한국어명만 등장해야 함
    ctx_kr_names = {d["feature"] for d in drivers}

    return {
        "n_raw_candidates": len(raw_candidates),
        "n_raw_in_context": len(raw_in_ctx),
        "n_raw_in_dataset_only": len(raw_inside_dataset_outside_ctx),
        "n_raw_outside_dataset": len(raw_outside),
        "raw_outside_dataset_list": sorted(list(raw_outside)),
        "raw_in_dataset_outside_context_list": sorted(list(raw_inside_dataset_outside_ctx)),
        # hallucination = "실제 데이터셋에도 없는 가짜 변수명을 만들어낸 비율"
        "hallucination_rate_strict": (
            len(raw_outside) / max(len(raw_candidates), 1)
        ),
        # broader = "컨텍스트에 없는 변수를 인용한 비율"
        "hallucination_rate_broad": (
            (len(raw_outside) + len(raw_inside_dataset_outside_ctx))
            / max(len(raw_candidates), 1)
        ),
    }


# ─────────────────────────────────────────────────────────────
# 3. G-Eval (LLM-as-a-Judge)
# ─────────────────────────────────────────────────────────────
G_EVAL_RUBRIC = """당신은 신용 평가 LLM 설명의 평가자입니다.
아래 [생성된 설명]을 [참조 컨텍스트]에 비추어 4개 차원별 1~5점 척도로 평가하세요.

[참조 컨텍스트]
{context_json}

[생성된 설명]
{explanation_text}

평가 차원 (각 1~5점):
1. factual_accuracy : 설명의 모든 변수명·수치·SHAP 부호가 컨텍스트와 일치하는가
   (1=다수 불일치 또는 환각, 5=완벽 일치)
2. completeness : 컨텍스트의 top driver를 충분히 다루며 결정 사유를 명확히 전달하는가
   (1=핵심 누락, 5=완결)
3. sensitive_leak : 성별·연령·인종·종교 등 민감 변수를 직접 언급하지 않았는가
   (1=직접 언급, 5=완벽 마스킹)
4. style : 고객에게 전달하기 적절한 친절·중립적 톤인가
   (1=부적절, 5=매우 적절)

오직 JSON 한 개만 출력하세요. 다른 텍스트는 금지.
{{
  "factual_accuracy": <int 1-5>,
  "completeness": <int 1-5>,
  "sensitive_leak": <int 1-5>,
  "style": <int 1-5>,
  "rationale": "<한 줄 사유>"
}}
"""


def g_eval_one(client, explanation_text: str, context: Dict,
                model: str = "gemini-2.5-flash") -> Dict:
    """단일 샘플에 대한 G-Eval. JSON 응답 파싱 포함."""
    ctx_for_prompt = {k: v for k, v in context.items() if not k.startswith("_meta")}
    prompt = G_EVAL_RUBRIC.format(
        context_json=json.dumps(ctx_for_prompt, ensure_ascii=False, indent=2),
        explanation_text=explanation_text,
    )
    resp = client.models.generate_content(model=model, contents=prompt)
    text = resp.text.strip()

    # JSON 추출 — ```json ... ``` 블록이거나 plain JSON
    json_match = re.search(r"\{[^{}]*?\"factual_accuracy\".*?\}", text, flags=re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            parsed = {"raw": text, "parse_error": True}
    else:
        parsed = {"raw": text, "parse_error": True}

    return {
        "model": model,
        "elapsed_sec": None,  # 타이밍은 호출부에서
        "parsed": parsed,
        "raw_response": text,
    }


def make_geval_client():
    import os
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 누락")
    return genai.Client(api_key=api_key)


# ─────────────────────────────────────────────────────────────
# 4. 메인 평가 루프
# ─────────────────────────────────────────────────────────────
def evaluate_all(explanations_dir: Path, skip_geval: bool = False,
                  geval_sleep: float = 5.0) -> pd.DataFrame:
    all_features = load_all_feature_names()
    rows = []

    files = sorted([p for p in explanations_dir.glob("*.json")
                     if p.name != "_index.json"])
    print(f"[1/3] 평가 대상: {len(files)}개 설명")
    geval_raw = {}

    if not skip_geval:
        client = make_geval_client()

    for i, exp_path in enumerate(files, start=1):
        with open(exp_path, "r", encoding="utf-8") as f:
            exp = json.load(f)

        ctx = exp["context_sent"]
        text = exp["explanation"]
        sample_id = exp_path.stem

        # 1. Faithfulness
        drivers = extract_drivers_from_context(ctx)
        per_driver = faithfulness_per_driver(text, drivers)
        f_scores = faithfulness_score(per_driver)

        # 2. Hallucination
        h_scores = hallucination_rate(text, ctx, all_features)

        row = {
            "sample_id": sample_id,
            "decision": exp.get("decision"),
            "true_label": exp.get("true_label"),
            "model_llm": exp.get("model"),
            "default_proba": ctx.get("default_probability"),
            **f_scores,
            "halluc_rate_strict": h_scores["hallucination_rate_strict"],
            "halluc_rate_broad": h_scores["hallucination_rate_broad"],
            "n_raw_candidates": h_scores["n_raw_candidates"],
            "n_raw_outside_dataset": h_scores["n_raw_outside_dataset"],
        }

        # 3. G-Eval
        if not skip_geval:
            print(f"[2/3] G-Eval [{i}/{len(files)}] {sample_id} ...")
            try:
                t0 = time.time()
                ge = g_eval_one(client, text, ctx)
                el = time.time() - t0
                ge["elapsed_sec"] = round(el, 2)
                geval_raw[sample_id] = ge
                p = ge["parsed"]
                if not p.get("parse_error"):
                    row.update({
                        "geval_factual_accuracy": p.get("factual_accuracy"),
                        "geval_completeness": p.get("completeness"),
                        "geval_sensitive_leak": p.get("sensitive_leak"),
                        "geval_style": p.get("style"),
                        "geval_rationale": p.get("rationale", "")[:200],
                    })
                else:
                    row.update({"geval_parse_error": True})
                if i < len(files):
                    time.sleep(geval_sleep)
            except Exception as e:
                print(f"  ERROR: {e}")
                row.update({"geval_error": str(e)[:200]})

        rows.append(row)

    df = pd.DataFrame(rows)
    if geval_raw:
        with open(RESULTS_DIR / "eval_geval_raw.json", "w", encoding="utf-8") as f:
            json.dump(geval_raw, f, indent=2, ensure_ascii=False)

    return df


def summarize(df: pd.DataFrame) -> Dict:
    """차원별 mean ± std 요약."""
    cols = [c for c in df.columns if c.startswith(("feat_", "val_", "sign_",
              "full_", "halluc_", "geval_factual", "geval_completeness",
              "geval_sensitive", "geval_style"))]
    out = {}
    for c in cols:
        s = df[c].dropna()
        if pd.api.types.is_numeric_dtype(s) and len(s) > 0:
            out[c] = {"mean": float(s.mean()), "std": float(s.std()),
                       "n": int(len(s))}
    return out


def plot_eval_metrics(df: pd.DataFrame, out_name: str = "20_eval_metrics") -> Path:
    metrics = ["full_match_rate", "halluc_rate_strict",
                "geval_factual_accuracy", "geval_completeness",
                "geval_sensitive_leak", "geval_style"]
    available = [m for m in metrics if m in df.columns]
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, available):
        s = df[m].dropna()
        if len(s) == 0:
            ax.set_visible(False)
            continue
        sns.boxplot(data=s, ax=ax, color="#4C72B0")
        sns.stripplot(data=s, ax=ax, color="black", size=4, alpha=0.6)
        ax.set_title(f"{m}\nmean={s.mean():.3f}, std={s.std():.3f}",
                      fontsize=10)
        ax.set_ylabel("score")
    plt.suptitle("LLM 설명 정량 평가 — 10 샘플 분포")
    return savefig(fig, out_name)


def main(explanations_dir: str | None = None,
         skip_geval: bool = False,
         suffix: str | None = None,
         geval_sleep: float = 15.0) -> None:
    exp_dir = Path(explanations_dir) if explanations_dir else EXPLANATIONS_DIR_DEFAULT
    if suffix is None:
        # 디렉토리 이름에서 자동 추출: explanations → "gemini",
        # explanations_anthropic → "anthropic"
        name = exp_dir.name
        if name == "explanations":
            suffix = "gemini"
        elif name.startswith("explanations_"):
            suffix = name[len("explanations_"):]
        else:
            suffix = name

    df = evaluate_all(exp_dir, skip_geval=skip_geval, geval_sleep=geval_sleep)
    csv_path = RESULTS_DIR / f"explanation_eval_{suffix}.csv"
    json_path = RESULTS_DIR / f"explanation_eval_summary_{suffix}.json"
    geval_path = RESULTS_DIR / f"eval_geval_raw_{suffix}.json"
    fig_name = f"20_eval_metrics_{suffix}"

    df.to_csv(csv_path, index=False)
    summary = summarize(df)
    summary["_provider"] = suffix
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # G-Eval raw 파일 이름도 변경 (evaluate_all에서 만든 게 있으면 rename)
    default_geval = RESULTS_DIR / "eval_geval_raw.json"
    if default_geval.exists():
        default_geval.replace(geval_path)

    print("\n[3/3] 요약")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    plot_eval_metrics(df, out_name=fig_name)
    print(f"\n[OK] Day 7 평가 완료 ({suffix})")
    print(f"     - {csv_path}")
    print(f"     - {json_path}")
    print(f"     - {geval_path}")
    print(f"     - figures/{fig_name}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--explanations-dir", default=None,
                    help="설명 JSON 디렉토리 (기본: results/explanations)")
    ap.add_argument("--skip-geval", action="store_true",
                    help="G-Eval(LLM-as-Judge) 단계 건너뛰기")
    ap.add_argument("--suffix", default=None,
                    help="결과 파일 suffix (기본: dir 이름 기반 자동)")
    ap.add_argument("--geval-sleep", type=float, default=15.0,
                    help="G-Eval 호출 간 sleep (free tier RPM 회피)")
    args = ap.parse_args()
    main(explanations_dir=args.explanations_dir, skip_geval=args.skip_geval,
         suffix=args.suffix, geval_sleep=args.geval_sleep)
