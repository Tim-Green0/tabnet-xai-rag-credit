"""텍스트 유사도 측정 헬퍼 (한국어 지원).

backbone: sentence-transformers 다국어 모델
  - paraphrase-multilingual-MiniLM-L12-v2 (한국어 OK, 빠름)

용도:
  - Counterfactual Test: 컨텍스트 변경 후 출력 차이 (낮을수록 모델이 컨텍스트에 의존)
  - Robustness Test: 프롬프트 변형 후 출력 일관성 (높을수록 안정)
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np


@lru_cache(maxsize=1)
def get_encoder(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """sentence-transformers 인코더 한 번만 로딩 (캐시)."""
    from sentence_transformers import SentenceTransformer
    print(f"[text_similarity] loading {model_name} (first call only)")
    return SentenceTransformer(model_name)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def embed_texts(texts: List[str]) -> np.ndarray:
    enc = get_encoder()
    return enc.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def similarity_pair(text_a: str, text_b: str) -> float:
    """두 텍스트의 cosine 유사도 (0~1, 1=완전 동일 의미)."""
    embs = embed_texts([text_a, text_b])
    return cosine_sim(embs[0], embs[1])


def similarity_pairs(pairs: List[tuple[str, str]]) -> List[float]:
    """대량 효율 — 한 번에 인코딩 후 매칭."""
    flat = []
    for a, b in pairs:
        flat.extend([a, b])
    embs = embed_texts(flat)
    out = []
    for i in range(len(pairs)):
        out.append(cosine_sim(embs[2*i], embs[2*i + 1]))
    return out


def rouge_l(text_a: str, text_b: str) -> float:
    """ROUGE-L F1 (간단 버전)."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    s = scorer.score(text_a, text_b)
    return float(s["rougeL"].fmeasure)
