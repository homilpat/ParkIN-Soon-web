"""Evaluate dense and hybrid retrieval against representative service questions."""

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from rag_chatbot import ParkinsonRAG, expand_cross_lingual_query, get_local_embedder
from rag_retrieval import hybrid_search

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "modeling/rag_retrieval_evaluation.json"
CASES = [
    ("후각 저하는 파킨슨병의 초기 신호인가요?", "후각과 파킨슨 종단연구"),
    ("냄새 검사와 머신러닝으로 파킨슨 위험을 구분할 수 있나요?", "후각머신러닝과 파킨슨"),
    ("변비는 파킨슨병보다 먼저 생길 수 있나요?", "변비와 파킨슨"),
    ("파킨슨병의 전구기 비운동 증상에는 무엇이 있나요?", "전구기종합파킨슨증상"),
    ("나선 그리기로 파킨슨 위험 신호를 볼 수 있나요?", "나선형그리기와 파킨슨"),
    ("나선 그림 AI가 어느 부분을 보고 판단하나요?", "나선형 그리기그래드캠"),
    ("고령자에게 검사 결과를 어떻게 쉽게 설명해야 하나요?", "노인친화적 설명"),
]


def reciprocal_rank(results, expected):
    for rank, result in enumerate(results, 1):
        if expected in result["source"]:
            return 1.0 / rank
    return 0.0


def source_diversity(results):
    return len({item["source"] for item in results[:3]}) / 3.0


def main():
    rag = ParkinsonRAG()
    embedder = get_local_embedder()
    rows = []
    for query, expected in CASES:
        expanded = expand_cross_lingual_query(query)
        embedding = np.asarray(embedder.encode(expanded, normalize_embeddings=True), dtype=np.float32)
        dense_scores = rag.db["embeddings"] @ embedding
        dense_indices = np.argsort(dense_scores)[::-1][:5]
        dense = [{"source": rag.db["metadata"][index]["source"]} for index in dense_indices]
        hybrid = hybrid_search(rag.db, embedding, expanded, top_k=5)
        rows.append({
            "query": query, "expected_source": expected,
            "dense_rr": reciprocal_rank(dense, expected),
            "hybrid_rr": reciprocal_rank(hybrid, expected),
            "dense_top3": [item["source"] for item in dense[:3]],
            "hybrid_top3": [item["source"] for item in hybrid[:3]],
            "dense_source_diversity_at_3": source_diversity(dense),
            "hybrid_source_diversity_at_3": source_diversity(hybrid),
        })
    irrelevant_query = "오늘 서울 날씨와 주식 가격을 알려줘"
    irrelevant_embedding = np.asarray(
        embedder.encode(irrelevant_query, normalize_embeddings=True), dtype=np.float32
    )
    irrelevant_results = hybrid_search(rag.db, irrelevant_embedding, irrelevant_query, top_k=3)
    report = {
        "cases": rows,
        "dense_recall_at_3": float(np.mean([row["dense_rr"] >= 1 / 3 for row in rows])),
        "hybrid_recall_at_3": float(np.mean([row["hybrid_rr"] >= 1 / 3 for row in rows])),
        "dense_mrr_at_5": float(np.mean([row["dense_rr"] for row in rows])),
        "hybrid_mrr_at_5": float(np.mean([row["hybrid_rr"] for row in rows])),
        "dense_source_diversity_at_3": float(np.mean([row["dense_source_diversity_at_3"] for row in rows])),
        "hybrid_source_diversity_at_3": float(np.mean([row["hybrid_source_diversity_at_3"] for row in rows])),
        "irrelevant_query_max_similarity": float(max(item["similarity"] for item in irrelevant_results)),
        "irrelevant_query_abstains_at_0_48": bool(max(item["similarity"] for item in irrelevant_results) < .48),
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
