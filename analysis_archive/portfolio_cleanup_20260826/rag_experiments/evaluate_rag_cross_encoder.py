"""Evaluate BGE cross-encoder reranking only on the development set."""

import json
import pickle
import time
from pathlib import Path

import numpy as np

from rag_chatbot import (CHILD_DB_PATH, EVIDENCE_DB_PATH, expand_cross_lingual_query,
                         get_local_embedder)
from rag_cross_encoder import CrossEncoderReranker
from rag_evidence_index import routed_candidates

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "modeling" / "rag_cross_encoder_evaluation.json"


def load_cases():
    cases = json.loads((ROOT / "rag_grounding_cases.json").read_text(encoding="utf-8"))
    cases += json.loads((ROOT / "rag_grounding_cases_extra.json").read_text(encoding="utf-8"))
    return cases


def hit(case, item, require_page=False):
    if item["source"] != case["source"]:
        return False
    return not require_page or any(
        item["page"] <= page <= item["page_end"] for page in case["pages"]
    )


def normalized(values):
    low, high = min(values), max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def blended_top3(pool, reranker_scores, alpha):
    retrieval = normalized([item["retrieval_score"] for item in pool])
    reranker = normalized(reranker_scores)
    ranked = sorted(
        zip(pool, retrieval, reranker),
        key=lambda row: (1.0 - alpha) * row[1] + alpha * row[2],
        reverse=True,
    )
    selected, seen_pages = [], set()
    for item, _, _ in ranked:
        page_key = (item["source"], item["page"])
        if page_key not in seen_pages:
            selected.append(item)
            seen_pages.add(page_key)
        if len(selected) == 3:
            break
    return selected


def main():
    cases = load_cases()
    with CHILD_DB_PATH.open("rb") as handle:
        db = pickle.load(handle)
    with EVIDENCE_DB_PATH.open("rb") as handle:
        evidence_db = pickle.load(handle)
    embedder = get_local_embedder()
    expanded = [expand_cross_lingual_query(case["question"]) for case in cases]
    vectors = embedder.encode(expanded, normalize_embeddings=True)
    pools = [
        routed_candidates(db, evidence_db, np.asarray(vector, dtype=np.float32), query, 28)
        for vector, query in zip(vectors, expanded)
    ]
    reranker = CrossEncoderReranker()
    scored_pools, elapsed = [], []
    for case, pool, expanded_query in zip(cases, pools, expanded):
        started = time.perf_counter()
        scores = reranker.scores(expanded_query, pool)
        elapsed.append(time.perf_counter() - started)
        scored_pools.append((pool, scores))
    variants, rows_by_alpha = {}, {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        rows = []
        for case, (pool, scores) in zip(cases, scored_pools):
            selected = blended_top3(pool, scores, alpha)
            rows.append({
                "id": case["id"],
                "source_hit": any(hit(case, item) for item in selected),
                "source_page_hit": any(hit(case, item, True) for item in selected),
                "selected": [{"source": item["source"], "page": item["page"]}
                             for item in selected],
            })
        key = str(alpha)
        rows_by_alpha[key] = rows
        variants[key] = {
            "source_recall_at_3": sum(row["source_hit"] for row in rows) / len(rows),
            "source_page_recall_at_3": sum(row["source_page_hit"] for row in rows) / len(rows),
        }
    best_alpha = max(variants, key=lambda key: (
        variants[key]["source_page_recall_at_3"], variants[key]["source_recall_at_3"]
    ))
    rows = rows_by_alpha[best_alpha]
    summary = {
        "cases": len(rows),
        "selected_alpha": float(best_alpha),
        **variants[best_alpha],
        "mean_rerank_seconds": sum(elapsed) / len(elapsed),
    }
    report = {"dataset_role": "development_regression", "candidate_k": 28,
              "model": "BAAI/bge-reranker-v2-m3", "summary": summary,
              "blend_variants": variants, "results": rows}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("page misses:", [row["id"] for row in rows if not row["source_page_hit"]])


if __name__ == "__main__":
    main()
