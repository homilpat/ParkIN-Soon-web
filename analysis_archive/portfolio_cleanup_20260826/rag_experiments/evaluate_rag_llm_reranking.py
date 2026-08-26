"""Evaluate LLM reranking on the development grounding set."""

import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI

from rag_chatbot import CHILD_DB_PATH, expand_cross_lingual_query, get_local_embedder
from rag_hierarchical import candidate_results
from rag_llm_reranker import rerank_evidence

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "modeling/rag_llm_reranking_evaluation.json"
MODEL = "parkinsoon-eval"


def main():
    cases = json.loads((ROOT / "rag_grounding_cases.json").read_text(encoding="utf-8"))
    cases += json.loads((ROOT / "rag_grounding_cases_extra.json").read_text(encoding="utf-8"))
    with CHILD_DB_PATH.open("rb") as handle:
        db = pickle.load(handle)
    embedder = get_local_embedder()
    expanded = [expand_cross_lingual_query(case["question"]) for case in cases]
    query_embeddings = embedder.encode(expanded, normalize_embeddings=True)
    pools = [
        candidate_results(db, np.asarray(vector, dtype=np.float32), query, candidate_k=20)
        for vector, query in zip(query_embeddings, expanded)
    ]
    client = OpenAI(api_key="lm-studio", base_url="http://127.0.0.1:1234/v1")

    def evaluate(args):
        case, pool = args
        selected = rerank_evidence(client, MODEL, case["question"], pool, top_k=3)
        page_hit = any(
            item["source"] == case["source"]
            and any(item["page"] <= page <= item["page_end"] for page in case["pages"])
            for item in selected
        )
        return {
            "id": case["id"], "source_page_hit": page_hit,
            "selected": [{"source": item["source"], "page": item["page"]} for item in selected],
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(evaluate, zip(cases, pools)))
    summary = {
        "cases": len(rows),
        "source_page_recall_at_3": sum(row["source_page_hit"] for row in rows) / len(rows),
    }
    OUTPUT.write_text(json.dumps({"dataset_role": "development_regression", "summary": summary,
                                  "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("misses:", [row["id"] for row in rows if not row["source_page_hit"]])


if __name__ == "__main__":
    main()
