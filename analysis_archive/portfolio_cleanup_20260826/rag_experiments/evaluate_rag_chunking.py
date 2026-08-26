"""Measure whether gold evidence survives PDF chunking intact."""

import json
import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "modeling/vector_db.pkl"
CASES_PATH = ROOT / "rag_grounding_cases.json"
EXTRA_CASES_PATH = ROOT / "rag_grounding_cases_extra.json"
OUTPUT_PATH = ROOT / "modeling/rag_chunking_evaluation.json"


def tokens(text):
    return set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower()))


def numeric_tokens(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def evaluate_case(case, chunks):
    candidates = [
        chunk for chunk in chunks
        if chunk["source"] == case["source"]
        and any(chunk["page"] <= page <= chunk.get("page_end", chunk["page"])
                for page in case["pages"])
    ]
    gold_tokens = tokens(case["evidence"])
    ranked = sorted(
        candidates,
        key=lambda item: len(tokens(item["text"]) & gold_tokens),
        reverse=True,
    )[:3]
    found_tokens = set().union(*(tokens(item["text"]) for item in ranked)) if ranked else set()
    gold_numbers = numeric_tokens(case["evidence"])
    found_numbers = set().union(*(numeric_tokens(item["text"]) for item in ranked)) if ranked else set()
    return {
        "id": case["id"],
        "source": case["source"],
        "pages": case["pages"],
        "evidence_chunk_pages": [
            [item["page"], item.get("page_end", item["page"])] for item in ranked
        ],
        "evidence_token_recall": len(found_tokens & gold_tokens) / max(len(gold_tokens), 1),
        "numeric_tokens_expected": sorted(gold_numbers),
        "numeric_tokens_preserved": sorted(gold_numbers & found_numbers),
        "all_numbers_preserved": gold_numbers <= found_numbers,
        "evidence_chunk_texts": [item["text"] for item in ranked],
    }


def main():
    with DB_PATH.open("rb") as handle:
        db = pickle.load(handle)
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases += json.loads(EXTRA_CASES_PATH.read_text(encoding="utf-8"))
    chunks = db["metadata"]
    rows = [evaluate_case(case, chunks) for case in cases]
    spans = [item.get("page_end", item["page"]) - item["page"] for item in chunks]
    summary = {
        "strategy": db.get("chunk_strategy", "unknown"),
        "chunks": len(chunks),
        "mean_evidence_token_recall": sum(row["evidence_token_recall"] for row in rows) / len(rows),
        "numeric_preservation_rate": sum(row["all_numbers_preserved"] for row in rows) / len(rows),
        "single_page_chunk_rate": sum(span == 0 for span in spans) / max(len(spans), 1),
        "multi_page_chunks": sum(span > 0 for span in spans),
    }
    report = {"summary": summary, "results": rows}
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
