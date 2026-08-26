"""Evaluate source/page retrieval for every groundedness gold case."""

import argparse
import hashlib
import json
from pathlib import Path

from rag_chatbot import ParkinsonRAG

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "modeling/rag_grounding_retrieval_evaluation.json"
BLIND_OUTPUT = ROOT / "modeling/rag_blind_retrieval_evaluation.json"
BLIND_V2_OUTPUT = ROOT / "modeling/rag_blind_v2_retrieval_evaluation.json"


def load_cases():
    cases = json.loads((ROOT / "rag_grounding_cases.json").read_text(encoding="utf-8"))
    cases += json.loads((ROOT / "rag_grounding_cases_extra.json").read_text(encoding="utf-8"))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--blind-v2", action="store_true")
    args = parser.parse_args()
    rag = ParkinsonRAG()
    if args.blind_v2:
        cases = json.loads(
            (ROOT / "rag_blind_grounding_cases_v2.json").read_text(encoding="utf-8")
        )
    elif args.blind:
        cases = json.loads(
            (ROOT / "rag_blind_grounding_cases.json").read_text(encoding="utf-8")
        )
    else:
        cases = load_cases()
    rows = []
    for case in cases:
        results = rag.search_similar_chunks(case["question"], top_k=3)
        source_hits = [item for item in results if item["source"] == case["source"]]
        page_hit = any(
            any(item["page"] <= page <= item["page_end"] for page in case["pages"])
            for item in source_hits
        )
        rows.append({
            "id": case["id"],
            "source_hit": bool(source_hits),
            "source_page_hit": page_hit,
            "retrieved": [
                {"source": item["source"], "page": item["page"],
                 "page_end": item["page_end"], "evidence_type": item["evidence_type"]}
                for item in results
            ],
        })
    summary = {
        "cases": len(rows),
        "source_recall_at_3": sum(row["source_hit"] for row in rows) / len(rows),
        "source_page_recall_at_3": sum(row["source_page_hit"] for row in rows) / len(rows),
    }
    is_blind = args.blind or args.blind_v2
    report = {"dataset_role": "locked_blind_test_v2" if args.blind_v2 else (
                  "locked_blind_test" if args.blind else "development_regression"),
              "independent_test": is_blind,
              "dataset_sha256": hashlib.sha256(
                  json.dumps(cases, ensure_ascii=False, sort_keys=True).encode("utf-8")
              ).hexdigest(),
              "summary": summary, "results": rows}
    output_path = BLIND_V2_OUTPUT if args.blind_v2 else (BLIND_OUTPUT if args.blind else OUTPUT)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("page misses:", [row["id"] for row in rows if not row["source_page_hit"]])


if __name__ == "__main__":
    main()
