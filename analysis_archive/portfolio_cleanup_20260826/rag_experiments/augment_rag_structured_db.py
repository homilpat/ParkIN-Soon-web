"""Append dedicated numeric and table evidence to the existing vector DB."""

import pickle
from pathlib import Path

import numpy as np
import pypdf

from rag_chatbot import DB_PATH, PAPERS_DIR, chunk_academic_pdf, get_local_embedder
from rag_structured_evidence import extract_numeric_chunks, extract_table_chunks


def collect_structured_chunks():
    chunks = []
    for pdf_path in PAPERS_DIR.glob("*.pdf"):
        body_chunks = chunk_academic_pdf(pypdf.PdfReader(pdf_path))
        evidence = extract_numeric_chunks(body_chunks) + extract_table_chunks(pdf_path)
        chunks.extend({**chunk, "source": pdf_path.name} for chunk in evidence)
    return chunks


def main():
    with DB_PATH.open("rb") as handle:
        db = pickle.load(handle)
    body_indices = [
        index for index, item in enumerate(db["metadata"])
        if item.get("evidence_type", "body") == "body"
    ]
    body_metadata = [db["metadata"][index] for index in body_indices]
    body_embeddings = np.asarray(db["embeddings"], dtype=np.float32)[body_indices]
    structured = collect_structured_chunks()
    embedder = get_local_embedder()
    structured_embeddings = embedder.encode(
        [item["text"] for item in structured],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    db["metadata"] = body_metadata + structured
    db["embeddings"] = np.vstack([
        body_embeddings, np.asarray(structured_embeddings, dtype=np.float32)
    ])
    db["structured_evidence"] = "numeric_context_and_tables_v1"
    db["structured_counts"] = {
        "numeric": sum(item["evidence_type"] == "numeric" for item in structured),
        "table": sum(item["evidence_type"] == "table" for item in structured),
    }
    with DB_PATH.open("wb") as handle:
        pickle.dump(db, handle)
    print({"body": len(body_metadata), **db["structured_counts"], "total": len(db["metadata"])})


if __name__ == "__main__":
    main()
