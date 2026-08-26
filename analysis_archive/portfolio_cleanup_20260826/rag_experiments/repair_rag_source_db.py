"""Rebuild one PDF source after parser fixes without re-embedding every paper."""

import pickle

import numpy as np
import pypdf

from rag_chatbot import DB_PATH, PAPERS_DIR, chunk_academic_pdf, get_local_embedder
from rag_structured_evidence import extract_numeric_chunks, extract_table_chunks

SOURCE = "노인친화적 설명.pdf"


def main():
    pdf_path = PAPERS_DIR / SOURCE
    body = chunk_academic_pdf(pypdf.PdfReader(pdf_path))
    for item in body:
        item["evidence_type"] = "body"
    rebuilt = body + extract_numeric_chunks(body) + extract_table_chunks(pdf_path)
    rebuilt = [{**item, "source": SOURCE} for item in rebuilt]

    with DB_PATH.open("rb") as handle:
        db = pickle.load(handle)
    keep = [index for index, item in enumerate(db["metadata"]) if item["source"] != SOURCE]
    metadata = [db["metadata"][index] for index in keep]
    embeddings = np.asarray(db["embeddings"], dtype=np.float32)[keep]

    embedder = get_local_embedder()
    new_embeddings = embedder.encode(
        [item["text"] for item in rebuilt], batch_size=16,
        normalize_embeddings=True, show_progress_bar=True,
    )
    db["metadata"] = metadata + rebuilt
    db["embeddings"] = np.vstack([embeddings, np.asarray(new_embeddings, dtype=np.float32)])
    db.setdefault("source_repairs", {})[SOURCE] = "toc_reference_guard_v1"
    with DB_PATH.open("wb") as handle:
        pickle.dump(db, handle)
    print({"source": SOURCE, "rebuilt_chunks": len(rebuilt), "total_chunks": len(db["metadata"])})


if __name__ == "__main__":
    main()
