"""Dedicated structured-evidence index and numeric-query candidate routing."""

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from rag_hierarchical import candidate_results

NUMERIC_QUERY_RE = re.compile(
    r"얼마|어느\s*정도|몇\s*(?:명|개|건|년|점)|수치|점수|성능|비율|범위|유병률|정확도|민감도|특이도|"
    r"임계값|백분위|효과크기|오즈|신뢰구간|표\s*\d*|"
    r"\b(?:AUC|OR|RR|HR|CI|p[ -]?value|percent|accuracy|sensitivity|specificity)\b",
    re.IGNORECASE,
)
CAPTION_RE = re.compile(
    r"(?:^|\s)(?:table|figure|fig\.)\s*\d+[.:]?\s+", re.IGNORECASE
)
STRUCTURED_TYPES = {"numeric", "table", "caption"}


def is_numeric_query(query: str) -> bool:
    return bool(NUMERIC_QUERY_RE.search(query))


def build_evidence_db(child_db: Dict[str, Any], output: Path) -> Dict[str, int]:
    """Create a physically separate index without recomputing unchanged embeddings."""
    indices, metadata = [], []
    for index, item in enumerate(child_db["metadata"]):
        evidence_type = item.get("evidence_type", "body")
        if evidence_type not in {"numeric", "table"} and not CAPTION_RE.search(item["text"]):
            continue
        copied = dict(item)
        if evidence_type not in {"numeric", "table"}:
            copied["evidence_type"] = "caption"
            copied["section"] = "Caption"
        indices.append(index)
        metadata.append(copied)
    data = {
        "metadata": metadata,
        "embeddings": np.asarray(child_db["embeddings"], dtype=np.float32)[indices],
        "embedding_model": child_db["embedding_model"],
        "index_role": "table_numeric_caption_v1",
        "source_child_strategy": child_db.get("chunk_strategy"),
    }
    with output.open("wb") as handle:
        pickle.dump(data, handle)
    counts = {kind: sum(x.get("evidence_type") == kind for x in metadata)
              for kind in STRUCTURED_TYPES}
    return {"total": len(metadata), **counts}


def routed_candidates(
    child_db: Dict[str, Any], evidence_db: Dict[str, Any] | None,
    query_embedding: np.ndarray, query: str, candidate_k: int = 28,
) -> List[Dict[str, Any]]:
    """Reserve structured slots for numeric questions, then merge general evidence."""
    general = candidate_results(child_db, query_embedding, query, candidate_k=40)
    if not evidence_db or not is_numeric_query(query):
        return general[:min(candidate_k, 20)]
    structured = candidate_results(evidence_db, query_embedding, query, candidate_k=20)
    merged, seen = [], set()
    for item in structured[:14] + general:
        key = (item["source"], item["page"], item["text"])
        if key in seen:
            continue
        copied = dict(item)
        if item.get("evidence_type") in STRUCTURED_TYPES:
            copied["retrieval_score"] = float(item["retrieval_score"]) + 0.06
            copied["structured_priority"] = True
        merged.append(copied)
        seen.add(key)
        if len(merged) == candidate_k:
            break
    return merged
