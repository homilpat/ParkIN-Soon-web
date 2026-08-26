"""Build and search sentence-level children while retaining parent context."""

import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from rag_retrieval import EVIDENCE_SECTIONS, bm25_scores, normalize

SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def child_chunks(metadata: List[Dict[str, Any]], target_size: int = 320) -> List[Dict[str, Any]]:
    children = []
    for parent_id, parent in enumerate(metadata):
        body = re.sub(r"^\[(?:Section|Evidence type): [^]]+\]\s*", "", parent["text"])
        raw_parts = [part.strip() for part in SENTENCE_RE.split(body) if len(part.strip()) >= 20]
        sentences = []
        for part in raw_parts:
            while len(part) > target_size:
                split_at = max(part.rfind(" ", 0, target_size), part.rfind(" | ", 0, target_size))
                split_at = split_at if split_at >= target_size // 2 else target_size
                sentences.append(part[:split_at].strip())
                part = part[split_at:].strip(" |")
            if part:
                sentences.append(part)
        current = []
        for sentence in sentences:
            if current and len(" ".join(current + [sentence])) > target_size:
                children.append(_child(parent, parent_id, " ".join(current)))
                current = current[-1:] if len(current[-1]) + len(sentence) + 1 <= target_size else []
            current.append(sentence)
        if current:
            children.append(_child(parent, parent_id, " ".join(current)))
    return children


def _child(parent: Dict[str, Any], parent_id: int, text: str) -> Dict[str, Any]:
    return {
        "text": text,
        "parent_text": parent["text"],
        "parent_id": parent_id,
        "source": parent["source"],
        "page": parent["page"],
        "page_end": parent.get("page_end", parent["page"]),
        "section": parent.get("section", "Body"),
        "evidence_type": parent.get("evidence_type", "body"),
    }


def save_child_db(parent_db: Dict[str, Any], embedder: Any, output: Path) -> int:
    children = child_chunks(parent_db["metadata"])
    embeddings = embedder.encode(
        [item["text"] for item in children], batch_size=64,
        normalize_embeddings=True, show_progress_bar=True,
    )
    data = {
        "metadata": children,
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "embedding_model": parent_db["embedding_model"],
        "chunk_strategy": "sentence_children_v1",
        "child_size": 320,
        "parent_strategy": parent_db.get("chunk_strategy"),
    }
    with output.open("wb") as handle:
        pickle.dump(data, handle)
    return len(children)


def hierarchical_search(
    db: Dict[str, Any], query_embedding: np.ndarray, query: str, top_k: int = 3,
) -> List[Dict[str, Any]]:
    candidates = hierarchical_candidates(db, query_embedding, query, candidate_k=80)
    metadata = candidates["metadata"]
    embeddings = candidates["embeddings"]
    ranked_indices = candidates["indices"]
    selected, page_counts = [], defaultdict(int)
    for index in ranked_indices:
        item = metadata[index]
        page_key = (item["source"], item["page"])
        if page_counts[page_key] >= 1:
            continue
        redundancy = max(
            (float(embeddings[index] @ embeddings[other]) for other in selected), default=0.0
        )
        if selected and redundancy > 0.94:
            continue
        selected.append(index)
        page_counts[page_key] += 1
        if len(selected) == top_k:
            break
    return [_result(metadata, candidates, index) for index in selected]


def hierarchical_candidates(
    db: Dict[str, Any], query_embedding: np.ndarray, query: str, candidate_k: int = 20,
) -> Dict[str, Any]:
    metadata = db["metadata"]
    embeddings = np.asarray(db["embeddings"], dtype=np.float32)
    dense = embeddings @ np.asarray(query_embedding, dtype=np.float32)
    lexical = bm25_scores(query, [item["text"] for item in metadata])
    query_numbers = set(NUMBER_RE.findall(query))
    exact = np.array([
        0.12 if query_numbers & set(NUMBER_RE.findall(item["text"])) else 0.0
        for item in metadata
    ], dtype=np.float32)
    section = np.array([
        0.025 if item.get("section") in EVIDENCE_SECTIONS else 0.0 for item in metadata
    ], dtype=np.float32)
    score = 0.82 * normalize(dense) + 0.18 * normalize(lexical) + exact + section
    return {
        "metadata": metadata, "embeddings": embeddings,
        "indices": [int(index) for index in np.argsort(score)[::-1][:candidate_k]],
        "dense": dense, "lexical": lexical, "score": score,
    }


def _result(metadata, candidates, index):
    return {
        **metadata[index],
        "similarity": float(candidates["dense"][index]),
        "retrieval_score": float(candidates["score"][index]),
        "lexical_score": float(candidates["lexical"][index]),
    }


def candidate_results(db, query_embedding, query, candidate_k=20):
    candidates = hierarchical_candidates(db, query_embedding, query, candidate_k)
    return [
        _result(candidates["metadata"], candidates, index)
        for index in candidates["indices"]
    ]
