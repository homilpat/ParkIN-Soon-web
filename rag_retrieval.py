"""Hybrid dense/lexical retrieval with source-aware reranking."""

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np

TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9-]{1,}|[가-힣]{2,}")
NUMERIC_QUERY_RE = re.compile(
    r"\d|수치|비율|몇\s*(?:배|년|개월|명)|퍼센트|통계|표|표본|"
    r"AUC|효과크기|임계값|민감도|특이도|정확도",
    re.IGNORECASE,
)
EVIDENCE_SECTIONS = {"Abstract", "Results", "Discussion", "Conclusion", "Conclusions", "Limitation", "Limitations"}


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def bm25_scores(query: str, documents: List[str], k1: float = 1.4, b: float = 0.75) -> np.ndarray:
    tokenized = [tokenize(text) for text in documents]
    query_terms = list(dict.fromkeys(tokenize(query)))
    lengths = np.array([len(tokens) for tokens in tokenized], dtype=np.float32)
    average_length = max(float(lengths.mean()), 1.0)
    frequencies = [Counter(tokens) for tokens in tokenized]
    document_frequency = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    scores = np.zeros(len(documents), dtype=np.float32)
    for term in query_terms:
        df = document_frequency.get(term, 0)
        if not df:
            continue
        idf = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
        for index, counts in enumerate(frequencies):
            frequency = counts.get(term, 0)
            if frequency:
                denominator = frequency + k1 * (1.0 - b + b * lengths[index] / average_length)
                scores[index] += idf * frequency * (k1 + 1.0) / denominator
    return scores


def normalize(values: np.ndarray) -> np.ndarray:
    minimum, maximum = float(values.min()), float(values.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def hybrid_search(
    db: Dict[str, Any], query_embedding: np.ndarray, expanded_query: str,
    top_k: int = 3, candidate_k: int = 30,
) -> List[Dict[str, Any]]:
    embeddings = np.asarray(db["embeddings"], dtype=np.float32)
    metadata = db["metadata"]
    dense = embeddings @ np.asarray(query_embedding, dtype=np.float32)
    lexical = bm25_scores(expanded_query, [item["text"] for item in metadata])
    section_bonus = np.array([
        0.035 if item.get("section") in EVIDENCE_SECTIONS else 0.0 for item in metadata
    ], dtype=np.float32)
    wants_numeric = bool(NUMERIC_QUERY_RE.search(expanded_query))
    raw_query_numbers = re.findall(r"\d+(?:\.\d+)?", expanded_query)
    query_numbers = {
        number for number in raw_query_numbers
        if "." in number or int(number) >= 10
    }
    structured_bonus = np.array([
        0.04 if wants_numeric and item.get("evidence_type") in {"numeric", "table"} else 0.0
        for item in metadata
    ], dtype=np.float32)
    exact_number_bonus = np.array([
        0.10 if query_numbers & set(re.findall(r"\d+(?:\.\d+)?", item["text"])) else 0.0
        for item in metadata
    ], dtype=np.float32)
    combined = (
        0.88 * normalize(dense) + 0.12 * normalize(lexical)
        + 0.4 * section_bonus + structured_bonus + exact_number_bonus
    )
    candidates = np.argsort(combined)[::-1][:max(candidate_k, top_k)]
    max_per_source = 3 if wants_numeric else 2

    # Preserve the strongest semantic match; hybrid signals only rerank the
    # remaining evidence candidates so lexical noise cannot displace rank 1.
    first_index = int(np.argmax(combined)) if wants_numeric else int(np.argmax(dense))
    selected: List[int] = [first_index] if top_k else []
    source_counts = defaultdict(int)
    if selected:
        source_counts[metadata[selected[0]]["source"]] = 1
    while len(selected) < top_k and len(selected) < len(candidates):
        best_index, best_score = None, -float("inf")
        for index in candidates:
            index = int(index)
            if index in selected or source_counts[metadata[index]["source"]] >= max_per_source:
                continue
            redundancy = max((float(embeddings[index] @ embeddings[other]) for other in selected), default=0.0)
            same_source_penalty = (
                0.005 if wants_numeric and source_counts[metadata[index]["source"]] else
                0.02 if source_counts[metadata[index]["source"]] else 0.0
            )
            score = float(combined[index]) - 0.06 * max(redundancy, 0.0) - same_source_penalty
            if score > best_score:
                best_index, best_score = index, score
        if best_index is None:
            break
        selected.append(best_index)
        source_counts[metadata[best_index]["source"]] += 1

    results = []
    for index in selected:
        item = metadata[index]
        results.append({
            "text": item["text"], "source": item["source"], "page": item["page"],
            "page_end": item.get("page_end", item["page"]),
            "section": item.get("section", "Body"),
            "evidence_type": item.get("evidence_type", "body"),
            "similarity": float(dense[index]), "retrieval_score": float(combined[index]),
            "lexical_score": float(lexical[index]),
        })
    return results
