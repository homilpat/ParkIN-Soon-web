"""Conservative evidence admission for user-facing RAG answers."""

import re
from typing import Any, Dict, Iterable, List

NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|년|개월|주|일|명|점)?\b")
KOREAN_TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{3,}")
STOPWORDS = {
    "얼마", "어느", "정도", "무엇", "뭐가", "관련", "대해", "알려", "인가요",
    "파킨슨", "파킨슨병", "결과", "검사", "사용자", "그리고", "하지만",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower() for token in KOREAN_TOKEN_RE.findall(text)
        if token.lower() not in STOPWORDS
    }


def _numbers(text: str) -> set[str]:
    return {match.group(0).replace(" ", "") for match in NUMBER_RE.finditer(text)}


def evidence_decision(query: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Return an auditable decision using independent retrieval and content signals."""
    text = " ".join(str(chunk.get(key, "")) for key in ("text", "parent_text", "section"))
    query_tokens = _tokens(query)
    overlap = query_tokens & _tokens(text)
    entity_match = not query_tokens or bool(overlap)

    query_numbers = _numbers(query)
    number_match = not query_numbers or bool(query_numbers & _numbers(text))
    similarity = float(chunk.get("similarity", 0.0))
    retrieval = float(chunk.get("retrieval_score", 0.0))
    combined = float(chunk.get("combined_score", 0.0))
    reranker_present = "reranker_score" in chunk or "combined_score" in chunk

    retrieval_ok = similarity >= 0.48 or retrieval >= 0.65
    reranker_ok = not reranker_present or combined >= 0.45
    accepted = retrieval_ok and reranker_ok and entity_match and number_match
    return {
        "accepted": accepted,
        "retrieval_ok": retrieval_ok,
        "reranker_ok": reranker_ok,
        "entity_match": entity_match,
        "number_match": number_match,
        "matched_terms": sorted(overlap),
    }


def admitted_evidence(
    query: str, chunks: Iterable[Dict[str, Any]], limit: int = 3,
) -> List[Dict[str, Any]]:
    """Keep only evidence passing the gate and attach non-user-facing audit data."""
    admitted = []
    for chunk in chunks:
        decision = evidence_decision(query, chunk)
        if not decision["accepted"]:
            continue
        admitted.append({**chunk, "evidence_gate": decision})
        if len(admitted) == limit:
            break
    return admitted
