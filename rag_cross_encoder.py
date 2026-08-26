"""Multilingual cross-encoder reranking for hierarchical RAG candidates."""

import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "modeling" / "bge-reranker-v2-m3"


class CrossEncoderReranker:
    """Score Korean-query/English-passage pairs with a local BGE reranker."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_DIR):
        model_ref = str(model_path if model_path.exists() else MODEL_ID)
        local_only = model_path.exists() or os.getenv("HF_HUB_OFFLINE") == "1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ref, local_files_only=local_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_ref, local_files_only=local_only
        ).to(self.device)
        self.model.eval()

    def scores(self, query: str, candidates: List[Dict[str, Any]], batch_size: int = 8):
        values: List[float] = []
        passages = [self._passage(item) for item in candidates]
        for start in range(0, len(passages), batch_size):
            pairs = [[query, passage] for passage in passages[start:start + batch_size]]
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**inputs, return_dict=True).logits.view(-1)
            values.extend(float(value) for value in logits.detach().cpu())
        return values

    @staticmethod
    def _passage(item: Dict[str, Any]) -> str:
        return (
            f"Source: {item['source']}\nSection: {item.get('section', 'Body')}\n"
            f"Passage: {item['text']}"
        )

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        low, high = min(values), max(values)
        if high <= low:
            return [0.0 for _ in values]
        return [(value - low) / (high - low) for value in values]

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3,
        blend_alpha: float = 0.25,
    ):
        cross_scores = self.scores(query, candidates)
        retrieval_scores = self._normalize(
            [float(item["retrieval_score"]) for item in candidates]
        )
        normalized_cross = self._normalize(cross_scores)
        scored = []
        for item, raw, retrieval, cross in zip(
            candidates, cross_scores, retrieval_scores, normalized_cross
        ):
            combined = (1.0 - blend_alpha) * retrieval + blend_alpha * cross
            scored.append({**item, "reranker_score": raw, "combined_score": combined})
        scored.sort(key=lambda item: item["combined_score"], reverse=True)
        selected, seen_pages = [], set()
        for item in scored:
            page_key = (item["source"], item["page"])
            if page_key in seen_pages:
                continue
            selected.append(item)
            seen_pages.add(page_key)
            if len(selected) == top_k:
                break
        return selected
