"""
Cross-Encoder Reranker Module.
Performs fine-grained token-level cross-attention scoring over top candidate chunks
to eliminate false positives from vector search and prioritize exact entity matches.
"""

from typing import List, Optional
import numpy as np
import re
from src.dag import DocumentChunk


class CrossEncoderReranker:
    """
    Reranks candidate DocumentChunks using cross-encoder scoring.
    Supports HuggingFace CrossEncoder models with a robust token-interaction fallback.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=True)
            self.model.eval()
        except Exception:
            self.model = None
            self.tokenizer = None

    def rerank(self, query: str, chunks: List[DocumentChunk], top_k: Optional[int] = None) -> List[DocumentChunk]:
        """
        Reranks candidates and updates their rerank_score and final_score.
        """
        if not chunks:
            return []

        if self.model is not None and self.tokenizer is not None:
            try:
                import torch
                pairs = [[query, f"{c.title}: {c.text}"] for c in chunks]
                inputs = self.tokenizer(pairs, padding=True, truncation=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    scores = self.model(**inputs).logits.squeeze(-1)
                    if scores.ndim == 0:
                        scores = scores.unsqueeze(0)
                    scores = torch.sigmoid(scores).cpu().numpy().tolist()
                for c, s in zip(chunks, scores):
                    c.rerank_score = round(float(s), 4)
                    # Blend RRF with Cross-Encoder
                    c.final_score = round(0.4 * c.rrf_score + 0.6 * c.rerank_score, 5)
            except Exception:
                self._fallback_rerank(query, chunks)
        else:
            self._fallback_rerank(query, chunks)

        # Sort by final score
        sorted_chunks = sorted(chunks, key=lambda c: (c.rerank_score, c.final_score), reverse=True)
        if top_k is not None:
            sorted_chunks = sorted_chunks[:top_k]

        # Re-assign chunk identifiers [Doc 1], [Doc 2], etc.
        for i, c in enumerate(sorted_chunks, start=1):
            c.chunk_id = f"Doc {i}"

        return sorted_chunks

    def _fallback_rerank(self, query: str, chunks: List[DocumentChunk]):
        """
        Deterministic, fast cross-attention approximation:
        Measures exact entity matching, sequence overlap, and title relevance.
        """
        q_tokens = re.findall(r"\w+", query.lower())
        q_set = set(q_tokens)

        # Extract capitalized entities from query
        proper_nouns = set(re.findall(r"\b[A-Z][a-zA-Z0-9_'-]+\b", query))

        for c in chunks:
            text_full = f"{c.title} {c.text}".lower()
            text_tokens = set(re.findall(r"\w+", text_full))

            # 1. Exact entity presence
            entity_matches = 0
            if proper_nouns:
                for pn in proper_nouns:
                    if pn.lower() in text_full:
                        entity_matches += 1
                entity_score = entity_matches / len(proper_nouns)
            else:
                entity_score = 0.5

            # 2. Token overlap & title boost
            token_overlap = len(q_set & text_tokens) / len(q_set) if q_set else 0.0
            title_boost = 1.0 if any(t in c.title.lower() for t in q_tokens if len(t) > 3) else 0.0

            # 3. Normalized cross score
            score = 0.45 * token_overlap + 0.35 * entity_score + 0.20 * title_boost
            c.rerank_score = round(float(score), 4)
            c.final_score = round(0.3 * c.rrf_score + 0.7 * c.rerank_score, 5)
