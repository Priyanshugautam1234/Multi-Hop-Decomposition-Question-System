"""
Unit tests for Hybrid Retrieval (FAISS + BM25 + RRF) and Cross-Encoder Reranker.
"""

import pytest
from src.retriever import HybridRetriever, ReciprocalRankFusion
from src.reranker import CrossEncoderReranker


SAMPLE_DOCS = [
    {
        "id": "doc_1",
        "title": "Inception Movie",
        "text": "Inception is a science fiction film written and directed by Christopher Nolan.",
    },
    {
        "id": "doc_2",
        "title": "Interstellar Movie",
        "text": "Interstellar is an epic science fiction film directed by Christopher Nolan, starring Matthew McConaughey.",
    },
    {
        "id": "doc_3",
        "title": "The Dark Knight",
        "text": "The Dark Knight is a superhero film directed by Christopher Nolan based on DC Comics.",
    },
    {
        "id": "doc_4",
        "title": "London Geography",
        "text": "London is the capital city of England and the United Kingdom.",
    },
]


def test_rrf_scoring():
    dense_ranks = ["doc_1", "doc_2", "doc_3"]
    bm25_ranks = ["doc_2", "doc_1", "doc_4"]
    
    rrf_scores = ReciprocalRankFusion.fuse(dense_ranks, bm25_ranks, k=60)
    
    assert "doc_1" in rrf_scores
    assert "doc_2" in rrf_scores
    # Both ranked high in dense and BM25, so doc_1 and doc_2 score highest
    assert rrf_scores["doc_1"] > rrf_scores["doc_4"]
    assert rrf_scores["doc_2"] > rrf_scores["doc_4"]


def test_hybrid_retriever_search():
    retriever = HybridRetriever()
    retriever.index_documents(SAMPLE_DOCS)
    
    # Query for exact movie title
    results = retriever.search("Inception director", top_k=2)
    assert len(results) == 2
    # doc_1 contains both Inception and director
    assert results[0].doc_id == "doc_1"
    assert results[0].chunk_id == "Doc 1"
    assert results[0].final_score > 0.0


def test_cross_encoder_reranker():
    reranker = CrossEncoderReranker()
    retriever = HybridRetriever()
    retriever.index_documents(SAMPLE_DOCS)
    
    candidates = retriever.search("Inception director", top_k=3)
    reranked = reranker.rerank("Inception director", candidates, top_k=2)
    
    assert len(reranked) == 2
    assert reranked[0].doc_id == "doc_1"
    assert reranked[0].rerank_score > 0.0
