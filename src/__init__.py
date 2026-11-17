"""
Multi-Hop Decompose Question System
Stateful DAG, Adaptive Iterative Retrieve-Reason-Retrieve Loop, Hybrid Search (FAISS + BM25 + RRF),
Cross-Encoder Reranking, and Grounded Source Citations.
"""

from src.dag import StatefulDAG, SubQuestion, ReasoningTraceHop, DocumentChunk
from src.retriever import HybridRetriever, ReciprocalRankFusion
from src.reranker import CrossEncoderReranker
from src.synthesizer import EvidenceSynthesizer
from src.decomposer import AdaptiveDecomposer
from src.loop import AdaptiveIterativeLoop
from src.pipeline import MultiHopPipeline

__all__ = [
    "StatefulDAG",
    "SubQuestion",
    "ReasoningTraceHop",
    "DocumentChunk",
    "HybridRetriever",
    "ReciprocalRankFusion",
    "CrossEncoderReranker",
    "EvidenceSynthesizer",
    "AdaptiveDecomposer",
    "AdaptiveIterativeLoop",
    "MultiHopPipeline",
]
