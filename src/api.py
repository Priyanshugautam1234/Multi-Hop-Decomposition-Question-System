"""
Production FastAPI Service for Multi-Hop Question Decomposition & Retrieval.
Endpoints:
- POST /query: Executes adaptive iterative multi-hop reasoning and returns final answer + full reasoning trace.
- GET /health: Health status and index statistics.
- POST /index: Dynamically adds and indexes new documents into hybrid store.
- GET /benchmarks: Retrieves latest 3-way ablation benchmark table & metrics.
"""

from typing import List, Dict, Any, Optional
import os
import json
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline import MultiHopPipeline, DEFAULT_CORPUS


app = FastAPI(
    title="Multi-Hop Decomposition & Hybrid Retrieval API",
    description="Adaptive Iterative Retrieve-Reason-Retrieve Engine with Dynamic Variable Binding and Citation Grounding",
    version="2.0.0",
)

# Enable CORS for frontend visualizers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline instance
pipeline = MultiHopPipeline()


class QueryRequest(BaseModel):
    query: str = Field(..., example="Which city was the director of Inception born in?")
    max_hops: int = Field(default=4, ge=1, le=10, description="Maximum hop budget")
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder reranking stage")


class DocumentItem(BaseModel):
    id: str = Field(..., example="doc_custom_1")
    title: str = Field(..., example="Christopher Nolan")
    text: str = Field(..., example="Christopher Nolan was born in London, United Kingdom.")


class IndexRequest(BaseModel):
    documents: List[DocumentItem]


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Returns engine health, vector index size, and loaded models."""
    return {
        "status": "healthy",
        "service": "multi-hop-decomposition-engine",
        "version": "2.0.0",
        "indexed_documents_count": len(pipeline.retriever.documents),
        "embedding_dim": pipeline.retriever.embed_dim,
        "hybrid_retrieval": ["FAISS", "BM25Okapi", "ReciprocalRankFusion"],
        "reranker_active": pipeline.reranker is not None,
    }


@app.post("/query", status_code=status.HTTP_200_OK)
def process_query(req: QueryRequest):
    """
    Executes adaptive iterative multi-hop reasoning loop.
    Returns grounded answer with inline citations [Doc X] and full reasoning trace.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        # Update hop ceiling if specified
        pipeline.loop.guardrails.max_hops = req.max_hops
        result = pipeline.query(req.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")


@app.post("/index", status_code=status.HTTP_201_CREATED)
def index_documents(req: IndexRequest):
    """Dynamically registers and indexes new documents into hybrid FAISS + BM25 stores."""
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided to index.")

    docs_dict = [d.model_dump() for d in req.documents]
    pipeline.add_documents(docs_dict)
    return {
        "message": f"Successfully indexed {len(docs_dict)} documents.",
        "total_documents": len(pipeline.retriever.documents),
    }


@app.get("/benchmarks", status_code=status.HTTP_200_OK)
def get_benchmarks():
    """Returns precomputed 3-way ablation benchmark results and markdown table."""
    results_path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results", "ablation_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": "Benchmark results not yet generated. Run python benchmarks/run_multihop_eval.py"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
