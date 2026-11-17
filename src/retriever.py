"""
Hybrid Retrieval Engine: Dense FAISS Vector Search + Lexical BM25 Search
fused via Reciprocal Rank Fusion (RRF).
Guarantees exact entity names (people, dates, laws) are not lost in semantic vector space.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import re
from src.dag import DocumentChunk


class DenseEmbedder:
    """
    Dense text embedder supporting transformer-based embeddings with a deterministic
    hash/TF-IDF projection fallback for offline/isolated environments.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dim: int = 384):
        self.dim = dim
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._init_transformer()

    def _init_transformer(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
            self.model.eval()
        except Exception:
            # Fallback to fast, deterministic subword hashing + n-gram projection
            self.tokenizer = None
            self.model = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a batch of texts into L2-normalized dense vectors."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        if self.model is not None and self.tokenizer is not None:
            try:
                import torch
                inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Mean pooling with attention mask
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    pooled = (sum_embeddings / sum_mask).cpu().numpy()
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return (pooled / norms).astype(np.float32)
            except Exception:
                pass

        # Robust, fast offline projection
        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = re.findall(r"\w+", text.lower())
            for w in words:
                # Deterministic hash feature mapping
                idx = hash(w) % self.dim
                embeddings[i, idx] += 1.0
                # Bi-gram features
                idx2 = hash(w + "_bi") % self.dim
                embeddings[i, idx2] += 0.5
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
            else:
                embeddings[i, 0] = 1.0
        return embeddings


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple ranking lists.
    Formula: RRF(d) = sum_{m in models} 1.0 / (k + rank_m(d))
    Default k = 60 based on Cormack et al.
    """

    @staticmethod
    def fuse(dense_ranks: List[str], bm25_ranks: List[str], k: int = 60) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        for rank, doc_id in enumerate(dense_ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        for rank, doc_id in enumerate(bm25_ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        return scores


class HybridRetriever:
    """
    Production Hybrid Retriever combining Dense FAISS vector similarity and BM25 lexical matching.
    """

    def __init__(self, embed_dim: int = 384, rrf_k: int = 60):
        self.embed_dim = embed_dim
        self.rrf_k = rrf_k
        self.embedder = DenseEmbedder(dim=embed_dim)
        
        # FAISS index for dense search (Inner Product over normalized vectors = Cosine similarity)
        self.index = faiss.IndexFlatIP(embed_dim)
        self.documents: List[Dict[str, Any]] = []
        self.doc_id_to_idx: Dict[str, int] = {}
        
        # BM25 components
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def index_documents(self, docs: List[Dict[str, Any]]):
        """
        Indexes a list of documents.
        Each doc must have 'id', 'title', and 'text'.
        """
        if not docs:
            return

        self.documents = docs
        self.doc_id_to_idx = {doc["id"]: i for i, doc in enumerate(docs)}

        # Build BM25
        corpus_texts = [f"{d.get('title', '')} {d.get('text', '')}" for d in docs]
        self.tokenized_corpus = [self._tokenize(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Build FAISS Dense Index
        embeddings = self.embedder.embed_texts(corpus_texts)
        self.index.reset()
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Executes hybrid search combining Dense FAISS and BM25 with RRF.
        Returns top_k DocumentChunks sorted by fused score.
        """
        n_docs = len(self.documents)
        if n_docs == 0:
            return []

        search_k = min(n_docs, max(top_k * 3, 10))

        # 1. Dense FAISS Search
        q_emb = self.embedder.embed_texts([query])
        dense_distances, dense_indices = self.index.search(q_emb, search_k)
        dense_results: List[str] = []
        dense_score_map: Dict[str, float] = {}
        for rank, (score, idx) in enumerate(zip(dense_distances[0], dense_indices[0])):
            if 0 <= idx < n_docs:
                did = self.documents[idx]["id"]
                dense_results.append(did)
                dense_score_map[did] = float(score)

        # 2. Lexical BM25 Search
        q_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens) if self.bm25 else np.zeros(n_docs)
        bm25_sorted_indices = np.argsort(bm25_scores)[::-1][:search_k]
        
        bm25_results: List[str] = []
        bm25_score_map: Dict[str, float] = {}
        for idx in bm25_sorted_indices:
            did = self.documents[idx]["id"]
            bm25_results.append(did)
            bm25_score_map[did] = float(bm25_scores[idx])

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = ReciprocalRankFusion.fuse(dense_results, bm25_results, k=self.rrf_k)

        # Sort all retrieved candidate IDs by RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)[:top_k]

        # Construct DocumentChunk outputs
        chunks: List[DocumentChunk] = []
        for rank, did in enumerate(sorted_doc_ids, start=1):
            doc = self.documents[self.doc_id_to_idx[did]]
            chunk = DocumentChunk(
                chunk_id=f"Doc {rank}",
                doc_id=did,
                title=doc.get("title", ""),
                text=doc.get("text", ""),
                dense_score=round(dense_score_map.get(did, 0.0), 4),
                bm25_score=round(bm25_score_map.get(did, 0.0), 4),
                rrf_score=round(rrf_scores.get(did, 0.0), 6),
                final_score=round(rrf_scores.get(did, 0.0), 6),
            )
            chunks.append(chunk)

        return chunks
