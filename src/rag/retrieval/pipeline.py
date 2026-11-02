"""Retrieval coordinator supporting Dense, Sparse, Hybrid RRF, and Reranking."""

import time
import logging
from typing import List, Tuple, Dict, Any, Optional
from src.rag.chunking import DocumentChunk
from src.rag.vector_store import ChromaVectorStore
from src.rag.retrieval.sparse import BM25SparseRetriever
from src.rag.retrieval.hybrid import HybridRRFMerger
from src.rag.retrieval.reranker import FlashRankReranker
from src.rag.config import config

logger = logging.getLogger(__name__)


class RetrievalProfile:
    """Detailed timing profiling of retrieval stages."""

    def __init__(self):
        self.dense_time_ms: float = 0.0
        self.sparse_time_ms: float = 0.0
        self.fusion_time_ms: float = 0.0
        self.rerank_time_ms: float = 0.0
        self.total_retrieval_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "dense_time_ms": round(self.dense_time_ms, 2),
            "sparse_time_ms": round(self.sparse_time_ms, 2),
            "fusion_time_ms": round(self.fusion_time_ms, 2),
            "rerank_time_ms": round(self.rerank_time_ms, 2),
            "total_retrieval_ms": round(self.total_retrieval_ms, 2),
        }


class RetrievalPipeline:
    """Two-stage retrieval pipeline: Hybrid (Dense + Sparse) followed by Cross-Encoder Rerank."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        chunks: List[DocumentChunk],
        reranker: Optional[FlashRankReranker] = None,
    ):
        self.vector_store = vector_store
        self.chunks = chunks
        self.sparse_retriever = BM25SparseRetriever(chunks) if chunks else None
        self.fusion = HybridRRFMerger(
            rrf_k=config.rrf_k,
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight,
        )
        self.reranker = reranker or FlashRankReranker(model_name=config.flashrank_model_name)

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid_rerank",
        top_k: int = 10,
        rerank_top_n: int = 4,
    ) -> Tuple[List[Tuple[DocumentChunk, float]], RetrievalProfile]:
        """Execute retrieval under the specified mode.
        
        Modes:
          - 'dense_only': Dense vector search via Chroma
          - 'sparse_only': BM25 lexical search
          - 'hybrid_rrf': Parallel Dense + BM25 merged with RRF
          - 'hybrid_rerank': Dense + BM25 -> RRF -> FlashRank Rerank
        """
        profile = RetrievalProfile()
        start_total = time.perf_counter()

        dense_results: List[Tuple[DocumentChunk, float]] = []
        sparse_results: List[Tuple[DocumentChunk, float]] = []

        # 1. Dense retrieval
        if mode in ("dense_only", "hybrid_rrf", "hybrid_rerank"):
            t0 = time.perf_counter()
            dense_results = self.vector_store.similarity_search(query, top_k=top_k)
            profile.dense_time_ms = (time.perf_counter() - t0) * 1000

        # 2. Sparse retrieval
        if mode in ("sparse_only", "hybrid_rrf", "hybrid_rerank"):
            t0 = time.perf_counter()
            if self.sparse_retriever:
                sparse_results = self.sparse_retriever.search(query, top_k=top_k)
            profile.sparse_time_ms = (time.perf_counter() - t0) * 1000

        # Return single-retriever results directly if requested
        if mode == "dense_only":
            profile.total_retrieval_ms = (time.perf_counter() - start_total) * 1000
            return dense_results[:rerank_top_n], profile

        if mode == "sparse_only":
            profile.total_retrieval_ms = (time.perf_counter() - start_total) * 1000
            return sparse_results[:rerank_top_n], profile

        # 3. Hybrid RRF Fusion
        t0 = time.perf_counter()
        fused_candidates = self.fusion.fuse(dense_results, sparse_results, top_k=top_k)
        profile.fusion_time_ms = (time.perf_counter() - t0) * 1000

        if mode == "hybrid_rrf":
            profile.total_retrieval_ms = (time.perf_counter() - start_total) * 1000
            return fused_candidates[:rerank_top_n], profile

        # 4. FlashRank Reranking
        candidate_chunks = [c for c, _score in fused_candidates]
        t0 = time.perf_counter()
        final_ranked = self.reranker.rerank(query, candidate_chunks, top_n=rerank_top_n)
        profile.rerank_time_ms = (time.perf_counter() - t0) * 1000

        profile.total_retrieval_ms = (time.perf_counter() - start_total) * 1000
        return final_ranked, profile

