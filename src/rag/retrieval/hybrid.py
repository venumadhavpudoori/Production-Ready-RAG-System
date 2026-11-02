"""Hybrid retrieval with Reciprocal Rank Fusion (RRF)."""

from typing import List, Tuple, Dict
from src.rag.chunking import DocumentChunk


class HybridRRFMerger:
    """Combines dense and sparse ranked lists using Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = 60, dense_weight: float = 0.5, sparse_weight: float = 0.5):
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
        top_k: int = 10,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Compute RRF scores across dense and sparse ranked result lists."""
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        # Process dense ranking
        for rank, (chunk, _score) in enumerate(dense_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            reciprocal_rank = self.dense_weight / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + reciprocal_rank

        # Process sparse ranking
        for rank, (chunk, _score) in enumerate(sparse_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            reciprocal_rank = self.sparse_weight / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + reciprocal_rank

        # Sort by combined RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        fused = [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_cids[:top_k]]
        return fused

