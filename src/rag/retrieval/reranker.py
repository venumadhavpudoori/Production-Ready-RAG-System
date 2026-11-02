"""Reranker module using FlashRank cross-encoder with fallback."""

import logging
from typing import List, Tuple
from src.rag.chunking import DocumentChunk

logger = logging.getLogger(__name__)


class FlashRankReranker:
    """Cross-encoder reranker leveraging FlashRank ONNX runtime."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            from flashrank import Ranker
            # Initialize Ranker (downloads or uses cached model)
            self.client = Ranker(model_name=self.model_name, cache_dir="./data/cache/flashrank")
            logger.info("FlashRank cross-encoder initialized with model %s", self.model_name)
        except Exception as e:
            logger.warning("FlashRank initialization failed (%s); fallback scorer will be used.", e)
            self.client = None

    def rerank(
        self, query: str, candidates: List[DocumentChunk], top_n: int = 4
    ) -> List[Tuple[DocumentChunk, float]]:
        """Rerank candidates based on deep cross-encoder query-document relevance."""
        if not candidates:
            return []

        if self.client is not None:
            try:
                from flashrank import RerankRequest

                passages = [
                    {"id": idx, "text": chunk.text, "meta": chunk.metadata}
                    for idx, chunk in enumerate(candidates)
                ]
                req = RerankRequest(query=query, passages=passages)
                results = self.client.rerank(req)

                ranked: List[Tuple[DocumentChunk, float]] = []
                for item in results[:top_n]:
                    idx = item["id"]
                    score = float(item["score"])
                    ranked.append((candidates[idx], score))
                return ranked
            except Exception as e:
                logger.warning("FlashRank rerank execution failed (%s), using heuristic fallback.", e)

        # Fallback scoring: lexical term frequency + proximity match
        q_terms = set(query.lower().split())
        scored: List[Tuple[DocumentChunk, float]] = []
        for chunk in candidates:
            c_text = chunk.text.lower()
            matches = sum(1 for t in q_terms if t in c_text)
            coverage = matches / max(1, len(q_terms))
            score = coverage
            scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

