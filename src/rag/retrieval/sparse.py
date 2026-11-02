"""Sparse Lexical Retriever using BM25Okapi."""

import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from src.rag.chunking import DocumentChunk


class BM25SparseRetriever:
    """Inverted index BM25 retriever for exact keyword and terminology matching."""

    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.corpus = [self._tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Normalize and tokenize text into alphanumeric tokens."""
        clean = re.sub(r"[^\w\s-]", " ", text.lower())
        tokens = [t for t in clean.split() if len(t) > 1]
        return tokens

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Search BM25 corpus and return sorted (chunk, score) pairs."""
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        doc_scores = self.bm25.get_scores(tokenized_query)
        scored_pairs = list(zip(self.chunks, doc_scores))
        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        # Filter out 0.0 scores
        filtered = [pair for pair in scored_pairs if pair[1] > 0.0][:top_k]
        return filtered

