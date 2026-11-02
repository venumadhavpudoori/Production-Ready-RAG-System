"""ChromaDB vector store integration with Gemini & local embedding options."""

import logging
import math
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from src.rag.chunking import DocumentChunk
from src.rag.config import config

logger = logging.getLogger(__name__)


class HashEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic local embedding function for offline/fallback operation."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def name(self) -> str:
        return "hash_embedding"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: List[List[float]] = []
        for text in input:
            vec = [0.0] * self.dim
            words = text.lower().split()
            if not words:
                embeddings.append(vec)
                continue
            for word in words:
                # Murmur-like polynomial hash into dimension buckets
                h = abs(hash(word))
                idx = h % self.dim
                sign = 1.0 if ((h >> 8) & 1) else -1.0
                vec[idx] += sign

            # L2 normalize
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            embeddings.append(vec)
        return embeddings


class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    """Wrapper around Google GenAI Embeddings for ChromaDB."""

    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-001"):
        self.api_key = api_key
        self.model_name = model_name
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._embedder = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
        )
        self._fallback = HashEmbeddingFunction(dim=3072)

    def name(self) -> str:
        return "gemini_embedding"

    def __call__(self, input: Documents) -> Embeddings:
        try:
            return self._embedder.embed_documents(input)
        except Exception as e:
            logger.warning("Gemini embedding call failed (%s), using local fallback embeddings", e)
            return self._fallback(input)


class ChromaVectorStore:
    """Persistent ChromaDB vector store for dense retrieval."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = "nasa_mars_reports",
        use_gemini: bool = True,
    ):
        self.persist_dir = Path(persist_dir or config.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self.embedding_fn: EmbeddingFunction
        if use_gemini and config.gemini_api_key and config.embedding_provider == "gemini":
            try:
                self.embedding_fn = GeminiEmbeddingFunction(
                    api_key=config.gemini_api_key,
                    model_name=config.embedding_model_name,
                )
                logger.info("Initialized Gemini embeddings (%s)", config.embedding_model_name)
            except Exception as e:
                logger.warning("Falling back to local embeddings: %s", e)
                self.embedding_fn = HashEmbeddingFunction()
        else:
            self.embedding_fn = HashEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 50) -> int:
        """Add document chunks to the collection with batching."""
        if not chunks:
            return 0

        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [
                {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "mission": c.metadata.get("mission", "Mars"),
                    "section_heading": c.section_heading,
                    "citation_tag": c.citation_tag,
                    "source_url": c.metadata.get("source_url", ""),
                    "token_estimate": c.token_estimate,
                }
                for c in batch
            ]

            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            total_added += len(batch)

        logger.info("Indexed %d chunks into Chroma collection '%s'", total_added, self.collection_name)
        return total_added

    def similarity_search(
        self, query: str, top_k: int = 10, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Perform cosine similarity search and return chunks with similarity scores."""
        query_params: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(top_k, max(1, self.collection.count())),
        }
        if filter_dict:
            query_params["where"] = filter_dict

        if self.collection.count() == 0:
            return []

        try:
            results = self.collection.query(**query_params)
        except Exception as e:
            logger.warning("Vector collection query error (%s), returning empty results", e)
            return []

        chunks_with_scores: List[Tuple[DocumentChunk, float]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

        for cid, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # Cosine distance in [0, 2]. Cosine similarity = 1 - distance.
            sim_score = max(0.0, 1.0 - float(dist))
            chunk = DocumentChunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                title=meta.get("title", ""),
                section_heading=meta.get("section_heading", ""),
                citation_tag=meta.get("citation_tag", f"[{cid}]"),
                text=doc_text,
                token_estimate=meta.get("token_estimate", len(doc_text) // 4),
                metadata=meta,
            )
            chunks_with_scores.append((chunk, sim_score))

        return chunks_with_scores

    def count(self) -> int:
        """Return total document chunks in collection."""
        return self.collection.count()

    def clear(self):
        """Reset the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )
