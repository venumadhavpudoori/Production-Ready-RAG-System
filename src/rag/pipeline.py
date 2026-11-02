"""End-to-end RAG Pipeline for NASA technical reports."""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.rag.config import config
from src.rag.data_loader import NASAReportLoader
from src.rag.chunking import MarkdownSectionChunker, DocumentChunk
from src.rag.vector_store import ChromaVectorStore
from src.rag.retrieval.pipeline import RetrievalPipeline, RetrievalProfile
from src.rag.generation.generator import CitationGenerator
from src.rag.generation.citation_validator import CitationValidator, CitationValidationResult

logger = logging.getLogger(__name__)


class RAGLatencyBreakdown(BaseModel):
    """Complete latency accounting across all pipeline stages."""

    dense_time_ms: float = 0.0
    sparse_time_ms: float = 0.0
    fusion_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_pipeline_ms: float = 0.0


class RAGResponse(BaseModel):
    """Structured response object containing answer, citations, and metrics."""

    query: str
    answer: str
    retrieval_mode: str
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    citations: CitationValidationResult
    latency: RAGLatencyBreakdown


class ProductionRAGPipeline:
    """Production-ready RAG pipeline supporting indexing, hybrid retrieval, reranking, and citation validation."""

    def __init__(
        self,
        raw_data_dir: Optional[Path] = None,
        persist_dir: Optional[Path] = None,
        use_gemini: bool = True,
    ):
        self.raw_data_dir = Path(raw_data_dir or config.raw_data_dir)
        self.persist_dir = Path(persist_dir or config.chroma_persist_dir)

        self.loader = NASAReportLoader(self.raw_data_dir)
        self.chunker = MarkdownSectionChunker(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        self.vector_store = ChromaVectorStore(
            persist_dir=self.persist_dir,
            use_gemini=use_gemini,
        )

        self.chunks: List[DocumentChunk] = []
        self.retrieval_pipeline: Optional[RetrievalPipeline] = None
        self.generator = CitationGenerator()
        self.validator = CitationValidator()

        # Initialize from existing corpus if available
        self._load_and_index_corpus()

    def _load_and_index_corpus(self, force_reindex: bool = False):
        """Load reports from raw directory and index into vector store and BM25."""
        reports = self.loader.load_all_reports()
        all_chunks: List[DocumentChunk] = []
        for r in reports:
            chunks = self.chunker.chunk_report(r)
            all_chunks.extend(chunks)

        self.chunks = all_chunks

        # Check if vector store needs indexing
        if force_reindex or self.vector_store.count() == 0:
            if force_reindex:
                self.vector_store.clear()
            if self.chunks:
                self.vector_store.add_chunks(self.chunks)

        self.retrieval_pipeline = RetrievalPipeline(
            vector_store=self.vector_store,
            chunks=self.chunks,
        )
        logger.info("Pipeline ready with %d chunks across %d reports", len(self.chunks), len(reports))

    def query(
        self,
        query: str,
        retrieval_mode: str = "hybrid_rerank",
        top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ) -> RAGResponse:
        """Execute full end-to-end query answering."""
        start_time = time.perf_counter()
        k = top_k or config.dense_top_k
        top_n = rerank_top_n or config.rerank_top_n

        if not self.retrieval_pipeline:
            self._load_and_index_corpus()

        # 1. Retrieval
        ranked_chunks, ret_profile = self.retrieval_pipeline.retrieve(
            query=query,
            mode=retrieval_mode,
            top_k=k,
            rerank_top_n=top_n,
        )

        # 2. Generation
        answer_text, gen_latency_ms = self.generator.generate(query, ranked_chunks)

        # 3. Citation Validation
        raw_chunks = [c for c, _score in ranked_chunks]
        citation_result = self.validator.validate(answer_text, raw_chunks)

        total_pipeline_ms = (time.perf_counter() - start_time) * 1000

        latency_breakdown = RAGLatencyBreakdown(
            dense_time_ms=ret_profile.dense_time_ms,
            sparse_time_ms=ret_profile.sparse_time_ms,
            fusion_time_ms=ret_profile.fusion_time_ms,
            rerank_time_ms=ret_profile.rerank_time_ms,
            total_retrieval_ms=ret_profile.total_retrieval_ms,
            generation_ms=round(gen_latency_ms, 2),
            total_pipeline_ms=round(total_pipeline_ms, 2),
        )

        formatted_chunks = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "section": c.section_heading,
                "citation_tag": c.citation_tag,
                "score": round(score, 4),
                "text": c.text,
                "preview": c.text[:150] + ("..." if len(c.text) > 150 else ""),
            }
            for c, score in ranked_chunks
        ]

        return RAGResponse(
            query=query,
            answer=answer_text,
            retrieval_mode=retrieval_mode,
            retrieved_chunks=formatted_chunks,
            citations=citation_result,
            latency=latency_breakdown,
        )
