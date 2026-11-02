"""Unit and integration tests for citation validation, latency profiler, and pipeline."""

from src.rag.chunking import DocumentChunk
from src.rag.generation.citation_validator import CitationValidator
from src.evaluation.latency_profiler import LatencyProfiler
from src.rag.pipeline import RAGLatencyBreakdown, ProductionRAGPipeline


def test_citation_validator_valid():
    validator = CitationValidator()
    chunks = [
        DocumentChunk(
            chunk_id="NASA-M2020-EDL_c001",
            doc_id="NASA-M2020-EDL",
            title="EDL Telemetry",
            section_heading="MEDLI2",
            citation_tag="[NASA-M2020-EDL §MEDLI2]",
            text="The peak heat flux was 132 W/cm2.",
            token_estimate=20,
        )
    ]

    answer = "The peak heat flux reached 132 W/cm2 [NASA-M2020-EDL §MEDLI2]."
    res = validator.validate(answer, chunks)

    assert res.has_citations is True
    assert res.valid_citations == 1
    assert res.invalid_citations == 0
    assert res.citation_precision == 1.0
    assert "NASA-M2020-EDL" in res.cited_doc_ids


def test_citation_validator_invalid_and_negative_rejection():
    validator = CitationValidator()
    chunks = [
        DocumentChunk(
            chunk_id="NASA-M2020-MOXIE_c001",
            doc_id="NASA-M2020-MOXIE",
            title="MOXIE Specs",
            section_heading="SOEC",
            citation_tag="[NASA-M2020-MOXIE §SOEC]",
            text="Operating temp is 800 C.",
            token_estimate=15,
        )
    ]

    # Fabricated citation
    answer = "Apollo 11 landed on the moon [NASA-APOLLO-11 §Landing]."
    res = validator.validate(answer, chunks)
    assert res.invalid_citations == 1
    assert res.citation_precision == 0.0

    # Negative rejection
    ans_abstained = "Based on the provided NASA technical reports, this information is not available."
    res_abstain = validator.validate(ans_abstained, chunks)
    assert res_abstain.citation_precision == 1.0


def test_latency_profiler():
    profiler = LatencyProfiler()
    for i in range(5):
        profiler.record(
            RAGLatencyBreakdown(
                dense_time_ms=10.0 + i,
                sparse_time_ms=2.0,
                fusion_time_ms=1.0,
                rerank_time_ms=15.0,
                total_retrieval_ms=28.0 + i,
                generation_ms=100.0,
                total_pipeline_ms=128.0 + i,
            )
        )

    summary = profiler.compute_summary()
    assert "dense_time_ms" in summary
    assert summary["dense_time_ms"]["mean"] == 12.0
    assert summary["sparse_time_ms"]["p50"] == 2.0


def test_end_to_end_pipeline_query():
    pipeline = ProductionRAGPipeline(use_gemini=False)
    assert len(pipeline.chunks) > 0

    response = pipeline.query("What was the peak heat flux recorded by MEDLI2?", retrieval_mode="hybrid_rerank")
    assert response.query is not None
    assert response.answer is not None
    assert len(response.retrieved_chunks) > 0
    assert response.latency.total_pipeline_ms > 0.0

