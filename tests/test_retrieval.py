"""Unit tests for dense, sparse, hybrid, and reranking retrieval."""

from src.rag.chunking import DocumentChunk
from src.rag.retrieval.sparse import BM25SparseRetriever
from src.rag.retrieval.hybrid import HybridRRFMerger
from src.rag.retrieval.reranker import FlashRankReranker
from src.rag.vector_store import ChromaVectorStore


def create_sample_chunks():
    return [
        DocumentChunk(
            chunk_id="DOC1_c001",
            doc_id="DOC1",
            title="Perseverance Heatshield",
            section_heading="MEDLI2 Telemetry",
            citation_tag="[DOC1 §MEDLI2 Telemetry]",
            text="The MEDLI2 instrument recorded a peak heat flux of 132 W/cm2 on the PICA heatshield.",
            token_estimate=25,
            metadata={"doc_id": "DOC1", "title": "Perseverance Heatshield"},
        ),
        DocumentChunk(
            chunk_id="DOC2_c001",
            doc_id="DOC2",
            title="MOXIE Oxygen Production",
            section_heading="SOEC Reactor",
            citation_tag="[DOC2 §SOEC Reactor]",
            text="The SOEC solid oxide electrolysis cells produce oxygen from Martian atmospheric CO2 at 800 C.",
            token_estimate=26,
            metadata={"doc_id": "DOC2", "title": "MOXIE Oxygen Production"},
        ),
        DocumentChunk(
            chunk_id="DOC3_c001",
            doc_id="DOC3",
            title="InSight Marsquakes",
            section_heading="SEIS S1222a Event",
            citation_tag="[DOC3 §SEIS S1222a Event]",
            text="The monster marsquake event S1222a reached magnitude Mw 4.7 and shook Elysium Planitia.",
            token_estimate=24,
            metadata={"doc_id": "DOC3", "title": "InSight Marsquakes"},
        ),
    ]


def test_bm25_sparse_retrieval():
    chunks = create_sample_chunks()
    retriever = BM25SparseRetriever(chunks)

    results = retriever.search("MEDLI2 heatshield", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "DOC1_c001"
    assert score > 0.0


def test_hybrid_rrf_merger():
    chunks = create_sample_chunks()
    dense_results = [(chunks[0], 0.95), (chunks[1], 0.70)]
    sparse_results = [(chunks[1], 5.2), (chunks[0], 3.1)]

    merger = HybridRRFMerger(rrf_k=60, dense_weight=0.5, sparse_weight=0.5)
    fused = merger.fuse(dense_results, sparse_results, top_k=2)

    assert len(fused) == 2
    # Both items should be included in fused results
    cids = [c.chunk_id for c, _score in fused]
    assert "DOC1_c001" in cids
    assert "DOC2_c001" in cids


def test_flashrank_reranker():
    chunks = create_sample_chunks()
    reranker = FlashRankReranker()

    results = reranker.rerank(query="Marsquake magnitude S1222a", candidates=chunks, top_n=2)
    assert len(results) <= 2
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "DOC3_c001"


def test_chroma_vector_store(tmp_path):
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma_test", use_gemini=False)
    chunks = create_sample_chunks()

    added = store.add_chunks(chunks)
    assert added == 3
    assert store.count() == 3

    search_res = store.similarity_search("electrolysis oxygen", top_k=2)
    assert len(search_res) > 0

