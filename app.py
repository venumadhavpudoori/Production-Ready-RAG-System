"""Streamlit Web Application for NASA Mars Exploration RAG System."""

import os
import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.config import config
from src.rag.pipeline import ProductionRAGPipeline
from src.evaluation.ragas_evaluator import RagasEvaluator

# Page Configuration
st.set_page_config(
    page_title="NASA Mars RAG Explorer",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E24A4A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #A0AEC0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1A202C;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #E24A4A;
        margin-bottom: 10px;
    }
    .citation-badge {
        display: inline-block;
        background-color: #2D3748;
        color: #63B3ED;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        margin-right: 5px;
        margin-bottom: 5px;
        font-family: monospace;
    }
    .latency-pill {
        font-size: 0.85rem;
        color: #48BB78;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Cache the pipeline instance across sessions."""
    return ProductionRAGPipeline()


# Load Pipeline
with st.spinner("Initializing NASA Mars RAG Knowledge Base & Index..."):
    pipeline = get_pipeline()

# Sidebar
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg",
        width=120,
    )
    st.title("Settings & Telemetry")

    st.subheader("Retrieval Architecture")
    mode_options = {
        "hybrid_rerank": "Hybrid + FlashRank Rerank (Production)",
        "hybrid_rrf": "Hybrid Fusion (Dense + BM25 RRF)",
        "dense_only": "Dense Vector Search (Chroma)",
        "sparse_only": "Sparse Lexical Search (BM25)",
    }
    selected_mode = st.selectbox(
        "Retrieval Mode",
        options=list(mode_options.keys()),
        format_func=lambda x: mode_options[x],
        index=0,
    )

    st.divider()
    st.subheader("Hyperparameters")
    top_k = st.slider("Dense & Sparse Top-K", min_value=5, max_value=25, value=config.dense_top_k)
    rerank_top_n = st.slider(
        "Reranked Top-N Chunks", min_value=2, max_value=10, value=config.rerank_top_n
    )

    st.divider()
    st.subheader("Corpus Statistics")
    total_chunks = len(pipeline.chunks)
    st.metric("Total Indexed Chunks", total_chunks)
    st.metric("Vector Database", "ChromaDB (Cosine)")
    st.metric("Embedding Model", config.embedding_model_name.replace("models/", ""))
    st.metric("Generation Model", config.generator_model_name)
    st.metric("Cross-Encoder", config.flashrank_model_name)

    st.divider()
    if st.button("🔄 Force Re-Index Documents", use_container_width=True):
        with st.spinner("Re-indexing NASA Technical Corpus..."):
            pipeline._load_and_index_corpus(force_reindex=True)
            st.success("Re-indexed successfully!")
            st.rerun()

# Main Header
st.markdown('<div class="main-header">🪐 NASA Mars Exploration Technical RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Production-grade hybrid retrieval & cross-encoder reranking over official NASA technical reports: '
    '<b>Perseverance EDL/MEDLI2</b>, <b>MOXIE Oxygen ISRU</b>, <b>Curiosity SAM</b>, and <b>InSight SEIS</b>.'
    '</div>',
    unsafe_allow_html=True,
)

# Tabs
tab_query, tab_benchmark, tab_corpus = st.tabs(["💬 Technical Q&A", "📊 Benchmark & Evaluation", "📂 Mission Corpus"])

# Initialize Chat State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick Preset Questions
with tab_query:
    st.markdown("**Sample Technical Queries:**")
    preset_cols = st.columns(3)
    preset_query = None

    if preset_cols[0].button("🔥 MEDLI2 Peak Heat Flux (EDL)"):
        preset_query = "What was the peak convective and radiative heat flux recorded on the heatshield by MEDLI2 during Mars 2020 entry?"
    if preset_cols[1].button("🧪 MOXIE SOEC Oxygen Production"):
        preset_query = "What is the operating temperature of the Solid Oxide Electrolysis Cells in MOXIE and what are the steady-state oxygen production rates?"
    if preset_cols[2].button("🌋 InSight Marsquake S1222a & Crust"):
        preset_query = "What was the magnitude of event S1222a and what is the estimated Martian crustal thickness?"

    preset_cols_2 = st.columns(2)
    if preset_cols_2[0].button("🔬 Curiosity SAM Organic Detections"):
        preset_query = "What organic compounds were detected by the SAM instrument on Curiosity in Gale Crater mudstone?"
    if preset_cols_2[1].button("🚫 Out-of-Scope Negative Rejection"):
        preset_query = "What are the specs of the polyethylene composite radiation shield used on the crewed interplanetary transit habitat?"

    # Display past conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "response_data" in msg:
                resp = msg["response_data"]
                # Display citations
                cit = resp.get("citations", {})
                if cit.get("has_citations") and cit.get("cited_chunks"):
                    st.markdown("**Cited Sources:**")
                    badges_html = "".join(
                        [
                            f'<span class="citation-badge">[{c["doc_id"]} §{c["section"]}]</span>'
                            for c in cit.get("cited_chunks", [])
                        ]
                    )
                    st.markdown(badges_html, unsafe_allow_html=True)

                # Latency metrics
                lat = resp.get("latency", {})
                st.caption(
                    f"⏱ Latency: Total **{lat.get('total_pipeline_ms', 0):.1f} ms** | "
                    f"Dense: {lat.get('dense_time_ms', 0):.1f} ms | "
                    f"BM25: {lat.get('sparse_time_ms', 0):.1f} ms | "
                    f"Rerank: {lat.get('rerank_time_ms', 0):.1f} ms | "
                    f"LLM Gen: {lat.get('generation_ms', 0):.1f} ms"
                )

    # Input Box
    query_input = st.chat_input("Ask a technical question about NASA Mars exploration missions...")
    active_query = preset_query or query_input

    if active_query:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Generate RAG response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving hybrid contexts & synthesizing answer..."):
                response = pipeline.query(
                    query=active_query,
                    retrieval_mode=selected_mode,
                    top_k=top_k,
                    rerank_top_n=rerank_top_n,
                )

            st.markdown(response.answer)

            # Citations Section
            cit = response.citations
            if cit.has_citations and cit.cited_chunks:
                st.markdown("---")
                st.markdown(
                    f"**Validated Sources** (Precision: `{cit.citation_precision * 100:.1f}%`):"
                )
                badges_html = "".join(
                    [
                        f'<span class="citation-badge">[{c["doc_id"]} §{c["section"]}]</span>'
                        for c in cit.cited_chunks
                    ]
                )
                st.markdown(badges_html, unsafe_allow_html=True)

                with st.expander("🔍 View Retrieved Context Passages & Match Scores"):
                    for idx, chunk in enumerate(response.retrieved_chunks, start=1):
                        st.markdown(
                            f"**Rank {idx}**: `{chunk['citation_tag']}` (Score: `{chunk['score']:.4f}`)"
                        )
                        st.markdown(f"*{chunk['title']}*")
                        st.text(chunk.get("text", chunk.get("preview", "")))
                        st.divider()

            # Latency Metrics Row
            lat = response.latency
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Total E2E", f"{lat.total_pipeline_ms:.1f} ms")
            col_m2.metric("Dense Search", f"{lat.dense_time_ms:.1f} ms")
            col_m3.metric("Sparse BM25", f"{lat.sparse_time_ms:.1f} ms")
            col_m4.metric("Rerank (FlashRank)", f"{lat.rerank_time_ms:.1f} ms")
            col_m5.metric("LLM Generation", f"{lat.generation_ms:.1f} ms")

            # Store in session state
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "response_data": {
                        "citations": cit.model_dump(),
                        "latency": lat.model_dump(),
                    },
                }
            )

# Benchmark Tab
with tab_benchmark:
    st.subheader("Quantitative Quality & Latency Benchmark")
    st.markdown(
        "Evaluation comparing 4 retrieval configurations across the golden Mars technical dataset "
        "measuring **Faithfulness**, **Context Precision**, **Context Recall**, and **Phase Latencies**."
    )

    report_path = config.reports_dir / "evaluation_report.md"
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("No benchmark report generated yet. Click below to run the evaluation suite.")

    if st.button("🚀 Run Live Ragas Benchmark Suite", use_container_width=True):
        with st.spinner("Running full 4-architecture evaluation benchmark (this may take ~2 minutes)..."):
            evaluator = RagasEvaluator(pipeline, config.eval_dataset_path)
            benchmark_results = evaluator.run_benchmark()
            evaluator.generate_report(benchmark_results, report_path)
            st.success("Benchmark completed and report updated!")
            st.rerun()

# Mission Corpus Tab
with tab_corpus:
    st.subheader("Ingested NASA Technical Mission Reports")
    reports = pipeline.loader.load_all_reports()

    for r in reports:
        with st.expander(f"🛰 {r.document_id}: {r.title} ({r.publication_year})"):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Mission:** {r.mission}")
            c2.markdown(f"**Author/Center:** {r.author}")
            c3.markdown(f"**Landing Site:** {r.landing_site}")

            if r.source_url:
                st.markdown(f"[NASA NTRS Technical Report Link]({r.source_url})")

            st.text_area("Document Preview", r.body_text[:1200] + "\n...", height=180, disabled=True)

