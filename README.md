# 🪐 NASA Mars Technical RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** system engineered over authentic NASA technical reports on Mars exploration missions. 

The system features **hybrid retrieval** (ChromaDB vector search + BM25 lexical search) with **Reciprocal Rank Fusion (RRF)**, lightweight CPU-optimized **FlashRank cross-encoder reranking**, **strict citation grounding**, anti-hallucination guardrails, an interactive **Streamlit web frontend**, and an automated **Ragas-style evaluation benchmark suite**.

---

## 🏗️ Architecture Diagram

```
                                 User Question
                                       │
                      ┌────────────────┴────────────────┐
                      │                                 │
                      ▼                                 ▼
           Dense Vector Retrieval             Sparse BM25 Retrieval
          (ChromaDB + Gemini Embed)           (BM25Okapi Tokenized)
                      │                                 │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                        Reciprocal Rank Fusion (RRF)
                         Score(d) = Σ w / (k + rank)
                                       │
                                       ▼ (Top-10 Candidates)
                           FlashRank Cross-Encoder
                           (ms-marco-TinyBERT-L-2)
                                       │
                                       ▼ (Top-4 Precision Chunks)
                       Citation-Grounded Generator
                         (Gemini 3.5 Flash Lite)
                                       │
                                       ▼
                        Output Parser & Provenance Guard
                        - Grounded Synthesized Response
                        - Verified Citations [Doc-ID §Section]
                        - Stage Latency Breakdown (ms)
```

---

## ✨ Key Capabilities

1. **Authentic Technical Domain Corpus (`data/raw/`)**:
   - **Mars 2020 Perseverance EDL & MEDLI2**: Entry telemetry, heatshield convective/radiative heat flux, PICA bondline margins, Range Trigger mortar timings, Terrain-Relative Navigation (TRN), and Sky Crane landing.
   - **Mars 2020 MOXIE (Oxygen ISRU)**: Solid Oxide Electrolysis Cells (SOEC), atmospheric scroll compressor, oxygen purity ($\ge 98.0\%$), production rates (6 to 12 g/hr), and Nernst voltage clamping to prevent catalytic coking.
   - **MSL Curiosity Gale Crater Mineralogy & SAM**: Quadrupole Mass Spectrometer (QMS), chlorobenzene/thiophene detection, and Yellowknife Bay lacustrine habitability.
   - **Mars InSight SEIS Geophysics**: Very Broad Band (VBB) seismometer, monster marsquake event S1222a ($M_W = 4.7$), Martian crustal thickness models, and liquid metallic core radius ($1,830 \pm 40$ km).

2. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**:
   - Combines semantic understanding from dense vector embeddings (`models/gemini-embedding-001` with deterministic local hash fallback) with exact keyword recall from BM25Okapi.
   - Scale-invariant RRF fusion ($k=60$) guarantees zero catastrophic misses on mission acronyms, sensor codes, and numbers.

3. **Ultra-Fast Cross-Encoder Reranking**:
   - Integrates **FlashRank** (`ms-marco-TinyBERT-L-2-v2`) running on local ONNX runtime.
   - Eliminates semantically noisy candidates in ~180 ms on CPU without requiring GPU infrastructure or cloud reranker costs.

4. **Strict Inline Citations & Provenance Verification**:
   - Generates answers where every technical specification and factual claim is tied to an inline citation tag, e.g., `[NASA-M2020-EDL §3. MEDLI2 Sensor Suite Instrumentation & Thermal Telemetry]`.
   - Automated citation validator checks that citations correspond to retrieved chunks, computing **Citation Precision**.
   - Negative rejection rule: Explicitly abstains on out-of-scope questions instead of hallucinating.

5. **Modern Dual User Interfaces**:
   - **Streamlit Web Application (`app.py`)**: Real-time chat, one-click preset questions, expandable source chunk inspector, live latency meters, and corpus browser.
   - **Terminal CLI (`src.cli`)**: Fast command-line operations for `chat`, `query`, `index`, and `evaluate`.

---

## 📊 Evaluation & Benchmark Results

The system includes a golden dataset of 10 Mars mission queries (including multi-hop, numeric lookups, and negative edge cases) evaluated across 4 retrieval configurations:

| Retrieval Architecture | Faithfulness | Context Precision | Context Recall | Citation Precision | Median E2E Latency (P50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Dense Vector (Chroma)** | 0.872 | 1.000 | 1.000 | 1.000 | 2,310 ms |
| **2. Sparse Lexical (BM25)** | 0.871 | 0.953 | 1.000 | 1.000 | 1,650 ms |
| **3. Hybrid (Dense + BM25 RRF)** | 0.932 | 0.983 | 1.000 | 1.000 | 1,952 ms |
| **4. Hybrid + FlashRank Rerank (Production)** | **0.935** | **0.992** | **1.000** | **0.900** | **2,056 ms** |

### Per-Stage Latency Breakdown (Production Pipeline):
- **Dense Vector Search**: ~545 ms *(includes API network roundtrip)*
- **Sparse BM25 Search**: ~0.8 ms
- **RRF Merge**: ~0.05 ms
- **FlashRank Cross-Encoder Rerank**: ~205 ms *(local CPU ONNX)*
- **LLM Synthesis**: ~1,420 ms
- **Total End-to-End Latency**: **~2.05 s**

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended package manager) or standard `pip`

### 1. Installation
Clone or navigate to the repository directory and install dependencies:
```powershell
uv sync
```

### 2. Environment Setup
Configure your environment variables:
```powershell
Copy-Item .env.example .env
```
Ensure your `GEMINI_API_KEY` is set in `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
CHROMA_PERSIST_DIR=./data/chroma_db
EMBEDDING_PROVIDER=gemini
```

### 3. Ingest and Index Documents
Ingest the NASA Mars reports into ChromaDB and BM25:
```powershell
uv run python -m src.cli index
```

---

## 💻 Running the Application

### Option A: Launch the Interactive Web Frontend (UI)
Start the Streamlit application:
```powershell
uv run streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### Option B: Interactive Terminal Chat
Chat directly within your terminal:
```powershell
uv run python -m src.cli chat
```

### Option C: Single Question Query
Execute a single query with formatted response, citation table, and latency breakdown:
```powershell
uv run python -m src.cli query "What was the peak convective and radiative heat flux recorded on the heatshield by MEDLI2 during Mars 2020 entry?"
```

To compare against other retrieval architectures, pass `--mode`:
```powershell
uv run python -m src.cli query "What is the operating temperature of SOEC in MOXIE?" --mode dense_only
uv run python -m src.cli query "What is the operating temperature of SOEC in MOXIE?" --mode sparse_only
uv run python -m src.cli query "What is the operating temperature of SOEC in MOXIE?" --mode hybrid_rrf
```

### Option D: Run Full Ragas Benchmark Suite
Execute the 10-query benchmark across all 4 architectures and generate a report:
```powershell
uv run python -m src.cli evaluate
```
The report is saved to `reports/evaluation_report.md`.

### Option E: Run Automated Tests
Execute the Pytest test suite:
```powershell
uv run pytest tests/ -v
```

---

## 📁 Repository Structure

```
.
├── app.py                     # Streamlit interactive web frontend
├── pyproject.toml             # Project dependencies and build config
├── .env.example               # Environment variables template
├── data/
│   ├── raw/                   # Authentic NASA Mars technical reports (markdown)
│   │   ├── nasa_perseverance_edl_telemetry.md
│   │   ├── nasa_moxie_oxygen_production.md
│   │   ├── nasa_curiosity_gale_crater_sam.md
│   │   └── nasa_insight_seis_crust.md
│   ├── evaluation/            # Golden test dataset (10 queries + ground truth)
│   │   └── mars_golden_dataset.json
│   └── chroma_db/             # Persistent ChromaDB vector index
├── reports/
│   └── evaluation_report.md   # Benchmark report with quality & latency tables
├── src/
│   ├── cli.py                 # Terminal CLI (chat, query, index, evaluate)
│   ├── rag/
│   │   ├── config.py          # Pydantic settings & hyperparameters
│   │   ├── data_loader.py     # Frontmatter parser & SHA-256 validator
│   │   ├── chunking.py        # Section-aware markdown splitter
│   │   ├── vector_store.py    # ChromaDB integration with Gemini & local fallback
│   │   ├── pipeline.py        # Unified end-to-end RAG pipeline
│   │   ├── retrieval/
│   │   │   ├── sparse.py      # BM25Okapi lexical retriever
│   │   │   ├── hybrid.py      # Reciprocal Rank Fusion (RRF) algorithm
│   │   │   ├── reranker.py    # FlashRank cross-encoder reranker
│   │   │   └── pipeline.py    # Multi-stage retrieval coordinator
│   │   └── generation/
│   │       ├── prompts.py     # Strict citation prompt templates
│   │       ├── generator.py   # Gemini 3.5 Flash Lite generator with fallback
│   │       └── citation_validator.py # Provenance & citation precision parser
│   └── evaluation/
│       ├── latency_profiler.py # P50/P90 statistical profiler
│       └── ragas_evaluator.py  # 4-architecture benchmark runner
└── tests/
    ├── test_chunking.py       # Unit tests for parsing and chunking
    ├── test_retrieval.py      # Unit tests for BM25, Chroma, RRF, FlashRank
    └── test_pipeline.py       # Integration tests for end-to-end pipeline
```

---

## 🐍 Programmatic Python API

You can import and use the pipeline directly in your Python applications:

```python
from src.rag.pipeline import ProductionRAGPipeline

# Initialize pipeline (loads existing ChromaDB index)
pipeline = ProductionRAGPipeline()

# Query the pipeline
response = pipeline.query(
    query="What clay minerals were detected by CheMin in Gale Crater and what was the water pH?",
    retrieval_mode="hybrid_rerank", # 'hybrid_rerank', 'hybrid_rrf', 'dense_only', 'sparse_only'
    top_k=10,
    rerank_top_n=4,
)

# Access structured response
print("Grounded Answer:\n", response.answer)
print("\nCitation Precision:", response.citations.citation_precision)
print("E2E Latency (ms):", response.latency.total_pipeline_ms)

# Access cited sources
for source in response.citations.cited_chunks:
    print(f"- [{source['doc_id']}] {source['title']} (§{source['section']})")
```

---

## 📜 License
This project is open-source under the Apache-2.0 License. NASA technical documentation and data are in the public domain.

