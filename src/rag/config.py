"""Configuration management for NASA Mars RAG system."""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load from .env if present
load_dotenv()


class RAGConfig(BaseSettings):
    """System-wide configuration settings with Pydantic validation."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API Keys
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data")
    raw_data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "raw")
    processed_data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "processed")
    chroma_persist_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db")
    eval_dataset_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "mars_golden_dataset.json"
    )
    reports_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "reports")

    # Models & Providers
    embedding_provider: str = Field(default="gemini", description="'gemini' or 'local'")
    embedding_model_name: str = Field(default="models/gemini-embedding-001")
    generator_model_name: str = Field(default="gemini-3.5-flash-lite")
    flashrank_model_name: str = Field(default="ms-marco-TinyBERT-L-2-v2")

    # Chunking Hyperparameters
    chunk_size: int = Field(default=600, ge=100, le=2000)
    chunk_overlap: int = Field(default=120, ge=0, le=500)

    # Retrieval Hyperparameters
    dense_top_k: int = Field(default=10, ge=1, le=50)
    sparse_top_k: int = Field(default=10, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, le=100)
    rerank_top_n: int = Field(default=4, ge=1, le=20)
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    # Generation Parameters
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=1024, ge=128, le=4096)


config = RAGConfig()
