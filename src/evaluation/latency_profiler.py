"""Statistical latency profiler for RAG pipeline benchmarking."""

import numpy as np
from typing import List, Dict, Any
from src.rag.pipeline import RAGLatencyBreakdown


class LatencyProfiler:
    """Computes statistical metrics (P50, P90, P99, mean) across RAG pipeline executions."""

    def __init__(self):
        self.records: List[RAGLatencyBreakdown] = []

    def record(self, breakdown: RAGLatencyBreakdown):
        """Append a latency measurement record."""
        self.records.append(breakdown)

    def compute_summary(self) -> Dict[str, Dict[str, float]]:
        """Compute statistical breakdown for each latency component."""
        if not self.records:
            return {}

        fields = [
            "dense_time_ms",
            "sparse_time_ms",
            "fusion_time_ms",
            "rerank_time_ms",
            "total_retrieval_ms",
            "generation_ms",
            "total_pipeline_ms",
        ]

        summary: Dict[str, Dict[str, float]] = {}

        for field in fields:
            values = [getattr(r, field) for r in self.records]
            arr = np.array(values)
            summary[field] = {
                "mean": round(float(np.mean(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "p50": round(float(np.percentile(arr, 50)), 2),
                "p90": round(float(np.percentile(arr, 90)), 2),
                "p99": round(float(np.percentile(arr, 99)), 2),
                "min": round(float(np.min(arr)), 2),
                "max": round(float(np.max(arr)), 2),
            }

        return summary

