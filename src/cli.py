"""Command-Line Interface for NASA Mars RAG System."""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.rag.config import config
from src.rag.pipeline import ProductionRAGPipeline
from src.evaluation.ragas_evaluator import RagasEvaluator

console = Console(highlight=False)


def cmd_index(args):
    """Index or re-index the NASA Mars technical reports."""
    console.print("[bold cyan][*] Starting Document Ingestion and Indexing...[/bold cyan]")
    pipeline = ProductionRAGPipeline()
    total_chunks = len(pipeline.chunks)
    console.print(f"[bold green][+] Ingested and indexed {total_chunks} chunks successfully![/bold green]")
    console.print(f"[dim]ChromaDB Directory: {config.chroma_persist_dir}[/dim]")


def cmd_query(args):
    """Run an interactive or one-shot query with citation tracing and latency metrics."""
    query_text = args.query
    mode = args.mode

    console.print(f"[bold cyan][?] Querying RAG System ({mode}):[/bold cyan] [italic]{query_text}[/italic]\n")
    pipeline = ProductionRAGPipeline()
    response = pipeline.query(query_text, retrieval_mode=mode)

    # Print Answer Panel
    console.print(
        Panel(
            response.answer,
            title="[bold green]Grounded Technical Response[/bold green]",
            border_style="green",
        )
    )

    # Print Citations & Provenance Table
    citations = response.citations
    if citations.has_citations:
        cit_table = Table(title="Validated Citations & Source Chunks", show_header=True)
        cit_table.add_column("Doc ID", style="cyan", width=18)
        cit_table.add_column("Section", style="magenta")
        cit_table.add_column("Title", style="white")

        for c in citations.cited_chunks:
            cit_table.add_row(c["doc_id"], c["section"], c["title"])

        console.print(cit_table)
        console.print(
            f"[dim]Citation Precision: [bold]{citations.citation_precision * 100:.1f}%[/bold] "
            f"({citations.valid_citations}/{citations.total_citations} valid citations)[/dim]\n"
        )
    else:
        console.print("[yellow][!] No inline citations detected in response.[/yellow]\n")

    # Print Latency Breakdown Table
    lat_table = Table(title="Latency Breakdown (milliseconds)", show_header=True)
    lat_table.add_column("Phase", style="cyan")
    lat_table.add_column("Duration (ms)", justify="right", style="green")

    lat = response.latency
    lat_table.add_row("Dense Search (Chroma)", f"{lat.dense_time_ms:.2f}")
    lat_table.add_row("Sparse Search (BM25)", f"{lat.sparse_time_ms:.2f}")
    lat_table.add_row("Reciprocal Rank Fusion", f"{lat.fusion_time_ms:.2f}")
    lat_table.add_row("Cross-Encoder Rerank (FlashRank)", f"{lat.rerank_time_ms:.2f}")
    lat_table.add_row("Total Retrieval Phase", f"{lat.total_retrieval_ms:.2f}")
    lat_table.add_row("LLM Generation", f"{lat.generation_ms:.2f}")
    lat_table.add_row("[bold]Total Pipeline E2E[/bold]", f"[bold]{lat.total_pipeline_ms:.2f}[/bold]")

    console.print(lat_table)


def cmd_evaluate(args):
    """Run full benchmark across 4 architectures and generate report."""
    console.print("[bold cyan][*] Initializing Ragas & Latency Benchmark Suite...[/bold cyan]")
    pipeline = ProductionRAGPipeline()
    evaluator = RagasEvaluator(pipeline, config.eval_dataset_path)

    console.print("[yellow]Running evaluation across 4 retrieval configurations...[/yellow]")
    benchmark_results = evaluator.run_benchmark()

    report_path = config.reports_dir / "evaluation_report.md"
    report_content = evaluator.generate_report(benchmark_results, report_path)

    console.print(f"[bold green][+] Benchmark Complete! Report saved to: {report_path}[/bold green]\n")
    console.print(Panel(report_content[:1500] + "\n\n... (see full report on disk)", title="Report Preview"))


def cmd_chat(args):
    """Interactive continuous chat session with the RAG system."""
    console.print("[bold cyan][*] Initializing Interactive Mars RAG Chat Session...[/bold cyan]")
    console.print("[dim]Type your question and hit Enter. Type 'exit' or 'quit' to end.[/dim]\n")
    pipeline = ProductionRAGPipeline()
    mode = args.mode

    while True:
        try:
            query_text = console.input("[bold green]Ask Mars RAG > [/bold green]").strip()
            if not query_text:
                continue
            if query_text.lower() in ("exit", "quit", "q"):
                console.print("[bold yellow]Exiting chat session. Farewell![/bold yellow]")
                break

            response = pipeline.query(query_text, retrieval_mode=mode)
            console.print(
                Panel(
                    response.answer,
                    title="[bold green]Grounded Technical Response[/bold green]",
                    border_style="green",
                )
            )

            citations = response.citations
            if citations.has_citations:
                console.print(
                    f"[dim]📚 Cited {len(citations.cited_chunks)} sources | "
                    f"Precision: {citations.citation_precision * 100:.1f}% | "
                    f"E2E Latency: {response.latency.total_pipeline_ms:.1f} ms[/dim]\n"
                )
            else:
                console.print(f"[dim]E2E Latency: {response.latency.total_pipeline_ms:.1f} ms[/dim]\n")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Exiting chat session.[/bold yellow]")
            break


def main():
    parser = argparse.ArgumentParser(description="Production-Ready NASA Mars RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index command
    subparsers.add_parser("index", help="Index NASA Mars technical documents")

    # query command
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline")
    query_parser.add_argument("query", type=str, help="Question to ask")
    query_parser.add_argument(
        "--mode",
        choices=["hybrid_rerank", "hybrid_rrf", "dense_only", "sparse_only"],
        default="hybrid_rerank",
        help="Retrieval architecture mode (default: hybrid_rerank)",
    )

    # chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument(
        "--mode",
        choices=["hybrid_rerank", "hybrid_rrf", "dense_only", "sparse_only"],
        default="hybrid_rerank",
        help="Retrieval architecture mode (default: hybrid_rerank)",
    )

    # evaluate command
    subparsers.add_parser("evaluate", help="Run Ragas benchmark and generate comparison report")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
