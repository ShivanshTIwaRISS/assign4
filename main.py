#!/usr/bin/env python3
"""
Track 03: AI Automation & Agent Development
AI Research & Discovery Platform — Main Entry Point

Deliverables:
  1. Autonomous Multi-Agent Scripts     (CrewAI swarm)
  2. Self-Correcting LLM Pipelines      (LangChain + retry)
  3. Semantic Vector Search Engine      (ChromaDB + RAG)

Usage:
  python main.py                  # Full demo (all 3 deliverables)
  python main.py --mode agents    # Run multi-agent crew only
  python main.py --mode pipeline  # Run LLM pipeline only
  python main.py --mode search    # Run semantic search only
  python main.py --mode api       # Start FastAPI server
  python main.py --mode ingest    # Ingest demo data into vector store
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

console = Console()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Banner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BANNER = """
[bold cyan]  ╔══════════════════════════════════════════════════════════════╗
  ║     🤖  AI Research & Discovery Platform  —  Track 03         ║
  ║         AI Automation & Agent Development                     ║
  ╚══════════════════════════════════════════════════════════════╝[/bold cyan]

  [dim]Deliverables:[/dim]
  [green]①[/green] Autonomous Multi-Agent Swarm      [dim](CrewAI + web scraping)[/dim]
  [green]②[/green] Self-Correcting LLM Pipeline      [dim](LangChain + retry logic)[/dim]
  [green]③[/green] Semantic Vector Search Engine     [dim](ChromaDB + RAG)[/dim]
"""


def print_banner():
    console.print(BANNER)
    console.print(
        Panel(
            f"Mode: [bold]{settings.mode_label}[/bold]   "
            f"Model: [cyan]{settings.OPENAI_MODEL}[/cyan]   "
            f"Embeddings: [cyan]{settings.OPENAI_EMBEDDING_MODEL}[/cyan]",
            border_style="dim",
        )
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Deliverable 1 — Autonomous Multi-Agent Swarm
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_agents(seed_urls: list[str] | None = None) -> dict:
    """
    Deliverable 1: Autonomous Multi-Agent Swarm.

    Launches a 4-agent CrewAI crew:
      • Scraper Agent   → autonomous web exploration
      • Parser Agent    → structural entity parsing
      • Researcher Agent→ ISBN/author verification
      • Validator Agent → confidence scoring & QA
    """
    console.print(Rule("[bold cyan]① Autonomous Multi-Agent Swarm[/bold cyan]", style="cyan"))
    console.print(
        "[dim]Agents: Scraper → Parser → Researcher → Validator[/dim]\n"
    )

    from agents.crew import ResearchDiscoveryCrew

    crew = ResearchDiscoveryCrew()
    results = crew.run(seed_urls=seed_urls)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Deliverable 2 — Self-Correcting LLM Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_pipeline(custom_texts: list[str] | None = None):
    """
    Deliverable 2: Self-Correcting LLM Pipeline.

    Runs the 3-stage pipeline:
      Stage 1 — Entity Extraction   (LangChain + GPT)
      Stage 2 — Data Screening      (rule-based validation)
      Stage 3 — Self-Correction     (automatic retry on failure)
    """
    console.print(Rule("[bold magenta]② Self-Correcting LLM Pipeline[/bold magenta]", style="magenta"))
    console.print(
        "[dim]Stages: Extraction → Screening → Self-Correction → Entity Construction[/dim]\n"
    )

    from pipelines.pipeline_manager import PipelineManager

    pm = PipelineManager()
    if custom_texts:
        results = pm.process_batch(custom_texts)
    else:
        results = pm.demo()

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Deliverable 3 — Semantic Vector Search Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_search_demo():
    """
    Deliverable 3: Semantic Vector Search Engine.

    Demonstrates:
      • Similarity search (ChromaDB + embeddings)
      • RAG-powered Q&A (retrieval + LLM generation)
      • Filtered entity-type search
    """
    console.print(Rule("[bold green]③ Semantic Vector Search Engine[/bold green]", style="green"))
    console.print(
        "[dim]Pipeline: Query → Embed → ChromaDB Retrieval → RAG Generation[/dim]\n"
    )

    from search_engine.query_engine import QueryEngine

    engine = QueryEngine()
    stats = engine.vector_store.get_stats()

    if stats["total"] == 0:
        console.print(
            "[yellow]⚠ Vector store is empty. Running ingest first...[/yellow]"
        )
        ingest_pipeline_data()
        engine = QueryEngine()  # re-init after ingest

    # --- Demo queries ---
    demo_queries = [
        ("search", "deep learning neural networks textbook", None),
        ("search", "university press artificial intelligence", None),
        ("ask", "Who wrote the book Deep Learning and what publisher released it?", None),
        ("search", "machine learning", "author"),
        ("search", "computer science engineering", "book"),
    ]

    for mode, query, entity_filter in demo_queries:
        console.print(f"\n[bold]Query:[/bold] [cyan]{query}[/cyan]"
                      + (f" [dim](filter: {entity_filter})[/dim]" if entity_filter else ""))

        if mode == "ask":
            response = engine.ask(query)
            console.print(
                Panel(
                    response.answer,
                    title="[green]💡 RAG Answer[/green]",
                    border_style="green",
                    padding=(0, 1),
                )
            )
            console.print(
                f"[dim]  ⏱ {response.duration_seconds:.2f}s | "
                f"sources: {len(response.sources)} | model: {response.model_used}[/dim]"
            )
        else:
            results = engine.search(query, n_results=3, entity_type=entity_filter)
            engine._display_search_results(results, query)


def run_interactive_search():
    """Launch the interactive search REPL."""
    console.print(Rule("[bold green]③ Interactive Semantic Search[/bold green]", style="green"))

    from search_engine.query_engine import QueryEngine

    engine = QueryEngine()
    stats = engine.vector_store.get_stats()

    if stats["total"] == 0:
        console.print("[yellow]⚠ Vector store is empty. Ingesting demo data...[/yellow]")
        ingest_pipeline_data()
        engine = QueryEngine()

    engine.interactive()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ingest — Populate vector store from pipeline results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def ingest_pipeline_data(pipeline_results=None, crew_results: dict | None = None):
    """
    Ingest entities from pipeline / crew results into ChromaDB.

    If no results are provided, runs both the pipeline demo and crew demo
    to generate data, then indexes all entities.
    """
    console.print(Rule("[bold yellow]📥 Vector Store Ingest[/bold yellow]", style="yellow"))

    from search_engine.vector_store import VectorStore
    from models.schemas import Author, Publisher, Book

    vs = VectorStore()
    total_indexed = 0

    # ── From LLM Pipeline results ──────────────────────────────────────────
    if pipeline_results is None:
        console.print("[dim]Running pipeline demo to generate entities...[/dim]")
        from pipelines.pipeline_manager import PipelineManager
        pm = PipelineManager()
        pipeline_results = pm.demo()

    all_authors: list[Author] = []
    all_publishers: list[Publisher] = []
    all_books: list[Book] = []

    for result in pipeline_results:
        all_authors.extend(result.final_authors)
        all_publishers.extend(result.final_publishers)
        all_books.extend(result.final_books)

    # ── Also ingest crew results if available ─────────────────────────────
    if crew_results:
        _merge_crew_results(crew_results, all_authors, all_publishers, all_books)

    # ── Deduplicate by name/title ─────────────────────────────────────────
    all_authors = _dedup_authors(all_authors)
    all_publishers = _dedup_publishers(all_publishers)
    all_books = _dedup_books(all_books)

    # ── Index into ChromaDB ───────────────────────────────────────────────
    console.print(
        f"\n[bold]Indexing:[/bold] {len(all_authors)} authors, "
        f"{len(all_publishers)} publishers, {len(all_books)} books"
    )
    total_indexed += vs.add_authors(all_authors)
    total_indexed += vs.add_publishers(all_publishers)
    total_indexed += vs.add_books(all_books)

    stats = vs.get_stats()
    _print_ingest_summary(stats, total_indexed)
    return total_indexed


def _merge_crew_results(
    crew_results: dict,
    authors: list,
    publishers: list,
    books: list,
):
    """Parse crew JSON results into Pydantic models and merge into lists."""
    from models.schemas import Author, Publisher, Book

    for a in crew_results.get("authors", []):
        if isinstance(a, dict) and a.get("name"):
            try:
                authors.append(Author(**{k: v for k, v in a.items() if k != "confidence_score"},
                                      confidence_score=a.get("confidence_score", 0.8)))
            except Exception:
                pass

    for p in crew_results.get("publishers", []):
        if isinstance(p, dict) and p.get("name"):
            try:
                publishers.append(Publisher(**{k: v for k, v in p.items() if k != "confidence_score"},
                                             confidence_score=p.get("confidence_score", 0.8)))
            except Exception:
                pass

    for b in crew_results.get("books", []):
        if isinstance(b, dict) and b.get("title"):
            try:
                books.append(Book(**{k: v for k, v in b.items() if k != "confidence_score"},
                                   confidence_score=b.get("confidence_score", 0.8)))
            except Exception:
                pass


def _dedup_authors(authors: list) -> list:
    seen = {}
    for a in authors:
        key = a.name.lower().strip()
        if key not in seen:
            seen[key] = a
    return list(seen.values())


def _dedup_publishers(publishers: list) -> list:
    seen = {}
    for p in publishers:
        key = p.name.lower().strip()
        if key not in seen:
            seen[key] = p
    return list(seen.values())


def _dedup_books(books: list) -> list:
    seen = {}
    for b in books:
        key = b.title.lower().strip()
        if key not in seen:
            seen[key] = b
    return list(seen.values())


def _print_ingest_summary(stats: dict, total_indexed: int):
    table = Table(title="Vector Store Status", border_style="yellow")
    table.add_column("Collection", style="bold")
    table.add_column("Count", justify="right")

    table.add_row("👤 Authors", str(stats.get("author", 0)))
    table.add_row("🏢 Publishers", str(stats.get("publisher", 0)))
    table.add_row("📚 Books", str(stats.get("book", 0)))
    table.add_row("[bold]Total[/bold]", f"[bold cyan]{stats.get('total', 0)}[/bold cyan]")

    console.print()
    console.print(table)
    console.print(f"\n[green]✅ Indexed {total_indexed} new entities into ChromaDB[/green]\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full Demo — All 3 deliverables in sequence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_full_demo():
    """
    Run all three deliverables end-to-end.

    Flow:
      ① Multi-agent swarm discovers entities
      ② LLM pipeline extracts & validates entities
      ③ Entities are indexed → semantic search demo
    """
    t0 = time.time()

    # ─── Deliverable 1: Agents ─────────────────────────────────────────────
    crew_results = run_agents()
    console.print()

    # ─── Deliverable 2: LLM Pipeline ──────────────────────────────────────
    pipeline_results = run_pipeline()
    console.print()

    # ─── Ingest into vector store ─────────────────────────────────────────
    ingest_pipeline_data(
        pipeline_results=pipeline_results,
        crew_results=crew_results,
    )

    # ─── Deliverable 3: Search demo ───────────────────────────────────────
    run_search_demo()

    elapsed = time.time() - t0
    console.print(
        Panel(
            f"[bold green]🎉 Full demo complete![/bold green]\n\n"
            f"All 3 Track-03 deliverables executed successfully in [cyan]{elapsed:.1f}s[/cyan].\n\n"
            f"  [green]①[/green] Autonomous Multi-Agent Swarm    ✅\n"
            f"  [green]②[/green] Self-Correcting LLM Pipeline    ✅\n"
            f"  [green]③[/green] Semantic Vector Search Engine   ✅\n\n"
            f"[dim]Run [cyan]python main.py --mode search[/cyan] for interactive search.[/dim]\n"
            f"[dim]Run [cyan]python main.py --mode api[/cyan] to start the REST API.[/dim]",
            title="Track 03 — Summary",
            border_style="green",
        )
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API Server
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_api_server():
    """Launch the FastAPI REST server."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install uvicorn[standard][/red]")
        sys.exit(1)

    console.print(
        Panel(
            f"[bold green]🚀 Starting FastAPI Server[/bold green]\n\n"
            f"  Host: [cyan]{settings.API_HOST}:{settings.API_PORT}[/cyan]\n"
            f"  Docs: [cyan]http://{settings.API_HOST}:{settings.API_PORT}/docs[/cyan]\n"
            f"  Mode: {settings.mode_label}",
            title="API Server",
            border_style="green",
        )
    )

    from api.server import app

    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI Argument Parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Track 03: AI Research & Discovery Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                       # Full demo (all deliverables)
  python main.py --mode agents         # Multi-agent swarm only
  python main.py --mode pipeline       # LLM pipeline only
  python main.py --mode search         # Interactive semantic search
  python main.py --mode ingest         # Ingest data into vector store
  python main.py --mode api            # Start FastAPI REST server
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "agents", "pipeline", "search", "ingest", "api"],
        default="full",
        help="Execution mode (default: full)",
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        default=None,
        help="Seed URLs for the agent swarm (agents mode only)",
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear the vector store before running",
    )
    return parser


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    parser = build_parser()
    args = parser.parse_args()

    print_banner()

    # Optionally clear the vector store
    if args.clear_db:
        from search_engine.vector_store import VectorStore
        VectorStore().clear()
        console.print("[yellow]🗑️  Vector store cleared.[/yellow]\n")

    mode = args.mode

    if mode == "full":
        run_full_demo()

    elif mode == "agents":
        run_agents(seed_urls=args.urls)

    elif mode == "pipeline":
        run_pipeline()

    elif mode == "search":
        run_interactive_search()

    elif mode == "ingest":
        ingest_pipeline_data()

    elif mode == "api":
        run_api_server()


if __name__ == "__main__":
    main()
