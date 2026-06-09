"""
Query Engine — High-Level Search Interface

Provides a unified search interface supporting natural language queries,
filtered searches, and interactive search mode with Rich output.
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import settings
from models.schemas import EntityType, SearchResult, RAGResponse
from search_engine.vector_store import VectorStore
from search_engine.rag import RAGPipeline

console = Console()


class QueryEngine:
    """
    High-level search interface for the knowledge base.

    Supports:
    - Semantic similarity search
    - RAG-powered Q&A
    - Filtered search by entity type
    - Interactive search mode
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self.rag = RAGPipeline(self.vector_store)

    def search(
        self,
        query: str,
        n_results: int = 5,
        entity_type: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Semantic search across all entities.

        Args:
            query: Natural language search query.
            n_results: Number of results to return.
            entity_type: Optional filter — "author", "publisher", or "book".

        Returns:
            List of SearchResult objects.
        """
        et = None
        if entity_type:
            try:
                et = EntityType(entity_type.lower())
            except ValueError:
                console.print(f"[yellow]Unknown entity type: {entity_type}. Searching all.[/yellow]")

        results = self.vector_store.search(
            query=query,
            n_results=n_results,
            entity_type=et,
        )

        return results

    def ask(
        self,
        question: str,
        n_results: int = 5,
        entity_type: Optional[str] = None,
    ) -> RAGResponse:
        """
        Ask a question using RAG (Retrieval-Augmented Generation).

        Args:
            question: Natural language question.
            n_results: Number of documents to retrieve for context.
            entity_type: Optional entity type filter.

        Returns:
            RAGResponse with answer and source citations.
        """
        et = None
        if entity_type:
            try:
                et = EntityType(entity_type.lower())
            except ValueError:
                pass

        return self.rag.query(
            question=question,
            n_results=n_results,
            entity_type=et,
        )

    def interactive(self):
        """
        Run an interactive search session in the terminal.
        """
        console.print(
            Panel(
                "[bold]Semantic Search Engine[/bold]\n\n"
                "Commands:\n"
                "  [cyan]search <query>[/cyan]     — Semantic similarity search\n"
                "  [cyan]ask <question>[/cyan]      — RAG-powered Q&A\n"
                "  [cyan]authors <query>[/cyan]     — Search only authors\n"
                "  [cyan]books <query>[/cyan]       — Search only books\n"
                "  [cyan]publishers <query>[/cyan]  — Search only publishers\n"
                "  [cyan]stats[/cyan]               — Show database statistics\n"
                "  [cyan]quit[/cyan]                — Exit\n",
                title="🔍 Interactive Search",
                border_style="cyan",
            )
        )

        stats = self.vector_store.get_stats()
        console.print(f"  Knowledge base: [cyan]{stats['total']}[/cyan] entities indexed\n")

        while True:
            try:
                user_input = console.input("[bold cyan]search>[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "stats":
                self._show_stats()
                continue

            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            query = parts[1] if len(parts) > 1 else ""

            if command == "ask" and query:
                response = self.ask(query)
                self._display_rag_response(response)
            elif command == "authors" and query:
                results = self.search(query, entity_type="author")
                self._display_search_results(results, query)
            elif command == "books" and query:
                results = self.search(query, entity_type="book")
                self._display_search_results(results, query)
            elif command == "publishers" and query:
                results = self.search(query, entity_type="publisher")
                self._display_search_results(results, query)
            elif command == "search" and query:
                results = self.search(query)
                self._display_search_results(results, query)
            else:
                # Default: treat entire input as a search query
                results = self.search(user_input)
                self._display_search_results(results, user_input)

    def _display_search_results(self, results: list[SearchResult], query: str):
        """Display search results in a formatted table."""
        if not results:
            console.print("[yellow]No results found.[/yellow]\n")
            return

        table = Table(
            title=f"Search Results for: \"{query}\"",
            border_style="cyan",
            show_lines=True,
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Type", style="bold", width=10)
        table.add_column("Entity", width=40)
        table.add_column("Score", justify="right", width=8)

        for i, result in enumerate(results, 1):
            type_emoji = {
                EntityType.AUTHOR: "👤",
                EntityType.PUBLISHER: "🏢",
                EntityType.BOOK: "📚",
            }
            emoji = type_emoji.get(result.entity_type, "📄")
            entity_name = (
                result.metadata.get("name")
                or result.metadata.get("title")
                or result.text[:50]
            )
            score_color = "green" if result.similarity_score > 0.7 else "yellow" if result.similarity_score > 0.4 else "red"
            table.add_row(
                str(i),
                f"{emoji} {result.entity_type.value}",
                entity_name,
                f"[{score_color}]{result.similarity_score:.2f}[/{score_color}]",
            )

        console.print(table)
        console.print()

    def _display_rag_response(self, response: RAGResponse):
        """Display a RAG response with answer and sources."""
        console.print(
            Panel(
                response.answer,
                title="💡 Answer",
                border_style="green",
                padding=(1, 2),
            )
        )

        if response.sources:
            console.print(f"\n[dim]Sources ({len(response.sources)} documents retrieved):[/dim]")
            for i, source in enumerate(response.sources, 1):
                name = (
                    source.metadata.get("name")
                    or source.metadata.get("title")
                    or "Unknown"
                )
                console.print(
                    f"  [dim]{i}. [{source.entity_type.value}] {name} "
                    f"(score: {source.similarity_score:.2f})[/dim]"
                )

        console.print(
            f"[dim]  ⏱ {response.duration_seconds:.2f}s | Model: {response.model_used}[/dim]\n"
        )

    def _show_stats(self):
        """Display database statistics."""
        stats = self.vector_store.get_stats()

        table = Table(title="Knowledge Base Statistics", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Count", justify="right")

        table.add_row("Total Entities", str(stats.get("total", 0)))
        table.add_row("👤 Authors", str(stats.get("author", 0)))
        table.add_row("🏢 Publishers", str(stats.get("publisher", 0)))
        table.add_row("📚 Books", str(stats.get("book", 0)))

        console.print(table)
        console.print()
