"""
Pipeline Manager — Central Pipeline Orchestrator

Chains the extraction → screening → correction stages into a unified
pipeline with comprehensive logging and progress tracking.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config.settings import settings
from models.schemas import (
    Author,
    Publisher,
    Book,
    EntityType,
    ExtractionResult,
    PipelineResult,
)
from pipelines.entity_extraction import EntityExtractor
from pipelines.data_screening import DataScreener
from pipelines.self_correction import SelfCorrectionEngine

console = Console()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sample texts for demo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAMPLE_TEXTS = [
    """
    Dr. Andrew Ng is a renowned machine learning researcher and educator at Stanford University.
    He co-founded Coursera and founded DeepLearning.AI. His research focuses on machine learning,
    deep learning, and AI education. He has published numerous papers and the book
    "Machine Learning Yearning". His h-index is approximately 165.

    The textbook "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville was
    published by MIT Press in 2016 with ISBN 978-0262035613. It covers neural networks,
    optimization, and regularization techniques across 800 pages.

    MIT Press, founded in 1962, is a university press based in Cambridge, Massachusetts.
    They specialize in computer science, artificial intelligence, cognitive science,
    and economics publications.

    "Pattern Recognition and Machine Learning" by Christopher M. Bishop was published
    by Springer in 2006 (ISBN: 978-0387310732). It covers Bayesian methods, kernel methods,
    and graphical models across 738 pages.

    Springer Nature, headquartered in Berlin, Germany, is one of the world's largest
    academic publishers, founded in 1842. They specialize in science, technology,
    medicine, and engineering publications.
    """,
    """
    Dr. Yoshua Bengio, a professor at the Université de Montréal and scientific director
    at Mila (Quebec AI Institute), is a Turing Award winner known for his work on deep
    learning and neural networks. His h-index exceeds 200.

    "Artificial Intelligence: A Modern Approach" (4th Edition) by Stuart Russell and Peter Norvig,
    published by Pearson in 2020 (ISBN: 978-0134610993), is the standard AI textbook
    with over 1115 pages covering search, knowledge representation, and machine learning.

    O'Reilly Media, founded in 1978 in Sebastopol, California, is a leading technical
    publisher specializing in programming, data science, AI, and cloud computing.
    """,
]


class PipelineManager:
    """
    Central orchestrator for the LLM processing pipeline.

    Pipeline stages:
    1. Entity Extraction — LLM-based structured extraction
    2. Data Screening — Rule-based validation and normalization
    3. Self-Correction — Automatic error correction with retry
    4. Entity Construction — Build validated Pydantic models
    """

    def __init__(self):
        self.extractor = EntityExtractor()
        self.screener = DataScreener()
        self.corrector = SelfCorrectionEngine()

    def process(self, text: str) -> PipelineResult:
        """
        Run the full pipeline on input text.

        Args:
            text: Unstructured text to process.

        Returns:
            PipelineResult with extracted, screened, and validated entities.
        """
        start_time = time.time()
        pipeline_result = PipelineResult(input_text=text[:500])

        console.print(
            Panel(
                f"[bold]Processing text[/bold] ({len(text)} chars)\n"
                f"Mode: {settings.mode_label}",
                title="🔄 Pipeline Started",
                border_style="cyan",
            )
        )

        try:
            # ── Stage 1: Extraction ───────────────
            console.print("\n[bold cyan]Stage 1/3:[/bold cyan] Entity Extraction")
            extraction_result = self.extractor.extract(text)

            if not extraction_result.success:
                # Trigger self-correction
                console.print("[yellow]  Extraction failed, triggering self-correction...[/yellow]")
                corrected_output, corrections = self.corrector.correct_extraction(
                    original_text=text,
                    error=extraction_result.error_message or "Extraction failed",
                    previous_output=extraction_result.raw_llm_output,
                )
                pipeline_result.corrections.extend(corrections)

                if corrected_output:
                    # Re-parse corrected output
                    extraction_result = self._reparse_extraction(corrected_output)
                else:
                    pipeline_result.success = False
                    pipeline_result.error_message = "Extraction failed after self-correction"
                    pipeline_result.duration_seconds = round(time.time() - start_time, 2)
                    return pipeline_result

            pipeline_result.extraction = extraction_result
            console.print(
                f"  [green]✅ Extracted {len(extraction_result.entities)} entities "
                f"({extraction_result.duration_seconds:.1f}s)[/green]"
            )

            # ── Stage 2: Screening ────────────────
            console.print("\n[bold cyan]Stage 2/3:[/bold cyan] Data Screening")
            screening_results = self.screener.screen_batch(extraction_result.entities)
            pipeline_result.screening_results = screening_results

            valid_count = sum(1 for r in screening_results if r.is_valid)
            issue_count = sum(len(r.issues) for r in screening_results)
            console.print(
                f"  [green]✅ Screened {len(screening_results)} entities: "
                f"{valid_count} valid, {issue_count} issues found[/green]"
            )

            # Show issues
            for sr in screening_results:
                if sr.issues:
                    for issue in sr.issues:
                        color = "red" if "CRITICAL" in issue or "ERROR" in issue else "yellow" if "WARNING" in issue else "green"
                        console.print(f"    [{color}]{issue}[/{color}]")

            # ── Stage 3: Entity Construction ──────
            console.print("\n[bold cyan]Stage 3/3:[/bold cyan] Entity Construction")
            authors, publishers, books = self._build_entities(screening_results)

            pipeline_result.final_authors = authors
            pipeline_result.final_publishers = publishers
            pipeline_result.final_books = books
            pipeline_result.total_entities = len(authors) + len(publishers) + len(books)
            pipeline_result.success = True

            console.print(
                f"  [green]✅ Built {len(authors)} authors, "
                f"{len(publishers)} publishers, {len(books)} books[/green]"
            )

        except Exception as e:
            pipeline_result.success = False
            pipeline_result.error_message = str(e)
            console.print(f"\n[bold red]❌ Pipeline error: {str(e)}[/bold red]")

        pipeline_result.duration_seconds = round(time.time() - start_time, 2)
        self._print_summary(pipeline_result)
        return pipeline_result

    def process_batch(self, texts: list[str]) -> list[PipelineResult]:
        """Process multiple texts through the pipeline."""
        results = []
        for i, text in enumerate(texts, 1):
            console.print(f"\n{'━' * 60}")
            console.print(f"[bold]Processing text {i}/{len(texts)}[/bold]")
            console.print(f"{'━' * 60}")
            result = self.process(text)
            results.append(result)
        return results

    def demo(self) -> list[PipelineResult]:
        """Run the pipeline on sample academic texts."""
        console.print(
            Panel(
                "[bold]Running pipeline demo with sample academic texts[/bold]",
                title="📝 Pipeline Demo",
                border_style="magenta",
            )
        )
        return self.process_batch(SAMPLE_TEXTS)

    def _reparse_extraction(self, corrected_output: str) -> ExtractionResult:
        """Re-parse a corrected output string into an ExtractionResult."""
        from models.schemas import ExtractedEntity

        entities = self.extractor._parse_llm_output(corrected_output)
        return ExtractionResult(
            entities=entities,
            raw_llm_output=corrected_output,
            model_used=settings.OPENAI_MODEL,
            success=True,
        )

    def _build_entities(self, screening_results):
        """Build validated Pydantic entities from screening results."""
        authors = []
        publishers = []
        books = []

        for sr in screening_results:
            data = sr.screened_data if sr.screened_data else sr.original_data
            if not sr.is_valid:
                continue

            try:
                if sr.entity_type == EntityType.AUTHOR:
                    # Remove non-schema fields
                    clean = {k: v for k, v in data.items() if k != "confidence"}
                    clean["confidence_score"] = data.get("confidence", 0.8)
                    authors.append(Author(**clean))
                elif sr.entity_type == EntityType.PUBLISHER:
                    clean = {k: v for k, v in data.items() if k != "confidence"}
                    clean["confidence_score"] = data.get("confidence", 0.8)
                    publishers.append(Publisher(**clean))
                elif sr.entity_type == EntityType.BOOK:
                    clean = {k: v for k, v in data.items() if k != "confidence"}
                    clean["confidence_score"] = data.get("confidence", 0.8)
                    books.append(Book(**clean))
            except Exception as e:
                console.print(f"  [yellow]⚠ Skipped entity: {str(e)[:100]}[/yellow]")

        return authors, publishers, books

    def _print_summary(self, result: PipelineResult):
        """Print a formatted pipeline summary."""
        table = Table(title="Pipeline Results", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Status", "✅ Success" if result.success else "❌ Failed")
        table.add_row("Authors", str(len(result.final_authors)))
        table.add_row("Publishers", str(len(result.final_publishers)))
        table.add_row("Books", str(len(result.final_books)))
        table.add_row("Total Entities", str(result.total_entities))
        table.add_row("Corrections Applied", str(len(result.corrections)))
        table.add_row("Duration", f"{result.duration_seconds:.2f}s")

        if result.extraction:
            table.add_row("Model Used", result.extraction.model_used)
            table.add_row("Tokens Used", str(result.extraction.tokens_used))

        console.print()
        console.print(table)
