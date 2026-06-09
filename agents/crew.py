"""
Research & Discovery Crew — Multi-Agent Orchestrator

Orchestrates the Scraper → Parser → Researcher → Validator agent pipeline
using CrewAI for autonomous academic entity discovery.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import settings

console = Console()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mock crew for demo mode (no API key)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MOCK_CREW_OUTPUT = {
    "authors": [
        {
            "name": "Andrew Ng",
            "affiliation": "Stanford University, DeepLearning.AI",
            "research_areas": ["Machine Learning", "Deep Learning", "AI Education"],
            "publications": ["Machine Learning Yearning", "CS229 Lecture Notes"],
            "h_index": 165,
            "profile_url": "https://scholar.google.com/citations?user=mG4imMEAAAAJ",
            "confidence_score": 0.95,
        },
        {
            "name": "Yoshua Bengio",
            "affiliation": "Université de Montréal, Mila",
            "research_areas": ["Deep Learning", "Neural Networks", "Generative Models"],
            "publications": ["Deep Learning (MIT Press)", "Generative Deep Learning"],
            "h_index": 202,
            "profile_url": "https://scholar.google.com/citations?user=kukA0LcAAAAJ",
            "confidence_score": 0.97,
        },
        {
            "name": "Geoffrey Hinton",
            "affiliation": "University of Toronto, Google Brain",
            "research_areas": ["Neural Networks", "Deep Learning", "Boltzmann Machines"],
            "publications": ["Learning representations by back-propagating errors"],
            "h_index": 186,
            "profile_url": "https://scholar.google.com/citations?user=JicYPdAAAAAJ",
            "confidence_score": 0.93,
        },
    ],
    "publishers": [
        {
            "name": "MIT Press",
            "publisher_type": "University Press",
            "location": "Cambridge, Massachusetts, USA",
            "website": "https://mitpress.mit.edu",
            "specializations": ["Computer Science", "AI", "Cognitive Science", "Economics"],
            "founded_year": 1962,
            "confidence_score": 0.98,
        },
        {
            "name": "Springer Nature",
            "publisher_type": "Academic Publisher",
            "location": "Berlin, Germany",
            "website": "https://www.springer.com",
            "specializations": ["Science", "Technology", "Medicine", "Engineering"],
            "founded_year": 1842,
            "confidence_score": 0.96,
        },
        {
            "name": "O'Reilly Media",
            "publisher_type": "Technical Publisher",
            "location": "Sebastopol, California, USA",
            "website": "https://www.oreilly.com",
            "specializations": ["Programming", "Data Science", "AI", "Cloud Computing"],
            "founded_year": 1978,
            "confidence_score": 0.94,
        },
    ],
    "books": [
        {
            "title": "Deep Learning",
            "authors": ["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"],
            "publisher": "MIT Press",
            "isbn": "978-0262035613",
            "year": 2016,
            "subjects": ["Artificial Intelligence", "Machine Learning", "Neural Networks"],
            "pages": 800,
            "confidence_score": 0.99,
        },
        {
            "title": "Pattern Recognition and Machine Learning",
            "authors": ["Christopher M. Bishop"],
            "publisher": "Springer",
            "isbn": "978-0387310732",
            "year": 2006,
            "subjects": ["Machine Learning", "Pattern Recognition", "Bayesian Methods"],
            "pages": 738,
            "confidence_score": 0.97,
        },
        {
            "title": "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow",
            "authors": ["Aurélien Géron"],
            "publisher": "O'Reilly Media",
            "isbn": "978-1492032649",
            "year": 2022,
            "edition": "3rd Edition",
            "subjects": ["Machine Learning", "Deep Learning", "Python", "TensorFlow"],
            "pages": 856,
            "confidence_score": 0.95,
        },
        {
            "title": "Artificial Intelligence: A Modern Approach",
            "authors": ["Stuart Russell", "Peter Norvig"],
            "publisher": "Pearson",
            "isbn": "978-0134610993",
            "year": 2020,
            "edition": "4th Edition",
            "subjects": ["Artificial Intelligence", "Search Algorithms", "Machine Learning"],
            "pages": 1115,
            "confidence_score": 0.98,
        },
    ],
}


class ResearchDiscoveryCrew:
    """
    Multi-agent crew for academic entity discovery.

    Orchestrates four specialized agents:
    1. Scraper Agent → explores web pages
    2. Parser Agent → extracts structured entities
    3. Researcher Agent → enriches & verifies entities
    4. Validator Agent → quality assurance & scoring
    """

    def __init__(self):
        self.results: dict = {}

    def run(self, seed_urls: Optional[list[str]] = None) -> dict:
        """
        Execute the multi-agent exploration pipeline.

        Args:
            seed_urls: Optional list of URLs to explore.
                      Uses default seed URLs if not provided.

        Returns:
            Dictionary with 'authors', 'publishers', and 'books' lists.
        """
        if not settings.is_live_mode:
            return self._run_demo()

        return self._run_live(seed_urls)

    def _run_demo(self) -> dict:
        """Run in demo mode with simulated agent execution."""
        console.print(
            Panel(
                "[bold yellow]🟡 DEMO MODE[/bold yellow]\n"
                "Running with simulated agents (no API key configured).\n"
                "Set OPENAI_API_KEY in .env for live mode.",
                title="Agent Mode",
                border_style="yellow",
            )
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Simulate agent execution stages
            task = progress.add_task("🔍 Scraper Agent: Exploring seed URLs...", total=None)
            time.sleep(1.5)
            progress.update(task, description="✅ Scraper Agent: Found 6 content pages")

            task2 = progress.add_task("📋 Parser Agent: Extracting entities...", total=None)
            time.sleep(1.5)
            progress.update(task2, description="✅ Parser Agent: Extracted 10 entities")

            task3 = progress.add_task("🔬 Researcher Agent: Enriching data...", total=None)
            time.sleep(1.5)
            progress.update(task3, description="✅ Researcher Agent: Verified 10 entities")

            task4 = progress.add_task("✔️  Validator Agent: Quality check...", total=None)
            time.sleep(1.5)
            progress.update(task4, description="✅ Validator Agent: All entities validated")

        self.results = MOCK_CREW_OUTPUT
        self._print_summary()
        return self.results

    def _run_live(self, seed_urls: Optional[list[str]] = None) -> dict:
        """Run with live CrewAI agents and OpenAI API."""
        # crewai >=0.80 ships Process inside crewai; 0.11.x may not
        try:
            from crewai import Crew, Task, Process
        except ImportError:
            from crewai import Crew, Task  # type: ignore
            Process = None  # type: ignore

        from agents.scraper_agent import create_scraper_agent
        from agents.parser_agent import create_parser_agent
        from agents.researcher_agent import create_researcher_agent
        from agents.validator_agent import create_validator_agent

        console.print(
            Panel(
                "[bold green]🟢 LIVE MODE[/bold green]\n"
                f"Model: {settings.OPENAI_MODEL}\n"
                "Agents will make real API calls and web requests.",
                title="Agent Mode",
                border_style="green",
            )
        )

        # Load seed URLs
        if not seed_urls:
            import json as _json
            from config.settings import DATA_DIR
            seed_file = DATA_DIR / "seed_urls.json"
            if seed_file.exists():
                with open(seed_file) as f:
                    seed_data = _json.load(f)
                seed_urls = [s["url"] for s in seed_data[:3]]
            else:
                seed_urls = [
                    "https://openlibrary.org/subjects/artificial_intelligence.json?limit=5"
                ]

        urls_text = "\n".join(f"- {url}" for url in seed_urls)

        # Create agents
        scraper = create_scraper_agent()
        parser = create_parser_agent()
        researcher = create_researcher_agent()
        validator = create_validator_agent()

        # Define tasks
        scrape_task = Task(
            description=(
                f"Explore the following seed URLs and collect raw content about "
                f"academic authors, publishers, and books:\n{urls_text}\n\n"
                f"For each URL, scrape the page content and extract any relevant "
                f"links that may lead to more academic data. Return all collected "
                f"raw content organized by source URL."
            ),
            expected_output=(
                "A structured collection of raw web content from academic pages, "
                "organized by source URL, containing author profiles, book listings, "
                "and publisher information."
            ),
            agent=scraper,
        )

        parse_task = Task(
            description=(
                "Analyze the raw web content collected by the scraper and extract "
                "structured entities. For each entity found, extract:\n\n"
                "AUTHORS: name, affiliation, research_areas, publications, h_index\n"
                "PUBLISHERS: name, publisher_type, location, website, specializations\n"
                "BOOKS: title, authors, publisher, isbn, year, subjects\n\n"
                "Output as a JSON object with 'authors', 'publishers', and 'books' arrays."
            ),
            expected_output=(
                "A JSON object containing arrays of structured entities: "
                '{"authors": [...], "publishers": [...], "books": [...]}'
            ),
            agent=parser,
            context=[scrape_task],
        )

        research_task = Task(
            description=(
                "Take the parsed entities and enrich them:\n"
                "1. Verify book ISBNs using the isbn_lookup tool\n"
                "2. Search for additional author information using the author_search tool\n"
                "3. Fill in missing fields where possible\n"
                "4. Cross-reference data between entities\n\n"
                "Return the enriched JSON with all verified and additional data."
            ),
            expected_output=(
                "An enriched JSON object with verified and augmented entity data, "
                "including any additional fields discovered through research."
            ),
            agent=researcher,
            context=[parse_task],
        )

        validate_task = Task(
            description=(
                "Validate all entities for data quality:\n"
                "1. Check completeness — flag entities with missing required fields\n"
                "2. Validate formats — ISBN checksums, year ranges (1900-2026), URLs\n"
                "3. Detect duplicates — merge entities that refer to the same real-world entity\n"
                "4. Assign confidence_score (0.0-1.0) to each entity based on data quality\n\n"
                "Return the final validated JSON dataset with confidence scores.\n"
                "The output MUST be valid JSON with this structure:\n"
                '{"authors": [...], "publishers": [...], "books": [...]}\n'
                "Each entity must include a confidence_score field."
            ),
            expected_output=(
                'A validated JSON object: {"authors": [...], "publishers": [...], "books": [...]}'
                " where each entity has a confidence_score between 0.0 and 1.0."
            ),
            agent=validator,
            context=[research_task],
        )

        # Create and run the crew
        crew_kwargs = {
            "agents": [scraper, parser, researcher, validator],
            "tasks": [scrape_task, parse_task, research_task, validate_task],
            "verbose": True,
        }
        if Process:
            crew_kwargs["process"] = Process.sequential
            
        crew = Crew(**crew_kwargs)

        console.print("\n[bold cyan]🚀 Launching multi-agent crew...[/bold cyan]\n")
        result = crew.kickoff()

        # Parse the crew output
        self.results = self._parse_crew_output(str(result))
        self._print_summary()
        return self.results

    def _parse_crew_output(self, output: str) -> dict:
        """Parse the raw crew output into structured data."""
        try:
            # Try to find JSON in the output
            start = output.find("{")
            end = output.rfind("}") + 1
            if start != -1 and end > start:
                json_str = output[start:end]
                data = json.loads(json_str)
                return {
                    "authors": data.get("authors", []),
                    "publishers": data.get("publishers", []),
                    "books": data.get("books", []),
                }
        except json.JSONDecodeError:
            pass

        # Fallback: return raw output wrapped
        return {
            "authors": [],
            "publishers": [],
            "books": [],
            "raw_output": output,
        }

    def _print_summary(self):
        """Print a summary of the crew results."""
        authors = self.results.get("authors", [])
        publishers = self.results.get("publishers", [])
        books = self.results.get("books", [])

        summary = (
            f"[bold green]📊 Crew Execution Complete[/bold green]\n\n"
            f"  👤 Authors discovered:    [cyan]{len(authors)}[/cyan]\n"
            f"  🏢 Publishers discovered: [cyan]{len(publishers)}[/cyan]\n"
            f"  📚 Books discovered:      [cyan]{len(books)}[/cyan]\n"
            f"  📈 Total entities:        [bold cyan]{len(authors) + len(publishers) + len(books)}[/bold cyan]"
        )

        console.print(Panel(summary, title="Multi-Agent Results", border_style="green"))

        # Print entity details
        if authors:
            console.print("\n[bold]👤 Authors:[/bold]")
            for a in authors:
                name = a.get("name", "Unknown") if isinstance(a, dict) else str(a)
                affil = a.get("affiliation", "—") if isinstance(a, dict) else ""
                score = a.get("confidence_score", 0) if isinstance(a, dict) else 0
                console.print(f"  • {name} ({affil}) [dim]confidence: {score:.2f}[/dim]")

        if books:
            console.print("\n[bold]📚 Books:[/bold]")
            for b in books:
                title = b.get("title", "Unknown") if isinstance(b, dict) else str(b)
                isbn = b.get("isbn", "—") if isinstance(b, dict) else ""
                score = b.get("confidence_score", 0) if isinstance(b, dict) else 0
                console.print(f"  • {title} (ISBN: {isbn}) [dim]confidence: {score:.2f}[/dim]")

        if publishers:
            console.print("\n[bold]🏢 Publishers:[/bold]")
            for p in publishers:
                name = p.get("name", "Unknown") if isinstance(p, dict) else str(p)
                ptype = p.get("publisher_type", "—") if isinstance(p, dict) else ""
                score = p.get("confidence_score", 0) if isinstance(p, dict) else 0
                console.print(f"  • {name} ({ptype}) [dim]confidence: {score:.2f}[/dim]")
