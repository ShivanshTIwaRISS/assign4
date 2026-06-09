"""
Parser Agent — Structural Entity Parser

Takes raw web content and extracts structured entities (authors, publishers,
books) using LLM-based understanding with defined schemas.
"""

from __future__ import annotations

from crewai import Agent

from config.settings import settings
from tools.web_tools import PageContentTool


def create_parser_agent() -> Agent:
    """Create the entity parser agent."""
    return Agent(
        role="Structural Entity Parser",
        goal=(
            "Analyze raw web content and extract structured entities: "
            "authors (with name, affiliation, research areas), "
            "publishers (with name, type, location, specializations), and "
            "books (with title, ISBN, authors, publisher, year, subjects). "
            "Output clean, structured data for each entity found."
        ),
        backstory=(
            "You are an expert at understanding unstructured academic web pages "
            "and extracting structured information from them. You can identify "
            "author profiles, book listings, and publisher information even when "
            "the data is presented in varied formats. You always maintain data "
            "integrity and flag uncertain extractions. You produce structured JSON "
            "output following precise schemas."
        ),
        tools=[PageContentTool()],
        verbose=settings.AGENT_VERBOSE,
        max_iter=settings.AGENT_MAX_ITER,
        allow_delegation=False,
    )
