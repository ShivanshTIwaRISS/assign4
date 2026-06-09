"""
Researcher Agent — Deep Research & Enrichment Specialist

Enriches extracted entities with additional data by cross-referencing
multiple sources, verifying ISBNs, and finding additional publications.
"""

from __future__ import annotations

from crewai import Agent

from config.settings import settings
from tools.search_tools import ISBNLookupTool, AuthorSearchTool


def create_researcher_agent() -> Agent:
    """Create the researcher agent."""
    return Agent(
        role="Deep Research & Enrichment Specialist",
        goal=(
            "Enrich and verify extracted entities by cross-referencing multiple "
            "academic sources. Verify ISBNs using the ISBN lookup tool, find "
            "additional author publications and metrics, and validate publisher "
            "information. Add missing fields and increase data completeness."
        ),
        backstory=(
            "You are a meticulous academic researcher who excels at finding "
            "and verifying information across multiple databases. You use ISBN "
            "lookup tools to verify book data, search author databases for "
            "additional publications and metrics, and cross-reference publisher "
            "catalogs. You never fabricate data — you only add verified information."
        ),
        tools=[ISBNLookupTool(), AuthorSearchTool()],
        verbose=settings.AGENT_VERBOSE,
        max_iter=settings.AGENT_MAX_ITER,
        allow_delegation=False,
    )
