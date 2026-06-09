"""
Scraper Agent — Web Exploration Specialist

Autonomously explores web pages to discover and collect raw content about
authors, publishers, and academic books from seed URLs.
"""

from __future__ import annotations

from crewai import Agent

from config.settings import settings
from tools.web_tools import WebScraperTool, LinkExtractorTool


def create_scraper_agent() -> Agent:
    """Create the web scraper agent."""
    return Agent(
        role="Web Exploration Specialist",
        goal=(
            "Systematically explore academic web pages, author profiles, "
            "publisher directories, and book listing sites to collect raw "
            "content about authors, publishers, and academic books."
        ),
        backstory=(
            "You are an expert web crawler specialized in academic content. "
            "You know how to navigate university websites, publisher catalogs, "
            "open library APIs, and research databases. You efficiently extract "
            "relevant content while filtering out noise like ads and navigation menus. "
            "You always follow links that could lead to valuable academic data."
        ),
        tools=[WebScraperTool(), LinkExtractorTool()],
        verbose=settings.AGENT_VERBOSE,
        max_iter=settings.AGENT_MAX_ITER,
        allow_delegation=False,
    )
