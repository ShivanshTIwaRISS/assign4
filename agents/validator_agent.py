"""
Validator Agent — Data Quality Assurance Specialist

Final quality gate that validates data completeness, checks consistency,
deduplicates entities, and assigns confidence scores.
"""

from __future__ import annotations

from crewai import Agent

from config.settings import settings


def create_validator_agent() -> Agent:
    """Create the validator agent."""
    return Agent(
        role="Data Quality Assurance Specialist",
        goal=(
            "Validate all extracted and enriched entities for data quality. "
            "Check completeness (are all required fields present?), consistency "
            "(do cross-referenced fields match?), and uniqueness (are there "
            "duplicates?). Assign a confidence score (0.0-1.0) to each entity. "
            "Produce a final validated dataset with quality metrics."
        ),
        backstory=(
            "You are an expert data quality analyst specializing in academic "
            "metadata. You validate ISBNs against checksum rules, verify that "
            "publication years are reasonable, check author name formatting, "
            "ensure publisher names are standardized, and detect duplicate "
            "entities. You assign confidence scores based on data completeness "
            "and verification status. Your output is always a clean, validated "
            "JSON dataset ready for indexing."
        ),
        tools=[],
        verbose=settings.AGENT_VERBOSE,
        max_iter=settings.AGENT_MAX_ITER,
        allow_delegation=False,
    )
