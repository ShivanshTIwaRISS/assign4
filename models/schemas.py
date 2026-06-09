"""
Pydantic data models for the AI Research & Discovery Platform.

Domain entities: Author, Publisher, Book
Pipeline entities: ExtractionResult, ScreeningResult, CorrectionRecord, PipelineResult
Search entities: SearchResult, RAGResponse
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Domain Entities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""
    AUTHOR = "author"
    PUBLISHER = "publisher"
    BOOK = "book"


class Author(BaseModel):
    """Represents an academic or textbook author."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(..., description="Full name of the author")
    affiliation: Optional[str] = Field(None, description="University or institution")
    research_areas: list[str] = Field(default_factory=list, description="Areas of expertise")
    publications: list[str] = Field(default_factory=list, description="Notable publications")
    email: Optional[str] = Field(None, description="Contact email if available")
    h_index: Optional[int] = Field(None, description="H-index if available")
    profile_url: Optional[str] = Field(None, description="Academic profile URL")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Data quality confidence")

    def to_text(self) -> str:
        """Convert to searchable text representation."""
        parts = [f"Author: {self.name}"]
        if self.affiliation:
            parts.append(f"Affiliation: {self.affiliation}")
        if self.research_areas:
            parts.append(f"Research Areas: {', '.join(self.research_areas)}")
        if self.publications:
            parts.append(f"Publications: {', '.join(self.publications[:5])}")
        if self.h_index is not None:
            parts.append(f"H-Index: {self.h_index}")
        return " | ".join(parts)


class Publisher(BaseModel):
    """Represents an academic publisher."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(..., description="Publisher name")
    publisher_type: Optional[str] = Field(None, description="e.g., Academic, Trade, University Press")
    location: Optional[str] = Field(None, description="Headquarters location")
    website: Optional[str] = Field(None, description="Official website URL")
    specializations: list[str] = Field(default_factory=list, description="Subject specializations")
    catalog_size: Optional[int] = Field(None, description="Approximate catalog size")
    founded_year: Optional[int] = Field(None, description="Year of establishment")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Data quality confidence")

    def to_text(self) -> str:
        """Convert to searchable text representation."""
        parts = [f"Publisher: {self.name}"]
        if self.publisher_type:
            parts.append(f"Type: {self.publisher_type}")
        if self.location:
            parts.append(f"Location: {self.location}")
        if self.specializations:
            parts.append(f"Specializations: {', '.join(self.specializations)}")
        if self.founded_year:
            parts.append(f"Founded: {self.founded_year}")
        return " | ".join(parts)


class Book(BaseModel):
    """Represents an academic or textbook publication."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = Field(..., description="Book title")
    authors: list[str] = Field(default_factory=list, description="Author name(s)")
    publisher: Optional[str] = Field(None, description="Publisher name")
    isbn: Optional[str] = Field(None, description="ISBN-10 or ISBN-13")
    year: Optional[int] = Field(None, description="Publication year")
    edition: Optional[str] = Field(None, description="Edition information")
    subjects: list[str] = Field(default_factory=list, description="Subject areas")
    language: str = Field(default="English", description="Language")
    pages: Optional[int] = Field(None, description="Number of pages")
    description: Optional[str] = Field(None, description="Brief description")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Data quality confidence")

    def to_text(self) -> str:
        """Convert to searchable text representation."""
        parts = [f"Book: {self.title}"]
        if self.authors:
            parts.append(f"Authors: {', '.join(self.authors)}")
        if self.publisher:
            parts.append(f"Publisher: {self.publisher}")
        if self.isbn:
            parts.append(f"ISBN: {self.isbn}")
        if self.year:
            parts.append(f"Year: {self.year}")
        if self.subjects:
            parts.append(f"Subjects: {', '.join(self.subjects)}")
        if self.description:
            parts.append(f"Description: {self.description[:200]}")
        return " | ".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline Entities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ExtractedEntity(BaseModel):
    """A single entity extracted by the LLM pipeline."""
    entity_type: EntityType
    data: dict = Field(..., description="Raw extracted data")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_text: Optional[str] = Field(None, description="Source text snippet")


class ExtractionResult(BaseModel):
    """Result from the entity extraction pipeline stage."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    raw_llm_output: Optional[str] = Field(None, description="Raw LLM response")
    model_used: str = Field(default="gpt-4o-mini")
    tokens_used: int = Field(default=0)
    duration_seconds: float = Field(default=0.0)
    success: bool = Field(default=True)
    error_message: Optional[str] = None


class ScreeningResult(BaseModel):
    """Result from the data screening pipeline stage."""
    entity_type: EntityType
    original_data: dict
    is_valid: bool = Field(default=True)
    issues: list[str] = Field(default_factory=list, description="List of validation issues")
    corrections_suggested: dict = Field(default_factory=dict, description="Suggested corrections")
    screened_data: dict = Field(default_factory=dict, description="Data after screening")


class CorrectionRecord(BaseModel):
    """Record of a self-correction attempt."""
    attempt_number: int
    original_error: str
    correction_prompt: str
    corrected_output: Optional[str] = None
    success: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)


class PipelineResult(BaseModel):
    """Complete result from a full pipeline run."""
    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    input_text: str
    extraction: Optional[ExtractionResult] = None
    screening_results: list[ScreeningResult] = Field(default_factory=list)
    corrections: list[CorrectionRecord] = Field(default_factory=list)
    final_authors: list[Author] = Field(default_factory=list)
    final_publishers: list[Publisher] = Field(default_factory=list)
    final_books: list[Book] = Field(default_factory=list)
    total_entities: int = Field(default=0)
    success: bool = Field(default=True)
    error_message: Optional[str] = None
    duration_seconds: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def summary(self) -> str:
        return (
            f"Pipeline {self.pipeline_id}: "
            f"{len(self.final_authors)} authors, "
            f"{len(self.final_publishers)} publishers, "
            f"{len(self.final_books)} books | "
            f"{len(self.corrections)} corrections | "
            f"{'✅' if self.success else '❌'}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Search Entities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SearchResult(BaseModel):
    """A single result from the semantic search engine."""
    entity_type: EntityType
    entity_id: str
    text: str
    metadata: dict = Field(default_factory=dict)
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)


class RAGResponse(BaseModel):
    """Response from the RAG pipeline."""
    query: str
    answer: str
    sources: list[SearchResult] = Field(default_factory=list)
    model_used: str = Field(default="gpt-4o-mini")
    tokens_used: int = Field(default=0)
    duration_seconds: float = Field(default=0.0)
