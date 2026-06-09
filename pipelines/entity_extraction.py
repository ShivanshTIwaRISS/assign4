"""
Entity Extraction Pipeline Stage

LLM-powered extraction of structured entities (authors, publishers, books)
from unstructured text using LangChain with Pydantic output parsing.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from rich.console import Console

from config.settings import settings
from models.schemas import EntityType, ExtractedEntity, ExtractionResult

console = Console()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Extraction prompt template
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXTRACTION_PROMPT = """You are an expert academic data extractor. Analyze the following text and extract ALL entities you can find.

Extract these entity types:
1. **AUTHORS**: Academic/textbook authors with fields:
   - name (required), affiliation, research_areas (list), publications (list), h_index, email, profile_url

2. **PUBLISHERS**: Academic publishers with fields:
   - name (required), publisher_type, location, website, specializations (list), founded_year

3. **BOOKS**: Academic books/textbooks with fields:
   - title (required), authors (list), publisher, isbn, year, edition, subjects (list), pages, description

RULES:
- Extract ONLY information explicitly present in the text
- Do NOT invent or hallucinate data
- Assign a confidence score (0.0-1.0) based on how clearly the entity is described
- If a field is not found in the text, omit it

Return your response as a JSON object with this EXACT structure:
{{
    "authors": [
        {{"name": "...", "affiliation": "...", "research_areas": [...], "confidence": 0.95}},
        ...
    ],
    "publishers": [
        {{"name": "...", "publisher_type": "...", "location": "...", "confidence": 0.9}},
        ...
    ],
    "books": [
        {{"title": "...", "authors": [...], "publisher": "...", "isbn": "...", "confidence": 0.85}},
        ...
    ]
}}

TEXT TO ANALYZE:
{text}

JSON OUTPUT:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mock extraction results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MOCK_EXTRACTION = {
    "authors": [
        {
            "name": "Andrew Ng",
            "affiliation": "Stanford University",
            "research_areas": ["Machine Learning", "Deep Learning"],
            "confidence": 0.95,
        },
        {
            "name": "Ian Goodfellow",
            "affiliation": "DeepMind",
            "research_areas": ["Generative Adversarial Networks", "Deep Learning"],
            "confidence": 0.88,
        },
    ],
    "publishers": [
        {
            "name": "MIT Press",
            "publisher_type": "University Press",
            "location": "Cambridge, MA",
            "confidence": 0.92,
        },
    ],
    "books": [
        {
            "title": "Deep Learning",
            "authors": ["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"],
            "publisher": "MIT Press",
            "isbn": "978-0262035613",
            "year": 2016,
            "confidence": 0.97,
        },
    ],
}


class EntityExtractor:
    """
    Extracts structured entities from unstructured text using LLMs.

    Uses LangChain with OpenAI to parse text into Author, Publisher,
    and Book entities with confidence scores.
    """

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        """Lazy-initialize the LLM client."""
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=settings.EXTRACTION_TEMPERATURE,
                api_key=settings.OPENAI_API_KEY,
            )
        return self._llm

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract entities from text.

        Args:
            text: Unstructured text containing academic entity information.

        Returns:
            ExtractionResult with extracted entities and metadata.
        """
        start_time = time.time()

        if not settings.is_live_mode:
            return self._mock_extract(text, start_time)

        return self._live_extract(text, start_time)

    def _live_extract(self, text: str, start_time: float) -> ExtractionResult:
        """Extract using live OpenAI API."""
        try:
            from langchain.schema import HumanMessage

            llm = self._get_llm()
            prompt = EXTRACTION_PROMPT.format(text=text[:6000])

            response = llm.invoke([HumanMessage(content=prompt)])
            raw_output = response.content

            # Parse the JSON response
            entities = self._parse_llm_output(raw_output)

            duration = time.time() - start_time
            tokens = response.response_metadata.get("token_usage", {}).get("total_tokens", 0)

            return ExtractionResult(
                entities=entities,
                raw_llm_output=raw_output,
                model_used=settings.OPENAI_MODEL,
                tokens_used=tokens,
                duration_seconds=round(duration, 2),
                success=True,
            )

        except Exception as e:
            duration = time.time() - start_time
            return ExtractionResult(
                success=False,
                error_message=str(e),
                duration_seconds=round(duration, 2),
            )

    def _mock_extract(self, text: str, start_time: float) -> ExtractionResult:
        """Extract using mock data for demo mode."""
        time.sleep(0.5)  # Simulate processing

        entities = []
        for author in MOCK_EXTRACTION["authors"]:
            entities.append(ExtractedEntity(
                entity_type=EntityType.AUTHOR,
                data=author,
                confidence=author.get("confidence", 0.8),
                source_text=text[:100],
            ))
        for pub in MOCK_EXTRACTION["publishers"]:
            entities.append(ExtractedEntity(
                entity_type=EntityType.PUBLISHER,
                data=pub,
                confidence=pub.get("confidence", 0.8),
                source_text=text[:100],
            ))
        for book in MOCK_EXTRACTION["books"]:
            entities.append(ExtractedEntity(
                entity_type=EntityType.BOOK,
                data=book,
                confidence=book.get("confidence", 0.8),
                source_text=text[:100],
            ))

        duration = time.time() - start_time
        return ExtractionResult(
            entities=entities,
            raw_llm_output=json.dumps(MOCK_EXTRACTION, indent=2),
            model_used="mock-gpt-4o-mini",
            tokens_used=0,
            duration_seconds=round(duration, 2),
            success=True,
        )

    def _parse_llm_output(self, output: str) -> list[ExtractedEntity]:
        """Parse LLM JSON output into ExtractedEntity objects."""
        entities = []

        # Find JSON in the response
        try:
            start = output.find("{")
            end = output.rfind("}") + 1
            if start == -1 or end <= start:
                raise ValueError("No JSON found in LLM output")

            data = json.loads(output[start:end])
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse LLM output as JSON: {e}\nOutput: {output[:500]}")

        # Parse authors
        for author_data in data.get("authors", []):
            confidence = author_data.pop("confidence", 0.8)
            entities.append(ExtractedEntity(
                entity_type=EntityType.AUTHOR,
                data=author_data,
                confidence=confidence,
            ))

        # Parse publishers
        for pub_data in data.get("publishers", []):
            confidence = pub_data.pop("confidence", 0.8)
            entities.append(ExtractedEntity(
                entity_type=EntityType.PUBLISHER,
                data=pub_data,
                confidence=confidence,
            ))

        # Parse books
        for book_data in data.get("books", []):
            confidence = book_data.pop("confidence", 0.8)
            entities.append(ExtractedEntity(
                entity_type=EntityType.BOOK,
                data=book_data,
                confidence=confidence,
            ))

        return entities
