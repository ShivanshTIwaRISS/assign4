"""
Data Screening Pipeline Stage

Context-aware validation and screening of extracted entities.
Checks data quality, validates formats, normalizes values,
and flags issues for the self-correction stage.
"""

from __future__ import annotations

import re
from typing import Optional

from rich.console import Console

from models.schemas import EntityType, ExtractedEntity, ScreeningResult

console = Console()


class DataScreener:
    """
    Context-aware data screening for extracted academic entities.

    Applies domain-specific validation rules:
    - ISBN format validation (ISBN-10 and ISBN-13 checksums)
    - Year range validation (1900-2026)
    - Required field completeness checks
    - Publisher name standardization
    - Author name format validation
    """

    # Known publisher name normalizations
    PUBLISHER_ALIASES = {
        "mit": "MIT Press",
        "mit press": "MIT Press",
        "springer": "Springer Nature",
        "springer nature": "Springer Nature",
        "springer-verlag": "Springer Nature",
        "oreilly": "O'Reilly Media",
        "o'reilly": "O'Reilly Media",
        "o'reilly media": "O'Reilly Media",
        "pearson": "Pearson Education",
        "wiley": "John Wiley & Sons",
        "john wiley": "John Wiley & Sons",
        "cambridge": "Cambridge University Press",
        "oxford": "Oxford University Press",
        "mcgraw-hill": "McGraw-Hill Education",
        "mcgraw hill": "McGraw-Hill Education",
        "elsevier": "Elsevier",
        "academic press": "Academic Press (Elsevier)",
    }

    def screen(self, entity: ExtractedEntity) -> ScreeningResult:
        """
        Screen a single extracted entity for data quality.

        Args:
            entity: The extracted entity to screen.

        Returns:
            ScreeningResult with validation status, issues, and suggested corrections.
        """
        if entity.entity_type == EntityType.AUTHOR:
            return self._screen_author(entity)
        elif entity.entity_type == EntityType.PUBLISHER:
            return self._screen_publisher(entity)
        elif entity.entity_type == EntityType.BOOK:
            return self._screen_book(entity)
        else:
            return ScreeningResult(
                entity_type=entity.entity_type,
                original_data=entity.data,
                is_valid=False,
                issues=[f"Unknown entity type: {entity.entity_type}"],
            )

    def screen_batch(self, entities: list[ExtractedEntity]) -> list[ScreeningResult]:
        """Screen a batch of entities."""
        return [self.screen(entity) for entity in entities]

    # ── Author Screening ──────────────────────

    def _screen_author(self, entity: ExtractedEntity) -> ScreeningResult:
        """Screen an author entity."""
        data = entity.data.copy()
        issues = []
        corrections = {}
        screened = data.copy()

        # Required: name
        name = data.get("name", "").strip()
        if not name:
            issues.append("CRITICAL: Author name is missing")
        elif len(name) < 2:
            issues.append(f"WARNING: Author name too short: '{name}'")
        else:
            # Normalize name formatting
            normalized = self._normalize_name(name)
            if normalized != name:
                corrections["name"] = normalized
                screened["name"] = normalized

        # Validate h_index
        h_index = data.get("h_index")
        if h_index is not None:
            if not isinstance(h_index, (int, float)):
                issues.append(f"WARNING: H-index should be numeric, got: {type(h_index).__name__}")
                try:
                    screened["h_index"] = int(h_index)
                    corrections["h_index"] = int(h_index)
                except (ValueError, TypeError):
                    issues.append("ERROR: Cannot convert h_index to integer")
            elif h_index < 0 or h_index > 500:
                issues.append(f"WARNING: Unusual h_index value: {h_index}")

        # Validate email format
        email = data.get("email", "")
        if email and not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
            issues.append(f"WARNING: Invalid email format: {email}")

        # Check research_areas is a list
        research_areas = data.get("research_areas", [])
        if isinstance(research_areas, str):
            screened["research_areas"] = [a.strip() for a in research_areas.split(",")]
            corrections["research_areas"] = screened["research_areas"]
            issues.append("FIXED: Converted research_areas from string to list")

        is_valid = not any(issue.startswith("CRITICAL") for issue in issues)

        return ScreeningResult(
            entity_type=EntityType.AUTHOR,
            original_data=data,
            is_valid=is_valid,
            issues=issues,
            corrections_suggested=corrections,
            screened_data=screened,
        )

    # ── Publisher Screening ───────────────────

    def _screen_publisher(self, entity: ExtractedEntity) -> ScreeningResult:
        """Screen a publisher entity."""
        data = entity.data.copy()
        issues = []
        corrections = {}
        screened = data.copy()

        # Required: name
        name = data.get("name", "").strip()
        if not name:
            issues.append("CRITICAL: Publisher name is missing")
        else:
            # Normalize publisher name
            normalized = self._normalize_publisher(name)
            if normalized != name:
                corrections["name"] = normalized
                screened["name"] = normalized
                issues.append(f"FIXED: Normalized publisher name: '{name}' → '{normalized}'")

        # Validate founded_year
        year = data.get("founded_year")
        if year is not None:
            if isinstance(year, str):
                try:
                    year = int(year)
                    screened["founded_year"] = year
                    corrections["founded_year"] = year
                except ValueError:
                    issues.append(f"ERROR: Invalid founded_year: {year}")
            if isinstance(year, (int, float)):
                if year < 1400 or year > 2026:
                    issues.append(f"WARNING: Unusual founded_year: {year}")

        # Validate website URL
        website = data.get("website", "")
        if website and not website.startswith(("http://", "https://")):
            corrected_url = f"https://{website}"
            corrections["website"] = corrected_url
            screened["website"] = corrected_url
            issues.append(f"FIXED: Added https:// to website URL")

        # Check specializations is a list
        specs = data.get("specializations", [])
        if isinstance(specs, str):
            screened["specializations"] = [s.strip() for s in specs.split(",")]
            corrections["specializations"] = screened["specializations"]
            issues.append("FIXED: Converted specializations from string to list")

        is_valid = not any(issue.startswith("CRITICAL") for issue in issues)

        return ScreeningResult(
            entity_type=EntityType.PUBLISHER,
            original_data=data,
            is_valid=is_valid,
            issues=issues,
            corrections_suggested=corrections,
            screened_data=screened,
        )

    # ── Book Screening ────────────────────────

    def _screen_book(self, entity: ExtractedEntity) -> ScreeningResult:
        """Screen a book entity."""
        data = entity.data.copy()
        issues = []
        corrections = {}
        screened = data.copy()

        # Required: title
        title = data.get("title", "").strip()
        if not title:
            issues.append("CRITICAL: Book title is missing")

        # Validate ISBN
        isbn = data.get("isbn", "")
        if isbn:
            isbn_clean = isbn.replace("-", "").replace(" ", "")
            isbn_valid, isbn_msg = self._validate_isbn(isbn_clean)
            if not isbn_valid:
                issues.append(f"WARNING: {isbn_msg}")
        else:
            issues.append("INFO: No ISBN provided")

        # Validate year
        year = data.get("year")
        if year is not None:
            if isinstance(year, str):
                try:
                    year = int(year)
                    screened["year"] = year
                    corrections["year"] = year
                except ValueError:
                    issues.append(f"ERROR: Invalid year format: {year}")
            if isinstance(year, (int, float)):
                if year < 1900 or year > 2026:
                    issues.append(f"WARNING: Unusual publication year: {year}")

        # Validate authors is a list
        authors = data.get("authors", [])
        if isinstance(authors, str):
            screened["authors"] = [a.strip() for a in authors.split(",")]
            corrections["authors"] = screened["authors"]
            issues.append("FIXED: Converted authors from string to list")

        # Normalize publisher name
        publisher = data.get("publisher", "")
        if publisher:
            normalized_pub = self._normalize_publisher(publisher)
            if normalized_pub != publisher:
                corrections["publisher"] = normalized_pub
                screened["publisher"] = normalized_pub
                issues.append(f"FIXED: Normalized publisher: '{publisher}' → '{normalized_pub}'")

        # Validate pages
        pages = data.get("pages")
        if pages is not None:
            if isinstance(pages, str):
                try:
                    pages = int(pages)
                    screened["pages"] = pages
                    corrections["pages"] = pages
                except ValueError:
                    issues.append(f"WARNING: Invalid pages value: {pages}")
            if isinstance(pages, (int, float)) and (pages < 1 or pages > 10000):
                issues.append(f"WARNING: Unusual page count: {pages}")

        is_valid = not any(issue.startswith("CRITICAL") for issue in issues)

        return ScreeningResult(
            entity_type=EntityType.BOOK,
            original_data=data,
            is_valid=is_valid,
            issues=issues,
            corrections_suggested=corrections,
            screened_data=screened,
        )

    # ── Helper Methods ────────────────────────

    def _normalize_name(self, name: str) -> str:
        """Normalize author name formatting."""
        # Remove extra whitespace
        name = " ".join(name.split())
        # Title case if all uppercase or all lowercase
        if name.isupper() or name.islower():
            name = name.title()
        return name

    def _normalize_publisher(self, name: str) -> str:
        """Normalize publisher name to standard form."""
        lookup = name.lower().strip()
        return self.PUBLISHER_ALIASES.get(lookup, name)

    def _validate_isbn(self, isbn: str) -> tuple[bool, str]:
        """
        Validate ISBN-10 or ISBN-13 checksum.

        Returns (is_valid, message).
        """
        if len(isbn) == 10:
            return self._validate_isbn10(isbn)
        elif len(isbn) == 13:
            return self._validate_isbn13(isbn)
        else:
            return False, f"Invalid ISBN length: {len(isbn)} (expected 10 or 13)"

    def _validate_isbn10(self, isbn: str) -> tuple[bool, str]:
        """Validate ISBN-10 checksum."""
        if not isbn[:9].isdigit():
            return False, f"ISBN-10 contains non-digit characters: {isbn}"
        total = 0
        for i, ch in enumerate(isbn[:9]):
            total += int(ch) * (10 - i)
        check = isbn[9]
        if check in ("X", "x"):
            total += 10
        elif check.isdigit():
            total += int(check)
        else:
            return False, f"Invalid ISBN-10 check digit: {check}"
        if total % 11 == 0:
            return True, "Valid ISBN-10"
        return False, f"ISBN-10 checksum failed for: {isbn}"

    def _validate_isbn13(self, isbn: str) -> tuple[bool, str]:
        """Validate ISBN-13 checksum."""
        if not isbn.isdigit():
            return False, f"ISBN-13 contains non-digit characters: {isbn}"
        total = sum(
            int(ch) * (1 if i % 2 == 0 else 3)
            for i, ch in enumerate(isbn)
        )
        if total % 10 == 0:
            return True, "Valid ISBN-13"
        return False, f"ISBN-13 checksum failed for: {isbn}"
