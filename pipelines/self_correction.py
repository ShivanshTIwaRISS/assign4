"""
Self-Correction Engine

Implements automatic error detection and retry logic for LLM pipelines.
When extraction or screening fails, the engine:
1. Captures error context
2. Generates a corrective prompt with error details
3. Re-runs the LLM with enhanced instructions
4. Validates the corrected output
5. Retries up to N times with exponential backoff
6. Logs the full correction chain
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel

from config.settings import settings
from models.schemas import CorrectionRecord, ExtractionResult

console = Console()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Self-correction prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORRECTION_PROMPT = """You are a self-correcting AI system. Your previous attempt to extract entities from text FAILED.

PREVIOUS ERROR:
{error}

PREVIOUS OUTPUT (if any):
{previous_output}

ORIGINAL TEXT:
{original_text}

Please try again, carefully fixing the issues mentioned above.

IMPORTANT RULES:
1. Return ONLY valid JSON — no markdown, no extra text
2. Use this exact structure:
{{
    "authors": [{{"name": "...", "affiliation": "...", "research_areas": [...], "confidence": 0.9}}],
    "publishers": [{{"name": "...", "publisher_type": "...", "location": "...", "confidence": 0.9}}],
    "books": [{{"title": "...", "authors": [...], "publisher": "...", "isbn": "...", "year": 2024, "confidence": 0.9}}]
}}
3. Extract ONLY entities clearly present in the text
4. Every entity MUST have a "confidence" score between 0.0 and 1.0

CORRECTED JSON OUTPUT:"""


class SelfCorrectionEngine:
    """
    Automatically detects and corrects LLM pipeline errors.

    Supports:
    - JSON parsing error correction
    - Missing field correction
    - Schema validation correction
    - Exponential backoff retry
    - Full correction chain logging
    """

    def __init__(self):
        self._llm = None
        self.correction_history: list[CorrectionRecord] = []

    def _get_llm(self):
        """Lazy-initialize the LLM for corrections."""
        if self._llm is None and settings.is_live_mode:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=settings.CORRECTION_TEMPERATURE,
                api_key=settings.OPENAI_API_KEY,
            )
        return self._llm

    def correct_extraction(
        self,
        original_text: str,
        error: str,
        previous_output: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> tuple[Optional[str], list[CorrectionRecord]]:
        """
        Attempt to correct a failed extraction.

        Args:
            original_text: The original input text that was being processed.
            error: The error message from the failed extraction.
            previous_output: The raw LLM output from the failed attempt.
            max_retries: Maximum retry attempts (defaults to settings.MAX_RETRIES).

        Returns:
            Tuple of (corrected_output, correction_records).
            corrected_output is None if all retries failed.
        """
        max_retries = max_retries or settings.MAX_RETRIES
        records: list[CorrectionRecord] = []

        if not settings.is_live_mode:
            return self._mock_correct(error, records)

        return self._live_correct(
            original_text, error, previous_output, max_retries, records
        )

    def _live_correct(
        self,
        original_text: str,
        error: str,
        previous_output: Optional[str],
        max_retries: int,
        records: list[CorrectionRecord],
    ) -> tuple[Optional[str], list[CorrectionRecord]]:
        """Live self-correction with OpenAI API."""
        from langchain.schema import HumanMessage

        current_error = error
        current_output = previous_output or ""

        for attempt in range(1, max_retries + 1):
            console.print(
                f"  [yellow]🔄 Self-correction attempt {attempt}/{max_retries}...[/yellow]"
            )

            # Generate corrective prompt
            prompt = CORRECTION_PROMPT.format(
                error=current_error,
                previous_output=current_output[:2000],
                original_text=original_text[:4000],
            )

            record = CorrectionRecord(
                attempt_number=attempt,
                original_error=current_error,
                correction_prompt=prompt[:500] + "...",
            )

            try:
                # Exponential backoff
                if attempt > 1:
                    delay = settings.RETRY_DELAY * (2 ** (attempt - 1))
                    console.print(f"  [dim]Waiting {delay:.1f}s before retry...[/dim]")
                    time.sleep(delay)

                llm = self._get_llm()
                response = llm.invoke([HumanMessage(content=prompt)])
                corrected_output = response.content

                # Validate the corrected output
                self._validate_json_output(corrected_output)

                record.corrected_output = corrected_output
                record.success = True
                records.append(record)

                console.print(f"  [green]✅ Self-correction succeeded on attempt {attempt}[/green]")
                self.correction_history.extend(records)
                return corrected_output, records

            except Exception as e:
                current_error = str(e)
                current_output = corrected_output if 'corrected_output' in dir() else ""
                record.corrected_output = current_output if current_output else None
                record.success = False
                records.append(record)
                console.print(f"  [red]❌ Attempt {attempt} failed: {str(e)[:100]}[/red]")

        console.print(f"  [bold red]⛔ All {max_retries} correction attempts failed[/bold red]")
        self.correction_history.extend(records)
        return None, records

    def _mock_correct(
        self,
        error: str,
        records: list[CorrectionRecord],
    ) -> tuple[Optional[str], list[CorrectionRecord]]:
        """Mock self-correction for demo mode."""
        console.print("  [yellow]🔄 Self-correction attempt 1/3 (demo)...[/yellow]")
        time.sleep(0.5)

        # Simulate first attempt failing
        record1 = CorrectionRecord(
            attempt_number=1,
            original_error=error,
            correction_prompt="[Demo] Corrective prompt v1",
            corrected_output=None,
            success=False,
        )
        records.append(record1)
        console.print("  [red]❌ Attempt 1 failed (simulated)[/red]")

        time.sleep(0.5)
        console.print("  [yellow]🔄 Self-correction attempt 2/3 (demo)...[/yellow]")

        # Simulate second attempt succeeding
        corrected = json.dumps({
            "authors": [
                {
                    "name": "Andrew Ng",
                    "affiliation": "Stanford University",
                    "research_areas": ["Machine Learning", "Deep Learning"],
                    "confidence": 0.92,
                }
            ],
            "publishers": [
                {
                    "name": "MIT Press",
                    "publisher_type": "University Press",
                    "location": "Cambridge, MA",
                    "confidence": 0.90,
                }
            ],
            "books": [
                {
                    "title": "Deep Learning",
                    "authors": ["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"],
                    "publisher": "MIT Press",
                    "isbn": "978-0262035613",
                    "year": 2016,
                    "confidence": 0.95,
                }
            ],
        }, indent=2)

        record2 = CorrectionRecord(
            attempt_number=2,
            original_error="JSON parse error (simulated)",
            correction_prompt="[Demo] Corrective prompt v2",
            corrected_output=corrected,
            success=True,
        )
        records.append(record2)

        console.print("  [green]✅ Self-correction succeeded on attempt 2[/green]")
        self.correction_history.extend(records)
        return corrected, records

    def _validate_json_output(self, output: str) -> dict:
        """Validate that the output is valid JSON with expected structure."""
        # Find JSON in the output
        start = output.find("{")
        end = output.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON object found in output")

        data = json.loads(output[start:end])

        # Validate structure
        if not isinstance(data, dict):
            raise ValueError("Output is not a JSON object")

        expected_keys = {"authors", "publishers", "books"}
        found_keys = set(data.keys()) & expected_keys
        if not found_keys:
            raise ValueError(f"Missing expected keys. Found: {list(data.keys())}")

        # Validate each entity has required fields
        for author in data.get("authors", []):
            if "name" not in author:
                raise ValueError("Author missing required 'name' field")

        for book in data.get("books", []):
            if "title" not in book:
                raise ValueError("Book missing required 'title' field")

        for pub in data.get("publishers", []):
            if "name" not in pub:
                raise ValueError("Publisher missing required 'name' field")

        return data

    def execute_with_correction(
        self,
        func: Callable,
        original_text: str,
        *args,
        **kwargs,
    ) -> tuple[any, list[CorrectionRecord]]:
        """
        Execute a function with automatic self-correction on failure.

        Args:
            func: The function to execute (should raise on failure).
            original_text: The original input text for correction context.
            *args, **kwargs: Arguments to pass to the function.

        Returns:
            Tuple of (result, correction_records).
        """
        records = []

        try:
            result = func(*args, **kwargs)
            return result, records
        except Exception as e:
            console.print(
                Panel(
                    f"[red]Pipeline error detected:[/red] {str(e)[:200]}",
                    title="Self-Correction Triggered",
                    border_style="yellow",
                )
            )
            corrected, records = self.correct_extraction(
                original_text=original_text,
                error=str(e),
            )
            if corrected is not None:
                return corrected, records
            raise RuntimeError(
                f"Self-correction exhausted after {settings.MAX_RETRIES} attempts"
            ) from e
