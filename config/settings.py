"""
Centralized configuration for the AI Research & Discovery Platform.
Loads from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ──────────────────────────────────────────────
# Base Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))


class Settings:
    """Application-wide settings loaded from environment."""

    # ── OpenAI ────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # ── ChromaDB ──────────────────────────────
    CHROMA_PERSIST_DIR: Path = CHROMA_DIR
    CHROMA_COLLECTION_AUTHORS: str = "authors"
    CHROMA_COLLECTION_PUBLISHERS: str = "publishers"
    CHROMA_COLLECTION_BOOKS: str = "books"
    CHROMA_COLLECTION_UNIFIED: str = "unified_entities"

    # ── Pipeline ──────────────────────────────
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY: float = float(os.getenv("RETRY_DELAY", "1.0"))
    EXTRACTION_TEMPERATURE: float = 0.1
    SCREENING_TEMPERATURE: float = 0.0
    CORRECTION_TEMPERATURE: float = 0.3

    # ── Agent ─────────────────────────────────
    AGENT_MAX_ITER: int = 10
    AGENT_VERBOSE: bool = True

    # ── API ───────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # ── Derived ───────────────────────────────
    @property
    def is_live_mode(self) -> bool:
        """True if a real OpenAI API key is configured."""
        return bool(self.OPENAI_API_KEY) and self.OPENAI_API_KEY != "your-openai-api-key-here"

    @property
    def mode_label(self) -> str:
        return "🟢 LIVE" if self.is_live_mode else "🟡 DEMO (mock)"


settings = Settings()
