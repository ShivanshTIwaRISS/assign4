"""
Embedding Generator

Handles embedding generation using OpenAI's text-embedding-3-small model
for semantic search. Supports batch embedding and local caching.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from config.settings import settings

console = Console()

# Simple in-memory cache for embeddings
_embedding_cache: dict[str, list[float]] = {}


class EmbeddingGenerator:
    """
    Generate embeddings using OpenAI or mock embeddings for demo mode.

    Features:
    - Single and batch embedding generation
    - In-memory caching to avoid duplicate API calls
    - Mock embeddings for demo mode (deterministic based on input hash)
    """

    def __init__(self):
        self._client = None
        self.model = settings.OPENAI_EMBEDDING_MODEL
        self.dimension = 1536  # text-embedding-3-small dimension

    def _get_client(self):
        """Lazy-initialize the OpenAI client."""
        if self._client is None and settings.is_live_mode:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        # Check cache
        cache_key = self._cache_key(text)
        if cache_key in _embedding_cache:
            return _embedding_cache[cache_key]

        if not settings.is_live_mode:
            embedding = self._mock_embed(text)
        else:
            embedding = self._live_embed(text)

        _embedding_cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not settings.is_live_mode:
            return [self._mock_embed(t) for t in texts]

        # Check cache, only send uncached texts to API
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in _embedding_cache:
                results[i] = _embedding_cache[cache_key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_embeddings = self._live_embed_batch(uncached_texts)
            for i, idx in enumerate(uncached_indices):
                results[idx] = new_embeddings[i]
                _embedding_cache[self._cache_key(texts[idx])] = new_embeddings[i]

        return results

    def _live_embed(self, text: str) -> list[float]:
        """Generate embedding using OpenAI API."""
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=text[:8000],  # Token limit safety
        )
        return response.data[0].embedding

    def _live_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch using OpenAI API."""
        client = self._get_client()
        # OpenAI supports batch embedding
        truncated = [t[:8000] for t in texts]
        response = client.embeddings.create(
            model=self.model,
            input=truncated,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def _mock_embed(self, text: str) -> list[float]:
        """
        Generate a deterministic mock embedding based on text hash.
        Produces consistent embeddings for the same input text.
        """
        import math

        # Create a deterministic seed from text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed_value = int(text_hash[:8], 16)

        # Generate pseudo-random but deterministic embedding
        embedding = []
        for i in range(self.dimension):
            # Use a combination of hash and index for variation
            val = math.sin(seed_value * (i + 1) * 0.001) * 0.5
            embedding.append(round(val, 6))

        # Normalize the vector
        magnitude = math.sqrt(sum(v * v for v in embedding))
        if magnitude > 0:
            embedding = [v / magnitude for v in embedding]

        return embedding

    def _cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def clear_cache(self):
        """Clear the embedding cache."""
        global _embedding_cache
        _embedding_cache = {}
