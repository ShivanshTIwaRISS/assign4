"""
Vector Store — ChromaDB Operations

Manages ChromaDB collections for authors, publishers, and books.
Supports upsert, delete, similarity search, and metadata filtering.
"""

from __future__ import annotations

import json
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from rich.console import Console

from config.settings import settings
from models.schemas import Author, Publisher, Book, EntityType, SearchResult
from search_engine.embeddings import EmbeddingGenerator

console = Console()


class VectorStore:
    """
    ChromaDB-based vector store for academic entities.

    Manages separate collections for different entity types
    and a unified collection for cross-entity search.
    """

    def __init__(self, persist: bool = True):
        """
        Initialize the vector store.

        Args:
            persist: Whether to persist data to disk (default: True).
        """
        self.embedding_generator = EmbeddingGenerator()

        if persist:
            settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(settings.CHROMA_PERSIST_DIR),
            )
        else:
            self._client = chromadb.Client()

        # Get or create collections
        self._unified = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_UNIFIED,
            metadata={"description": "Unified collection for all entity types"},
        )

    @property
    def collection(self):
        """Get the unified collection."""
        return self._unified

    def add_authors(self, authors: list[Author]) -> int:
        """Add author entities to the vector store."""
        if not authors:
            return 0

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        texts = [a.to_text() for a in authors]
        embed_vectors = self.embedding_generator.embed_batch(texts)

        for author, text, embedding in zip(authors, texts, embed_vectors):
            ids.append(f"author_{author.id}")
            documents.append(text)
            metadatas.append({
                "entity_type": EntityType.AUTHOR.value,
                "name": author.name,
                "affiliation": author.affiliation or "",
                "confidence_score": author.confidence_score,
                "research_areas": ", ".join(author.research_areas),
            })
            embeddings.append(embedding)

        self._unified.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        console.print(f"  [green]📥 Indexed {len(authors)} authors[/green]")
        return len(authors)

    def add_publishers(self, publishers: list[Publisher]) -> int:
        """Add publisher entities to the vector store."""
        if not publishers:
            return 0

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        texts = [p.to_text() for p in publishers]
        embed_vectors = self.embedding_generator.embed_batch(texts)

        for pub, text, embedding in zip(publishers, texts, embed_vectors):
            ids.append(f"publisher_{pub.id}")
            documents.append(text)
            metadatas.append({
                "entity_type": EntityType.PUBLISHER.value,
                "name": pub.name,
                "publisher_type": pub.publisher_type or "",
                "location": pub.location or "",
                "confidence_score": pub.confidence_score,
                "specializations": ", ".join(pub.specializations),
            })
            embeddings.append(embedding)

        self._unified.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        console.print(f"  [green]📥 Indexed {len(publishers)} publishers[/green]")
        return len(publishers)

    def add_books(self, books: list[Book]) -> int:
        """Add book entities to the vector store."""
        if not books:
            return 0

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        texts = [b.to_text() for b in books]
        embed_vectors = self.embedding_generator.embed_batch(texts)

        for book, text, embedding in zip(books, texts, embed_vectors):
            ids.append(f"book_{book.id}")
            documents.append(text)
            metadatas.append({
                "entity_type": EntityType.BOOK.value,
                "title": book.title,
                "publisher": book.publisher or "",
                "isbn": book.isbn or "",
                "year": book.year or 0,
                "confidence_score": book.confidence_score,
                "subjects": ", ".join(book.subjects),
                "authors": ", ".join(book.authors),
            })
            embeddings.append(embedding)

        self._unified.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        console.print(f"  [green]📥 Indexed {len(books)} books[/green]")
        return len(books)

    def search(
        self,
        query: str,
        n_results: int = 5,
        entity_type: Optional[EntityType] = None,
    ) -> list[SearchResult]:
        """
        Semantic similarity search.

        Args:
            query: Natural language search query.
            n_results: Number of results to return.
            entity_type: Optional filter by entity type.

        Returns:
            List of SearchResult objects sorted by similarity.
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.embed(query)

        # Build where filter
        where_filter = None
        if entity_type:
            where_filter = {"entity_type": entity_type.value}

        try:
            results = self._unified.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self._unified.count() or 1),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            console.print(f"  [red]Search error: {str(e)}[/red]")
            return []

        # Convert to SearchResult objects
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""
                distance = results["distances"][0][i] if results["distances"] else 1.0

                # Convert distance to similarity score (ChromaDB uses L2 distance)
                similarity = max(0.0, 1.0 - (distance / 2.0))

                entity_type_val = metadata.get("entity_type", "book")

                search_results.append(SearchResult(
                    entity_type=EntityType(entity_type_val),
                    entity_id=doc_id,
                    text=document,
                    metadata=metadata,
                    similarity_score=round(similarity, 4),
                ))

        return search_results

    def get_stats(self) -> dict:
        """Get collection statistics."""
        total = self._unified.count()

        # Count by type
        stats = {"total": total}
        for et in EntityType:
            try:
                result = self._unified.get(
                    where={"entity_type": et.value},
                    include=[],
                )
                stats[et.value] = len(result["ids"]) if result["ids"] else 0
            except Exception:
                stats[et.value] = 0

        return stats

    def clear(self):
        """Clear all data from the vector store."""
        self._client.delete_collection(settings.CHROMA_COLLECTION_UNIFIED)
        self._unified = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_UNIFIED,
        )
        console.print("  [yellow]🗑️  Vector store cleared[/yellow]")
