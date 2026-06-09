"""
RAG Pipeline — Retrieval-Augmented Generation

Retrieves relevant documents from the vector store and uses an LLM
to generate synthesized answers with source citations.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from config.settings import settings
from models.schemas import EntityType, SearchResult, RAGResponse
from search_engine.vector_store import VectorStore

console = Console()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAG prompt template
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAG_PROMPT = """You are an expert academic research assistant. Answer the user's question using ONLY the information provided in the context below.

CONTEXT (retrieved from the knowledge base):
{context}

RULES:
1. Answer ONLY based on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite specific entities (authors, books, publishers) by name
4. Be concise but comprehensive
5. If multiple sources are relevant, synthesize the information

USER QUESTION:
{query}

ANSWER:"""


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Flow:
    1. User query → embed
    2. Retrieve top-K similar documents from ChromaDB
    3. Construct context-rich prompt with retrieved docs
    4. Generate answer via LLM
    5. Return answer + source citations
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self._llm = None

    def _get_llm(self):
        """Lazy-initialize the LLM."""
        if self._llm is None and settings.is_live_mode:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=0.3,
                api_key=settings.OPENAI_API_KEY,
            )
        return self._llm

    def query(
        self,
        question: str,
        n_results: int = 5,
        entity_type: Optional[EntityType] = None,
    ) -> RAGResponse:
        """
        Run a RAG query.

        Args:
            question: Natural language question.
            n_results: Number of documents to retrieve.
            entity_type: Optional entity type filter.

        Returns:
            RAGResponse with answer and source citations.
        """
        start_time = time.time()

        # Step 1: Retrieve relevant documents
        search_results = self.vector_store.search(
            query=question,
            n_results=n_results,
            entity_type=entity_type,
        )

        if not search_results:
            return RAGResponse(
                query=question,
                answer="No relevant information found in the knowledge base. Please run the agent pipeline first to populate the database.",
                sources=[],
                duration_seconds=round(time.time() - start_time, 2),
            )

        # Step 2: Build context from retrieved documents
        context = self._build_context(search_results)

        # Step 3: Generate answer
        if not settings.is_live_mode:
            answer = self._mock_generate(question, search_results)
        else:
            answer = self._live_generate(question, context)

        duration = round(time.time() - start_time, 2)

        return RAGResponse(
            query=question,
            answer=answer,
            sources=search_results,
            model_used=settings.OPENAI_MODEL if settings.is_live_mode else "mock",
            duration_seconds=duration,
        )

    def _build_context(self, results: list[SearchResult]) -> str:
        """Build a context string from search results."""
        context_parts = []
        for i, result in enumerate(results, 1):
            entity_label = result.entity_type.value.upper()
            context_parts.append(
                f"[Source {i} — {entity_label}] (similarity: {result.similarity_score:.2f})\n"
                f"{result.text}"
            )
        return "\n\n".join(context_parts)

    def _live_generate(self, question: str, context: str) -> str:
        """Generate answer using live LLM."""
        from langchain.schema import HumanMessage

        llm = self._get_llm()
        prompt = RAG_PROMPT.format(context=context, query=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _mock_generate(self, question: str, results: list[SearchResult]) -> str:
        """Generate a mock answer for demo mode."""
        time.sleep(0.3)

        # Build answer from search results
        question_lower = question.lower()

        # Collect relevant info from results
        authors = []
        books = []
        publishers = []

        for r in results:
            if r.entity_type == EntityType.AUTHOR:
                authors.append(r.metadata.get("name", "Unknown"))
            elif r.entity_type == EntityType.BOOK:
                books.append(r.metadata.get("title", "Unknown"))
            elif r.entity_type == EntityType.PUBLISHER:
                publishers.append(r.metadata.get("name", "Unknown"))

        # Generate contextual answer
        answer_parts = [f"Based on the knowledge base, here's what I found:\n"]

        if authors:
            answer_parts.append(
                f"**Relevant Authors**: {', '.join(authors)}. "
                f"These researchers are active in fields related to your query."
            )

        if books:
            answer_parts.append(
                f"\n\n**Relevant Books**: {', '.join(books)}. "
                f"These publications cover topics related to your question."
            )

        if publishers:
            answer_parts.append(
                f"\n\n**Relevant Publishers**: {', '.join(publishers)}. "
                f"These publishers have catalogs related to your area of interest."
            )

        if not (authors or books or publishers):
            answer_parts.append(
                "I found some relevant entries in the knowledge base. "
                "The results above show the most semantically similar entities."
            )

        return "\n".join(answer_parts)
