"""
FastAPI REST Server — AI Research & Discovery Platform

Endpoints:
  POST /api/agents/run          Run the multi-agent crew
  POST /api/pipeline/process    Run the LLM extraction pipeline
  POST /api/search              Semantic similarity search
  POST /api/ask                 RAG-powered Q&A
  POST /api/ingest              Ingest entities into vector store
  GET  /api/stats               Vector store statistics
  DELETE /api/store             Clear the vector store
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from config.settings import settings
from models.schemas import EntityType

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Research & Discovery Platform",
    description=(
        "Track 03 — AI Automation & Agent Development\n\n"
        "Endpoints for multi-agent web exploration, self-correcting LLM pipelines, "
        "and semantic vector search over academic entities."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Request / Response Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AgentRunRequest(BaseModel):
    seed_urls: Optional[list[str]] = Field(
        None,
        description="Optional list of seed URLs for the agent swarm",
        example=["https://openlibrary.org/subjects/artificial_intelligence.json?limit=5"],
    )


class PipelineRequest(BaseModel):
    text: str = Field(
        ...,
        description="Unstructured text to extract entities from",
        example="Dr. Andrew Ng from Stanford University authored the book 'Machine Learning Yearning'.",
        max_length=50_000,
    )


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query", example="deep learning textbooks")
    n_results: int = Field(5, ge=1, le=20, description="Number of results to return")
    entity_type: Optional[str] = Field(
        None,
        description="Filter by entity type: 'author', 'publisher', or 'book'",
        example="book",
    )


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question", example="Who wrote the book Deep Learning?")
    n_results: int = Field(5, ge=1, le=20, description="Number of documents to retrieve for context")
    entity_type: Optional[str] = Field(None, description="Optional entity type filter")


class IngestRequest(BaseModel):
    run_pipeline_demo: bool = Field(
        True, description="Run the pipeline demo to generate entities before ingesting"
    )
    run_crew_demo: bool = Field(
        False, description="Also run the agent crew demo before ingesting"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Routes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.get("/", tags=["Health"])
def root():
    """Platform health check."""
    return {
        "platform": "AI Research & Discovery Platform",
        "track": "03 — AI Automation & Agent Development",
        "version": "1.0.0",
        "mode": settings.mode_label,
        "model": settings.OPENAI_MODEL,
        "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
        "docs": "/docs",
    }


@app.get("/api/stats", tags=["Vector Store"])
def get_stats():
    """Return vector store statistics."""
    from search_engine.vector_store import VectorStore
    vs = VectorStore()
    stats = vs.get_stats()
    return {
        "total_entities": stats.get("total", 0),
        "authors": stats.get("author", 0),
        "publishers": stats.get("publisher", 0),
        "books": stats.get("book", 0),
        "chroma_dir": str(settings.CHROMA_PERSIST_DIR),
    }


@app.post("/api/agents/run", tags=["Deliverable 1 — Multi-Agent Swarm"])
def run_agents(request: AgentRunRequest):
    """
    **Deliverable 1**: Run the Autonomous Multi-Agent Swarm.

    Launches the 4-agent CrewAI crew:
    - **Scraper Agent** — explores seed URLs
    - **Parser Agent** — extracts structured entities
    - **Researcher Agent** — verifies ISBNs & enriches data
    - **Validator Agent** — quality assurance & scoring
    """
    try:
        from agents.crew import ResearchDiscoveryCrew
        crew = ResearchDiscoveryCrew()
        results = crew.run(seed_urls=request.seed_urls)
        return {
            "status": "success",
            "deliverable": "1 — Autonomous Multi-Agent Swarm",
            "authors_discovered": len(results.get("authors", [])),
            "publishers_discovered": len(results.get("publishers", [])),
            "books_discovered": len(results.get("books", [])),
            "total_entities": sum(
                len(results.get(k, [])) for k in ("authors", "publishers", "books")
            ),
            "data": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipeline/process", tags=["Deliverable 2 — Self-Correcting Pipeline"])
def run_pipeline(request: PipelineRequest):
    """
    **Deliverable 2**: Run the Self-Correcting LLM Pipeline on custom text.

    Stages:
    1. **Entity Extraction** — LLM-based structured extraction
    2. **Data Screening** — rule-based validation (ISBN checksum, year range, etc.)
    3. **Self-Correction** — automatic retry with exponential backoff
    """
    try:
        from pipelines.pipeline_manager import PipelineManager
        pm = PipelineManager()
        result = pm.process(request.text)
        return {
            "status": "success" if result.success else "failed",
            "deliverable": "2 — Self-Correcting LLM Pipeline",
            "pipeline_id": result.pipeline_id,
            "authors_extracted": len(result.final_authors),
            "publishers_extracted": len(result.final_publishers),
            "books_extracted": len(result.final_books),
            "total_entities": result.total_entities,
            "corrections_applied": len(result.corrections),
            "duration_seconds": result.duration_seconds,
            "authors": [a.model_dump() for a in result.final_authors],
            "publishers": [p.model_dump() for p in result.final_publishers],
            "books": [b.model_dump() for b in result.final_books],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", tags=["Deliverable 3 — Semantic Search"])
def semantic_search(request: SearchRequest):
    """
    **Deliverable 3**: Semantic Vector Search.

    Generates a query embedding and performs cosine similarity search
    against the ChromaDB vector store. Optionally filter by entity type.
    """
    try:
        from search_engine.query_engine import QueryEngine
        engine = QueryEngine()
        results = engine.search(
            query=request.query,
            n_results=request.n_results,
            entity_type=request.entity_type,
        )
        return {
            "query": request.query,
            "entity_type_filter": request.entity_type,
            "result_count": len(results),
            "results": [
                {
                    "entity_type": r.entity_type.value,
                    "entity_id": r.entity_id,
                    "similarity_score": r.similarity_score,
                    "text": r.text,
                    "metadata": r.metadata,
                }
                for r in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask", tags=["Deliverable 3 — Semantic Search"])
def rag_ask(request: AskRequest):
    """
    **Deliverable 3**: RAG-powered Q&A.

    Retrieves the most relevant documents from ChromaDB and uses an LLM
    to synthesize a grounded answer with source citations.
    """
    try:
        from search_engine.query_engine import QueryEngine
        engine = QueryEngine()
        response = engine.ask(
            question=request.question,
            n_results=request.n_results,
            entity_type=request.entity_type,
        )
        return {
            "query": response.query,
            "answer": response.answer,
            "model_used": response.model_used,
            "duration_seconds": response.duration_seconds,
            "sources": [
                {
                    "entity_type": s.entity_type.value,
                    "entity_id": s.entity_id,
                    "similarity_score": s.similarity_score,
                    "metadata": s.metadata,
                }
                for s in response.sources
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest", tags=["Vector Store"])
def ingest_data(request: IngestRequest):
    """
    Ingest entities into the ChromaDB vector store.

    Runs the pipeline demo and/or crew demo to generate entities,
    then indexes them all into ChromaDB.
    """
    try:
        from search_engine.vector_store import VectorStore
        from models.schemas import Author, Publisher, Book

        pipeline_results = None
        crew_results = None

        if request.run_pipeline_demo:
            from pipelines.pipeline_manager import PipelineManager
            pm = PipelineManager()
            pipeline_results = pm.demo()

        if request.run_crew_demo:
            from agents.crew import ResearchDiscoveryCrew
            crew = ResearchDiscoveryCrew()
            crew_results = crew.run()

        vs = VectorStore()
        all_authors: list[Author] = []
        all_publishers: list[Publisher] = []
        all_books: list[Book] = []

        if pipeline_results:
            for r in pipeline_results:
                all_authors.extend(r.final_authors)
                all_publishers.extend(r.final_publishers)
                all_books.extend(r.final_books)

        n_authors = vs.add_authors(all_authors)
        n_publishers = vs.add_publishers(all_publishers)
        n_books = vs.add_books(all_books)
        stats = vs.get_stats()

        return {
            "status": "success",
            "indexed": {
                "authors": n_authors,
                "publishers": n_publishers,
                "books": n_books,
                "total": n_authors + n_publishers + n_books,
            },
            "vector_store_total": stats.get("total", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/store", tags=["Vector Store"])
def clear_store():
    """Clear all data from the vector store."""
    try:
        from search_engine.vector_store import VectorStore
        VectorStore().clear()
        return {"status": "success", "message": "Vector store cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
