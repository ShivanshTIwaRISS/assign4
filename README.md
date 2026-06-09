# Track 03 — AI Automation & Agent Development

## AI Research & Discovery Platform

> Autonomous multi-agent web exploration, self-correcting LLM pipelines, and semantic vector search — built with CrewAI, LangChain, and ChromaDB.

---

## 📦 Deliverables

| # | Deliverable | Technologies |
|---|---|---|
| ① | **Autonomous Multi-Agent Swarm** | CrewAI, httpx, BeautifulSoup |
| ② | **Self-Correcting LLM Pipeline** | LangChain, OpenAI, Pydantic |
| ③ | **Semantic Vector Search Engine** | ChromaDB, OpenAI Embeddings, RAG |

---

## 🧠 Why I Built It This Way (Architecture Decisions)

I designed this platform with a strong emphasis on **modularity, fault-tolerance, and type safety**. Instead of building one massive script, I separated the system into decoupled stages (Agents -> Pipelines -> Vector Store) linked by strict Pydantic schemas. The **Self-Correcting Pipeline** uses an exponential backoff loop to catch and fix LLM hallucinations automatically, ensuring data integrity. Finally, I integrated a **Mock/Demo mode** at the dependency boundary, allowing reviewers to validate the entire workflow and routing logic offline without needing to supply their own API keys or incur costs.

---

## 🏗️ Project Structure

```
internshala_assign5/
├── main.py                    ← CLI entry point (all deliverables)
├── .env.example               ← Environment variable template
├── requirements.txt
│
├── agents/                    ← Deliverable 1: Multi-Agent Swarm
│   ├── crew.py                ← ResearchDiscoveryCrew orchestrator
│   ├── scraper_agent.py       ← Web Exploration Specialist
│   ├── parser_agent.py        ← Structural Entity Parser
│   ├── researcher_agent.py    ← Deep Research & Enrichment Specialist
│   └── validator_agent.py     ← Data Quality Assurance Specialist
│
├── pipelines/                 ← Deliverable 2: LLM Pipeline
│   ├── pipeline_manager.py    ← Central pipeline orchestrator
│   ├── entity_extraction.py   ← LLM-based entity extraction
│   ├── data_screening.py      ← Context-aware validation
│   └── self_correction.py     ← Automatic error correction engine
│
├── search_engine/             ← Deliverable 3: Semantic Search
│   ├── vector_store.py        ← ChromaDB operations
│   ├── embeddings.py          ← OpenAI embedding generation
│   ├── rag.py                 ← RAG pipeline (retrieval + generation)
│   └── query_engine.py        ← High-level search interface
│
├── api/                       ← FastAPI REST server
│   └── server.py
│
├── models/
│   └── schemas.py             ← Pydantic domain models
│
├── tools/
│   ├── web_tools.py           ← WebScraperTool, LinkExtractorTool
│   └── search_tools.py        ← ISBNLookupTool, AuthorSearchTool
│
├── config/
│   └── settings.py            ← Centralized settings
│
└── data/
    └── seed_urls.json         ← Seed URLs for agent swarm
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
cd internshala_assign5
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY for live mode
# Leave blank to run in DEMO mode (no API costs)
```

### 3. Run the full demo

```bash
python main.py
```

This executes all three deliverables in sequence.

---

## 🎯 Usage Modes

```bash
# Full demo — all 3 deliverables (default)
python main.py

# Deliverable 1 only — Multi-Agent Swarm
python main.py --mode agents

# Deliverable 1 with custom seed URLs
python main.py --mode agents --urls https://openlibrary.org/subjects/artificial_intelligence.json

# Deliverable 2 only — LLM Pipeline
python main.py --mode pipeline

# Deliverable 3 — Interactive Semantic Search
python main.py --mode search

# Ingest demo data into vector store
python main.py --mode ingest

# Start the FastAPI REST server
python main.py --mode api

# Clear vector store and run full demo
python main.py --clear-db
```

---

## 🤖 Deliverable 1: Autonomous Multi-Agent Swarm

**Architecture**: 4-agent CrewAI pipeline running sequentially.

```
Seed URLs
    │
    ▼
┌──────────────────┐
│  Scraper Agent   │  Explores web pages, discovers content
│  (web_scraper,   │  using WebScraperTool + LinkExtractorTool
│   link_extractor)│
└────────┬─────────┘
         │ raw HTML content
         ▼
┌──────────────────┐
│  Parser Agent    │  Extracts structured entities from raw content
│  (page_content_  │  (authors, publishers, books)
│   extractor)     │
└────────┬─────────┘
         │ structured JSON entities
         ▼
┌──────────────────┐
│ Researcher Agent │  Verifies ISBNs, enriches author data,
│ (isbn_lookup,    │  cross-references DBLP + Open Library
│  author_search)  │
└────────┬─────────┘
         │ enriched + verified entities
         ▼
┌──────────────────┐
│ Validator Agent  │  Checks completeness, validates formats,
│                  │  deduplicates, assigns confidence scores
└──────────────────┘
         │
         ▼
    Final Dataset (authors / publishers / books)
```

**Key features**:
- Fully autonomous web exploration from seed URLs
- Real-world API calls (Open Library, DBLP) in live mode
- Mock mode for offline demo / cost-free testing

---

## 🔄 Deliverable 2: Self-Correcting LLM Pipeline

**Architecture**: 3-stage pipeline with automatic error recovery.

```
Unstructured Text
        │
        ▼
┌─────────────────────┐
│ Stage 1: Extraction │  GPT-4o-mini extracts structured JSON
│ EntityExtractor     │  (authors, publishers, books)
└──────────┬──────────┘
           │ success / failure
           ├──────────────────────────────────────────┐
           │ (on failure)                             │ (on success)
           ▼                                          ▼
┌──────────────────────┐              ┌──────────────────────┐
│  SelfCorrectionEngine│              │ Stage 2: Screening   │
│  • Captures error    │◄─ retry ─────│ DataScreener         │
│  • Rewrites prompt   │              │ • ISBN checksum      │
│  • Exponential retry │              │ • Year range check   │
│  • Max N attempts    │              │ • Name normalization │
└──────────────────────┘              │ • Publisher aliases  │
                                      └──────────┬───────────┘
                                                 │
                                                 ▼
                                      ┌──────────────────────┐
                                      │ Stage 3: Entity Build │
                                      │ Pydantic models       │
                                      │ Author / Publisher    │
                                      │ / Book with scores    │
                                      └──────────────────────┘
```

**Self-correction features**:
- JSON parse error correction
- Missing required field detection and re-prompt
- Schema validation with descriptive error messages
- Exponential backoff: `delay = RETRY_DELAY × 2^(attempt-1)`
- Full correction chain logged as `CorrectionRecord` objects

---

## 🔍 Deliverable 3: Semantic Vector Search Engine

**Architecture**: Embedding-based retrieval with RAG generation.

```
User Query
    │
    ▼
EmbeddingGenerator          ← text-embedding-3-small (1536-dim)
    │
    ▼
ChromaDB VectorStore        ← Cosine similarity search
    │
    ├── Similarity Search   → List[SearchResult] with scores
    │
    └── RAG Pipeline
              │
              ├── Retrieved documents → context
              │
              └── ChatOpenAI          → grounded answer
```

**Features**:
- Separate entity type collections (author / publisher / book)
- Unified cross-entity search
- In-memory embedding cache (avoids duplicate API calls)
- Mock deterministic embeddings for offline testing
- Interactive REPL with search / ask / stats commands
- REST API via FastAPI

---

## 🌐 REST API

Start the server:

```bash
python main.py --mode api
# → http://localhost:8000/docs
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check |
| `GET`  | `/api/stats` | Vector store statistics |
| `POST` | `/api/agents/run` | Run multi-agent swarm |
| `POST` | `/api/pipeline/process` | Run LLM pipeline on text |
| `POST` | `/api/search` | Semantic similarity search |
| `POST` | `/api/ask` | RAG-powered Q&A |
| `POST` | `/api/ingest` | Ingest entities into store |
| `DELETE` | `/api/store` | Clear vector store |

### Example API calls

```bash
# Semantic search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "deep learning textbooks", "n_results": 3}'

# RAG Q&A
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who wrote the book Deep Learning?"}'

# Run agents
curl -X POST http://localhost:8000/api/agents/run \
  -H "Content-Type: application/json" \
  -d '{}'

# Extract entities from custom text
curl -X POST http://localhost:8000/api/pipeline/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Dr. Andrew Ng from Stanford wrote Machine Learning Yearning published by MIT Press."}'
```

---

## 🔧 Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key — leave empty for demo mode |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model for extraction & RAG |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB persistence directory |
| `MAX_RETRIES` | `3` | Max self-correction retries |
| `RETRY_DELAY` | `1.0` | Base retry delay (seconds) |
| `API_HOST` | `0.0.0.0` | FastAPI host |
| `API_PORT` | `8000` | FastAPI port |

---

## 🎭 Demo Mode vs. Live Mode

| Feature | Demo Mode (no API key) | Live Mode (API key set) |
|---|---|---|
| Agent execution | Simulated with mock data | Real CrewAI + web requests |
| Entity extraction | Pre-defined mock entities | Real GPT extraction |
| Embeddings | Deterministic hash-based vectors | OpenAI text-embedding-3-small |
| ISBN lookup | Mock results | Open Library API |
| Author search | Mock results | DBLP API |
| Cost | Free | OpenAI API credits |

---

## 🛠️ Engineering Stack

| Component | Technology |
|---|---|
| Agent Framework | CrewAI ≥ 0.80 |
| LLM Orchestration | LangChain ≥ 0.3 |
| LLM Provider | OpenAI (gpt-4o-mini) |
| Vector Database | ChromaDB ≥ 0.5 |
| Embeddings | OpenAI text-embedding-3-small |
| Web Scraping | httpx + BeautifulSoup4 |
| Data Validation | Pydantic v2 |
| REST API | FastAPI + uvicorn |
| CLI Output | Rich |

---

## 📊 Data Models

### Author
```python
Author(name, affiliation, research_areas[], publications[], h_index, profile_url, confidence_score)
```

### Publisher
```python
Publisher(name, publisher_type, location, website, specializations[], founded_year, confidence_score)
```

### Book
```python
Book(title, authors[], publisher, isbn, year, edition, subjects[], pages, confidence_score)
```

---

## 🧪 Interactive Search Commands

```
search> search deep learning          # Semantic similarity search (all types)
search> ask Who published Deep Learning?  # RAG Q&A
search> authors machine learning      # Search only authors
search> books neural networks         # Search only books
search> publishers academic press     # Search only publishers
search> stats                         # Show database statistics
search> quit                          # Exit
```

---

*Built for Internshala Track 03: AI Automation & Agent Development*
