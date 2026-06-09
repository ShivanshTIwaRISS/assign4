"""
Search and lookup tools for CrewAI agents.

Provides ISBN lookup and author search capabilities
using open academic APIs (Open Library, DBLP).

Compatible with both crewai>=0.80 and crewai 0.11.x.
"""

from __future__ import annotations

import json
from typing import Type

import httpx
from pydantic import BaseModel, Field

from config.settings import settings

# ── Tool base-class compatibility shim ───────────────────────────────────────
# crewai>=0.80 ships its own BaseTool; older versions delegate to langchain.
try:
    from crewai.tools import BaseTool  # crewai >=0.80
except ImportError:
    try:
        from langchain.tools import BaseTool  # type: ignore[no-redef]  # crewai 0.11.x
    except ImportError:
        from langchain_core.tools import BaseTool  # type: ignore[no-redef]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mock data for demo mode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MOCK_ISBN_RESULT = {
    "978-0262035613": {
        "title": "Deep Learning",
        "authors": ["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"],
        "publisher": "MIT Press",
        "year": 2016,
        "pages": 800,
        "subjects": ["Artificial Intelligence", "Machine Learning", "Neural Networks"],
    },
    "978-0387310732": {
        "title": "Pattern Recognition and Machine Learning",
        "authors": ["Christopher M. Bishop"],
        "publisher": "Springer",
        "year": 2006,
        "pages": 738,
        "subjects": ["Machine Learning", "Pattern Recognition", "Bayesian Methods"],
    },
}

MOCK_AUTHOR_RESULTS = [
    {
        "name": "Andrew Ng",
        "affiliation": "Stanford University",
        "research_areas": ["Machine Learning", "Deep Learning", "AI Education"],
        "profile_url": "https://scholar.google.com/citations?user=mG4imMEAAAAJ",
    },
    {
        "name": "Yoshua Bengio",
        "affiliation": "Université de Montréal",
        "research_areas": ["Deep Learning", "Neural Networks", "Generative Models"],
        "profile_url": "https://scholar.google.com/citations?user=kukA0LcAAAAJ",
    },
    {
        "name": "Geoffrey Hinton",
        "affiliation": "University of Toronto",
        "research_areas": ["Neural Networks", "Deep Learning", "Boltzmann Machines"],
        "profile_url": "https://scholar.google.com/citations?user=JicYPdAAAAAJ",
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ISBN Lookup Tool
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ISBNLookupInput(BaseModel):
    isbn: str = Field(description="The ISBN (10 or 13 digit) to look up")


class ISBNLookupTool(BaseTool):
    name: str = "isbn_lookup"
    description: str = (
        "Look up book details by ISBN using the Open Library API. "
        "Returns book title, authors, publisher, year, and subject information. "
        "Input should be an ISBN-10 or ISBN-13 number."
    )
    args_schema: Type[BaseModel] = ISBNLookupInput

    def _run(self, isbn: str) -> str:
        """Look up book by ISBN."""
        isbn_clean = isbn.replace("-", "").replace(" ", "").strip()

        if not settings.is_live_mode:
            return self._mock_lookup(isbn)

        try:
            url = f"https://openlibrary.org/isbn/{isbn_clean}.json"
            with httpx.Client(timeout=10, follow_redirects=True) as client:
                response = client.get(url)

                if response.status_code == 404:
                    return f"No book found for ISBN: {isbn}"

                response.raise_for_status()
                data = response.json()

            result = {
                "title": data.get("title", "Unknown"),
                "isbn": isbn,
                "publishers": data.get("publishers", []),
                "publish_date": data.get("publish_date", "Unknown"),
                "number_of_pages": data.get("number_of_pages"),
                "subjects": [s for s in data.get("subjects", [])[:10]],
            }

            # Try to get author names
            author_keys = data.get("authors", [])
            authors = []
            for ak in author_keys[:5]:
                key = ak.get("key", "")
                if key:
                    try:
                        author_resp = client.get(
                            f"https://openlibrary.org{key}.json"
                        )
                        if author_resp.status_code == 200:
                            authors.append(author_resp.json().get("name", "Unknown"))
                    except Exception:
                        pass
            result["authors"] = authors

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error looking up ISBN {isbn}: {str(e)}"

    def _mock_lookup(self, isbn: str) -> str:
        """Return mock ISBN data."""
        isbn_clean = isbn.replace("-", "").replace(" ", "")
        for key, data in MOCK_ISBN_RESULT.items():
            if isbn_clean == key.replace("-", ""):
                return json.dumps(data, indent=2)
        return json.dumps({
            "title": "Sample Academic Textbook",
            "authors": ["Dr. Sample Author"],
            "publisher": "Academic Press",
            "year": 2023,
            "isbn": isbn,
            "subjects": ["Computer Science", "Artificial Intelligence"],
        }, indent=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Author Search Tool
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AuthorSearchInput(BaseModel):
    query: str = Field(description="Author name or research area to search for")


class AuthorSearchTool(BaseTool):
    name: str = "author_search"
    description: str = (
        "Search for academic authors by name or research area using the DBLP API. "
        "Returns author names, affiliations, and profile URLs. "
        "Input should be an author name or topic keyword."
    )
    args_schema: Type[BaseModel] = AuthorSearchInput

    def _run(self, query: str) -> str:
        """Search for authors."""
        if not settings.is_live_mode:
            return self._mock_search(query)

        try:
            url = "https://dblp.org/search/author/api"
            params = {"q": query, "format": "json", "h": 5}

            with httpx.Client(timeout=10) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            hits = data.get("result", {}).get("hits", {}).get("hit", [])
            if not hits:
                return f"No authors found for query: {query}"

            results = []
            for hit in hits:
                info = hit.get("info", {})
                author = {
                    "name": info.get("author", "Unknown"),
                    "url": info.get("url", ""),
                    "affiliations": [],
                }
                notes = info.get("notes", {}).get("note", [])
                if isinstance(notes, dict):
                    notes = [notes]
                for note in notes:
                    if note.get("@type") == "affiliation":
                        author["affiliations"].append(note.get("text", ""))
                results.append(author)

            return json.dumps(results, indent=2)

        except Exception as e:
            return f"Error searching for authors '{query}': {str(e)}"

    def _mock_search(self, query: str) -> str:
        """Return mock author search results."""
        filtered = [
            a for a in MOCK_AUTHOR_RESULTS
            if query.lower() in a["name"].lower()
            or any(query.lower() in ra.lower() for ra in a["research_areas"])
        ]
        if not filtered:
            filtered = MOCK_AUTHOR_RESULTS  # Return all for demo
        return json.dumps(filtered, indent=2)
