"""
Web scraping tools for CrewAI agents.

Provides tools for fetching web pages, extracting content, and discovering links.
All tools work with both live HTTP requests and mock data for demo mode.

Compatible with both crewai>=0.80 and crewai 0.11.x.
"""

from __future__ import annotations

import re
from typing import Type
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from config.settings import settings

# ── Tool base-class compatibility shim ───────────────────────────────────────
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

MOCK_WEB_CONTENT = """
<html>
<body>
<h1>Academic Research Directory</h1>
<div class="author-profile">
    <h2>Dr. Andrew Ng</h2>
    <p>Affiliation: Stanford University, DeepLearning.AI</p>
    <p>Research Areas: Machine Learning, Deep Learning, Artificial Intelligence</p>
    <p>Notable Publications: Machine Learning Yearning, CS229 Lecture Notes</p>
    <p>H-Index: 165</p>
</div>
<div class="author-profile">
    <h2>Dr. Yoshua Bengio</h2>
    <p>Affiliation: Université de Montréal, Mila</p>
    <p>Research Areas: Deep Learning, Neural Networks, Generative Models</p>
    <p>Notable Publications: Deep Learning (MIT Press), Generative Deep Learning</p>
    <p>H-Index: 202</p>
</div>
<div class="book-listing">
    <h3>Deep Learning</h3>
    <p>Authors: Ian Goodfellow, Yoshua Bengio, Aaron Courville</p>
    <p>Publisher: MIT Press</p>
    <p>ISBN: 978-0262035613</p>
    <p>Year: 2016</p>
    <p>Subjects: Artificial Intelligence, Machine Learning, Neural Networks</p>
</div>
<div class="book-listing">
    <h3>Pattern Recognition and Machine Learning</h3>
    <p>Authors: Christopher M. Bishop</p>
    <p>Publisher: Springer</p>
    <p>ISBN: 978-0387310732</p>
    <p>Year: 2006</p>
    <p>Subjects: Machine Learning, Pattern Recognition, Statistics</p>
</div>
<div class="publisher-info">
    <h3>MIT Press</h3>
    <p>Type: University Press</p>
    <p>Location: Cambridge, Massachusetts, USA</p>
    <p>Specializations: Computer Science, AI, Cognitive Science, Economics</p>
    <p>Founded: 1962</p>
    <p>Website: https://mitpress.mit.edu</p>
</div>
<div class="publisher-info">
    <h3>Springer Nature</h3>
    <p>Type: Academic Publisher</p>
    <p>Location: Berlin, Germany</p>
    <p>Specializations: Science, Technology, Medicine, Engineering</p>
    <p>Founded: 1842</p>
    <p>Website: https://www.springer.com</p>
</div>
</body>
</html>
"""

MOCK_LINKS = [
    "https://scholar.google.com/citations?user=mG4imMEAAAAJ",
    "https://www.deeplearning.ai/",
    "https://openlibrary.org/works/OL17860744W",
    "https://mitpress.mit.edu/",
    "https://www.springer.com/",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Web Scraper Tool
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class WebScraperInput(BaseModel):
    url: str = Field(description="The URL to scrape content from")


class WebScraperTool(BaseTool):
    name: str = "web_scraper"
    description: str = (
        "Fetches a web page and returns its cleaned text content. "
        "Use this to scrape author profiles, publisher pages, book listings, "
        "and academic directories. Input should be a valid URL."
    )
    args_schema: Type[BaseModel] = WebScraperInput

    def _run(self, url: str) -> str:
        """Fetch and clean web page content."""
        if not settings.is_live_mode:
            return self._mock_scrape(url)

        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Academic Research Bot) AcademicDiscovery/1.0"
                }
                response = client.get(url, headers=headers)
                response.raise_for_status()
                return self._clean_html(response.text, url)
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"

    def _clean_html(self, html: str, base_url: str) -> str:
        """Extract meaningful text from HTML."""
        soup = BeautifulSoup(html, "lxml")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract text with structure
        text_parts = []
        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "span", "div"]):
            text = element.get_text(strip=True)
            if text and len(text) > 5:
                tag_name = element.name
                if tag_name in ("h1", "h2", "h3"):
                    text_parts.append(f"\n## {text}\n")
                elif tag_name == "h4":
                    text_parts.append(f"\n### {text}\n")
                else:
                    text_parts.append(text)

        content = "\n".join(text_parts)
        # Truncate to avoid token overflow
        return content[:8000] if len(content) > 8000 else content

    def _mock_scrape(self, url: str) -> str:
        """Return mock content for demo mode."""
        soup = BeautifulSoup(MOCK_WEB_CONTENT, "lxml")
        return soup.get_text(separator="\n", strip=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Link Extractor Tool
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LinkExtractorInput(BaseModel):
    url: str = Field(description="The URL to extract links from")


class LinkExtractorTool(BaseTool):
    name: str = "link_extractor"
    description: str = (
        "Extracts all relevant links from a web page. Returns a list of URLs "
        "found on the page that may lead to author profiles, book listings, "
        "or publisher information. Input should be a valid URL."
    )
    args_schema: Type[BaseModel] = LinkExtractorInput

    def _run(self, url: str) -> str:
        """Extract links from a web page."""
        if not settings.is_live_mode:
            return "\n".join(MOCK_LINKS)

        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Academic Research Bot) AcademicDiscovery/1.0"
                }
                response = client.get(url, headers=headers)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")
            links = set()

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                full_url = urljoin(url, href)
                parsed = urlparse(full_url)
                if parsed.scheme in ("http", "https"):
                    links.add(full_url)

            # Filter for academic/relevant links
            relevant_keywords = [
                "author", "book", "publisher", "isbn", "publication",
                "scholar", "research", "academic", "university", "press",
                "library", "catalog", "profile"
            ]
            filtered = [
                link for link in links
                if any(kw in link.lower() for kw in relevant_keywords)
            ]

            result = filtered[:20] if filtered else list(links)[:20]
            return "\n".join(result) if result else "No relevant links found."

        except Exception as e:
            return f"Error extracting links from {url}: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Content Tool
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PageContentInput(BaseModel):
    url: str = Field(description="The URL to extract structured content from")


class PageContentTool(BaseTool):
    name: str = "page_content_extractor"
    description: str = (
        "Extracts the main content from a web page in a structured format. "
        "Returns headings, paragraphs, and metadata in a clean format suitable "
        "for entity extraction. Input should be a valid URL."
    )
    args_schema: Type[BaseModel] = PageContentInput

    def _run(self, url: str) -> str:
        """Extract structured content from a page."""
        if not settings.is_live_mode:
            return self._mock_extract(url)

        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Academic Research Bot) AcademicDiscovery/1.0"
                }
                response = client.get(url, headers=headers)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Extract metadata
            title = soup.title.string if soup.title else "No title"
            meta_desc = ""
            meta_tag = soup.find("meta", attrs={"name": "description"})
            if meta_tag:
                meta_desc = meta_tag.get("content", "")

            # Remove non-content
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()

            # Build structured output
            content = f"PAGE TITLE: {title}\n"
            if meta_desc:
                content += f"DESCRIPTION: {meta_desc}\n"
            content += f"URL: {url}\n"
            content += "=" * 50 + "\n"

            # Extract sections
            for heading in soup.find_all(["h1", "h2", "h3"]):
                section_title = heading.get_text(strip=True)
                content += f"\n[{heading.name.upper()}] {section_title}\n"

                # Get sibling content
                for sibling in heading.find_next_siblings():
                    if sibling.name in ("h1", "h2", "h3"):
                        break
                    text = sibling.get_text(strip=True)
                    if text and len(text) > 5:
                        content += f"  {text}\n"

            return content[:8000]

        except Exception as e:
            return f"Error extracting content from {url}: {str(e)}"

    def _mock_extract(self, url: str) -> str:
        """Return structured mock content."""
        return (
            f"PAGE TITLE: Academic Research Directory\n"
            f"URL: {url}\n"
            f"{'=' * 50}\n"
            f"\n[H1] Academic Research Directory\n"
            f"\n[H2] Dr. Andrew Ng\n"
            f"  Affiliation: Stanford University, DeepLearning.AI\n"
            f"  Research Areas: Machine Learning, Deep Learning, AI\n"
            f"  Publications: Machine Learning Yearning, CS229 Notes\n"
            f"  H-Index: 165\n"
            f"\n[H2] Dr. Yoshua Bengio\n"
            f"  Affiliation: Université de Montréal, Mila\n"
            f"  Research Areas: Deep Learning, Neural Networks\n"
            f"  Publications: Deep Learning (MIT Press)\n"
            f"  H-Index: 202\n"
            f"\n[H3] Deep Learning (Book)\n"
            f"  Authors: Goodfellow, Bengio, Courville\n"
            f"  Publisher: MIT Press | ISBN: 978-0262035613 | Year: 2016\n"
            f"\n[H3] Pattern Recognition and Machine Learning (Book)\n"
            f"  Authors: Christopher M. Bishop\n"
            f"  Publisher: Springer | ISBN: 978-0387310732 | Year: 2006\n"
        )
