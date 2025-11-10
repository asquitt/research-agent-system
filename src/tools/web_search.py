# src/tools/web_search.py
"""
Web Search Tool with multiple provider support.

Supports:
- Tavily API (recommended - best for AI applications)
- Serper API (good alternative)
- DuckDuckGo (free, no API key needed)
"""

import os
import json
import logging
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured search result."""
    title: str
    url: str
    snippet: str
    source: str  # Domain name
    published_date: Optional[str] = None
    relevance_score: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "published_date": self.published_date,
            "relevance_score": self.relevance_score
        }


class WebSearchTool:
    """
    Multi-provider web search tool.
    
    Usage:
        search_tool = WebSearchTool(provider="tavily", api_key="your_key")
        results = await search_tool.search("quantum computing", num_results=5)
    """
    
    def __init__(
        self,
        provider: Literal["tavily", "serper", "duckduckgo"] = "tavily",
        api_key: Optional[str] = None,
        timeout: int = 10
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.timeout = timeout
        
        # Validate API key for paid providers
        if provider in ["tavily", "serper"] and not self.api_key:
            raise ValueError(
                f"{provider} requires an API key. "
                f"Set {provider.upper()}_API_KEY environment variable."
            )
        
        logger.info(f"Initialized WebSearchTool with provider: {provider}")
    
    async def search(
        self,
        query: str,
        num_results: int = 5,
        search_depth: Literal["basic", "advanced"] = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search the web for a query.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            search_depth: "basic" for faster results, "advanced" for more comprehensive
            include_domains: Only search these domains (e.g., ["wikipedia.org"])
            exclude_domains: Exclude these domains
            
        Returns:
            List of SearchResult objects
        """
        logger.info(f"Searching for: '{query}' (provider: {self.provider})")
        
        try:
            if self.provider == "tavily":
                return await self._search_tavily(
                    query, num_results, search_depth, include_domains, exclude_domains
                )
            elif self.provider == "serper":
                return await self._search_serper(
                    query, num_results, include_domains, exclude_domains
                )
            elif self.provider == "duckduckgo":
                return await self._search_duckduckgo(query, num_results)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def _search_tavily(
        self,
        query: str,
        num_results: int,
        search_depth: str,
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]]
    ) -> List[SearchResult]:
        """Search using Tavily API."""
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": num_results,
            "search_depth": search_depth,
            "include_answer": False,  # We'll synthesize our own answer
            "include_raw_content": False,  # Save tokens
        }
        
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()
        
        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source=self._extract_domain(item.get("url", "")),
                published_date=item.get("published_date"),
                relevance_score=item.get("score")
            ))
        
        logger.info(f"Tavily returned {len(results)} results")
        return results
    
    async def _search_serper(
        self,
        query: str,
        num_results: int,
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]]
    ) -> List[SearchResult]:
        """Search using Serper API."""
        url = "https://google.serper.dev/search"
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Build query with domain filters
        modified_query = query
        if include_domains:
            site_queries = " OR ".join([f"site:{domain}" for domain in include_domains])
            modified_query = f"{query} ({site_queries})"
        if exclude_domains:
            exclude_queries = " ".join([f"-site:{domain}" for domain in exclude_domains])
            modified_query = f"{modified_query} {exclude_queries}"
        
        payload = {
            "q": modified_query,
            "num": num_results
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()
        
        results = []
        for item in data.get("organic", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source=self._extract_domain(item.get("link", "")),
                published_date=item.get("date"),
                relevance_score=None  # Serper doesn't provide scores
            ))
        
        logger.info(f"Serper returned {len(results)} results")
        return results
    
    async def _search_duckduckgo(
        self,
        query: str,
        num_results: int
    ) -> List[SearchResult]:
        """
        Search using DuckDuckGo (free, no API key).
        Uses duckduckgo-search library.
        """
        try:
            from duckduckgo_search import AsyncDDGS
        except ImportError:
            raise ImportError(
                "duckduckgo-search not installed. "
                "Install with: pip install duckduckgo-search"
            )
        
        ddgs = AsyncDDGS()
        raw_results = await ddgs.text(query, max_results=num_results)
        
        results = []
        for item in raw_results:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("href", ""),
                snippet=item.get("body", ""),
                source=self._extract_domain(item.get("href", "")),
                published_date=None,
                relevance_score=None
            ))
        
        logger.info(f"DuckDuckGo returned {len(results)} results")
        return results
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""
    
    def format_results(self, results: List[SearchResult]) -> str:
        """
        Format search results as a string for LLM context.
        
        Returns:
            Formatted string with numbered results
        """
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"[{i}] {result.title}\n"
                f"    Source: {result.source}\n"
                f"    URL: {result.url}\n"
                f"    Snippet: {result.snippet}\n"
            )
        
        return "\n".join(formatted)
    
    def get_metadata(self, results: List[SearchResult]) -> Dict:
        """
        Extract metadata from search results.
        
        Returns:
            Dict with statistics about the results
        """
        if not results:
            return {
                "total_results": 0,
                "unique_sources": 0,
                "sources": []
            }
        
        sources = [r.source for r in results if r.source]
        
        return {
            "total_results": len(results),
            "unique_sources": len(set(sources)),
            "sources": list(set(sources)),
            "has_published_dates": any(r.published_date for r in results),
            "has_relevance_scores": any(r.relevance_score for r in results)
        }


# Example usage and testing
async def main():
    """Example usage of WebSearchTool."""
    
    # Initialize with your preferred provider
    # Option 1: Tavily (recommended, requires API key)
    search = WebSearchTool(
        provider="tavily",
        api_key="your_tavily_api_key"  # Or set TAVILY_API_KEY env var
    )
    
    # Option 2: DuckDuckGo (free, no API key)
    # search = WebSearchTool(provider="duckduckgo")
    
    # Basic search
    print("=== Basic Search ===")
    results = await search.search(
        query="latest developments in quantum computing",
        num_results=5
    )
    
    print(search.format_results(results))
    
    # Advanced search with filters
    print("\n=== Filtered Search ===")
    results = await search.search(
        query="climate change",
        num_results=3,
        include_domains=["nature.com", "science.org"],
        search_depth="advanced"
    )
    
    print(search.format_results(results))
    
    # Get metadata
    print("\n=== Metadata ===")
    metadata = search.get_metadata(results)
    print(json.dumps(metadata, indent=2))
    
    # Access individual results
    print("\n=== First Result Details ===")
    if results:
        first = results[0]
        print(f"Title: {first.title}")
        print(f"URL: {first.url}")
        print(f"Source: {first.source}")
        print(f"Snippet: {first.snippet[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
