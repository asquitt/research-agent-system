"""
Unit tests for WebSearchTool.

Run with: pytest tests/test_web_search.py -v
"""

import pytest
import os
from src.tools.web_search import WebSearchTool, SearchResult


class TestWebSearchTool:
    """Test suite for WebSearchTool."""
    
    def test_initialization_duckduckgo(self):
        """Test that DuckDuckGo provider works without API key."""
        search = WebSearchTool(provider="duckduckgo")
        assert search.provider == "duckduckgo"
        assert search.api_key is None
    
    def test_initialization_tavily_requires_key(self):
        """Test that Tavily requires an API key."""
        # Clear any existing env var
        old_key = os.environ.pop("TAVILY_API_KEY", None)
        
        with pytest.raises(ValueError, match="tavily requires an API key"):
            WebSearchTool(provider="tavily")
        
        # Restore old key if it existed
        if old_key:
            os.environ["TAVILY_API_KEY"] = old_key
    
    def test_initialization_with_explicit_key(self):
        """Test initialization with explicit API key."""
        search = WebSearchTool(provider="tavily", api_key="test_key")
        assert search.api_key == "test_key"
    
    @pytest.mark.asyncio
    async def test_duckduckgo_search(self):
        """Test actual DuckDuckGo search (integration test)."""
        search = WebSearchTool(provider="duckduckgo")
        
        results = await search.search(
            query="Python programming",
            num_results=3
        )
        
        # Verify we got results
        assert len(results) > 0
        assert len(results) <= 3
        
        # Verify result structure
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.title
            assert result.url
            assert result.snippet
            assert result.source
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("TAVILY_API_KEY"),
        reason="TAVILY_API_KEY not set"
    )
    async def test_tavily_search(self):
        """Test Tavily search (requires API key)."""
        search = WebSearchTool(provider="tavily")
        
        results = await search.search(
            query="artificial intelligence",
            num_results=5,
            search_depth="basic"
        )
        
        assert len(results) > 0
        assert len(results) <= 5
        
        # Tavily includes relevance scores
        assert any(r.relevance_score is not None for r in results)
    
    @pytest.mark.asyncio
    async def test_search_with_domain_filter(self):
        """Test search with domain filtering."""
        search = WebSearchTool(provider="duckduckgo")
        
        # This is more of a smoke test - actual filtering depends on provider
        results = await search.search(
            query="machine learning",
            num_results=3,
            include_domains=["wikipedia.org"]
        )
        
        assert isinstance(results, list)
    
    def test_extract_domain(self):
        """Test domain extraction from URLs."""
        tool = WebSearchTool(provider="duckduckgo")
        
        test_cases = [
            ("https://www.example.com/path", "example.com"),
            ("https://example.com", "example.com"),
            ("http://subdomain.example.com/page", "subdomain.example.com"),
            ("", ""),
        ]
        
        for url, expected_domain in test_cases:
            assert tool._extract_domain(url) == expected_domain
    
    @pytest.mark.asyncio
    async def test_format_results(self):
        """Test result formatting."""
        search = WebSearchTool(provider="duckduckgo")
        
        results = await search.search("test query", num_results=2)
        formatted = search.format_results(results)
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        
        # Should contain numbered results
        if results:
            assert "[1]" in formatted
    
    @pytest.mark.asyncio
    async def test_get_metadata(self):
        """Test metadata extraction."""
        search = WebSearchTool(provider="duckduckgo")
        
        results = await search.search("test query", num_results=3)
        metadata = search.get_metadata(results)
        
        assert "total_results" in metadata
        assert "unique_sources" in metadata
        assert "sources" in metadata
        assert metadata["total_results"] >= 0
        assert isinstance(metadata["sources"], list)
    
    def test_format_empty_results(self):
        """Test formatting with no results."""
        search = WebSearchTool(provider="duckduckgo")
        formatted = search.format_results([])
        assert formatted == "No results found."
    
    def test_metadata_empty_results(self):
        """Test metadata with no results."""
        search = WebSearchTool(provider="duckduckgo")
        metadata = search.get_metadata([])
        
        assert metadata["total_results"] == 0
        assert metadata["unique_sources"] == 0
        assert metadata["sources"] == []
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self):
        """Test that search handles errors gracefully."""
        search = WebSearchTool(provider="duckduckgo", timeout=0.001)
        
        # Very short timeout should cause an error
        with pytest.raises(Exception):
            await search.search("test", num_results=1)


# Fixture for reusable search tool
@pytest.fixture
def search_tool():
    """Provide a DuckDuckGo search tool for tests."""
    return WebSearchTool(provider="duckduckgo")


# Parametrized tests for multiple providers
@pytest.mark.parametrize("provider", ["duckduckgo"])
@pytest.mark.asyncio
async def test_multiple_providers(provider):
    """Test that all providers return valid results."""
    # Skip providers that require API keys in CI
    if provider in ["tavily", "serper"] and not os.getenv(f"{provider.upper()}_API_KEY"):
        pytest.skip(f"{provider} API key not available")
    
    search = WebSearchTool(provider=provider)
    results = await search.search("test", num_results=2)
    
    assert isinstance(results, list)
    for result in results:
        assert isinstance(result, SearchResult)
