"""
Quick start script to test your WebSearchTool setup.

This script will:
1. Test DuckDuckGo search (no API key needed)
2. Test Tavily if you have an API key
3. Show you what the output looks like

Run with: python quickstart.py
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the tool (adjust path if needed)
try:
    from src.tools.web_search import WebSearchTool
except ImportError:
    print("❌ Could not import WebSearchTool")
    print("Make sure you're running from the project root directory")
    print("and that web_search.py is in src/tools/")
    exit(1)


async def test_duckduckgo():
    """Test free DuckDuckGo search."""
    print("=" * 60)
    print("Testing DuckDuckGo (Free, No API Key Required)")
    print("=" * 60)
    
    try:
        search = WebSearchTool(provider="duckduckgo")
        
        query = "What is machine learning?"
        print(f"\nSearching for: '{query}'")
        print("Please wait...")
        
        results = await search.search(query, num_results=3)
        
        print(f"\n✅ Success! Found {len(results)} results\n")
        
        # Show formatted results
        print(search.format_results(results))
        
        # Show metadata
        metadata = search.get_metadata(results)
        print("\nMetadata:")
        print(f"  Total results: {metadata['total_results']}")
        print(f"  Unique sources: {metadata['unique_sources']}")
        print(f"  Sources: {', '.join(metadata['sources'][:5])}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DuckDuckGo search failed: {e}")
        return False


async def test_tavily():
    """Test Tavily search if API key is available."""
    print("\n" + "=" * 60)
    print("Testing Tavily (Requires API Key)")
    print("=" * 60)
    
    api_key = os.getenv("TAVILY_API_KEY")
    
    if not api_key:
        print("\n⚠️  TAVILY_API_KEY not found in environment")
        print("To test Tavily:")
        print("1. Sign up at https://tavily.com")
        print("2. Get your API key")
        print("3. Add it to your .env file")
        print("4. Run this script again")
        return False
    
    try:
        search = WebSearchTool(provider="tavily", api_key=api_key)
        
        query = "latest AI developments 2025"
        print(f"\nSearching for: '{query}'")
        print("Please wait...")
        
        results = await search.search(
            query, 
            num_results=3,
            search_depth="basic"
        )
        
        print(f"\n✅ Success! Found {len(results)} results\n")
        
        # Show formatted results
        print(search.format_results(results))
        
        # Show Tavily-specific features
        print("\nTavily Features:")
        if any(r.relevance_score for r in results):
            print("  ✓ Relevance scores available")
            for i, r in enumerate(results, 1):
                if r.relevance_score:
                    print(f"    [{i}] Score: {r.relevance_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Tavily search failed: {e}")
        return False


async def compare_providers():
    """Compare results from different providers."""
    print("\n" + "=" * 60)
    print("Comparing Search Providers")
    print("=" * 60)
    
    query = "quantum computing"
    providers_to_test = ["duckduckgo"]
    
    if os.getenv("TAVILY_API_KEY"):
        providers_to_test.append("tavily")
    
    print(f"\nQuery: '{query}'")
    print(f"Testing providers: {', '.join(providers_to_test)}\n")
    
    for provider in providers_to_test:
        print(f"\n--- {provider.upper()} ---")
        try:
            if provider == "tavily":
                search = WebSearchTool(provider=provider)
            else:
                search = WebSearchTool(provider=provider)
            
            results = await search.search(query, num_results=3)
            
            print(f"Results: {len(results)}")
            print(f"Sources: {', '.join(search.get_metadata(results)['sources'])}")
            
            # Show first result title
            if results:
                print(f"Top result: {results[0].title[:60]}...")
                
        except Exception as e:
            print(f"Error: {e}")


async def interactive_mode():
    """Interactive search mode."""
    print("\n" + "=" * 60)
    print("Interactive Search Mode")
    print("=" * 60)
    print("\nType 'quit' to exit\n")
    
    # Choose provider
    providers = ["duckduckgo"]
    if os.getenv("TAVILY_API_KEY"):
        providers.append("tavily")
    
    if len(providers) > 1:
        print("Available providers:")
        for i, p in enumerate(providers, 1):
            print(f"  {i}. {p}")
        choice = input("\nChoose provider (1-{}): ".format(len(providers)))
        try:
            provider = providers[int(choice) - 1]
        except (ValueError, IndexError):
            provider = "duckduckgo"
    else:
        provider = "duckduckgo"
    
    search = WebSearchTool(provider=provider)
    print(f"\nUsing {provider}")
    
    while True:
        query = input("\nEnter search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            results = await search.search(query, num_results=5)
            print(f"\nFound {len(results)} results:\n")
            
            for i, r in enumerate(results, 1):
                print(f"[{i}] {r.title}")
                print(f"    {r.source}")
                print(f"    {r.snippet[:100]}...")
                print()
                
        except Exception as e:
            print(f"Error: {e}")


async def main():
    """Run all tests."""
    print("\n🚀 Multi-Agent Research System - Web Search Tool Test\n")
    
    # Test 1: DuckDuckGo (always available)
    ddg_success = await test_duckduckgo()
    
    # Test 2: Tavily (if API key available)
    tavily_success = await test_tavily()
    
    # Test 3: Compare providers
    if ddg_success or tavily_success:
        await compare_providers()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"DuckDuckGo: {'✅ Working' if ddg_success else '❌ Failed'}")
    print(f"Tavily: {'✅ Working' if tavily_success else '⚠️  API key not configured'}")
    
    if ddg_success or tavily_success:
        print("\n✅ Web search tool is working!")
        print("\nNext steps:")
        print("1. Review the code in src/tools/web_search.py")
        print("2. Run tests: pytest tests/test_web_search.py")
        print("3. Move on to building your first agent (Step 2)")
    else:
        print("\n❌ Setup needs attention")
        print("Check the error messages above")
    
    # Optional: Interactive mode
    interactive = input("\nWant to try interactive search? (y/n): ").strip().lower()
    if interactive == 'y':
        await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())