"""
Test script for the Researcher Agent.

This script demonstrates the full research workflow:
1. Initialize researcher with tools
2. Execute research queries
3. Display structured results

Run with: python test_researcher.py
"""

import asyncio
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our components
try:
    from src.tools.web_search import WebSearchTool
    from src.agents.researcher import ResearcherAgent, ResearchResult
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure:")
    print("1. You're running from the project root")
    print("2. All files are in the correct locations")
    print("3. You have __init__.py files in src/, src/tools/, src/agents/")
    exit(1)


async def test_basic_research():
    """Test basic research functionality."""
    print("=" * 70)
    print("TEST 1: Basic Research")
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: ANTHROPIC_API_KEY not found!")
        print("The researcher agent needs Claude API access to work.")
        print("\nTo fix this:")
        print("1. Sign up at https://console.anthropic.com/")
        print("2. Get your API key")
        print("3. Add to .env file: ANTHROPIC_API_KEY=sk-ant-...")
        return False
    
    try:
        # Initialize components
        print("\n1. Initializing researcher agent...")
        researcher = ResearcherAgent(max_searches=2)  # Limit searches to save costs
        
        print("2. Registering web search tool...")
        search_tool = WebSearchTool(provider="duckduckgo")
        researcher.register_tool("web_search", search_tool)
        
        # Execute research
        query = "What is quantum computing?"
        print(f"\n3. Researching: '{query}'")
        print("   (This will take 15-30 seconds...)\n")
        
        result = await researcher.research(query)
        
        # Display results
        print("\n" + "=" * 70)
        print("RESEARCH RESULTS")
        print("=" * 70)
        
        print(f"\n📊 Metadata:")
        print(f"   Query: {result.query}")
        print(f"   Findings: {len(result.findings)}")
        print(f"   Sources: {len(result.sources)}")
        print(f"   Confidence: {result.confidence}")
        
        print(f"\n📝 Summary:")
        print(f"   {result.summary}\n")
        
        print("🔍 Key Findings:")
        for i, finding in enumerate(result.findings[:3], 1):  # Show top 3
            print(f"\n   {i}. {finding.title}")
            print(f"      Relevance: {finding.relevance}")
            print(f"      Source: {finding.source}")
            print(f"      {finding.content[:200]}...")
            if finding.key_points:
                print(f"      Key points:")
                for point in finding.key_points[:2]:
                    print(f"        • {point}")
        
        print(f"\n📚 All Sources:")
        for source in result.sources:
            print(f"   - {source}")
        
        print("\n✅ Test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_complex_research():
    """Test with a more complex, multi-faceted query."""
    print("\n" + "=" * 70)
    print("TEST 2: Complex Research (Multi-perspective)")
    print("=" * 70)
    
    try:
        researcher = ResearcherAgent(max_searches=3)
        search_tool = WebSearchTool(provider="duckduckgo")
        researcher.register_tool("web_search", search_tool)
        
        query = "What are the pros and cons of remote work?"
        print(f"\nResearching: '{query}'")
        print("(Looking for multiple perspectives...)\n")
        
        result = await researcher.research(query)
        
        print(f"✅ Found {len(result.findings)} findings")
        print(f"   Confidence: {result.confidence}")
        print(f"\nSummary preview:")
        print(f"   {result.summary[:300]}...\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_save_results():
    """Test saving research results to JSON."""
    print("\n" + "=" * 70)
    print("TEST 3: Save Results to File")
    print("=" * 70)
    
    try:
        researcher = ResearcherAgent(max_searches=2)
        search_tool = WebSearchTool(provider="duckduckgo")
        researcher.register_tool("web_search", search_tool)
        
        query = "What is machine learning?"
        print(f"\nResearching: '{query}'")
        
        result = await researcher.research(query)
        
        # Save to JSON
        output_file = "research_results.json"
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"\n✅ Results saved to {output_file}")
        print(f"   File size: {os.path.getsize(output_file)} bytes")
        
        # Save formatted version
        formatted_file = "research_results.md"
        formatted = researcher._format_research_result(result)
        with open(formatted_file, "w") as f:
            f.write(formatted)
        
        print(f"✅ Formatted results saved to {formatted_file}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def interactive_research():
    """Interactive mode - research any query."""
    print("\n" + "=" * 70)
    print("INTERACTIVE RESEARCH MODE")
    print("=" * 70)
    print("\nType 'quit' to exit\n")
    
    # Initialize once
    researcher = ResearcherAgent(max_searches=2)
    search_tool = WebSearchTool(provider="duckduckgo")
    researcher.register_tool("web_search", search_tool)
    
    while True:
        query = input("Enter research question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            print("\n🔍 Researching...\n")
            result = await researcher.research(query)
            
            print(f"Confidence: {result.confidence}")
            print(f"Sources: {len(result.sources)}\n")
            print("Summary:")
            print(result.summary)
            print("\n" + "-" * 70 + "\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")


async def run_all_tests():
    """Run all tests in sequence."""
    print("\n🚀 RESEARCHER AGENT TEST SUITE\n")
    
    results = []
    
    # Test 1: Basic research
    results.append(("Basic Research", await test_basic_research()))
    
    # Test 2: Complex research (only if test 1 passed)
    if results[0][1]:
        results.append(("Complex Research", await test_complex_research()))
        results.append(("Save Results", await test_save_results()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Your researcher agent is working!\n")
        
        # Offer interactive mode
        interactive = input("Want to try interactive research mode? (y/n): ").strip().lower()
        if interactive == 'y':
            await interactive_research()
    else:
        print("\n⚠️  Some tests failed. Check the errors above.\n")


async def main():
    """Main entry point."""
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            await interactive_research()
        elif sys.argv[1] == "basic":
            await test_basic_research()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python test_researcher.py [interactive|basic]")
    else:
        # Run all tests
        await run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())