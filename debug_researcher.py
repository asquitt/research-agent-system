"""
Debug script to troubleshoot finding extraction issues.

This script helps identify where findings might be getting lost.

Run with: python debug_researcher.py
"""

import asyncio
import os
import json
import logging
from dotenv import load_dotenv

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

try:
    from src.tools.web_search import WebSearchTool
    from src.agents.researcher import ResearcherAgent
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


async def debug_search_results():
    """Test that search is returning results."""
    print("=" * 70)
    print("DEBUG STEP 1: Testing Search Tool")
    print("=" * 70)
    
    search_tool = WebSearchTool(provider="duckduckgo")
    
    query = "machine learning"
    print(f"\nSearching for: '{query}'")
    
    results = await search_tool.search(query, num_results=3)
    
    print(f"\n✅ Search returned {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Title: {result.title}")
        print(f"  Source: {result.source}")
        print(f"  URL: {result.url}")
        print(f"  Snippet: {result.snippet[:100]}...")
        print()
    
    return results


async def debug_finding_extraction():
    """Test the finding extraction process step by step."""
    print("\n" + "=" * 70)
    print("DEBUG STEP 2: Testing Finding Extraction")
    print("=" * 70)
    
    # Initialize
    researcher = ResearcherAgent(max_searches=1)
    search_tool = WebSearchTool(provider="duckduckgo")
    researcher.register_tool("web_search", search_tool)
    
    query = "What is artificial intelligence?"
    
    print(f"\n1. Planning searches for: '{query}'")
    search_queries = await researcher._plan_searches(query, {})
    print(f"   Planned queries: {search_queries}")
    
    print("\n2. Executing search...")
    search_results = []
    for sq in search_queries[:1]:  # Just do one search
        results = await search_tool.search(sq, num_results=3)
        search_results.extend(results)
        print(f"   Search '{sq}' returned {len(results)} results")
    
    print(f"\n3. Total search results: {len(search_results)}")
    
    print("\n4. Extracting findings...")
    print("   (This calls Claude to parse the search results)")
    
    findings = await researcher._extract_findings(query, search_results)
    
    print(f"\n5. Extracted {len(findings)} findings")
    
    if findings:
        print("\nFindings extracted successfully:")
        for i, finding in enumerate(findings, 1):
            print(f"\n  Finding {i}:")
            print(f"    Title: {finding.title}")
            print(f"    Content: {finding.content[:150]}...")
            print(f"    Relevance: {finding.relevance}")
            print(f"    Key points: {len(finding.key_points)}")
            if finding.key_points:
                for point in finding.key_points:
                    print(f"      - {point}")
    else:
        print("\n⚠️  WARNING: No findings extracted!")
        print("\nPossible causes:")
        print("  1. Claude's response wasn't valid JSON")
        print("  2. The JSON didn't have 'findings' key")
        print("  3. An exception occurred during parsing")
        print("\nCheck the DEBUG logs above for details.")
    
    return findings


async def debug_full_research():
    """Test the complete research flow."""
    print("\n" + "=" * 70)
    print("DEBUG STEP 3: Testing Full Research Flow")
    print("=" * 70)
    
    researcher = ResearcherAgent(max_searches=2)
    search_tool = WebSearchTool(provider="duckduckgo")
    researcher.register_tool("web_search", search_tool)
    
    query = "What is deep learning?"
    print(f"\nResearch query: '{query}'")
    print("Running full research pipeline...\n")
    
    result = await researcher.research(query)
    
    print("\nResearch completed!")
    print(f"  Query: {result.query}")
    print(f"  Findings: {len(result.findings)}")
    print(f"  Sources: {len(result.sources)}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Summary length: {len(result.summary)} chars")
    
    # Convert to dict and check
    result_dict = result.to_dict()
    print(f"\n  Dict 'findings' key has: {len(result_dict.get('findings', []))} items")
    
    if result.findings:
        print("\n✅ SUCCESS: Findings are present")
        
        # Save to file
        with open("debug_results.json", "w") as f:
            json.dump(result_dict, f, indent=2)
        print("\n  Saved to debug_results.json")
        
        # Verify the file
        with open("debug_results.json", "r") as f:
            loaded = json.load(f)
            print(f"  Verified file has {len(loaded.get('findings', []))} findings")
    else:
        print("\n❌ PROBLEM: No findings in result")
        print("\n  Diagnosis:")
        print("    - Search results were found (we saw them earlier)")
        print("    - But _extract_findings returned empty list")
        print("    - Check the logs for JSON parsing errors")
    
    return result


async def test_json_parsing():
    """Test that we can parse Claude's response format."""
    print("\n" + "=" * 70)
    print("DEBUG STEP 4: Testing JSON Parsing")
    print("=" * 70)
    
    # Simulate various response formats we might get from Claude
    test_cases = [
        # Case 1: Clean JSON
        '''{"findings": [{"title": "Test", "content": "Content", "source": "example.com", "url": "http://example.com", "relevance": "High", "key_points": ["point 1"]}]}''',
        
        # Case 2: JSON with markdown
        '''```json
{"findings": [{"title": "Test", "content": "Content", "source": "example.com", "url": "http://example.com", "relevance": "High", "key_points": ["point 1"]}]}
```''',
        
        # Case 3: JSON with preamble
        '''Here are the findings:
{"findings": [{"title": "Test", "content": "Content", "source": "example.com", "url": "http://example.com", "relevance": "High", "key_points": ["point 1"]}]}''',
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"Input: {test_case[:100]}...")
        
        try:
            # Simulate the parsing logic from _extract_findings
            response = test_case.strip()
            
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            if not response.startswith("{"):
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end > start:
                    response = response[start:end]
            
            data = json.loads(response)
            print(f"  ✅ Parsed successfully: {len(data.get('findings', []))} findings")
            
        except Exception as e:
            print(f"  ❌ Parse failed: {e}")


async def main():
    """Run all debug tests."""
    print("\n🔍 RESEARCHER AGENT DEBUG SUITE\n")
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set!")
        print("Cannot test finding extraction without it.")
        return
    
    try:
        # Step 1: Test search
        print("\nStep 1: Testing search tool...")
        await debug_search_results()
        input("\nPress Enter to continue to Step 2...")
        
        # Step 2: Test finding extraction
        print("\nStep 2: Testing finding extraction...")
        await debug_finding_extraction()
        input("\nPress Enter to continue to Step 3...")
        
        # Step 3: Test full flow
        print("\nStep 3: Testing full research flow...")
        await debug_full_research()
        
        # Step 4: Test JSON parsing
        print("\nStep 4: Testing JSON parsing...")
        await test_json_parsing()
        
        print("\n" + "=" * 70)
        print("DEBUG COMPLETE")
        print("=" * 70)
        print("\nIf findings are still empty:")
        print("1. Check the DEBUG logs above for 'Failed to parse findings'")
        print("2. Look for the 'Raw LLM response' log message")
        print("3. The issue is likely in how Claude is formatting its response")
        print("\nYou can share the logs with me to help diagnose!")
        
    except Exception as e:
        print(f"\n❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())