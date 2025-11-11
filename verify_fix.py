"""
Quick verification script to test the fix.

Run with: python verify_fix.py
"""

import asyncio
import os
import json
import logging
from dotenv import load_dotenv

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

load_dotenv()

from src.tools.web_search import WebSearchTool
from src.agents.researcher import ResearcherAgent


async def test_fix():
    """Test that findings are now being extracted."""
    
    print("=" * 70)
    print("TESTING FIX: Search Results → Findings → File")
    print("=" * 70)
    
    # Initialize
    print("\n1. Setting up researcher...")
    researcher = ResearcherAgent(max_searches=2)
    search_tool = WebSearchTool(provider="duckduckgo")
    researcher.register_tool("web_search", search_tool)
    
    # Test query
    query = "What is Python programming?"
    print(f"\n2. Researching: '{query}'")
    print("   Please wait 15-30 seconds...\n")
    
    # Run research
    result = await researcher.research(query)
    
    # Check results
    print("\n3. Results:")
    print(f"   ✓ Searches planned: {result.metadata.get('num_searches', 0)}")
    print(f"   ✓ Search results: {result.metadata.get('num_results', 0)}")
    print(f"   ✓ Findings extracted: {len(result.findings)}")
    print(f"   ✓ Sources: {len(result.sources)}")
    print(f"   ✓ Confidence: {result.confidence}")
    
    if len(result.findings) > 0:
        print("\n✅ SUCCESS! Findings are being extracted!")
        
        # Show first finding
        print(f"\n   First finding:")
        f = result.findings[0]
        print(f"   Title: {f.title}")
        print(f"   Relevance: {f.relevance}")
        print(f"   Content: {f.content[:150]}...")
        print(f"   Key points: {len(f.key_points)}")
        
        # Save to file
        print("\n4. Saving to files...")
        
        # JSON
        with open("test_results.json", "w") as file:
            json.dump(result.to_dict(), file, indent=2)
        
        # Verify JSON
        with open("test_results.json", "r") as file:
            data = json.load(file)
            findings_in_file = len(data.get("findings", []))
        
        print(f"   ✓ JSON saved: {findings_in_file} findings in file")
        
        # Markdown
        formatted = researcher._format_research_result(result)
        with open("test_results.md", "w") as file:
            file.write(formatted)
        
        print(f"   ✓ Markdown saved: {len(formatted)} characters")
        
        if findings_in_file > 0:
            print("\n🎉 FIXED! Findings are now being saved to files!")
            print("\nCheck these files:")
            print("  - test_results.json")
            print("  - test_results.md")
        else:
            print("\n⚠️  Findings extracted but not in JSON file")
            print("   This is weird - let me know if you see this!")
    
    else:
        print("\n❌ ISSUE PERSISTS: No findings extracted")
        print(f"\n   Metadata: {result.metadata}")
        print("\n   Possible issues:")
        print("   1. Search returned 0 results (check num_results)")
        print("   2. Finding extraction failed (check logs above)")
        print("   3. JSON parsing failed (look for ERROR messages)")
    
    return result


async def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not set")
        print("Set it in your .env file to test")
        return
    
    try:
        await test_fix()
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())