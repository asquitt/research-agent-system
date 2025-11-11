"""
Complete multi-agent research system test.

This demonstrates the full workflow with all agents working together.

Run with: python test_full_system.py
"""

import asyncio
import os
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

try:
    from src.orchestrator.orchestrator import ResearchOrchestrator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure:")
    print("1. All files are saved in correct locations")
    print("2. __init__.py files exist in all directories")
    print("3. You're running from project root")
    exit(1)


async def test_quick_research():
    """Test quick research mode (faster, cheaper)."""
    print("=" * 70)
    print("TEST 1: Quick Research Mode")
    print("=" * 70)
    
    orchestrator = ResearchOrchestrator(
        search_provider="duckduckgo",
        use_validation=False,  # Skip validation for speed
        max_searches=1
    )
    
    query = "What is artificial intelligence?"
    print(f"\nQuery: {query}")
    print("Mode: Quick (no validation, 1 search)")
    print("Estimated time: 10-15 seconds")
    print("Estimated cost: $0.01-0.02\n")
    
    result = await orchestrator.research(query, depth="quick")
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(orchestrator.get_summary(result))
    
    return result


async def test_comprehensive_research():
    """Test comprehensive research with all agents."""
    print("\n" + "=" * 70)
    print("TEST 2: Comprehensive Research Mode")
    print("=" * 70)
    
    orchestrator = ResearchOrchestrator(
        search_provider="duckduckgo",
        use_validation=True,  # Enable validation
        max_searches=2
    )
    
    query = "What are the benefits and drawbacks of renewable energy?"
    print(f"\nQuery: {query}")
    print("Mode: Comprehensive (with validation, 2 searches)")
    print("Estimated time: 30-45 seconds")
    print("Estimated cost: $0.05-0.10\n")
    
    result = await orchestrator.research(
        query=query,
        depth="comprehensive",
        save_results=True,
        output_prefix="test_comprehensive"
    )
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(orchestrator.get_summary(result))
    
    print("\n📁 Files saved:")
    print("   - test_comprehensive_full.json")
    print("   - test_comprehensive_report.md")
    
    # Show validation details
    if result.validated_findings:
        print("\n🔍 Validation Details:")
        for i, vf in enumerate(result.validated_findings[:3], 1):
            finding = vf["original_finding"]
            cred = vf["overall_credibility"]
            print(f"   {i}. {finding['title'][:50]}...")
            print(f"      Credibility: {cred:.2f}/1.0")
    
    return result


async def test_parallel_research():
    """Test researching multiple topics in parallel."""
    print("\n" + "=" * 70)
    print("TEST 3: Parallel Research")
    print("=" * 70)
    
    orchestrator = ResearchOrchestrator(
        search_provider="duckduckgo",
        use_validation=False,
        max_searches=1
    )
    
    queries = [
        "What is machine learning?",
        "What is blockchain?",
        "What is quantum computing?"
    ]
    
    print(f"\nResearching {len(queries)} topics in parallel:")
    for q in queries:
        print(f"  - {q}")
    
    print("\nEstimated time: 15-20 seconds (faster than sequential)")
    print("Estimated cost: $0.03-0.06\n")
    
    results = await orchestrator.research_parallel(queries, depth="quick")
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.query}")
        print(f"   Findings: {len(result.research_result.findings)}")
        print(f"   Sources: {len(result.research_result.sources)}")
        print(f"   Confidence: {result.final_report.confidence_level}")
        print(f"   Duration: {result.metadata.get('duration_seconds', 0):.1f}s")
    
    return results


async def test_validation_comparison():
    """Compare results with and without validation."""
    print("\n" + "=" * 70)
    print("TEST 4: Validation Comparison")
    print("=" * 70)
    
    query = "What are the health benefits of coffee?"
    
    print(f"\nQuery: {query}")
    print("\nRunning twice: with and without validation...\n")
    
    # Without validation
    print("1. Without validation...")
    orch_no_val = ResearchOrchestrator(use_validation=False, max_searches=2)
    result_no_val = await orch_no_val.research(query, depth="quick")
    
    # With validation
    print("\n2. With validation...")
    orch_with_val = ResearchOrchestrator(use_validation=True, max_searches=2)
    result_with_val = await orch_with_val.research(query, depth="quick")
    
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    print("\nWithout Validation:")
    print(f"  Duration: {result_no_val.metadata.get('duration_seconds', 0):.1f}s")
    print(f"  Confidence: {result_no_val.final_report.confidence_level}")
    print(f"  Insights: {len(result_no_val.final_report.key_insights)}")
    
    print("\nWith Validation:")
    print(f"  Duration: {result_with_val.metadata.get('duration_seconds', 0):.1f}s")
    print(f"  Confidence: {result_with_val.final_report.confidence_level}")
    print(f"  Insights: {len(result_with_val.final_report.key_insights)}")
    
    if result_with_val.validated_findings:
        avg_cred = sum(
            vf["overall_credibility"] for vf in result_with_val.validated_findings
        ) / len(result_with_val.validated_findings)
        print(f"  Avg Source Credibility: {avg_cred:.2f}/1.0")
    
    print("\n💡 Validation adds ~10-15s but provides credibility scores")
    
    return result_with_val


async def interactive_mode():
    """Interactive research mode."""
    print("\n" + "=" * 70)
    print("INTERACTIVE RESEARCH MODE")
    print("=" * 70)
    print("\nCommands:")
    print("  'quick' - Fast research without validation")
    print("  'full' - Comprehensive research with validation")
    print("  'save' - Save last results to file")
    print("  'quit' - Exit")
    print()
    
    orchestrator = ResearchOrchestrator(use_validation=False, max_searches=2)
    last_result = None
    
    while True:
        query = input("\nResearch query (or command): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if query.lower() == 'quick':
            mode = 'quick'
            query = input("Enter query: ").strip()
            orchestrator.use_validation = False
        elif query.lower() == 'full':
            mode = 'comprehensive'
            query = input("Enter query: ").strip()
            orchestrator.use_validation = True
        elif query.lower() == 'save':
            if last_result:
                last_result.save_json("interactive_results.json")
                last_result.save_markdown("interactive_report.md")
                print("✅ Saved to interactive_results.json and interactive_report.md")
            else:
                print("❌ No results to save yet")
            continue
        else:
            mode = 'quick'
        
        if not query:
            continue
        
        try:
            print(f"\n🔍 Researching in {mode} mode...")
            result = await orchestrator.research(query, depth=mode)
            last_result = result
            
            print("\n" + "=" * 70)
            print(orchestrator.get_summary(result))
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ Error: {e}")


async def run_all_tests():
    """Run all tests in sequence."""
    
    print("\n🚀 MULTI-AGENT RESEARCH SYSTEM - FULL TEST SUITE\n")
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set!")
        print("This system requires Claude API access.")
        return
    
    results = []
    
    try:
        # Test 1: Quick
        print("\n📌 Starting Test 1...")
        input("Press Enter to continue...")
        result1 = await test_quick_research()
        results.append(("Quick Research", True, result1))
        
        # Test 2: Comprehensive
        print("\n📌 Starting Test 2...")
        input("Press Enter to continue...")
        result2 = await test_comprehensive_research()
        results.append(("Comprehensive Research", True, result2))
        
        # Test 3: Parallel
        print("\n📌 Starting Test 3...")
        input("Press Enter to continue...")
        result3 = await test_parallel_research()
        results.append(("Parallel Research", True, result3))
        
        # Test 4: Validation comparison
        print("\n📌 Starting Test 4...")
        input("Press Enter to continue...")
        result4 = await test_validation_comparison()
        results.append(("Validation Comparison", True, result4))
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Failed test", False, None))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed, _ in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    
    if passed_count == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYour multi-agent research system is fully operational!")
        print("\nNext steps:")
        print("  1. Try interactive mode: python test_full_system.py interactive")
        print("  2. Experiment with different queries")
        print("  3. Build on top of this system for your use cases")
        
        # Offer interactive mode
        interactive = input("\nTry interactive mode now? (y/n): ").strip().lower()
        if interactive == 'y':
            await interactive_mode()
    else:
        print(f"\n⚠️  {len(results) - passed_count} test(s) failed")


async def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            await interactive_mode()
        elif sys.argv[1] == "quick":
            await test_quick_research()
        elif sys.argv[1] == "comprehensive":
            await test_comprehensive_research()
        elif sys.argv[1] == "parallel":
            await test_parallel_research()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python test_full_system.py [interactive|quick|comprehensive|parallel]")
    else:
        await run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())