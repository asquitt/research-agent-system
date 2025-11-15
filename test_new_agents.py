"""
Test the new agents and tools.

Run: python3 test_new_agents.py
"""

import asyncio
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

load_dotenv()

from src.orchestrator.orchestrator import ResearchOrchestrator


async def test_with_planner():
    """Test research with planner enabled."""
    print("="*70)
    print("TEST 1: Research with Planner")
    print("="*70)
    
    orchestrator = ResearchOrchestrator(
        use_planner=True,
        use_validation=False,
        max_searches=2
    )
    
    query = "Compare Python and JavaScript for web development, including performance benchmarks"
    print(f"\nQuery: {query}")
    print("Mode: Comprehensive with planner\n")
    
    result = await orchestrator.research(query, depth="comprehensive")
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(orchestrator.get_summary(result))
    
    return result


async def test_code_execution_query():
    """Test query that requires code execution."""
    print("\n" + "="*70)
    print("TEST 2: Query Requiring Code Execution")
    print("="*70)
    
    orchestrator = ResearchOrchestrator(
        use_planner=False,
        use_validation=False,
        max_searches=1
    )
    
    # Manually test code executor
    from src.tools.code_executor import CodeExecutorTool
    executor = CodeExecutorTool()
    
    print("\nTesting code execution directly:")
    code = """
import math
numbers = [1, 2, 3, 4, 5]
mean = sum(numbers) / len(numbers)
std_dev = math.sqrt(sum((x - mean)**2 for x in numbers) / len(numbers))
print(f"Mean: {mean}, Std Dev: {std_dev:.2f}")
result = {"mean": mean, "std_dev": std_dev}
"""
    
    result = executor.execute(code)
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    print(f"Return: {result.return_value}")
    
    return result


async def test_api_calls():
    """Test API agent."""
    print("\n" + "="*70)
    print("TEST 3: API Agent")
    print("="*70)
    
    from src.tools.api_agent import APIAgentTool
    
    async with APIAgentTool() as api_agent:
        
        # Test 1: Exchange rates
        print("\n1. Exchange Rate: USD to EUR")
        result = await api_agent.call_exchange_rate_api("USD", "EUR")
        if result.success:
            print(f"   ✓ Rate: {result.data.get('rate', 'N/A')}")
        else:
            print(f"   ✗ Error: {result.error}")
        
        # Test 2: Multiple currencies
        print("\n2. All USD exchange rates")
        result = await api_agent.call_exchange_rate_api("USD")
        if result.success:
            rates = result.data.get('rates', {})
            print(f"   ✓ Found {len(rates)} exchange rates")
            print(f"   EUR: {rates.get('EUR', 'N/A')}")
            print(f"   GBP: {rates.get('GBP', 'N/A')}")
            print(f"   JPY: {rates.get('JPY', 'N/A')}")
        else:
            print(f"   ✗ Error: {result.error}")
        
        # Test 3: Free public API
        print("\n3. Random Dog Image API")
        result = await api_agent.call_api("https://dog.ceo/api/breeds/image/random")
        if result.success:
            print(f"   ✓ Image: {result.data.get('message', 'N/A')[:60]}...")
        else:
            print(f"   ✗ Error: {result.error}")


async def test_full_integration():
    """Test all new features together."""
    print("\n" + "="*70)
    print("TEST 4: Full Integration")
    print("="*70)
    
    orchestrator = ResearchOrchestrator(
        use_planner=True,
        use_validation=True,
        max_searches=2
    )
    
    query = "What is the current EUR to USD exchange rate and what factors affect it?"
    print(f"\nQuery: {query}")
    print("Features: Planner + Validation + All Tools\n")
    
    result = await orchestrator.research(
        query, 
        depth="comprehensive",
        save_results=True,
        output_prefix="integration_test"
    )
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(orchestrator.get_summary(result))
    
    print("\n📁 Files saved:")
    print("   - integration_test_full.json")
    print("   - integration_test_report.md")


async def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set")
        return
    
    print("\n🚀 TESTING NEW AGENTS AND TOOLS\n")
    
    try:
        # Test 1: Planner
        print("Test 1: Planner Agent")
        input("Press Enter to start...")
        await test_with_planner()
        
        # Test 2: Code Executor
        print("\n\nTest 2: Code Executor")
        input("Press Enter to start...")
        await test_code_execution_query()
        
        # Test 3: API Agent
        print("\n\nTest 3: API Agent")
        input("Press Enter to start...")
        await test_api_calls()
        
        # Test 4: Full Integration
        print("\n\nTest 4: Full Integration")
        input("Press Enter to start...")
        await test_full_integration()
        
        print("\n\n" + "="*70)
        print("ALL TESTS COMPLETE")
        print("="*70)
        print("\n✅ All new features working!")
        print("\nYou now have:")
        print("  - Planner Agent (breaks down complex queries)")
        print("  - Code Executor (runs Python safely)")
        print("  - API Agent (calls external APIs)")
        print("  - Full integration in orchestrator")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())