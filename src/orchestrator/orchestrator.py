"""
Research Orchestrator - Coordinates multiple agents to complete research tasks.

This is the main entry point for the multi-agent research system.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from ..tools.web_search import WebSearchTool
from ..agents.researcher import ResearcherAgent, ResearchResult
from ..agents.validator import ValidatorAgent
from ..agents.synthesizer import SynthesizerAgent, SynthesizedReport

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Complete result from the orchestrated research process."""
    query: str
    research_result: ResearchResult
    validated_findings: List[Dict[str, Any]]
    final_report: SynthesizedReport
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "research_result": self.research_result.to_dict(),
            "validated_findings": self.validated_findings,
            "final_report": self.final_report.to_dict(),
            "metadata": self.metadata
        }
    
    def save_json(self, filename: str) -> None:
        """Save results to JSON file."""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved results to {filename}")
    
    def save_markdown(self, filename: str) -> None:
        """Save final report to Markdown file."""
        synthesizer = SynthesizerAgent()
        content = synthesizer._format_report(self.final_report)
        
        # Add metadata footer
        content += f"\n\n---\n\n"
        content += f"*Generated: {self.metadata.get('timestamp', 'Unknown')}*\n"
        content += f"*Total Sources: {len(self.research_result.sources)}*\n"
        content += f"*Research Confidence: {self.research_result.confidence}*\n"
        content += f"*Report Confidence: {self.final_report.confidence_level}*\n"
        
        with open(filename, "w") as f:
            f.write(content)
        logger.info(f"Saved report to {filename}")


class ResearchOrchestrator:
    """
    Orchestrates multiple agents to complete comprehensive research tasks.
    
    Workflow:
    1. Planner/Researcher: Gathers information
    2. Validator: Checks credibility and accuracy
    3. Synthesizer: Creates final report
    """
    
    def __init__(
        self,
        search_provider: str = "duckduckgo",
        use_validation: bool = True,
        max_searches: int = 3
    ):
        """
        Initialize the orchestrator.
        
        Args:
            search_provider: Which search provider to use
            use_validation: Whether to validate findings (costs more but better quality)
            max_searches: Maximum number of searches per query
        """
        self.use_validation = use_validation
        
        # Initialize tools
        logger.info(f"Initializing search tool: {search_provider}")
        self.search_tool = WebSearchTool(provider=search_provider)
        
        # Initialize agents
        logger.info("Initializing agents...")
        self.researcher = ResearcherAgent(max_searches=max_searches)
        self.researcher.register_tool("web_search", self.search_tool)
        
        if use_validation:
            self.validator = ValidatorAgent()
        else:
            self.validator = None
        
        self.synthesizer = SynthesizerAgent()
        
        logger.info("Orchestrator initialized successfully")
    
    async def research(
        self,
        query: str,
        depth: str = "comprehensive",
        save_results: bool = False,
        output_prefix: str = "research"
    ) -> OrchestrationResult:
        """
        Execute complete research workflow.
        
        Args:
            query: Research question
            depth: "quick" or "comprehensive"
            save_results: Whether to save to files
            output_prefix: Prefix for output files
            
        Returns:
            OrchestrationResult with all findings and reports
        """
        start_time = datetime.now()
        logger.info(f"Starting research: '{query}' (depth: {depth})")
        
        # Adjust parameters based on depth
        if depth == "quick":
            self.researcher.max_searches = 1
        else:
            self.researcher.max_searches = 3
        
        # Step 1: Research
        logger.info("STEP 1: Gathering information...")
        research_result = await self.researcher.research(query)
        
        logger.info(
            f"Research complete: {len(research_result.findings)} findings "
            f"from {len(research_result.sources)} sources"
        )
        
        # Step 2: Validate (optional)
        validated_findings = []
        if self.use_validation and self.validator and research_result.findings:
            logger.info("STEP 2: Validating findings...")
            validated_findings = await self.validator.validate_findings(
                research_result.findings
            )
            
            avg_cred = sum(
                vf["overall_credibility"] for vf in validated_findings
            ) / len(validated_findings)
            logger.info(f"Validation complete: avg credibility {avg_cred:.2f}/1.0")
        else:
            logger.info("STEP 2: Skipping validation")
        
        # Step 3: Synthesize
        logger.info("STEP 3: Synthesizing final report...")
        final_report = await self.synthesizer.synthesize(
            query=query,
            findings=research_result.findings,
            validated_findings=validated_findings if validated_findings else None
        )
        logger.info(
            f"Synthesis complete: {len(final_report.key_insights)} key insights, "
            f"confidence: {final_report.confidence_level}"
        )
        
        # Create result
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = OrchestrationResult(
            query=query,
            research_result=research_result,
            validated_findings=validated_findings,
            final_report=final_report,
            metadata={
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration,
                "depth": depth,
                "validation_used": self.use_validation,
                "num_findings": len(research_result.findings),
                "num_sources": len(research_result.sources)
            }
        )
        
        logger.info(f"Research complete in {duration:.1f} seconds")
        
        # Save if requested
        if save_results:
            result.save_json(f"{output_prefix}_full.json")
            result.save_markdown(f"{output_prefix}_report.md")
        
        return result
    
    async def research_parallel(
        self,
        queries: List[str],
        depth: str = "quick"
    ) -> List[OrchestrationResult]:
        """
        Research multiple queries in parallel.
        
        Args:
            queries: List of research questions
            depth: Research depth for each query
            
        Returns:
            List of OrchestrationResults
        """
        import asyncio
        
        logger.info(f"Starting parallel research on {len(queries)} queries")
        
        tasks = [self.research(query, depth=depth) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, OrchestrationResult)]
        failed_count = len(results) - len(valid_results)
        
        if failed_count > 0:
            logger.warning(f"{failed_count} queries failed")
        
        logger.info(f"Parallel research complete: {len(valid_results)} succeeded")
        return valid_results
    
    def get_summary(self, result: OrchestrationResult) -> str:
        """Get a quick summary of results."""
        return f"""
Research Query: {result.query}
Duration: {result.metadata.get('duration_seconds', 0):.1f}s
Findings: {result.metadata.get('num_findings', 0)}
Sources: {result.metadata.get('num_sources', 0)}
Research Confidence: {result.research_result.confidence}
Report Confidence: {result.final_report.confidence_level}

Executive Summary:
{result.final_report.executive_summary[:300]}...

Key Insights:
{chr(10).join(f"- {insight}" for insight in result.final_report.key_insights[:3])}
""".strip()


# Example usage
async def demo():
    """Demonstrate the full orchestrated research workflow."""
    
    # Initialize orchestrator
    orchestrator = ResearchOrchestrator(
        search_provider="duckduckgo",
        use_validation=True,  # Turn off to save costs
        max_searches=2
    )
    
    # Run research
    query = "What are the main arguments for and against nuclear energy?"
    print(f"Researching: {query}\n")
    print("This will take 30-60 seconds as all agents work together...\n")
    
    result = await orchestrator.research(
        query=query,
        depth="comprehensive",
        save_results=True,
        output_prefix="demo_research"
    )
    
    # Display summary
    print("=" * 70)
    print(orchestrator.get_summary(result))
    print("=" * 70)
    
    print("\nFull report saved to:")
    print("  - demo_research_full.json")
    print("  - demo_research_report.md")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())