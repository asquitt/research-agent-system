"""
Researcher Agent - Gathers information using various tools.

This agent:
1. Analyzes research queries
2. Decides which tools to use
3. Executes searches
4. Extracts and structures findings
"""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .base_agent import ToolUseAgent, AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class ResearchFinding:
    """Structured research finding."""
    title: str
    content: str
    source: str
    url: str
    relevance: str  # High, Medium, Low
    key_points: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "url": self.url,
            "relevance": self.relevance,
            "key_points": self.key_points
        }


@dataclass
class ResearchResult:
    """Complete research result with multiple findings."""
    query: str
    findings: List[ResearchFinding]
    summary: str
    sources: List[str]
    confidence: str  # High, Medium, Low
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "sources": self.sources,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class ResearcherAgent(ToolUseAgent):
    """
    Specialized agent for research tasks.
    
    Capabilities:
    - Web search
    - Information extraction
    - Source evaluation
    - Structured output
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.3,  # Lower for more focused research
        max_searches: int = 3
    ):
        super().__init__(
            name="Researcher",
            role="""You are an expert research agent. Your job is to:
1. Understand research queries and break them down if needed
2. Search for relevant, high-quality information
3. Extract key facts and insights
4. Cite sources properly
5. Assess the quality and relevance of information

Always prioritize:
- Accuracy over speed
- Quality sources (academic, official, reputable)
- Recent information when relevant
- Multiple perspectives when appropriate""",
            model=model,
            temperature=temperature
        )
        
        self.max_searches = max_searches
    
    async def _execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """
        Execute a research task.
        
        This is the main research flow:
        1. Analyze the query
        2. Perform searches
        3. Extract findings
        4. Structure results
        """
        logger.info(f"{self.name}: Starting research on: {task}")
        
        # Perform research
        research_result = await self.research(task, context)
        
        # Format for output
        formatted = self._format_research_result(research_result)
        
        return formatted
    
    async def research(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> ResearchResult:
        """
        Perform comprehensive research on a query.
        
        Args:
            query: Research question/topic
            context: Additional context or constraints
            
        Returns:
            ResearchResult with findings
        """
        context = context or {}
        
        # Step 1: Analyze query and plan searches
        search_queries = await self._plan_searches(query, context)
        logger.info(f"Planned {len(search_queries)} searches: {search_queries}")
        
        # Step 2: Execute searches
        all_search_results = []
        for search_query in search_queries[:self.max_searches]:
            if "web_search" in self.tools:
                try:
                    logger.info(f"Executing search: '{search_query}'")
                    
                    # Call the tool's search method directly
                    tool = self.tools["web_search"]
                    results = await tool.search(query=search_query, num_results=5)
                    
                    logger.info(f"Search '{search_query}': returned {len(results)} results")
                    
                    if results:
                        all_search_results.extend(results)
                        logger.debug(f"First result: {results[0].title}")
                    else:
                        logger.warning(f"Search '{search_query}' returned no results")
                        
                except Exception as e:
                    logger.error(f"Search failed for '{search_query}': {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.warning("No web_search tool available")
        
        logger.info(f"Total search results collected: {len(all_search_results)}")
        
        # Step 3: Extract and structure findings
        findings = await self._extract_findings(query, all_search_results)
        
        # Step 4: Generate summary
        summary = await self._generate_summary(query, findings)
        
        # Step 5: Assess confidence
        confidence = self._assess_confidence(findings)
        
        # Collect unique sources
        sources = list(set(f.source for f in findings))
        
        return ResearchResult(
            query=query,
            findings=findings,
            summary=summary,
            sources=sources,
            confidence=confidence,
            metadata={
                "num_searches": len(search_queries),
                "num_results": len(all_search_results),
                "num_findings": len(findings)
            }
        )
    
    async def _plan_searches(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Plan what to search for based on the query.
        
        Returns:
            List of search queries to execute
        """
        prompt = f"""
Research query: {query}

Context: {json.dumps(context, indent=2)}

Generate 1-3 specific search queries that will help answer this research question.
Make queries:
- Specific and targeted
- Diverse to cover different angles
- Optimized for web search (2-6 words each)

Respond in JSON format:
{{
    "queries": ["query1", "query2", "query3"],
    "reasoning": "why these queries will be effective"
}}
"""
        
        response = await self.call_llm(prompt, temperature=0.3, max_tokens=500)
        
        try:
            # Parse response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response)
            queries = data.get("queries", [query])  # Fallback to original query
            
            logger.info(f"Planned searches: {queries}")
            return queries[:self.max_searches]
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse search plan, using original query")
            return [query]
    
    async def _extract_findings(
        self,
        query: str,
        search_results: List[Any]
    ) -> List[ResearchFinding]:
        """
        Extract structured findings from search results.
        
        Args:
            query: Original research query
            search_results: Raw search results from tools
            
        Returns:
            List of structured ResearchFinding objects
        """
        if not search_results:
            logger.warning("No search results to extract findings from")
            return []
        
        logger.info(f"Extracting findings from {len(search_results)} search results")
        
        # Format search results for the LLM
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            formatted_results.append(
                f"[{i}] {result.title}\n"
                f"Source: {result.source}\n"
                f"URL: {result.url}\n"
                f"Content: {result.snippet}\n"
            )
        
        results_text = "\n".join(formatted_results)
        
        prompt = f"""
Research query: {query}

Search results:
{results_text}

Extract the most relevant and important findings from these search results.
For each finding:
1. Summarize the key information
2. Extract 2-4 key points
3. Assess relevance (High/Medium/Low)
4. Cite the source

Focus on:
- Direct answers to the query
- Important context and background
- Different perspectives if available
- Recent developments

You MUST respond with ONLY valid JSON in this exact format (no other text):
{{
    "findings": [
        {{
            "title": "Finding title",
            "content": "2-3 sentence summary",
            "source": "source name",
            "url": "source url",
            "relevance": "High",
            "key_points": ["point 1", "point 2", "point 3"]
        }}
    ]
}}

Include 3-5 most relevant findings.
"""
        
        response = await self.call_llm(prompt, temperature=0.3, max_tokens=2000)
        
        logger.debug(f"Raw LLM response for findings: {response[:200]}...")
        
        try:
            # Parse response - handle various formats
            response = response.strip()
            
            # Remove markdown code blocks
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            # Try to find JSON if there's extra text
            if not response.startswith("{"):
                # Look for JSON object in the response
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end > start:
                    response = response[start:end]
            
            logger.debug(f"Cleaned response: {response[:200]}...")
            
            data = json.loads(response)
            
            if "findings" not in data:
                logger.error("Response missing 'findings' key")
                logger.debug(f"Full response: {response}")
                return []
            
            findings = []
            for f in data.get("findings", []):
                finding = ResearchFinding(
                    title=f.get("title", "Untitled"),
                    content=f.get("content", ""),
                    source=f.get("source", "Unknown"),
                    url=f.get("url", ""),
                    relevance=f.get("relevance", "Medium"),
                    key_points=f.get("key_points", [])
                )
                findings.append(finding)
                logger.debug(f"Extracted finding: {finding.title}")
            
            logger.info(f"Successfully extracted {len(findings)} findings")
            return findings
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse findings JSON: {e}")
            logger.error(f"Problematic response: {response[:500]}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error extracting findings: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def _generate_summary(
        self,
        query: str,
        findings: List[ResearchFinding]
    ) -> str:
        """Generate a comprehensive summary of all findings."""
        if not findings:
            return "No findings available to summarize."
        
        findings_text = []
        for i, finding in enumerate(findings, 1):
            findings_text.append(
                f"{i}. {finding.title}\n"
                f"   {finding.content}\n"
                f"   Key points: {', '.join(finding.key_points)}\n"
            )
        
        prompt = f"""
Research query: {query}

Findings:
{chr(10).join(findings_text)}

Create a comprehensive 3-4 paragraph summary that:
1. Directly answers the research query
2. Synthesizes information from all findings
3. Highlights the most important points
4. Mentions any important caveats or limitations
5. Is clear and well-structured

Write in a professional, informative tone.
"""
        
        summary = await self.call_llm(prompt, temperature=0.5, max_tokens=1000)
        return summary.strip()
    
    def _assess_confidence(self, findings: List[ResearchFinding]) -> str:
        """
        Assess confidence in the research results.
        
        Based on:
        - Number of findings
        - Quality of sources
        - Consistency of information
        """
        if not findings:
            return "Low"
        
        high_relevance = sum(1 for f in findings if f.relevance == "High")
        
        if len(findings) >= 3 and high_relevance >= 2:
            return "High"
        elif len(findings) >= 2:
            return "Medium"
        else:
            return "Low"
    
    def _format_research_result(self, result: ResearchResult) -> str:
        """Format research result as readable text."""
        output = []
        
        output.append(f"# Research Results: {result.query}\n")
        output.append(f"**Confidence:** {result.confidence}\n")
        output.append(f"**Sources:** {len(result.sources)}\n")
        output.append(f"\n## Summary\n{result.summary}\n")
        
        output.append("\n## Key Findings\n")
        for i, finding in enumerate(result.findings, 1):
            output.append(f"\n### {i}. {finding.title}")
            output.append(f"**Source:** {finding.source}")
            output.append(f"**Relevance:** {finding.relevance}\n")
            output.append(finding.content + "\n")
            
            if finding.key_points:
                output.append("**Key Points:**")
                for point in finding.key_points:
                    output.append(f"- {point}")
            
            output.append(f"\n[Source]({finding.url})\n")
        
        output.append("\n## Sources")
        for source in result.sources:
            output.append(f"- {source}")
        
        return "\n".join(output)


# Example usage
async def demo():
    """Demonstrate the researcher agent."""
    from src.tools.web_search import WebSearchTool
    
    # Create researcher
    researcher = ResearcherAgent()
    
    # Register search tool
    search_tool = WebSearchTool(provider="duckduckgo")
    researcher.register_tool("web_search", search_tool)
    
    # Run research
    print("=== Research Demo ===\n")
    
    query = "What are the main benefits and drawbacks of nuclear energy?"
    print(f"Query: {query}\n")
    print("Researching... (this may take 10-20 seconds)\n")
    
    result = await researcher.research(query)
    
    # Print results
    print(f"Found {len(result.findings)} findings from {len(result.sources)} sources")
    print(f"Confidence: {result.confidence}\n")
    print(f"Summary:\n{result.summary}\n")
    
    print("\nFindings:")
    for i, finding in enumerate(result.findings, 1):
        print(f"\n{i}. {finding.title}")
        print(f"   Relevance: {finding.relevance}")
        print(f"   {finding.content[:150]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())