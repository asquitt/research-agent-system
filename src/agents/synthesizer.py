"""
Synthesizer Agent - Combines findings into coherent reports.

This agent:
1. Synthesizes multiple findings
2. Identifies patterns and themes
3. Resolves contradictions
4. Generates structured reports
"""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .base_agent import Agent, AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class SynthesizedReport:
    """Complete synthesized research report."""
    query: str
    executive_summary: str
    key_insights: List[str]
    detailed_analysis: str
    contradictions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    confidence_level: str = "Medium"
    sources_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "executive_summary": self.executive_summary,
            "key_insights": self.key_insights,
            "detailed_analysis": self.detailed_analysis,
            "contradictions": self.contradictions,
            "limitations": self.limitations,
            "confidence_level": self.confidence_level,
            "sources_used": self.sources_used
        }


class SynthesizerAgent(Agent):
    """
    Specialized agent for synthesizing research findings.
    
    Capabilities:
    - Multi-source synthesis
    - Pattern identification
    - Contradiction resolution
    - Structured report generation
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.5  # Balanced for creative synthesis
    ):
        super().__init__(
            name="Synthesizer",
            role="""You are an expert research synthesizer. Your job is to:
1. Combine information from multiple sources into coherent narratives
2. Identify key patterns, themes, and insights across findings
3. Highlight agreements and disagreements between sources
4. Acknowledge limitations and uncertainties
5. Create clear, well-structured reports

Always:
- Synthesize rather than just summarize
- Draw connections between different findings
- Present multiple perspectives fairly
- Be clear about confidence levels
- Write in clear, professional prose""",
            model=model,
            temperature=temperature
        )
    
    async def _execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """Execute synthesis task."""
        findings = context.get("findings", [])
        validated_findings = context.get("validated_findings")
        
        if not findings:
            return "No findings provided for synthesis."
        
        # Synthesize the research
        report = await self.synthesize(
            query=task,
            findings=findings,
            validated_findings=validated_findings
        )
        
        # Format as readable report
        return self._format_report(report)
    
    async def synthesize(
        self,
        query: str,
        findings: List[Any],
        validated_findings: Optional[List[Dict[str, Any]]] = None
    ) -> SynthesizedReport:
        """
        Synthesize findings into a comprehensive report.
        
        Args:
            query: Original research question
            findings: List of research findings
            validated_findings: Optional validation results
            
        Returns:
            SynthesizedReport
        """
        logger.info(f"Synthesizing {len(findings)} findings for query: {query}")
        
        # Prepare findings for synthesis
        findings_text = self._prepare_findings_text(findings, validated_findings)
        
        # Generate executive summary
        exec_summary = await self._generate_executive_summary(query, findings_text)
        
        # Extract key insights
        key_insights = await self._extract_key_insights(query, findings_text)
        
        # Generate detailed analysis
        detailed_analysis = await self._generate_detailed_analysis(
            query, findings_text, key_insights
        )
        
        # Identify contradictions
        contradictions = await self._identify_contradictions(findings_text)
        
        # Identify limitations (not async)
        limitations = self._identify_limitations(findings, validated_findings)
        
        # Assess confidence
        confidence = self._assess_confidence(findings, validated_findings)
        
        # Collect sources
        sources = list(set(f.source for f in findings if hasattr(f, 'source')))
        
        return SynthesizedReport(
            query=query,
            executive_summary=exec_summary,
            key_insights=key_insights,
            detailed_analysis=detailed_analysis,
            contradictions=contradictions,
            limitations=limitations,
            confidence_level=confidence,
            sources_used=sources
        )
    
    def _prepare_findings_text(
        self,
        findings: List[Any],
        validated_findings: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Prepare findings text for LLM processing."""
        
        parts = []
        for i, finding in enumerate(findings, 1):
            # Get validation info if available
            credibility_note = ""
            if validated_findings and i <= len(validated_findings):
                cred = validated_findings[i-1].get("overall_credibility", 0)
                credibility_note = f" [Credibility: {cred:.2f}/1.0]"
            
            parts.append(
                f"[{i}] {finding.title}{credibility_note}\n"
                f"Source: {finding.source}\n"
                f"Content: {finding.content}\n"
                f"Key Points: {', '.join(finding.key_points)}\n"
            )
        
        return "\n".join(parts)
    
    async def _generate_executive_summary(
        self,
        query: str,
        findings_text: str
    ) -> str:
        """Generate a concise executive summary."""
        
        prompt = f"""
Research Question: {query}

Findings:
{findings_text}

Write a concise 2-3 paragraph executive summary that:
1. Directly answers the research question
2. Highlights the most important points
3. Synthesizes information from all sources
4. Is accessible to a general audience

Do not list findings separately - synthesize them into a cohesive narrative.
"""
        
        summary = await self.call_llm(prompt, temperature=0.4, max_tokens=500)
        return summary.strip()
    
    async def _extract_key_insights(
        self,
        query: str,
        findings_text: str
    ) -> List[str]:
        """Extract 4-6 key insights from the findings."""
        
        prompt = f"""
Research Question: {query}

Findings:
{findings_text}

Extract 4-6 key insights that emerge from synthesizing these findings.
Each insight should:
- Be substantive and non-obvious
- Draw from multiple findings when possible
- Be clearly stated in 1-2 sentences

Respond in JSON format:
{{
    "insights": [
        "First key insight...",
        "Second key insight...",
        "Third key insight..."
    ]
}}
"""
        
        response = await self.call_llm(prompt, temperature=0.5, max_tokens=800)
        
        try:
            # Parse response
            response = response.strip()
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
            insights = data.get("insights", [])
            
            logger.info(f"Extracted {len(insights)} key insights")
            return insights
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse insights: {e}")
            return ["Unable to extract structured insights"]
    
    async def _generate_detailed_analysis(
        self,
        query: str,
        findings_text: str,
        key_insights: List[str]
    ) -> str:
        """Generate detailed analysis section."""
        
        insights_text = "\n".join(f"- {insight}" for insight in key_insights)
        
        prompt = f"""
Research Question: {query}

Key Insights Identified:
{insights_text}

All Findings:
{findings_text}

Write a detailed analysis (3-4 paragraphs) that:
1. Expands on the key insights with supporting evidence
2. Discusses important nuances and context
3. Compares and contrasts different perspectives
4. Explains implications or significance

Write in clear, professional prose. Cite sources naturally (e.g., "According to Nature.com...").
"""
        
        analysis = await self.call_llm(prompt, temperature=0.5, max_tokens=1500)
        return analysis.strip()
    
    async def _identify_contradictions(self, findings_text: str) -> List[str]:
        """Identify any contradictions between findings."""
        
        prompt = f"""
Findings:
{findings_text}

Analyze these findings for contradictions or disagreements.
Look for:
- Direct contradictions (one source says X, another says not-X)
- Different conclusions from similar evidence
- Conflicting claims about facts or figures

Respond in JSON format:
{{
    "contradictions": [
        "Description of contradiction 1",
        "Description of contradiction 2"
    ]
}}

If there are no significant contradictions, return empty array.
"""
        
        response = await self.call_llm(prompt, temperature=0.3, max_tokens=500)
        
        try:
            response = response.strip()
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
            return data.get("contradictions", [])
            
        except json.JSONDecodeError:
            return []
    
    def _identify_limitations(
        self,
        findings: List[Any],
        validated_findings: Optional[List[Dict[str, Any]]]
    ) -> List[str]:
        """Identify limitations of the research."""
        
        limitations = []
        
        # Check number of sources
        if len(findings) < 3:
            limitations.append("Limited number of sources consulted")
        
        # Check credibility if validation available
        if validated_findings:
            avg_cred = sum(
                vf.get("overall_credibility", 0) for vf in validated_findings
            ) / len(validated_findings)
            
            if avg_cred < 0.6:
                limitations.append("Some sources have lower credibility scores")
        
        # Check source diversity
        sources = [f.source for f in findings if hasattr(f, 'source')]
        if len(set(sources)) < len(sources) * 0.5:
            limitations.append("Limited source diversity")
        
        return limitations
    
    def _assess_confidence(
        self,
        findings: List[Any],
        validated_findings: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Assess overall confidence level."""
        
        if not findings:
            return "Low"
        
        # Base score on number and validation
        score = 0
        
        if len(findings) >= 4:
            score += 2
        elif len(findings) >= 2:
            score += 1
        
        if validated_findings:
            avg_cred = sum(
                vf.get("overall_credibility", 0) for vf in validated_findings
            ) / len(validated_findings)
            
            if avg_cred >= 0.7:
                score += 2
            elif avg_cred >= 0.5:
                score += 1
        else:
            score += 1  # Assume medium if no validation
        
        if score >= 3:
            return "High"
        elif score >= 2:
            return "Medium"
        else:
            return "Low"
    
    def _format_report(self, report: SynthesizedReport) -> str:
        """Format report as readable text."""
        
        output = []
        
        output.append(f"# Research Report: {report.query}\n")
        output.append(f"**Confidence Level:** {report.confidence_level}\n")
        
        output.append("\n## Executive Summary\n")
        output.append(report.executive_summary + "\n")
        
        output.append("\n## Key Insights\n")
        for i, insight in enumerate(report.key_insights, 1):
            output.append(f"{i}. {insight}\n")
        
        output.append("\n## Detailed Analysis\n")
        output.append(report.detailed_analysis + "\n")
        
        if report.contradictions:
            output.append("\n## Contradictions & Disagreements\n")
            for contradiction in report.contradictions:
                output.append(f"- {contradiction}\n")
        
        if report.limitations:
            output.append("\n## Limitations\n")
            for limitation in report.limitations:
                output.append(f"- {limitation}\n")
        
        output.append(f"\n## Sources Consulted ({len(report.sources_used)})\n")
        for source in report.sources_used:
            output.append(f"- {source}\n")
        
        return "\n".join(output)


# Example usage
async def demo():
    """Demonstrate the synthesizer agent."""
    from .researcher import ResearchFinding
    
    # Mock findings
    findings = [
        ResearchFinding(
            title="Climate Change Overview",
            content="Global temperatures have risen 1.1°C since pre-industrial times.",
            source="ipcc.ch",
            url="https://ipcc.ch/report",
            relevance="High",
            key_points=["1.1°C increase", "Human caused", "Urgent action needed"]
        ),
        ResearchFinding(
            title="Climate Impacts",
            content="Rising temperatures cause more extreme weather events.",
            source="noaa.gov",
            url="https://noaa.gov/climate",
            relevance="High",
            key_points=["Extreme weather", "Sea level rise", "Ecosystem disruption"]
        ),
    ]
    
    # Synthesize
    synthesizer = SynthesizerAgent()
    report = await synthesizer.synthesize(
        query="What is climate change and what are its impacts?",
        findings=findings
    )
    
    # Display
    print(synthesizer._format_report(report))


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
