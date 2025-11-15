"""
Validator Agent - Checks source credibility and fact accuracy.

This agent:
1. Evaluates source credibility
2. Cross-references facts
3. Identifies potential biases
4. Assigns confidence scores
"""

import json
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field

from .base_agent import Agent, AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a finding or source."""
    is_valid: bool
    credibility_score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reasoning: str = ""
    
    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "credibility_score": self.credibility_score,
            "issues": self.issues,
            "warnings": self.warnings,
            "reasoning": self.reasoning
        }


@dataclass
class SourceEvaluation:
    """Evaluation of a source's credibility."""
    source: str
    credibility_score: float
    source_type: str  # "Academic", "News", "Government", "Commercial", "Unknown"
    strengths: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "credibility_score": self.credibility_score,
            "source_type": self.source_type,
            "strengths": self.strengths,
            "concerns": self.concerns
        }


class ValidatorAgent(Agent):
    """
    Specialized agent for validation and fact-checking.
    
    Capabilities:
    - Source credibility assessment
    - Cross-reference verification
    - Bias detection
    - Confidence scoring
    """
    
    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",  # Using Sonnet (Sonnet: claude-sonnet-4-20250514)
        temperature: float = 0.1  # Low temperature for consistent evaluation
    ):
        super().__init__(
            name="Validator",
            role="""You are an expert fact-checker and source evaluator. Your job is to:
1. Assess the credibility of sources (academic > government > reputable news > commercial)
2. Identify potential biases or conflicts of interest
3. Cross-reference claims when possible
4. Flag inconsistencies or questionable information
5. Provide objective credibility scores

Be thorough but fair. Consider:
- Source reputation and expertise
- Potential biases or motivations
- Consistency with known facts
- Quality of evidence presented
- Recency and relevance""",
            model=model,
            temperature=temperature
        )
    
    async def _execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """Execute validation task."""
        findings = context.get("findings", [])
        
        if not findings:
            return "No findings provided for validation."
        
        # Validate all findings
        validated = await self.validate_findings(findings)
        
        # Format results
        return self._format_validation_results(validated)
    
    async def validate_findings(
        self,
        findings: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Validate a list of research findings.
        
        Args:
            findings: List of ResearchFinding objects
            
        Returns:
            List of findings with validation metadata added
        """
        logger.info(f"Validating {len(findings)} findings")
        
        validated_findings = []
        
        for finding in findings:
            # Evaluate the source
            source_eval = await self._evaluate_source(
                finding.source,
                finding.url
            )
            
            # Validate the content
            content_validation = await self._validate_content(
                finding.content,
                finding.key_points,
                finding.source
            )
            
            # Combine results
            validated_finding = {
                "original_finding": finding.to_dict(),
                "source_evaluation": source_eval.to_dict(),
                "content_validation": content_validation.to_dict(),
                "overall_credibility": (
                    source_eval.credibility_score * 0.6 +  # Source weight: 60%
                    content_validation.credibility_score * 0.4  # Content weight: 40%
                )
            }
            
            validated_findings.append(validated_finding)
            
            logger.info(
                f"Validated '{finding.title}': "
                f"credibility={validated_finding['overall_credibility']:.2f}"
            )
        
        return validated_findings
    
    async def _evaluate_source(
        self,
        source: str,
        url: str
    ) -> SourceEvaluation:
        """
        Evaluate the credibility of a source.
        
        Args:
            source: Source domain/name
            url: Full URL
            
        Returns:
            SourceEvaluation with credibility assessment
        """
        prompt = f"""
Evaluate the credibility of this source:

Source: {source}
URL: {url}

Assess:
1. Source type (Academic, Government, News, Commercial, Blog, Social Media, Unknown)
2. Overall credibility (0.0 to 1.0 scale)
3. Strengths (2-3 points about why this source might be trustworthy)
4. Concerns (1-3 points about potential issues or biases)

Consider:
- Is this a peer-reviewed academic source? (very high credibility)
- Is this a government or official organization? (high credibility)
- Is this established, reputable news? (medium-high credibility)
- Is this commercial or promotional content? (lower credibility)
- Are there potential conflicts of interest?

Respond in JSON format only:
{{
    "source_type": "Academic|Government|News|Commercial|Blog|Unknown",
    "credibility_score": 0.85,
    "strengths": ["point 1", "point 2"],
    "concerns": ["concern 1"]
}}
"""
        
        response = await self.call_llm(prompt, temperature=0.1, max_tokens=500)
        
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
            
            return SourceEvaluation(
                source=source,
                credibility_score=float(data.get("credibility_score", 0.5)),
                source_type=data.get("source_type", "Unknown"),
                strengths=data.get("strengths", []),
                concerns=data.get("concerns", [])
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse source evaluation: {e}")
            # Return default low-credibility evaluation
            return SourceEvaluation(
                source=source,
                credibility_score=0.3,
                source_type="Unknown",
                concerns=["Unable to verify source credibility"]
            )
    
    async def _validate_content(
        self,
        content: str,
        key_points: List[str],
        source: str
    ) -> ValidationResult:
        """
        Validate the content itself.
        
        Args:
            content: Finding content
            key_points: Extracted key points
            source: Source of the information
            
        Returns:
            ValidationResult with content assessment
        """
        prompt = f"""
Validate this content for accuracy and reliability:

Content: {content}

Key Points:
{chr(10).join(f"- {point}" for point in key_points)}

Source: {source}

Assess:
1. Are there any obvious factual errors or inconsistencies?
2. Does the content make exaggerated or unsupported claims?
3. Is the language objective or does it show clear bias?
4. Are the key points accurately extracted?
5. Overall content credibility (0.0 to 1.0)

Respond in JSON format only:
{{
    "is_valid": true,
    "credibility_score": 0.8,
    "issues": ["any serious problems"],
    "warnings": ["minor concerns"],
    "reasoning": "brief explanation of assessment"
}}
"""
        
        response = await self.call_llm(prompt, temperature=0.1, max_tokens=500)
        
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
            
            return ValidationResult(
                is_valid=data.get("is_valid", True),
                credibility_score=float(data.get("credibility_score", 0.5)),
                issues=data.get("issues", []),
                warnings=data.get("warnings", []),
                reasoning=data.get("reasoning", "")
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse content validation: {e}")
            return ValidationResult(
                is_valid=True,
                credibility_score=0.5,
                warnings=["Unable to fully validate content"]
            )
    
    def _format_validation_results(
        self,
        validated_findings: List[Dict[str, Any]]
    ) -> str:
        """Format validation results as readable text."""
        
        output = ["# Validation Results\n"]
        
        # Overall statistics
        avg_credibility = sum(
            f["overall_credibility"] for f in validated_findings
        ) / len(validated_findings)
        
        output.append(f"**Overall Credibility:** {avg_credibility:.2f}/1.0\n")
        output.append(f"**Findings Validated:** {len(validated_findings)}\n")
        
        # Detailed results
        output.append("\n## Individual Findings\n")
        
        for i, vf in enumerate(validated_findings, 1):
            finding = vf["original_finding"]
            source_eval = vf["source_evaluation"]
            content_val = vf["content_validation"]
            
            output.append(f"\n### {i}. {finding['title']}")
            output.append(f"**Overall Credibility:** {vf['overall_credibility']:.2f}/1.0\n")
            
            # Source evaluation
            output.append(f"**Source:** {source_eval['source']}")
            output.append(f"- Type: {source_eval['source_type']}")
            output.append(f"- Credibility: {source_eval['credibility_score']:.2f}/1.0")
            
            if source_eval["strengths"]:
                output.append("- Strengths:")
                for strength in source_eval["strengths"]:
                    output.append(f"  - {strength}")
            
            if source_eval["concerns"]:
                output.append("- Concerns:")
                for concern in source_eval["concerns"]:
                    output.append(f"  - {concern}")
            
            # Content validation
            output.append(f"\n**Content Validation:**")
            output.append(f"- Valid: {'✓' if content_val['is_valid'] else '✗'}")
            output.append(f"- Credibility: {content_val['credibility_score']:.2f}/1.0")
            
            if content_val["issues"]:
                output.append("- Issues:")
                for issue in content_val["issues"]:
                    output.append(f"  - ⚠️  {issue}")
            
            if content_val["warnings"]:
                output.append("- Warnings:")
                for warning in content_val["warnings"]:
                    output.append(f"  - ⚡ {warning}")
            
            output.append("")
        
        return "\n".join(output)


# Example usage
async def demo():
    """Demonstrate the validator agent."""
    from .researcher import ResearcherAgent, ResearchFinding
    
    # Create some mock findings to validate
    findings = [
        ResearchFinding(
            title="Quantum Computing Basics",
            content="Quantum computers use qubits which can be in superposition states.",
            source="nature.com",
            url="https://nature.com/articles/quantum",
            relevance="High",
            key_points=["Uses qubits", "Superposition states", "Quantum mechanics"]
        ),
        ResearchFinding(
            title="Amazing Quantum Breakthrough",
            content="Scientists discover quantum computers can solve any problem instantly!",
            source="techblog.example",
            url="https://techblog.example/quantum-miracle",
            relevance="Medium",
            key_points=["Solves all problems", "Instant solutions"]
        )
    ]
    
    # Validate
    validator = ValidatorAgent()
    validated = await validator.validate_findings(findings)
    
    # Display results
    print("=== Validation Results ===\n")
    for vf in validated:
        print(f"Finding: {vf['original_finding']['title']}")
        print(f"Overall Credibility: {vf['overall_credibility']:.2f}/1.0")
        print(f"Source Type: {vf['source_evaluation']['source_type']}")
        print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
