"""
Planner Agent - Breaks down complex queries into structured research plans.

This agent:
1. Analyzes query complexity
2. Identifies required information
3. Creates step-by-step research plan
4. Determines which agents/tools are needed
"""

import json
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field

from src.agents.base_agent import Agent

logger = logging.getLogger(__name__)


@dataclass
class ResearchTask:
    """A single research task in the plan."""
    id: int
    description: str
    agent: str  # Which agent should handle this
    tools: List[str]  # Which tools are needed
    dependencies: List[int] = field(default_factory=list)  # Must complete these first
    priority: str = "medium"  # high, medium, low
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "agent": self.agent,
            "tools": self.tools,
            "dependencies": self.dependencies,
            "priority": self.priority
        }


@dataclass
class ResearchPlan:
    """Complete research plan."""
    query: str
    complexity: str  # simple, moderate, complex
    tasks: List[ResearchTask]
    estimated_duration: str
    agents_needed: List[str]
    tools_needed: List[str]
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "complexity": self.complexity,
            "tasks": [t.to_dict() for t in self.tasks],
            "estimated_duration": self.estimated_duration,
            "agents_needed": self.agents_needed,
            "tools_needed": self.tools_needed
        }


class PlannerAgent(Agent):
    """
    Specialized agent for query planning and task decomposition.
    
    Capabilities:
    - Query complexity assessment
    - Task breakdown
    - Resource allocation
    - Dependency management
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.3
    ):
        super().__init__(
            name="Planner",
            role="""You are an expert research planner. Your job is to:
1. Analyze research queries for complexity and requirements
2. Break complex queries into manageable sub-tasks
3. Identify which agents and tools are needed for each task
4. Determine task dependencies and optimal execution order
5. Estimate time and resources required

Consider:
- Query scope (narrow vs broad)
- Information types needed (facts, analysis, comparisons)
- Data sources required (web, academic, code execution)
- Logical dependencies between tasks""",
            model=model,
            temperature=temperature
        )
    
    async def _execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """Execute planning task."""
        plan = await self.plan(task)
        return self._format_plan(plan)
    
    async def plan(self, query: str) -> ResearchPlan:
        """
        Create a research plan for a query.
        
        Args:
            query: Research question
            
        Returns:
            ResearchPlan with tasks and resource allocation
        """
        logger.info(f"Planning research for: {query}")
        
        # Assess complexity
        complexity = await self._assess_complexity(query)
        
        # Generate tasks
        tasks = await self._generate_tasks(query, complexity)
        
        # Identify resources needed
        agents_needed = list(set(task.agent for task in tasks))
        tools_needed = list(set(tool for task in tasks for tool in task.tools))
        
        # Estimate duration
        duration = self._estimate_duration(tasks, complexity)
        
        plan = ResearchPlan(
            query=query,
            complexity=complexity,
            tasks=tasks,
            estimated_duration=duration,
            agents_needed=agents_needed,
            tools_needed=tools_needed
        )
        
        logger.info(
            f"Plan created: {len(tasks)} tasks, "
            f"complexity={complexity}, "
            f"agents={agents_needed}"
        )
        
        return plan
    
    async def _assess_complexity(self, query: str) -> str:
        """Assess query complexity."""
        
        prompt = f"""
Analyze this research query for complexity:

Query: {query}

Assess complexity based on:
- Scope (narrow topic vs broad field)
- Depth (surface facts vs deep analysis)
- Multiple sub-questions vs single question
- Requires comparison/synthesis vs simple lookup

Respond in JSON:
{{
    "complexity": "simple|moderate|complex",
    "reasoning": "brief explanation"
}}

Examples:
- "What is Python?" = simple
- "Compare Python vs Java for web development" = moderate
- "Analyze the evolution of programming languages and predict future trends" = complex
"""
        
        response = await self.call_llm(prompt, temperature=0.2, max_tokens=300)
        
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
            complexity = data.get("complexity", "moderate")
            logger.info(f"Complexity: {complexity} - {data.get('reasoning', '')}")
            
            return complexity
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse complexity, defaulting to moderate")
            return "moderate"
    
    async def _generate_tasks(self, query: str, complexity: str) -> List[ResearchTask]:
        """Generate research tasks."""
        
        # Adjust number of tasks based on complexity
        max_tasks = {"simple": 2, "moderate": 4, "complex": 6}[complexity]
        
        prompt = f"""
Create a research plan for this query:

Query: {query}
Complexity: {complexity}

Break this into {max_tasks} or fewer concrete research tasks.

Available agents:
- researcher: Web search, information gathering
- validator: Source credibility, fact checking
- synthesizer: Combining findings, report generation
- code_executor: Python code execution, calculations
- api_agent: External API calls

Available tools:
- web_search: Search the web
- wikipedia: Wikipedia lookup
- arxiv: Academic papers
- calculator: Math calculations
- code_executor: Run Python code
- api_call: Call external APIs

For each task specify:
- description (clear, specific)
- agent (which agent handles this)
- tools (which tools needed)
- dependencies (which task IDs must complete first, empty array if none)
- priority (high/medium/low)

Respond in JSON:
{{
    "tasks": [
        {{
            "id": 1,
            "description": "Search for basic information about X",
            "agent": "researcher",
            "tools": ["web_search"],
            "dependencies": [],
            "priority": "high"
        }},
        {{
            "id": 2,
            "description": "Validate sources from task 1",
            "agent": "validator",
            "tools": [],
            "dependencies": [1],
            "priority": "medium"
        }}
    ]
}}
"""
        
        response = await self.call_llm(prompt, temperature=0.3, max_tokens=1000)
        
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
            
            tasks = []
            for t in data.get("tasks", []):
                task = ResearchTask(
                    id=t.get("id", len(tasks) + 1),
                    description=t.get("description", ""),
                    agent=t.get("agent", "researcher"),
                    tools=t.get("tools", []),
                    dependencies=t.get("dependencies", []),
                    priority=t.get("priority", "medium")
                )
                tasks.append(task)
            
            logger.info(f"Generated {len(tasks)} tasks")
            return tasks
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tasks: {e}")
            # Return basic fallback task
            return [
                ResearchTask(
                    id=1,
                    description=f"Research: {query}",
                    agent="researcher",
                    tools=["web_search"],
                    dependencies=[],
                    priority="high"
                )
            ]
    
    def _estimate_duration(self, tasks: List[ResearchTask], complexity: str) -> str:
        """Estimate research duration."""
        base_times = {
            "simple": 15,
            "moderate": 30,
            "complex": 60
        }
        
        base = base_times.get(complexity, 30)
        task_time = len(tasks) * 10
        total = base + task_time
        
        if total < 30:
            return "15-30 seconds"
        elif total < 60:
            return "30-60 seconds"
        elif total < 120:
            return "1-2 minutes"
        else:
            return "2-5 minutes"
    
    def _format_plan(self, plan: ResearchPlan) -> str:
        """Format plan as readable text."""
        
        output = []
        output.append(f"# Research Plan: {plan.query}\n")
        output.append(f"**Complexity:** {plan.complexity}")
        output.append(f"**Estimated Duration:** {plan.estimated_duration}")
        output.append(f"**Agents Needed:** {', '.join(plan.agents_needed)}")
        output.append(f"**Tools Needed:** {', '.join(plan.tools_needed)}\n")
        
        output.append("## Tasks\n")
        for task in plan.tasks:
            deps = f" (depends on: {', '.join(map(str, task.dependencies))})" if task.dependencies else ""
            output.append(f"**Task {task.id}** [{task.priority}]{deps}")
            output.append(f"- Description: {task.description}")
            output.append(f"- Agent: {task.agent}")
            output.append(f"- Tools: {', '.join(task.tools) if task.tools else 'none'}\n")
        
        return "\n".join(output)


async def demo():
    """Demonstrate planner agent."""
    planner = PlannerAgent()
    
    queries = [
        "What is machine learning?",
        "Compare renewable energy sources and calculate ROI",
        "Analyze the impact of AI on employment across industries and predict future trends"
    ]
    
    for query in queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        
        plan = await planner.plan(query)
        print(planner._format_plan(plan))


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
