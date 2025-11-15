"""
Base Agent class for the multi-agent research system.

All specialized agents (Researcher, Validator, Synthesizer) inherit from this.
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import aiohttp
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AgentResponse:
    """Structured response from an agent."""
    content: str
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict] = field(default_factory=list)
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "success": self.success,
            "metadata": self.metadata,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "cost": self.cost
        }


class Agent(ABC):
    """
    Base class for all agents in the system.
    
    Each agent has:
    - A name and role (personality/expertise)
    - Access to tools
    - Conversation history
    - LLM client for making API calls
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        model: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        self.name = name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        self.conversation_history: List[Message] = []
        self.tools: Dict[str, Any] = {}
        
        if not self.api_key:
            logger.warning(
                f"{self.name}: No API key found. "
                "Set ANTHROPIC_API_KEY environment variable."
            )
        
        logger.info(f"Initialized agent: {self.name} ({self.role})")
    
    def register_tool(self, name: str, tool: Any) -> None:
        """Register a tool that this agent can use."""
        self.tools[name] = tool
        logger.info(f"{self.name}: Registered tool '{name}'")
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.debug(f"{self.name}: Cleared conversation history")
    
    async def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call the LLM API with the given prompt.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt (defaults to agent's role)
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            LLM response text
        """
        if not self.api_key:
            raise ValueError("No API key configured")
        
        # Use role as system prompt if none provided
        if system_prompt is None:
            system_prompt = self.role
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Add conversation history if it exists
        if self.conversation_history:
            history_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in self.conversation_history[-10:]  # Last 10 messages
            ]
            messages = history_messages + messages
        
        # Prepare request
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "system": system_prompt,
            "messages": messages
        }
        
        logger.debug(f"{self.name}: Calling LLM with {len(messages)} messages")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            # Extract response
            content = data["content"][0]["text"]
            tokens_used = data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
            
            logger.debug(f"{self.name}: LLM response received ({tokens_used} tokens)")
            
            # Store in history
            self.conversation_history.append(Message("user", prompt))
            self.conversation_history.append(Message("assistant", content))
            
            return content
            
        except aiohttp.ClientError as e:
            logger.error(f"{self.name}: API call failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"{self.name}: Unexpected API response format: {e}")
            raise
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Execute a task. This is the main entry point for agents.
        
        Args:
            task: The task description
            context: Optional context/data to help with the task
            
        Returns:
            AgentResponse with results
        """
        logger.info(f"{self.name}: Executing task: {task[:100]}...")
        
        try:
            # Child classes implement specific behavior
            result = await self._execute_task(task, context or {})
            
            return AgentResponse(
                content=result,
                success=True,
                metadata={
                    "agent": self.name,
                    "task": task,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"{self.name}: Task execution failed: {e}")
            return AgentResponse(
                content=f"Error: {str(e)}",
                success=False,
                metadata={"agent": self.name, "error": str(e)}
            )
    
    @abstractmethod
    async def _execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """
        Implement task-specific logic in child classes.
        
        This is where each specialized agent does its unique work.
        """
        pass
    
    def get_tool_descriptions(self) -> str:
        """
        Get a formatted string describing available tools.
        Used to tell the LLM what tools it can use.
        """
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for name, tool in self.tools.items():
            # Try to get description from tool
            desc = getattr(tool, "description", f"Tool: {name}")
            descriptions.append(f"- {name}: {desc}")
        
        return "\n".join(descriptions)
    
    def __repr__(self) -> str:
        return f"Agent(name='{self.name}', role='{self.role[:50]}...', tools={list(self.tools.keys())})"


class ToolUseAgent(Agent):
    """
    Extended agent that can decide which tools to use.
    
    This adds the ability to:
    1. Analyze which tool(s) are needed
    2. Call tools with appropriate parameters
    3. Process tool results
    """
    
    async def decide_tool_use(self, task: str, context: Dict[str, Any]) -> Optional[Dict]:
        """
        Ask the LLM which tool to use and with what parameters.
        
        Returns:
            Dict with 'tool', 'arguments', and 'reasoning' or None if no tool needed
        """
        if not self.tools:
            return None
        
        tool_descriptions = self.get_tool_descriptions()
        
        prompt = f"""
Task: {task}

Available tools:
{tool_descriptions}

Context: {json.dumps(context, indent=2)}

Based on this task, decide if you need to use any tools.

Respond in JSON format:
{{
    "use_tool": true/false,
    "tool": "tool_name" (if use_tool is true),
    "arguments": {{"arg1": "value1"}} (if use_tool is true),
    "reasoning": "why you chose this tool or why no tool is needed"
}}

If multiple tools are needed, pick the most important one first.
"""
        
        response = await self.call_llm(
            prompt,
            temperature=0.3,  # Lower temperature for tool selection
            max_tokens=500
        )
        
        try:
            # Parse JSON response
            # Remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            decision = json.loads(response)
            
            if decision.get("use_tool"):
                logger.info(
                    f"{self.name}: Decided to use tool '{decision['tool']}' - "
                    f"{decision.get('reasoning', 'No reasoning provided')}"
                )
                return decision
            else:
                logger.info(f"{self.name}: No tool needed - {decision.get('reasoning', '')}")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"{self.name}: Failed to parse tool decision: {e}")
            logger.debug(f"Response was: {response}")
            return None
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool with given arguments.
        
        Args:
            tool_name: Name of the registered tool
            arguments: Dict of arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not registered")
        
        tool = self.tools[tool_name]
        
        logger.info(f"{self.name}: Executing tool '{tool_name}' with args: {arguments}")
        
        try:
            # Try different methods tools might have
            if hasattr(tool, "search") and callable(getattr(tool, "search")):
                # Web search tool
                result = await tool.search(**arguments)
            elif hasattr(tool, "execute") and callable(getattr(tool, "execute")):
                # Generic execute method
                if asyncio.iscoroutinefunction(tool.execute):
                    result = await tool.execute(**arguments)
                else:
                    result = tool.execute(**arguments)
            elif callable(tool):
                # Tool is directly callable
                if asyncio.iscoroutinefunction(tool):
                    result = await tool(**arguments)
                else:
                    result = tool(**arguments)
            else:
                raise ValueError(f"Tool '{tool_name}' doesn't have a callable method")
            
            logger.info(f"{self.name}: Tool '{tool_name}' executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"{self.name}: Tool execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    async def process_tool_result(self, task: str, tool_result: Any) -> str:
        """
        Process the tool result and generate a response.
        
        Args:
            task: The original task
            tool_result: Result from tool execution
            
        Returns:
            Processed response
        """
        prompt = f"""
Original task: {task}

Tool result:
{json.dumps(tool_result, indent=2, default=str)}

Based on the tool result above, provide a clear and concise answer to the original task.
Focus on the most relevant information and present it in a well-structured way.
"""
        
        response = await self.call_llm(prompt, temperature=0.5)
        return response


# Example usage
async def demo():
    """Demonstrate basic agent functionality."""
    
    # Create a simple agent
    agent = ToolUseAgent(
        name="Demo Agent",
        role="You are a helpful research assistant that answers questions clearly and concisely.",
        temperature=0.7
    )
    
    # Simple question without tools
    print("=== Simple Question (No Tools) ===")
    response = await agent.execute(
        task="What is the capital of France?",
        context={}
    )
    print(f"Response: {response.content}\n")
    
    # Question that might need tools
    print("=== Question That Needs Tools ===")
    response = await agent.execute(
        task="What are the latest developments in quantum computing?",
        context={}
    )
    print(f"Response: {response.content}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
