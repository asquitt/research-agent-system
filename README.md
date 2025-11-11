# Multi-Agent Research System

> A production-grade autonomous research system powered by collaborative LLM agents with advanced tool use capabilities.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This system orchestrates multiple specialized LLM agents that collaborate to conduct comprehensive research on complex topics. Unlike single-agent systems, this architecture distributes cognitive tasks across specialized agents, resulting in more thorough, accurate, and verifiable research outputs.

**Key Features:**
- 🤖 Multi-agent orchestration with specialized roles
- 🔧 Extensible tool system (web search, code execution, document analysis)
- ✅ Built-in fact validation and source verification
- 📊 Comprehensive logging and observability
- 💰 Cost-optimized API usage with intelligent caching
- 🧪 Production-ready with extensive testing

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Research Query                      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   Planner Agent      │  (Breaks down complex queries)
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Researcher Agent    │  (Gathers information)
          │   + Tool Selection   │
          └──────────┬───────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   [Web Search]            [Code Execution]
   [Wikipedia]             [Document Reader]
   [arXiv]                 [Calculator]
        │                         │
        └────────────┬────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Validator Agent     │  (Verifies sources & facts)
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Synthesizer Agent   │  (Creates final report)
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   Research Report    │
          │  + Sources + Metadata│
          └──────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9 or higher
- API keys for:
  - Anthropic Claude (recommended) or OpenAI GPT-4
  - Tavily Search API (free tier: 1000 requests/month)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/research-agent-system.git
cd research-agent-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```python
from src.orchestrator import ResearchOrchestrator

async def main():
    # Initialize the orchestrator
    orchestrator = ResearchOrchestrator()
    
    # Run research
    result = await orchestrator.research(
        query="What are the latest developments in quantum computing?",
        depth="comprehensive"  # or "quick" for faster results
    )
    
    # Access results
    print(f"Title: {result.title}")
    print(f"Summary: {result.summary}")
    print(f"Sources: {len(result.sources)}")
    print(f"Confidence: {result.confidence_score}")
    
    # Export report
    result.export_markdown("research_report.md")
    result.export_json("research_report.json")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Command Line Interface

```bash
# Quick research
python -m src.cli research "What is quantum entanglement?"

# Comprehensive research with all agents
python -m src.cli research "Compare nuclear vs renewable energy" --depth comprehensive

# Research with specific tools
python -m src.cli research "Solve x^2 + 5x + 6 = 0" --tools calculator,web_search

# Interactive mode
python -m src.cli interactive
```

## Agent System

### Agent Roles

| Agent | Role | Capabilities |
|-------|------|--------------|
| **Planner** | Query decomposition | Breaks complex queries into manageable subtasks |
| **Researcher** | Information gathering | Uses multiple tools to find relevant information |
| **Validator** | Fact checking | Verifies source credibility and cross-references facts |
| **Synthesizer** | Report generation | Combines findings into coherent, well-structured reports |

### Tool System

Built-in tools include:

- **Web Search**: Multi-provider search (Tavily, Serper, DuckDuckGo)
- **Wikipedia**: Structured knowledge retrieval
- **arXiv**: Academic paper search and analysis
- **Calculator**: Mathematical computation
- **Code Executor**: Safe Python code execution in sandboxed environment
- **Web Scraper**: Extract content from URLs
- **Document Reader**: PDF, DOCX, TXT analysis

#### Adding Custom Tools

```python
from src.tools import BaseTool

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "Description of what your tool does"
    
    def execute(self, param1: str, param2: int) -> dict:
        # Your tool logic here
        result = do_something(param1, param2)
        return {
            "success": True,
            "data": result
        }

# Register the tool
orchestrator.register_tool(MyCustomTool())
```

## Configuration

### Agent Configuration

```yaml
# config/agents.yaml
planner:
  model: "claude-sonnet-4"
  temperature: 0.7
  max_tokens: 2000

researcher:
  model: "claude-sonnet-4"
  temperature: 0.3
  max_tokens: 4000
  max_tool_calls: 10

validator:
  model: "claude-haiku-4"  # Cheaper model for validation
  temperature: 0.1
  max_tokens: 2000

synthesizer:
  model: "claude-sonnet-4"
  temperature: 0.5
  max_tokens: 6000
```

### Tool Configuration

```yaml
# config/tools.yaml
web_search:
  provider: "tavily"
  max_results: 5
  timeout: 10

code_executor:
  timeout: 5
  memory_limit_mb: 256
  allowed_imports: ["math", "statistics", "datetime"]
```

## Performance & Cost Optimization

### Caching Strategy

The system implements multi-level caching:
- **LLM response cache**: Identical prompts return cached responses
- **Tool result cache**: Search results cached for 24 hours
- **Embedding cache**: For similarity-based retrieval

```python
# Enable/disable caching
orchestrator = ResearchOrchestrator(
    cache_llm_responses=True,
    cache_tool_results=True,
    cache_ttl=86400  # 24 hours
)
```

### Model Selection Strategy

- **Planner**: Claude Sonnet (requires reasoning)
- **Researcher**: Claude Sonnet (complex tool selection)
- **Validator**: Claude Haiku (simple validation tasks - 5x cheaper)
- **Synthesizer**: Claude Sonnet (coherent writing)

**Estimated costs per research query:**
- Quick research: $0.02 - $0.05
- Comprehensive research: $0.10 - $0.30

### Rate Limiting

```python
# Configure rate limits
orchestrator = ResearchOrchestrator(
    max_concurrent_llm_calls=5,
    max_concurrent_tool_calls=10,
    requests_per_minute=60
)
```

## Evaluation & Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_agents.py
pytest tests/test_orchestrator.py
```

### Benchmark Suite

Evaluate system performance on standardized questions:

```bash
# Run benchmark
python -m src.evaluation.benchmark

# Results include:
# - Accuracy score (vs ground truth)
# - Source quality score
# - Response completeness
# - Average latency
# - Total cost
```

### Custom Evaluation

```python
from src.evaluation import Evaluator

evaluator = Evaluator()

# Add test cases
evaluator.add_test_case(
    query="What is photosynthesis?",
    expected_keywords=["chlorophyll", "sunlight", "glucose", "oxygen"],
    expected_sources_min=3
)

# Run evaluation
results = await evaluator.evaluate(orchestrator)
print(f"Accuracy: {results.accuracy_score}")
print(f"Average latency: {results.avg_latency}s")
```

## Observability

### Logging

All agent actions are logged with structured metadata:

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Logs include:
# - Agent name and action
# - Tool calls with parameters
# - Token usage and costs
# - Latency metrics
# - Error traces
```

### Tracing

Visualize agent execution flow:

```python
result = await orchestrator.research(query, trace=True)

# Export trace
result.trace.export_json("trace.json")
result.trace.visualize()  # Creates visualization in browser
```

### Metrics Dashboard

```bash
# Start metrics server
python -m src.monitoring.server

# View at http://localhost:8000
# Metrics include:
# - Requests per minute
# - Average latency
# - Error rate
# - Cost per query
# - Cache hit rate
```

## Advanced Usage

### Parallel Research

Research multiple topics simultaneously:

```python
queries = [
    "Latest AI breakthroughs",
    "Climate change solutions",
    "Quantum computing applications"
]

results = await orchestrator.research_parallel(queries, max_concurrent=3)
```

### Streaming Results

Get real-time updates as agents work:

```python
async for update in orchestrator.research_stream(query):
    print(f"[{update.agent}] {update.status}: {update.message}")
```

### Human-in-the-Loop

Enable agent to request clarification:

```python
def clarification_callback(question: str) -> str:
    return input(f"Agent asks: {question}\nYour response: ")

orchestrator = ResearchOrchestrator(
    human_in_the_loop=True,
    clarification_callback=clarification_callback
)
```

## Project Structure

```
multi-agent-research-system/
├── src/
│   ├── agents/
│   │   ├── base_agent.py        # Base agent class
│   │   ├── planner.py           # Query planning agent
│   │   ├── researcher.py        # Information gathering agent
│   │   ├── validator.py         # Fact validation agent
│   │   └── synthesizer.py       # Report synthesis agent
│   ├── tools/
│   │   ├── base_tool.py         # Tool interface
│   │   ├── web_search.py        # Web search implementations
│   │   ├── code_executor.py     # Safe code execution
│   │   ├── wikipedia.py         # Wikipedia integration
│   │   └── arxiv.py             # Academic paper search
│   ├── orchestrator/
│   │   ├── orchestrator.py      # Main orchestration logic
│   │   └── strategies.py        # Orchestration strategies
│   ├── utils/
│   │   ├── llm_client.py        # LLM API client
│   │   ├── cache.py             # Caching utilities
│   │   └── rate_limiter.py      # Rate limiting
│   ├── evaluation/
│   │   ├── benchmark.py         # Benchmark suite
│   │   └── metrics.py           # Evaluation metrics
│   ├── monitoring/
│   │   ├── tracer.py            # Execution tracing
│   │   └── server.py            # Metrics server
│   └── cli.py                   # Command line interface
├── tests/
│   ├── test_agents.py
│   ├── test_tools.py
│   ├── test_orchestrator.py
│   └── test_integration.py
├── examples/
│   ├── basic_research.py
│   ├── custom_tool.py
│   └── streaming_example.py
├── config/
│   ├── agents.yaml
│   └── tools.yaml
├── benchmarks/
│   └── question_sets/
├── requirements.txt
├── setup.py
├── .env.example
└── README.md
```

## Roadmap

- [ ] **v0.2**: Memory system for context across sessions
- [ ] **v0.3**: Support for additional LLM providers (Gemini, Mistral)
- [ ] **v0.4**: Multi-modal capabilities (image analysis)
- [ ] **v0.5**: Self-improving agents with feedback loops
- [ ] **v1.0**: Production deployment guides (Docker, K8s)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
black src/ tests/
flake8 src/ tests/
mypy src/
```

## Troubleshooting

### Common Issues

**Rate limit errors**
```python
# Reduce concurrent requests
orchestrator = ResearchOrchestrator(max_concurrent_llm_calls=2)
```

**High costs**
```python
# Use cheaper models and enable caching
orchestrator = ResearchOrchestrator(
    default_model="claude-haiku-4",
    cache_llm_responses=True
)
```

**Slow response times**
```python
# Use quick mode or parallel execution
result = await orchestrator.research(query, depth="quick")
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{multi_agent_research_system,
  title={Multi-Agent Research System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/multi-agent-research-system}
}
```

## Acknowledgments

- Built with [Anthropic Claude](https://www.anthropic.com/) and [OpenAI GPT-4](https://openai.com/)
- Search powered by [Tavily](https://tavily.com/)
- Inspired by research in multi-agent systems and autonomous agents

## Contact

- GitHub: [@asquitt](https://github.com/asquitt)
- Email: demarioasquitt@gmail.com
- Blog: [Technical deep-dive posts](https://medium.com/@demarioasquitt)

---

**Star this repo if you find it helpful! ⭐**
