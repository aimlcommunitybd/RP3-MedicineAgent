# MedicineCrew Multi-Agent System (CrewAI)

A multi-agent AI system for answering patient medication queries using CrewAI and OpenRouter.

## Overview

This system uses **5 specialized AI agents** that work together to provide accurate, safe medical information:

```
User Query
    ↓
[Triage Agent] - Handles simple or delegates complex
    ↓
    ├─→ ANSWER: Direct response (fast)
    └─→ DELEGATE:
            ↓
        [NER Agent] - Extract medicine names
            ↓
        [Researcher Agent] - Local DB + Web search with citations
            ↓
        [Expert Agent] - Generate advice with evidence
            ↓
        [Safety Agent] - Verify safety & citations
            ↓
Final Response
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Smart Routing** | Simple queries answered directly, complex delegated |
| **Hybrid Search** | Local drug database + Web search |
| **Evidence-Based** | All responses include source citations |
| **Safety First** | Dedicated safety verification |

## Architecture

### Agent Responsibilities

| Agent | Role | Tools | Decision |
|-------|------|-------|----------|
| **Triage** | Route queries | Web, Local DB | ANSWER or DELEGATE |
| **NER** | Extract medicines | Local DB | Always runs |
| **Researcher** | Find drug info | Web, Local DB | Runs if delegated |
| **Expert** | Generate advice | Web, Local DB | Runs if delegated |
| **Safety** | Verify safety | (review only) | Runs if delegated |

### Data Sources

1. **Local Database** (`dataset/db_drug_interactions.csv`)
   - Drug-drug interactions
   - Fast, reliable for known interactions

2. **Web Search** (SerperDevTool)
   - Latest drug information
   - Broader coverage

## Setup

```bash
# Install dependencies
uv venv --python 3.11
uv sync
uv add crewai crewai-tools litellm onnxruntime pandas

# Setup environment
cp .env.example .env
# Add OPENROUTER_APIKEY and SERPER_API_KEY
```

## Usage

### Python API

```python
from medicinecrew.orchestrator_crewai import run_medicine_agent

result = run_medicine_agent("What is Napa500?")
print(result)
```

### CLI Tests

```bash
# Test greeting (fast - 1 LLM call)
uv run python medicinecrew/orchestrator_crewai.py greeting

# Test off-topic (fast - 1 LLM call)  
uv run python medicinecrew/orchestrator_crewai.py offtopic

# Test complex (full pipeline - 5 LLM calls)
uv run python medicinecrew/orchestrator_crewai.py complex
```

## Query Flow Details

### Simple Query (Greeting/Off-topic)

```
User: "Hello!"
    ↓
Triage Agent
    ↓
[Decides: ANSWER]
    ↓
Response: "Hello! How can I help you today?"
```

- ~1 LLM call
- Fast response
- No agent delegation

### Complex Query (Drug Interaction)

```
User: "What is interaction between Napa500 and Fymoxil500?"
    ↓
Triage Agent
    ↓
[Decides: DELEGATE]
    ↓
NER Agent → Extract: ["Napa500", "Fymoxil500"]
    ↓
Researcher → Search Local DB + Web
    ↓
Expert → Generate advice with citations
    ↓
Safety → Verify + check citations
    ↓
Final Response
```

- ~5 LLM calls
- Evidence-based with citations

## Customization

### Changing Models

In `medicinecrew/agents/openrouter.py`:

```python
def create_llm(
    model: str = "qwen/qwen3-8b",  # Change here
    temperature: float = 0.2,
):
    ...
```

### Adjusting Temperature

| Agent | Temp | Reason |
|-------|------|--------|
| Triage | 0.2 | Balanced |
| NER | 0.2 | Consistent extraction |
| Researcher | 0.2 | Accurate info |
| Expert | 0.2 | Balanced |
| Safety | 0.1 | Strict verification |

### Adding New Agents

```python
# In medicinecrew/agents/multi_agent.py

def create_dosage_agent() -> Agent:
    llm = create_llm(temperature=0.2)
    
    return Agent(
        role="Dosage Specialist",
        goal="Provide accurate dosage information",
        backstory="You are a clinical pharmacist...",
        verbose=True,
        llm=llm,
        tools=[local_drug_tool],
    )
```

## Maintenance

### Monitoring

- Enable tracing: `CREWAI_TRACING_ENABLED=true` in `.env`
- Check CrewAI dashboard for execution traces
- Monitor OpenRouter API usage

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Triage not delegating | Check prompt for "DELEGATE:" keyword |
| No citations | Ensure Researcher uses [Source: ...] format |
| Local DB not found | Check dataset path in rag/__init__.py |

## File Structure

```
medicinecrew/
├── agents/
│   ├── multi_agent.py      # 5 agent definitions
│   └── openrouter.py       # LLM wrapper
├── orchestrator_crewai.py  # Main flow
├── prompts.py              # Task prompts
├── rag/
│   └── __init__.py        # Local DB tool
└── README.md               # This file

dataset/
└── db_drug_interactions.csv  # Drug interaction data
```

## Comparison

| Aspect | Simple Query | Complex Query |
|--------|-------------|---------------|
| LLM Calls | 1 | 5 |
| Speed | Fast | Slower |
| Tools Used | Triage only | All agents |
| Citations | No | Yes |
| Safety Check | No | Yes |

## License

See project root.
