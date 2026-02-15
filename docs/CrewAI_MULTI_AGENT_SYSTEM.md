# MedicineAgent Multi-Agent System (CrewAI)

A multi-agent AI system for answering patient medication queries using CrewAI and OpenRouter.

## Overview

This system uses **5 specialized AI agents** that work together to provide accurate, safe medical information:

```
User Query
    ↓
[Triage Agent]      → Classifies query (drug interaction, medication info, etc.)
    ↓
[NER Agent]        → Extracts medicine names from query
    ↓
[Researcher Agent] → Web searches for drug information
    ↓
[Expert Agent]     → Generates medical advice
    ↓
[Safety Agent]     → Verifies advice is safe & accurate
    ↓
Final Response
```

## Installation

```bash
# Ensure Python 3.11+ is available
uv venv --python 3.11
uv sync

# Install crewai (if not already installed)
uv add crewai crewai-tools litellm onnxruntime
```

## Usage

### Python API

```python
from medicineagent.orchestrator_crewai import run_medicine_agent

# Simple query
result = run_medicine_agent("What is Napa500?")

# Drug interaction query
result = run_medicine_agent("What is the interaction between Napa500 and Fymoxil500?")

print(result)
```

### CLI

```bash
# Add to scripts/chat_crewai.py
uv run python -c "
from medicineagent.orchestrator_crewai import run_medicine_agent
print(run_medicine_agent('Your query here'))
"
```

## Architecture

### Agents

| Agent | Role | Purpose |
|-------|------|---------|
| **Triage Agent** | Medical Triage Specialist | Classifies query as DRUG_INTERACTION, MEDICATION_INFO, GENERAL_HEALTH, or NON_MEDICAL |
| **NER Agent** | Medicine Name Extractor | Extracts medicine names (brand/generic) from user query |
| **Researcher Agent** | Drug Information Researcher | Web searches for drug info, interactions, safety |
| **Expert Agent** | Medical Expert Doctor | Generates evidence-based medical advice |
| **Safety Agent** | Medical Safety Verifier | Reviews advice for safety concerns, accuracy |

### LLM Configuration

All agents use **OpenRouter** via custom `OpenRouterLLM` wrapper:

```python
# medicineagent/agents/crewai_llm.py
class OpenRouterLLM(LLM):
    def __init__(
        self,
        model: str = "qwen/qwen3-8b",  # Default model
        api_key: str = None,  # Reads from OPENROUTER_APIKEY env
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
```

### Environment Variables

Create `.env` file:

```env
OPENROUTER_APIKEY=sk-or-v1-xxxxxxxxxxxxx
OPENROUTER_BASEURL=https://openrouter.ai/api/v1
SERPER_API_KEY=your_serper_api_key  # For web search
```

## Customization

### Changing Models

```python
# In multi_agent.py, modify create_*_agent() functions:

def create_expert_agent() -> Agent:
    llm = create_llm(
        model="openai/gpt-4o",  # Change model here
        temperature=0.1,  # Lower for safety-critical tasks
        max_tokens=4096
    )
    ...
```

### Adding New Agents

```python
# In medicineagent/agents/multi_agent.py

def create_pharmacist_agent() -> Agent:
    """Agent specialized in medication dosages"""
    llm = create_llm(temperature=0.2)
    
    return Agent(
        role="Clinical Pharmacist",
        goal="Provide dosage recommendations and timing",
        backstory="""You are a clinical pharmacist...""",
        verbose=True,
        llm=llm,
        allow_delegation=True,
    )
```

### Adding Tools to Agents

```python
from crewai_tools import SerperDevTool, WebsiteSearchTool, PDFSearchTool

def create_researcher_agent() -> Agent:
    search_tool = SerperDevTool()
    website_tool = WebsiteSearchTool()
    
    return Agent(
        ...
        tools=[search_tool, website_tool],  # Add tools here
    )
```

### Modifying Workflow

```python
# In orchestrator_crewai.py

# Change from sequential to hierarchical
crew = Crew(
    agents=[...],
    tasks=[...],
    process=Process.hierarchical,  # Manager assigns tasks
    manager_agent=create_manager_agent(),
)
```

## Maintenance

### Monitoring

1. **Enable Tracing**: Set `CREWAI_TRACING_ENABLED=true` in `.env`
2. **Check Logs**: CrewAI logs each agent's reasoning
3. **Track Costs**: Monitor OpenRouter API usage

### Common Issues

| Issue | Solution |
|-------|----------|
| API Key errors | Check `OPENROUTER_APIKEY` in `.env` |
| Slow responses | Reduce `max_tokens` or use faster model |
| Poor quality answers | Adjust temperature (lower = more accurate) |
| Missing drug info | Add more specific search prompts |

### Updating Dependencies

```bash
uv add crewai@latest crewai-tools@latest
```

## Improvements

### 1. Add Local Knowledge Base (RAG)

```python
from crewai_tools import RAGAgent

# Add RAG tool for local drug database
rag_tool = RAGAgent(
    knowledge_base=...,
    document_path="dataset/drug_database.pdf"
)
```

### 2. Add More Specialized Agents

- **Drug Interaction Checker**: Dedicated DDI verification
- **Dosage Calculator**: Calculate safe dosages
- **Side Effect Analyzer**: List common/rare side effects
- **Contraindication Checker**: Verify patient conditions

### 3. Implement Review Loop

Currently, the Safety Agent only verifies once. Add iterative refinement:

```python
# Modify in orchestrator_crewai.py
max_retries = 3
for i in range(max_retries):
    if safety_result["status"] == "SAFE":
        break
    # Send feedback back to Expert Agent
    advice_task = Task(description=f"Improve based on: {safety_result}")
```

### 4. Add Memory

```python
from crewai import Agent, Crew
from crewai.memory import Memory

crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,  # Enable memory
    memory_config={
        "provider": "redis",  # Or "postgres", "sqlite"
    }
)
```

### 5. Switch to Smaller Models (SLM)

For cost savings, use smaller models:

```python
# Replace qwen/qwen3-8b with:
model = "qwen/qwen3-4b"   # Smaller, faster
model = "mistralai/mistral-7b-instruct"  # Alternative
model = "meta-llama/llama-3.1-8b-instruct"  # Meta's model
```

## Comparison: Pipeline vs Multi-Agent

| Aspect | Pipeline (Original) | Multi-Agent (CrewAI) |
|--------|-------------------|---------------------|
| **Flexibility** | Fixed flow | Dynamic task assignment |
| **Specialization** | Functions | Autonomous agents |
| **Debugging** | Easier | More complex |
| **Cost** | Lower | Higher (more LLM calls) |
| **Quality** | Good | Better with review |
| **Speed** | Faster | Slower |

## File Structure

```
medicineagent/
├── agents/
│   ├── crewai_llm.py      # OpenRouter LLM wrapper
│   ├── multi_agent.py      # Agent definitions
│   ├── general.py          # Original pipeline agents
│   └── expert.py          # Original expert agent
├── orchestrator.py         # Original pipeline
├── orchestrator_crewai.py # CrewAI orchestration
└── engine/
    ├── slm_caller.py      # Original LLM caller
    └── openrouter.py      # Original OpenRouter client
```

## License

See project root for license information.
