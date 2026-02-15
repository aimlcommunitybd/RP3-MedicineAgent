# SLMs in Multi-Agent Systems for Medication Safety

This repository explores the design and evaluation of **Small Language Model (SLM)–based multi-agent systems** for answering medication-related queries.

---

## Research Objective

The primary objectives of this research are:

- Assess how effectively **SLM-based multi-agent architectures** can answer medication-related questions safely and accurately
- Compare **multi-agent SLM systems** against **single-agent LLM chatbots** in terms of reliability, hallucination control, and safety for drug information

---

## Technical Overview

A CrewAI-based multi-agent system for answering patient medication queries with evidence-based responses.

MedicineCrew uses **5 specialized AI agents** that collaborate to provide accurate, safe medical information:

```
User Query
    ↓
[Triage Agent] - Handles simple queries or delegates complex ones
    ↓
    ├─→ ANSWER: Direct response (greetings, off-topic)
    └─→ DELEGATE: Complex medical queries
            ↓
        [NER Agent] - Extract medicine names
            ↓
        [Researcher Agent] - Local DB + Web search with citations
            ↓
        [Expert Agent] - Medical advice with evidence
            ↓
        [Safety Agent] - Verify safety & citations
            ↓
        Final Response
```

## Features

- **Smart Routing**: Simple queries handled directly, complex queries delegate to specialists
- **Hybrid Data Sources**: Local drug interaction database + Web search
- **Evidence-Based**: All responses include source citations
- **Safety First**: Dedicated safety verification agent

## Requirements

- Python 3.11+
- OpenRouter API key
- Serper API key (for web search)

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/aimlcommunitybd/MediFlow.git
cd MediFlow
uv venv
uv sync
source .venv/bin/activate
```
**Note:** `uv` should be already installed in your system. Read [uv-astral installation guide](https://docs.astral.sh/uv/getting-started/installation/). Alternatively, You can use installation script using `bash setup.sh` that auto setup uv and dependencies in your ubuntu machine.

### 2. Environment Variables

Create `.env` file:
```bash
cp .env.example .env
```

```env
# OpenRouter (required)
OPENROUTER_APIKEY=sk-or-v1-xxxxxxxxxxxxx
OPENROUTER_BASEURL=https://openrouter.ai/api/v1

# Web Search (required for Researcher)
SERPER_API_KEY=your_serper_api_key
```

Get API keys:
- **OpenRouter**: https://openrouter.ai/keys
- **Serper**: https://serper.dev/api-key

### 3. Local Database

The system uses `dataset/db_drug_interactions.csv` for local drug interaction lookups.

## Usage

### Python API
```bash
uv run python
```
```python
from medicinecrew.orchestrator_crewai import run_medicine_agent

# Simple greeting
result = run_medicine_agent("Hello!")
# → "Hello! How can I help you today?"

# Complex medical query
result = run_medicine_agent("What is the interaction between Napa500 and Fymoxil500?")
# → Full response with citations
```

### CLI

TBA

### HTTP Server

```bash
# Run server
make run
```
#### On a separate terminal
```bash
# Query
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Napa500?"}'
```

## Configuration

### Changing Models

Edit `medicinecrew/agents/openrouter.py`:

```python
def create_llm(
    model: str = "qwen/qwen3-8b",  # Change model here
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> LLM:
    ...
```

Recommended models:
- `qwen/qwen3-8b` - Fast, good quality
- `meta-llama/llama-3.1-8b-instruct` - Open source option

### Adjusting Temperature

| Agent | Recommended Temperature |
|-------|----------------------|
| Triage | 0.2 |
| NER | 0.2 |
| Researcher | 0.2 |
| Expert | 0.2 |
| Safety | 0.1 (stricter) |

## Project Structure

```
medicinecrew/
├── agents/
│   ├── multi_agent.py      # Agent definitions
│   └── openrouter.py       # LLM wrapper
├── orchestrator_crewai.py  # Main orchestration
├── prompts.py              # Task prompts
└── rag/
    └── __init__.py        # Local drug database tool

dataset/
└── db_drug_interactions.csv  # Drug interaction data
```

## Architecture

### Agent Responsibilities

| Agent | Role | Tools |
|-------|------|-------|
| **Triage** | Route queries | Web Search, Local DB |
| **NER** | Extract medicine names | Local DB |
| **Researcher** | Find drug info | Web Search, Local DB |
| **Expert** | Generate advice | Web Search, Local DB |
| **Safety** | Verify safety | (None - review only) |

### Query Flow

1. **Simple Query** (greeting/off-topic):
   - Triage answers directly
   - ~1 LLM call

2. **Complex Query** (drug interactions):
   - Triage delegates
   - NER extracts medicines
   - Researcher searches (DB + Web)
   - Expert provides advice
   - Safety verifies
   - ~5 LLM calls

## Documentation

- [Multi-Agent System Documentation](docs/MULTI_AGENT_SYSTEM.md)
- [Changes Log](CHANGES.md)


## Sample Usage

```bash
make run
```

### Greetings
```bash
curl -X POST http://localhost:8000/api/chat/ -H "Content-Type: application/json" -d '{"query": "Hello"}'
```
```json
{
  "user": "Hello",
  "assistant": "Hello! How can I assist you today? If you have any questions about medications, drug interactions, or health-related topics, feel free to ask!"
}
```
### Simple Query
```bash
curl -X POST http://localhost:8000/api/chat/ -H "Content-Type: application/json" -d '{"query": "what type of medical advice you can provde?"}'
```
```json
{
  "user": "what type of medical advice you can provde?",
  "assistant": "I can provide general medical information, guidance on medication usage, side effects, drug interactions, and basic health advice. However, for complex medical questions, drug interaction analysis, or personalized recommendations, I recommend consulting a licensed healthcare professional or using the research agent for detailed analysis. Let me know how I can assist you specifically!"
}
```
### Complex Query
```sh
curl -X POST http://localhost:8000/api/chat/ -H "Content-Type: application/json" -d '{"query": "I am taking avocard10 for heart, mixturd30 for diabetics, prigaba75 for nerve, gastic and bp medicines everyday. is this safe to take naproxen for pain"}'
```
```json
{
  "user": "I am taking avocard10 for heart, mixturd30 for diabetics, prigaba75 for nerve, gastic and bp medicines everyday. is this safe to take naproxen for pain",
  "assistant": "Taking naproxen alongside your current medications carries potential risks and should be discussed with a healthcare provider before starting. Here's a detailed analysis:\n\n1. **Drug Interaction Warnings**:  \n   - **Acetaminophen/Caffeine (Avocard10) + Naproxen**: Both medications are hepatotoxic, increasing the risk of liver damage. Acetaminophen is metabolized in the liver, and naproxen may exacerbate this risk, especially with long-term use [Source: Local DB; Source: Web - https://www.drugs.com/mtm/avocet.html].  \n   - **Pregabalin (Prigaba75) + Naproxen**: This combination increases the risk of dizziness, sedation, and impaired renal function. Both drugs are renally excreted, so concurrent use may worsen kidney function. Monitor for signs of renal impairment (e.g., decreased urine output, swelling) [Source: Web - https://www.drugs.com/interaction/pregabalin-naproxen.html].  \n   - **Methocarbamol/Hydroxyzine (Mixturd30) + Naproxen**: Hydroxyzine (a CNS depressant) may enhance sedation when combined with naproxen, leading to increased drowsiness or impaired coordination [Source: Local DB; Source: Web - https://www.rxlist.com/mixturd-30-side-effects-drug-interactions.html].  \n\n2. **Safety Considerations**:  \n   - **Liver Toxicity**: Acetaminophen and naproxen both pose risks to liver function. Concurrent use may elevate serum aminotransferases or lead to acute liver injury, particularly in patients with pre-existing liver disease or alcohol use [Source: Web - https://www.drugs.com/mtm/avocet.html; Source: Web - https://www.mayoclinic.org/drugs-supplements/naproxen-oral dosage, side-effects, interactions].  \n   - **Gastrointestinal Risk**: Naproxen may irritate the stomach lining, increasing the risk of ulcers or bleeding, especially if you are on blood pressure medications (e.g., NSAID interactions with ACE inhibitors or ARBs) [Source: Web - https://www.mayoclinic.org/drugs-supplements/naproxen-oral dosage, side-effects, interactions].  \n   - **Blood Pressure Management**: Naproxen can cause fluid retention and elevate blood pressure, potentially counteracting your BP medications [Source: Web - https://www.mayoclinic.org/drugs-supplements/naproxen-oral dosage, side-effects, interactions].  \n\n3. **Recommendations**:  \n   - **Consult Your Healthcare Provider**: They can assess your renal and hepatic function, review all medications (including over-the-counter drugs), and determine if naproxen is appropriate for your condition.  \n   - **Monitor Symptoms**: If you start naproxen, watch for signs of liver toxicity (e.g., jaundice, dark urine), renal issues (e.g., swelling, fatigue), or excessive drowsiness.  \n   - **Avoid Self-Medication**: Do not adjust dosages or discontinue medications without medical guidance.  \n\n4. **Limitations**:  \n   - This analysis assumes your BP medications are non-NSAID types (e.g., ACE inhibitors, beta-blockers). If you are on NSAID-containing BP drugs (e.g., olmesartan/amlodipine), the risks would be compounded [Source: Web - https://www.mayoclinic.org/drugs-supplements/naproxen-oral dosage, side-effects, interactions].  \n   - The local database and cited sources may not capture all rare or idiosyncratic interactions.  \n\nAlways prioritize professional medical advice for personalized safety assessments."
}
```

