# SLMs in Multi-Agent Systems for Medication Safety

This repository explores the design and evaluation of **Small Language Model (SLM)–based multi-agent systems** for answering medication-related queries.

---

## Research Objective

The primary objectives of this research are:

- Assess how effectively **SLM-based multi-agent architectures** can answer medication-related questions safely and accurately
- Compare **multi-agent SLM systems** against **single-agent LLM chatbots** in terms of reliability, hallucination control, and safety for drug information

---

## Quick Setup

### 1. Setup Workspace
You can either setup manually or use the `setup.sh` script.

#### Option 1: Manual Setup
**Note:** `git` and `uv` should be already installed in your system.  
Read [uv-astral installation guide](https://docs.astral.sh/uv/getting-started/installation/) for more.

```bash
git clone https://github.com/aimlcommunitybd/RP3-MedicineAgent.git
cd RP3-MedicineAgent
uv venv
uv sync
```

#### Option 2: Automated Setup (Linux - Ubuntu)
Run the setup script to perform all steps automatically. 
In case you encounter any system level error, please setup manually.

```bash
git clone https://github.com/aimlcommunitybd/RP3-MedicineAgent.git
cd RP3-MedicineAgent
bash scripts/setup.sh
```

### 2. Update Environment Secret
```
cp .env.example .env # Later add actual environment variables in .env
```

### 3. Download Models
This command downloads the gguf formatted models to `src/models/downloads/` by default.   
After downloading a model update your `.env` file with proper filepath.  
We'll use an expert model for Doctor Agent. For that an expert model is required. **Skip to generate general response only.**  
**Format:** `uv run scripts/download_model.py <hf_repo_id> <gguf_filename>`  

```bash
uv run scripts/download_model.py "unsloth/medgemma-4b-it-GGUF" "medgemma-4b-it-BF16.gguf"
```

### 4. Run Chat Command
Format: `uv run chat.py <args>`
```bash
# For cotinuous chat: 
uv run chat.py --live_chat

# For single query: 
uv run chat.py --query "What is Type-2 Diabetes?"
```

---

## Architecture

![Draft Architecture](resources/architecture-1.png)

High-level flow:

* User queries are routed through an orchestrator
* Specialized agents (general reasoning, medical expert) collaborate
* Retrieval tools support evidence-grounded responses
* Safety-aware responses are returned to the user

---

## Project Structure

```text
.
├── chat.py                # Entry point for interactive chat
├── dataset/               # Drug–drug interaction datasets
├── resources/             # Diagrams and documentation assets
├── scripts/               # Utility scripts (model download, setup)
├── src/
│   ├── agents/            # Agent roles (general, expert)
│   ├── app/               # Orchestration logic
│   ├── engine/            # LLM tools
│   ├── models/            # Model files and metadata
│   ├── prompts/           # Prompt templates
|   ├── rag/               # rag and vector db logics
│   ├── tools/             # Vector and web search tools
│   └── settings.py        # Global configuration
└── pyproject.toml
```

---

## Models

1. MedGemma-4B (Instruction-Tuned)

* Official model: [https://deepmind.google/models/gemma/medgemma/](https://deepmind.google/models/gemma/medgemma/)
* GGUF source: [https://huggingface.co/unsloth/medgemma-4b-it-GGUF](https://huggingface.co/unsloth/medgemma-4b-it-GGUF)

2. OpenRouter  
We use several api based small models via openrouter
---

## Troubleshoot
- Module not found: Make sure your python path is correct.
```
export PYTHONPATH="${PYTHONPATH}:$(pwd)" # `pwd` should be project base dir
```

## External Medical Knowledge Sources

Publicly accessible sources used for retrieval and reference:

* medex.com.bd
* drugs.com
* rxlist.com
* medscape.com
* medsbd.com
* farmacoinc.com
* arogga.com

> These sources are used for **informational support only**.

---

## Help
1. How to install uv?  
Answer: Please refere to [resources/pkg.md](resources/pkg.md)

## Disclaimer

This project is intended **solely for research and educational purposes**.
It is **not a medical device** and must not be used for clinical diagnosis, treatment, or decision-making.

---



