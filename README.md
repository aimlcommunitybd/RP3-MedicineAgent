# SLMs in Multi-agent System

## Research Objective
Assess how well SLM-based multi-agent systems can answer medication-related questions safely and accurately, and determine whether multi-agent designs outperform LLM-based single-agent chatbots for drug information.

## Quick Setup
```bash
git clone 
uv venv
uv sync
```

---

### Global Dependencies
```bash
sudo apt-get install git-lfs
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install "transformers[torch]"
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh # not needed
```

### Models
- MedGemma4B: https://deepmind.google/models/gemma/medgemma/

```bash
git clone https://huggingface.co/google/medgemma-4b-it
```
- GGUF of MedGemma4b: https://huggingface.co/unsloth/medgemma-4b-it-GGUF/tree/main
```
uv run src/models/download_model.py "unsloth/medgemma-4b-it-GGUF" "medgemma-4b-it-BF16.gguf"
```

### Some Web-database about medicines
- medex.com.bd    
- Drugs.com  
- RxList  
- Medscape  
- medsbd.com  
- farmacoinc.com  
- www.arogga.com