# Changes Made to MedicineAgent

## Summary
Multiple improvements to fix response quality, JSON parsing, and agent behavior.

---

## 1. Temperature & Token Settings

**Files:** `medicineagent/agents/general.py`, `medicineagent/agents/expert.py`

- Reduced `temperature` from **0.8 → 0.2-0.3** across all functions to reduce hallucinations
- Increased `max_tokens` for better responses:
  - Classification: 100 → 256
  - Entity extraction: 100 → 256
  - Generic name lookup: 200 → 512
  - Drug interaction: 300 → 512
  - Expert advice: 500 → 1024
  - Evaluation: 1000 → 1024
  - Empathic rewrite: 150 → 512

---

## 2. System Message Fix

**File:** `medicineagent/engine/llamacpp.py`

- Changed default system message from "Python Coding Assistant" to medical-focused:
  ```python
  "You are a helpful and accurate medical assistant. Provide safe, evidence-based information. If you're unsure, say so."
  ```

---

## 3. Context Window

**File:** `medicineagent/settings.py`

- Increased from `1024*4 = 4096` → `8192` tokens

---

## 4. Improved NER (Named Entity Recognition)

**File:** `medicineagent/agents/general.py`

- Better prompt with examples for extracting medicine names:
  ```python
  prompt = f"""Extract ALL medicine/drug names from the user query, including brand names, generic names, and combination drugs. Look for patterns like:
  - Numbers in medicine names (e.g., Napa500, Fymoxil500, Napa, Fymoxil, Napa+)
  - Common drug suffixes (-cilin, -ox, -ine, -ol, -in, -ate)
  - Any words that could be medication names
  """
  ```

---

## 5. Improved Generic Name Lookup

**File:** `medicineagent/agents/general.py`

- Added common Bangladesh medicine references in prompt:
  ```python
  - Napa500, Napa, Paracetamol = Paracetamol (Acetaminophen)
  - Fymoxil500, Fymoxil = Amoxicillin with Clavulanic acid
  - Napa Plus = Paracetamol + Tramadol
  - Monocef = Ceftriaxone
  ```
- Added `example_response` to guide JSON output

---

## 6. Robust JSON Parsing

**File:** `medicineagent/engine/slm_caller.py`

- Fixed JSON extraction to handle malformed responses
- Added fallback to return error dict instead of crashing:
  ```python
  return {"error": "Failed to parse JSON", "raw": content[:500]}
  ```
- Improved `fix_json_format()` to handle edge cases

---

## 7. Improved Expert Prompt

**File:** `medicineagent/agents/expert.py`

- Clearer instructions for medication questions
- Emphasized use of ground knowledge
- Added safety warnings and disclaimer requirements

---

## 8. Improved Evaluator (Review Loop)

**File:** `medicineagent/agents/expert.py`

- Added stricter evaluation criteria:
  - Accuracy
  - Relevance
  - Clarity
  - Completeness
  - **Safety** (new)
  - Use of Ground Knowledge
- Made evaluator more strict: "patient safety comes first"
- Added `key_issues` and `strengths` to evaluation output

---

## 9. Review Loop in Orchestrator

**File:** `medicineagent/orchestrator.py`

- Fixed feedback loop to pass previous evaluation to expert:
  ```python
  previous_response = expert_response
  previous_response_evaluation = evaluation.get("suggestions", []) + evaluation.get("key_issues", [])
  ```
- Added error handling for failed JSON parsing
- Added fallback when all retries fail

---

## 10. Model Path Fix

**File:** `.env`

- Fixed incorrect model path from `src/models/` → `medicineagent/models/`

---

## 11. Removed Local Model Dependency

**File:** `scripts/chat.py`

- Changed to use OpenRouter for both general and expert models
- Removed local GGUF model loading for now

---

## Files Modified

| File | Changes |
|------|---------|
| `medicineagent/agents/general.py` | Temperature, tokens, NER prompt, generic name prompt |
| `medicineagent/agents/expert.py` | Temperature, tokens, expert prompt, evaluator prompt |
| `medicineagent/engine/llamacpp.py` | System message |
| `medicineagent/engine/slm_caller.py` | JSON parsing, error handling |
| `medicineagent/orchestrator.py` | Review loop, error handling |
| `medicineagent/settings.py` | Context window |
| `.env` | Model path |
| `scripts/chat.py` | Model selection |

---

## Testing

Run a test query:
```bash
uv run scripts/chat.py --query "What is the interaction between Napa500 and Fymoxil500?"
```
