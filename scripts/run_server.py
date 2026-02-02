import sys
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import structlog
from nanoserver import APIServer

from medicineagent.engine.openrouter import MODEL as OpenRouterModel
from medicineagent.orchestrator import chat

MODEL = OpenRouterModel
EXPERT_MODEL = OpenRouterModel  # FIXME: load_gguf_model(model_path=settings.EXPERT_MODEL_PATH)
logger = structlog.get_logger(__name__)

server = APIServer()

def generate_response(req, data):
    try:
        query = data["query"]
        chat_history = [{"role": "user", "content": query}]
        result = chat(
            text=query,
            general_model=MODEL, 
            expert_model=MODEL,
            chat_history=chat_history,
        )
        return result
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}

server.add_route(method="POST", path="/api/chat/", handler=generate_response)
server.run()


"""
Example usage:
curl -X POST http://localhost:8000/api/chat/ -H "Content-Type: application/json" -d '{"query": "What is the interaction between Napa and Fymoxil?"}'
"""