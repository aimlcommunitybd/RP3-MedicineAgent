import structlog
from nanoserver import APIServer

from src.engine.openrouter import MODEL as OpenRouterModel
from src.app.orchestrator import chat


MODEL = OpenRouterModel
EXPERT_MODEL = OpenRouterModel  # load_gguf_model(model_path=settings.EXPERT_MODEL_PATH)

logger = structlog.get_logger(__name__)

import traceback

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
$ curl -X POST http://localhost:8000/api/chat/ -H "Content-Type: application/json" -d '{"query": "What is the interaction between Napa and Fymoxil?"}'
"""