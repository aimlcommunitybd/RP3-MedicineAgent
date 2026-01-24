from typing import Union, List, Dict

import structlog
from llama_cpp import Llama

from src import settings
from src.agents.general import (
    classify_text,
    response_irrelevent_query,
    generate_relevent_response,  # test
)
from src.engine.slm_caller import generate_completion

logger = structlog.get_logger(__name__)


def chat(
    text: str,
    general_model: Union[object, str] = None,
    expert_model: Union[object, str] = None,
    chat_history: Union[List[dict], None] = None,
) -> str:
    # Classification
    result = classify_text(text=text, model=general_model)
    query_class = result["result"]
    if not query_class == "relevant":
        return response_irrelevent_query(
            text, model=general_model, query_class=query_class
        )
    return generate_relevent_response(text, general_model)

    # NER
    # Ground Knowledge
    # Expert Response Generation
    # Supervisor based Response Validation
    # Empathy rewrite
    # Return

    # Time and Resource Tracker
    # Shared Context Memory
    # ChatID based Context tracker

    # Chat UI (future)
    return print(f"Hard Coded Response: {text}")
