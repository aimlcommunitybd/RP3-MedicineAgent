from typing import Union

import structlog
from llama_cpp import Llama

from src.engine.llamacpp import (
    load_model, infer
)
from src import settings

logger = structlog.get_logger(__name__)

GENERAL_MODEL = load_model(model_path=settings.GENERAL_MODEL)
EXPERT_MODEL = load_model(model_path=settings.EXPERT_MODEL)


def chat(text):
    # Classification
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

