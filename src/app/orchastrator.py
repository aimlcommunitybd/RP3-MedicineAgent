from typing import Union

import structlog
from llama_cpp import Llama

from src.app import load_model
from src.engine.llamacpp import infer
from src import settings

logger = structlog.get_logger(__name__)

GENERAL_MODEL = load_model(model_path=settings.GENERAL_MODEL)
EXPERT_MODEL = load_model(model_path=settings.EXPERT_MODEL)


def chat(text):
    return infer(text)
    return generate_response(text)

