import os

import structlog
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger(__name__)

GENERAL_MODEL = os.getenv("GENERAL_MODEL_PATH")
EXPERT_MODEL = os.getenv("EXPERT_MODEL_PATH")
LINKUP_APIKEY = os.getenv("LINKUP_APIKEY")
CONTEXT_WINDOW = os.getenv("CONTEXT_WINDOW", 1024*4)
OPENROUTER_APIKEY = os.getenv("OPENROUTER_APIKEY")
OPENROUTER_BASEURL=os.getenv("OPENROUTER_BASEURL", "https://openrouter.ai/api/v1")