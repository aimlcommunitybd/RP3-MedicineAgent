import os

import structlog
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger(__name__)

GENERAL_MODEL = os.getenv("GENERAL_MODEL_PATH")
EXPERT_MODEL = os.getenv("EXPERT_MODEL_PATH")
LINKUP_APIKEY = os.getenv("LINKUP_APIKEY")
CONTEXT_WINDOW = os.getenv("CONTEXT_WINDOW", 1024*4)