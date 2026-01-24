import time
import json
from typing import Literal, Tuple, Union

import structlog

from src import settings
from src.engine.llamacpp import infer_local_model
from src.engine.openrouter import api_complete

logger = structlog.get_logger(__name__)


def generate_completion(
    prompt,
    model: Union[object, str],
    max_tokens: int = 1024, 
    temparature: float = 0.8, 
    response_format: Literal["text", "josn"] = "text",
    purpose: Literal["general", "expert"] = "expert",
    example_reponse: Union[str, dict, None] = None,
) -> Tuple[object, str]:
    if purpose=="expert":
        completion, content = infer_local_model(
            prompt=prompt, 
            max_tokens=max_tokens, 
            temperature=temparature,
            response_format=response_format,
            model=model,
        )
        return completion, content
    else:
        completion, content = api_complete(
            prompt=prompt, 
            max_tokens=max_tokens, 
            temperature=temparature,
            response_format=response_format,
            model=model,
        )
    return completion, extract_json(content=content) if response_format=="json" else content
    
    
def extract_json(content: str, example_response: dict = {}) -> Union[str, dict]:
    # FIXME: Handle failed extraction
    try:
        json_start = content.index("{")
        json_end = content.rfind("}")
        return json.loads(content[json_start : json_end + 1])
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("[JSONDecodeError] Can not extract JSON from content", content=content, exc=exc)
        # TODO: May use free api call to fix json output
        raise exc
    except Exception as exc:
        logger.error("[JSONDecodeError] Can not fix JSON", content=content, exc=exc)
        raise exc