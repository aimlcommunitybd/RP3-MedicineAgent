import time
import json
from typing import Literal, Tuple, Union

import structlog

from medicineagent import settings
from medicineagent.engine.llamacpp import infer_local_model
from medicineagent.engine.openrouter import api_complete, MODEL_XL

logger = structlog.get_logger(__name__)


def generate_completion(
    prompt,
    model: Union[object, str],
    search_prompt: str = None,
    max_tokens: int = 1024, 
    temparature: float = 0.8, 
    response_format: Literal["text", "json"] = "text",
    purpose: Literal["general", "expert"] = "expert",
    example_response: Union[str, dict, None] = None,
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
            search_prompt=search_prompt,
        )
    return completion, extract_json(content=content, example_response=example_response) if response_format=="json" else content
    
    
def extract_json(content: str, example_response: dict = {}, retry: bool = True) -> Union[str, dict]:
    try:
        json_start = content.index("{")
        json_end = content.rfind("}")
        return json.loads(content[json_start : json_end + 1])
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("[JSONDecodeError] Can not extract JSON from content", content=content, exc=exc)
        
        if not example_response or not retry:  # ← Add retry guard
            raise exc
            
        content = fix_json_format(content=content, example_response=example_response)
        logger.info("[Fixed] JSON Format", content=content)
        
        # Retry ONE more time without further fixing
        return extract_json(content=content, example_response=example_response, retry=False)


def fix_json_format(
    content: str,
    example_response: str,
    model: object = MODEL_XL,
    max_tokens: int = 2000,
    temparature: float = 0.3,
    response_format: Literal["json", "text"] = "json",
    purpose: str = "general",
):
    logger.info("Fixing JSON Format", content=content)
    prompt = f"""The following response needs to be reformatted to match the example JSON format provided. \n\nResponse: {content}\n\nExample Response: {example_response}\n\nPlease ensure the reformatted response adheres strictly to the JSON structure shown in the example."""
    completion, content = api_complete(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,        
    )
    logger.info("[Done] JSON Format Fixed", content=content)
    return content