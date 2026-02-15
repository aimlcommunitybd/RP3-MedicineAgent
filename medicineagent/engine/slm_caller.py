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
    temparature: float = 0.2,
    response_format: Literal["text", "json"] = "text",
    purpose: Literal["general", "expert"] = "general",
    example_response: dict = None,
) -> Tuple[object, Union[str, dict]]:
    completion, content = api_complete(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temparature,
        response_format=response_format,
        model=model if isinstance(model, str) else str(model),
        search_prompt=search_prompt,
    )
    if response_format == "json":
        result = extract_json(content=content, example_response=example_response)
        return completion, result
    return completion, content


def extract_json(
    content: str, example_response: dict = None, retry: bool = True
) -> Union[str, dict]:
    if example_response is None:
        example_response = {}
    try:
        json_start = content.index("{")
        json_end = content.rfind("}")
        return json.loads(content[json_start : json_end + 1])
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "[JSONDecodeError] Can not extract JSON from content",
            content=content[:200],
            exc=exc,
        )

        if not example_response or not retry:
            return {"error": "Failed to parse JSON", "raw": content[:500]}

        try:
            fixed_content = fix_json_format(
                content=content, example_response=json.dumps(example_response)
            )
            logger.info("[Fixed] JSON Format", content=fixed_content[:200])

            json_start = fixed_content.index("{")
            json_end = fixed_content.rfind("}")
            return json.loads(fixed_content[json_start : json_end + 1])
        except Exception as fix_err:
            logger.error("[JSONFixer] Failed too", error=str(fix_err))
            return {"error": "Failed to fix JSON", "raw": content[:500]}


def fix_json_format(
    content: str,
    example_response: str,
    model: str = MODEL_XL,
    max_tokens: int = 2000,
    temparature: float = 0.3,
    response_format: Literal["json", "text"] = "json",
):
    logger.info("Fixing JSON Format", content=content[:200])
    prompt = f"""Fix the malformed JSON below. Return ONLY valid JSON, no explanations.

Malformed JSON:
{content}

Example format:
{example_response}

Return valid JSON:"""
    try:
        completion, fixed_content = api_complete(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temparature,
            response_format=response_format,
        )
        logger.info("[Done] JSON Format Fixed", content=fixed_content[:200])
        return fixed_content
    except Exception as e:
        logger.error("JSON fix failed", error=str(e))
        return content
