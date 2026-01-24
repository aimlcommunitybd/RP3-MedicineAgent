import time
from typing import Literal, Dict, Union

import structlog

from src.prompts import general as template
from src.engine.slm_caller import generate_completion

PURPOSE = "general"
logger = structlog.get_logger(__name__)


def classify_text(
    text: str,
    model: Union[object, str],
    context: str = None,
    max_tokens: int = 100,
    temparature: float = 0.8,
    response_format: Literal["json", "text"] = "json",
    purpose: str = PURPOSE,
):
    prompt = template.classification.format(
        query=text,
        context=context,
        team_expertise=template.team_expertise,
        identity=template.admin_identity,
        response_format=response_format,
        classes=template.classification_classes,
        example_response=template.classification_example_response,
    )
    logger.info(
        "Classifying Givent Query",
        prompt=prompt,
        text=text,
        model=model,
        response_format=response_format,
    )
    completion, content = generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        # example_reponse=template.classification_example_response,
    )
    logger.info("[Done] Classification", query=text, content=content)
    return content


def response_irrelevent_query(
    text: str,
    model: Union[object, str],
    context: str = None,
    query_class: str = "General Query",
    max_tokens: int = 100,
    temparature: float = 0.8,
    response_format: Literal["json", "text"] = "text",
    purpose: str = PURPOSE,
):
    prompt = template.irrelevent_query_response.format(
        query=text,
        query_class=query_class,
        identity=template.assistant_identity,
    )
    logger.info(
        "Generating Answer for Irrelevent Query",
        query=text,
        prompt=prompt,
        context=context,
        purpose=purpose,
        model=model,
    )
    completion, content = generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        # example_reponse=template.classification_example_response,
    )
    logger.info(
        "[Done] Generated answer for irrelevent query", query=text, content=content
    )
    return content


def generate_relevent_response(
    text: str,
    model: Union[object, str],
    context: str = None,
    max_tokens: int = 100,
    temparature: float = 0.8,
    response_format: Literal["json", "text"] = "text",
    purpose: str = PURPOSE,
):
    prompt = template.general_relevent_response.format(
        query=text,
        context=context,
        identity=template.doctor_identity,
    )
    logger.info(
        "Generating relevent query response",
        query=text,
        context=context,
        model=model,
        prompt=prompt,
    )
    completion, content = generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temparature=temparature,
        response_format=response_format,
        purpose=purpose,
        # example_reponse=template.classification_example_response,
    )
    logger.info("[Done] Generated relevent query answer", query=text, content=content)
    return content
