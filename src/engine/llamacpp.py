import time
from typing import Union, List, Tuple

import structlog

from src.models import load_model


MODEL = load_model()

logger = structlog.get_logger(__name__)


def infer(
    prompt: Union[list, str],
    max_tokens: int = 1024,
    temperature: int = 0.7,
    model=MODEL,
) -> Tuple:
    """
    Main function for inferece.
    Sends prompt and other parameters to model.
    return generated output inside JSON.
    """
    start_time = time.perf_counter()
    logger.info(
        "Generating Response",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
    )
    messages = process_prompt(prompt)
    completion = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
    )
    content = completion["choices"][0]["message"]["content"]  # plain completion text
    runtime = round(time.perf_counter() - start_time, 2)
    logger.info(
        "Generation Done", content=content, completion=completion, runtime=f"{runtime}s"
    )
    return completion


def process_prompt(
    prompt: Union[List[dict], str],
    system_message: str = "You're a helpful Python Coding Assistant. Help user on his task.",
):
    """
    Prompt can be in 2 format.
        1. plain str: converted into role based messages
        2. list of dict containing role based messages: used as it is
    """
    if isinstance(prompt, str):
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    return prompt