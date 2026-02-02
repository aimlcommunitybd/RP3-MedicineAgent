import os
import time
from typing import Union, List, Tuple

import structlog
from llama_cpp import Llama

from medicineagent import settings


logger = structlog.get_logger(__name__)



def load_gguf_model(
    model_path: str = settings.EXPERT_MODEL_PATH,
    context_window: int = 1024,
) -> Llama:
    """
    load gguf formatted model from local
    """
    if not os.path.exists(model_path):
        logger.error(
            f"Recheck  MODEL_PATH': '{model_path}' is correct path",
            model_path=model_path,
        )
        raise FileNotFoundError(f"Model file not found at {model_path}")

    logger.info(f"Loading model", context_window=context_window, model_path=model_path)
    model = Llama(
        model_path=model_path,
        n_ctx=context_window,
    )
    return model


def infer_local_model(
    prompt: Union[list, str],
    max_tokens: int = 1024,
    temperature: int = 0.7,
    model=Llama,
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
    return completion, content


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


