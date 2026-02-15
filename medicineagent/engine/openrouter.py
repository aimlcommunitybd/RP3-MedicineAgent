import structlog
import time
import requests
import json
from enum import Enum
from typing import Literal, Tuple

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from medicineagent import settings

logger = structlog.get_logger(__name__)

client = OpenAI(
    base_url=settings.OPENROUTER_BASEURL,
    api_key=settings.OPENROUTER_APIKEY,
)
async_client = AsyncOpenAI(
    base_url=settings.OPENROUTER_BASEURL,
    api_key=settings.OPENROUTER_APIKEY,
)


class RouterModel(Enum):
    # NVDIA
    NEMO_9B = "nvidia/nemotron-nano-9b-v2"
    
    # Qwen
    QWEN3_8B = "qwen/qwen3-8b"                          # 0.05/0.4 1M IO
    QWEN3_8B_VL = "qwen/qwen3-vl-8b-instruct"            # 0.08/0.5 1M IO
    QWEN3_8B_THINK = "qwen/qwen3-vl-8b-thinking"      # 0.18/2.10 1M IO
    
    # Mistral
    MISTRAL_7B = "mistralai/mistral-7b-instruct"        # 0.2/0.2 1M IO
    MISTRAL3_8B = "mistralai/ministral-8b-2512"         # 0.15/0.15 1M IO
    
    # Google
    GEMMA_3N_4B = "google/gemma-3n-e4b-it:free"         # no json
    GEMMA_3_4B = "google/gemma-3-4b-it:free"            # no json
    
    # META
    LLAMA3_3B_IT = "meta-llama/llama-3.2-3b-instruct"    # 0.02/0.02 1M IO
    LLAMA3_8B = "meta-llama/llama-3.1-8b-instruct"       # 0.02/0.05 1M IO
    LLAMA3_GUARD = "meta-llama/llama-guard-3-8b"        # 0.02/0.06 1M IO
    # Liquid 
    LFM_1B_IT = "liquid/lfm-2.5-1.2b-instruct:free"
    LFM_1B_THINK = "liquid/lfm-2.5-1.2b-thinking:free"
    LFM_2B = "liquid/lfm-2.2-6b"                        # 0.01/0.02 1M I/O
    LFM_8B = "liquid/lfm-2.5-8b"                        # 0.01/0.02 1M I/O
    
    # BlackForest
    KLEIN_4b = "black-forest-labs/flux.2-klein-4b"
    
    # AllenAI
    OLMO3_7B_THINK = "allenai/olmo-3-7b-think"


class RouterConfig:
    FALLBACK = True
    QUANTIZATION = []
    PROVIDERS_IGNORED = []
    PROVIDERS_PRIORITY = [
        "google",
        "google-ai-studio",
        "modelrun",
        "together",
        "venice",
        "liquid",
    ]
    MODELS_PRIORITY = [
        RouterModel.QWEN3_8B.value,
        RouterModel.MISTRAL_7B.value,
        RouterModel.LLAMA3_8B.value,
    ]
    MODELS_PRIORITY_REASON = [
        RouterModel.QWEN3_8B_THINK.value,
        RouterModel.OLMO3_7B_THINK.value,
        RouterModel.LFM_1B_THINK.value,
    ]

    @classmethod
    def config(cls, MODEL: str = RouterModel.QWEN3_8B.value, search_prompt: str = None):
        PRIORITY_MODELS = (
            cls.MODELS_PRIORITY_REASON
            if MODEL in cls.MODELS_PRIORITY_REASON
            else cls.MODELS_PRIORITY
        )
        configartion = {
            "provider": {
                "order": cls.PROVIDERS_PRIORITY,
                "ignore": cls.PROVIDERS_IGNORED,
                "allow_fallbacks": cls.FALLBACK,
                "quantizations": cls.QUANTIZATION,
            },
            "models": PRIORITY_MODELS,
            "route": "fallback",
        }
        if search_prompt:
            configartion["plugins"] = [{
                "id": "web",
                "engine": "exa",                    # Optional: "native", "exa", or undefined
                "max_results": 5,
                "search_prompt": search_prompt
            }]
        return configartion


MODEL = RouterModel.QWEN3_8B.value
MODEL_THINK = RouterModel.QWEN3_8B_THINK.value
MODEL_XL = "openai/gpt-4.1-nano"                   # for json fixing only


def api_complete(
    prompt,
    model: str = MODEL,
    search_prompt: str = None,
    stop=None,
    frequency_penalty: int = 0,
    n: int = 1,
    max_tokens: int = 2000,
    temperature: float = 0.8,
    response_format: Literal["text", "json"] = "text",
    **kwargs,
) -> Tuple[ChatCompletion, str]:
    str_time = time.time()
    router_config = RouterConfig.config(model, search_prompt=search_prompt)
    messages = [{"role": "user", "content": prompt}] if type(prompt) == str else prompt
    response_format = (
        {"type": "json_object"} if response_format == "json" else {"type": "text"}
    )
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        n=n,
        frequency_penalty=frequency_penalty,
        temperature=temperature,
        stop=stop,
        response_format=response_format,
        extra_body=router_config,
    )
    choice = completion.choices[0]
    content = choice.message.content
    grounding_result = completion.choices[0].message.annotations
    runtime = round(time.time() - str_time, 2)

    logger.info(
        "Completion done",
        metric_name=f"openrouter.chat.completions.create.{completion.model}",
        usage=completion.usage,
        max_tokens=max_tokens,
        model_completion=completion.model,
        model_requested=model,
        provider=completion.provider,
        response_format=response_format,
        temperature=temperature,
        runtime=runtime,
        content=content,
        router_config=router_config,
    )
    return completion, content


