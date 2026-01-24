import structlog
import time
import requests
import json
from enum import Enum 
from typing import Literal, Tuple

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from src import settings


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
    NEMO_9B = "nvidia/nemotron-nano-9b-v2:free"
    # Qwen
    QWEN_4B = "qwen/qwen3-4b:free"
    QWEN_7B_IT="qwen/qwen-2.5-vl-7b-instruct:free"
    # Google
    GEMMA_3N_4B="google/gemma-3n-e4b-it:free"
    GEMMA_3_4B="google/gemma-3-4b-it:free"
    # META
    LLAMA_3B_IT="meta-llama/llama-3.2-3b-instruct:free"
    # Liquid
    LFM_1B_IT="liquid/lfm-2.5-1.2b-instruct:free"
    LFM_1B_THINK="liquid/lfm-2.5-1.2b-thinking:free"
    # BlackForest
    KLEIN_4b="black-forest-labs/flux.2-klein-4b"
    
    
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
    # TODO: Should have 2 separate priority list for Models and small Models
    MODELS_PRIORITY = [
        RouterModel.QWEN_7B_IT.value,
        RouterModel.GEMMA_3_4B.value,
        RouterModel.QWEN_4B.value,
    ]
    MODELS_PRIORITY_SM = [
        RouterModel.LLAMA_3B_IT.value,
        RouterModel.LFM_1B_IT.value,
        RouterModel.LFM_1B_THINK.value,
    ]
    
    @classmethod
    def config(cls, MODEL:str=RouterModel.QWEN_7B_IT.value):
        PRIORITY_MODELS = cls.MODELS_PRIORITY_SM if MODEL in cls.MODELS_PRIORITY_SM else cls.MODELS_PRIORITY
        return {
            "provider": {
                "order": cls.PROVIDERS_PRIORITY,
                "ignore": cls.PROVIDERS_IGNORED,
                "allow_fallbacks": cls.FALLBACK,
                "quantizations": cls.QUANTIZATION,
            },
            "models": PRIORITY_MODELS,
            "route": "fallback",
        }


MODEL = RouterModel.QWEN_7B_IT.value
MODEL_SM = RouterModel.LLAMA_3B_IT.value
# MODEL_LARGE_CONTEXT = RouterModel.NEMO_9B.value 


def api_complete(
    prompt,
    model:str=MODEL,
    stop=None,
    frequency_penalty:int=0,
    n:int=1,
    max_tokens:int=2000,
    temperature:float=0.8,
    response_format:Literal["text", "json_object"]="text",
    **kwargs,
) -> Tuple[ChatCompletion, str]:
    str_time = time.time()
    router_config = RouterConfig.config(model)
    messages = [{"role": "user", "content": prompt}] if type(prompt) == str else prompt
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        n=n,
        frequency_penalty=frequency_penalty,
        temperature=temperature,
        stop=stop,
        response_format={"type": response_format},
        extra_body = router_config,
    )
    choice = completion.choices[0]
    content = choice.message.content
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
    )
    return completion, content


async def async_api_complete(
    prompt,
    model:str=MODEL,
    stop=None,
    frequency_penalty:int=0,
    top_p:int=1,
    n:int=1,
    max_tokens:int=2000,
    temperature:float=0.8,
    response_format:Literal["text", "json_object"]="text",
    **kwargs,
) -> Tuple[ChatCompletion, str]:
    str_time = time.time()
    router_config = RouterConfig.config(model)
    messages = [{"role": "user", "content": prompt}] if type(prompt) == str else prompt
    completion = await async_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        n=n,
        frequency_penalty=frequency_penalty,
        temperature=temperature,
        stop=stop,
        response_format={"type": response_format},
        extra_body = router_config,
    )
    choice = completion.choices[0]
    content = choice.message.content
    runtime = round(time.time() - str_time, 2)
    
    logger.info(
        "[async] Completion done",
        metric_name=f"openrouter.chat.completions.create.{completion.model}",
        usage=completion.usage,
        max_tokens=max_tokens,
        model_completion=completion.model,
        model_requested=model,
        provider=completion.provider,
        response_format=response_format,
        temperature=temperature,
        runtime=runtime,
    )
    return completion, content


def api_limit(OPENROUTER_KEY=settings.OPENROUTER_APIKEY):
  response = requests.get(
    url="https://openrouter.ai/api/v1/auth/key",
    headers={
      "Authorization": f"Bearer {OPENROUTER_KEY}"
    }
  )
  formatted_response = json.dumps(response.json(), indent=2)
  logger.info("OPENROUTER_KEY API LIMIT", response=formatted_response)
  return formatted_response