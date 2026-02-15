import os
from typing import Optional
from crewai import LLM
from openai import OpenAI


class OpenRouterLLM(LLM):
    """Custom LLM wrapper for OpenRouter"""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs,
    ):
        super().__init__(
            model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        self._api_key = (
            api_key or os.getenv("OPENROUTER_APIKEY") or os.getenv("OPENAI_API_KEY")
        )
        self._base_url = base_url
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def call(self, messages: list, **kwargs):
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.choices[0].message.content

    def get_model_name(self):
        return self.model


def create_llm(
    model: str = "qwen/qwen3-8b", temperature: float = 0.2, max_tokens: int = 2048
) -> LLM:
    """Get OpenRouter LLM instance"""
    return OpenRouterLLM(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
