"""Provider-agnostic LLM factory.

Switching between local Ollama and free cloud tiers (Groq, Gemini) - or paid
providers later - is a matter of changing LLM_PROVIDER / LLM_MODEL in .env.
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from research_analyst.config import settings


def get_llm(
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    provider = provider or settings.llm_provider
    kwargs = {}
    if provider == "ollama":
        kwargs["base_url"] = settings.ollama_base_url

    return init_chat_model(
        model or settings.llm_model,
        model_provider=provider,
        temperature=settings.llm_temperature if temperature is None else temperature,
        **kwargs,
    )
