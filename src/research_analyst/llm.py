"""Provider-agnostic LLM factory.

Switching between local Ollama and free cloud tiers (Groq, Gemini) - or paid
providers later - is a matter of changing LLM_PROVIDER / LLM_MODEL in .env.
"""

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from ollama import ResponseError

from research_analyst.config import settings

# Errors worth retrying: the Ollama server restarting (dropped connection) or its
# model runner dying mid-request (HTTP 500). Some VPN tools that hook Windows
# networking (e.g. Astrill's ASProxy LSP) crash Ollama now and then; the tray app
# restarts it within a second or two.
RETRYABLE_ERRORS = (httpx.TransportError, ConnectionError, ResponseError)


def get_llm(
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    provider = provider or settings.llm_provider
    kwargs = {}
    if provider == "ollama":
        kwargs["base_url"] = settings.ollama_base_url
        # Room for several retrieved passages; Ollama's default window is smaller.
        kwargs["num_ctx"] = settings.ollama_num_ctx

    return init_chat_model(
        model or settings.llm_model,
        model_provider=provider,
        temperature=settings.llm_temperature if temperature is None else temperature,
        **kwargs,
    )


def with_retry(runnable: Runnable) -> Runnable:
    """Retry a chain's invoke() when the LLM server drops the connection."""
    return runnable.with_retry(
        retry_if_exception_type=RETRYABLE_ERRORS,
        stop_after_attempt=4,
        wait_exponential_jitter=True,
    )
