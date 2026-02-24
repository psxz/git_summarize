"""
app/services/llm_factory.py
────────────────────────────
Creates a LangChain BaseChatModel for the configured (or requested) provider.

Supported providers
───────────────────
  openai    → ChatOpenAI (GPT-4o, GPT-4o-mini, …)
  anthropic → ChatAnthropic (Claude 3.5 Sonnet, Haiku, …)
  google    → ChatGoogleGenerativeAI (Gemini 1.5 Flash/Pro, …)
  nebius    → ChatOpenAI pointed at Nebius Token Factory
               (OpenAI-compatible, default model: MiniMaxAI/MiniMax-M2.1)

Nebius Token Factory
────────────────────
Nebius AI Studio exposes an OpenAI-compatible inference API that hosts a
curated catalogue of open models including MiniMax-M2.1.  We reuse
ChatOpenAI with a custom base_url — no extra library needed.

Endpoint details (MiniMax-M2.1):
  Base URL : https://api.studio.nebius.com/v1/
  Model ID : MiniMaxAI/MiniMax-M2.1
  Auth     : Bearer NEBIUS_API_KEY
  Docs     : https://tokenfactory.nebius.com/models?model-id=MiniMaxAI/MiniMax-M2.1

MiniMax-M2.1 is a reasoning model (~10B active parameters) that interleaves
thinking tokens enclosed in <think>…</think> tags before the final answer.
LangChain's StrOutputParser strips these transparently, but if you want to
surface the reasoning chain set model_kwargs={"reasoning_split": True} and
read response.additional_kwargs["reasoning_details"].

The factory is intentionally thin — prompt engineering lives in summarizer.py.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.core.config import LLMProvider, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Default model for each provider — overridden by DEFAULT_LLM_MODEL env var
# or by the per-request llm_model field in SummarizeRequest.
_PROVIDER_DEFAULTS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.ANTHROPIC: "claude-3-5-haiku-20241022",
    LLMProvider.GOOGLE: "gemini-1.5-flash",
    LLMProvider.NEBIUS: "MiniMaxAI/MiniMax-M2.1",
}


def get_llm(
    provider: str | None = None,
    model: str | None = None,
) -> BaseChatModel:
    """
    Return an instantiated LangChain chat model.

    Args:
        provider: One of 'openai', 'anthropic', 'google', 'nebius'.
                  Defaults to DEFAULT_LLM_PROVIDER env var.
        model:    Model name override.  Defaults to DEFAULT_LLM_MODEL env var,
                  then to the provider's built-in default above.

    Raises:
        ValueError: Unknown provider or missing API key.
    """
    settings = get_settings()
    chosen_provider = LLMProvider(provider) if provider else settings.default_llm_provider

    # Model resolution order:
    #  1. Explicit argument to get_llm()
    #  2. DEFAULT_LLM_MODEL env var (only if it differs from the openai default,
    #     to avoid using "gpt-4o-mini" when the provider is nebius)
    #  3. Provider-specific default from _PROVIDER_DEFAULTS
    env_model = settings.default_llm_model
    if model:
        chosen_model = model
    elif env_model and env_model != "gpt-4o-mini":
        # User has set a custom default — respect it regardless of provider
        chosen_model = env_model
    else:
        chosen_model = _PROVIDER_DEFAULTS.get(chosen_provider, env_model)

    temperature = settings.llm_temperature
    max_tokens = settings.llm_max_tokens

    logger.info(
        "llm_factory_create",
        provider=chosen_provider.value,
        model=chosen_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # ── OpenAI ───────────────────────────────────────────────────────────────
    if chosen_provider == LLMProvider.OPENAI:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=chosen_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=settings.openai_api_key,
            streaming=True,
        )

    # ── Anthropic ─────────────────────────────────────────────────────────────
    if chosen_provider == LLMProvider.ANTHROPIC:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=chosen_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=settings.anthropic_api_key,
            streaming=True,
        )

    # ── Google ────────────────────────────────────────────────────────────────
    if chosen_provider == LLMProvider.GOOGLE:
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=chosen_model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=settings.google_api_key,
        )

    # ── Nebius Token Factory ──────────────────────────────────────────────────
    # Uses ChatOpenAI with Nebius's OpenAI-compatible base URL.
    # Model catalogue: https://tokenfactory.nebius.com/models
    # Default model  : MiniMaxAI/MiniMax-M2.1
    #
    # MiniMax-M2.1 notes:
    #   • Reasoning model — prepends <think>…</think> blocks before answers.
    #     LangChain's StrOutputParser strips them automatically.
    #   • 200k token context window, 10B active params, optimised for code
    #     and agentic tasks.
    #   • Supports streaming via SSE (same protocol as OpenAI).
    #   • For production: consider MiniMaxAI/MiniMax-M2.1-highspeed for
    #     lower latency at a small accuracy trade-off.
    if chosen_provider == LLMProvider.NEBIUS:
        if not settings.nebius_api_key:
            raise ValueError(
                "NEBIUS_API_KEY is not set. "
                "Generate one at https://studio.nebius.com/settings/api-keys"
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=chosen_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=settings.nebius_api_key,
            base_url=settings.nebius_api_base,
            streaming=True,
            # MiniMax-M2.1 specific: preserve <think> tags in streaming chunks
            # so the SSE stream receives reasoning tokens before the answer.
            # StrOutputParser in the summarizer strips them before LLM scoring.
            model_kwargs={
                "extra_headers": {
                    "X-Nebius-Model": chosen_model,  # aids Nebius routing logs
                }
            },
        )

    raise ValueError(f"Unknown LLM provider: {chosen_provider!r}")
