"""LLM providers behind one interface. OpenAI and Ollama share the OpenAI-compatible client."""

from collections.abc import Iterator
from functools import lru_cache
from typing import Protocol

from app.config import Settings, get_settings


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    def stream(self, messages: list[dict]) -> Iterator[str]: ...

    def generate(self, messages: list[dict]) -> str: ...


class OpenAICompatibleLLM:
    def __init__(
        self, model: str, api_key: str | None, base_url: str | None, max_tokens: int, token_param: str,
        reasoning_effort: str | None = None,
    ):
        from openai import OpenAI

        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set. Add it to .env, or set LLM_PROVIDER=ollama to run locally.")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        # OpenAI's newer models take max_completion_tokens; Ollama's endpoint takes max_tokens.
        self._limits = {token_param: max_tokens}
        if reasoning_effort:
            self._limits["reasoning_effort"] = reasoning_effort

    def stream(self, messages: list[dict]) -> Iterator[str]:
        try:
            response = self._client.chat.completions.create(
                model=self._model, messages=messages, stream=True, **self._limits
            )
            for event in response:
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
        except Exception as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

    def generate(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages)).strip()


class AnthropicLLM:
    def __init__(self, model: str, api_key: str | None, max_tokens: int):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("Install the Anthropic SDK: uv sync --extra anthropic") from exc
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def stream(self, messages: list[dict]) -> Iterator[str]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        try:
            with self._client.messages.stream(
                model=self._model, system=system, messages=chat, max_tokens=self._max_tokens
            ) as stream:
                yield from stream.text_stream
        except Exception as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

    def generate(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages)).strip()


def create_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        return OpenAICompatibleLLM(
            settings.llm_model, settings.openai_api_key, settings.openai_base_url,
            settings.llm_max_output_tokens, "max_completion_tokens", settings.llm_reasoning_effort,
        )
    if settings.llm_provider == "ollama":
        return OpenAICompatibleLLM(
            settings.llm_model, "ollama", settings.ollama_base_url, settings.llm_max_output_tokens, "max_tokens"
        )
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.llm_model, settings.anthropic_api_key, settings.llm_max_output_tokens)
    raise LLMError(f"Unknown LLM provider: {settings.llm_provider}")


@lru_cache
def get_llm() -> LLMProvider:
    return create_llm(get_settings())
