"""
utils/llm_client.py
Unified LLM client supporting Anthropic Claude, OpenAI GPT-4o,
and Groq for low-latency inference.
"""
from __future__ import annotations

import os
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger(__name__)

PRIMARY_LLM = os.getenv("PRIMARY_LLM", "anthropic")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")


class LLMClient:
    """
    Thin wrapper around Anthropic / OpenAI / Groq APIs.
    Automatically falls back to OpenAI if Anthropic is unavailable.
    Use Groq for fast (<5s) interactive queries.
    """

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or PRIMARY_LLM
        self._anthropic_client: Any = None
        self._openai_client: Any = None
        self._groq_client: Any = None

    def _get_anthropic(self) -> Any:
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        return self._anthropic_client

    def _get_openai(self) -> Any:
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        return self._openai_client

    def _get_groq(self) -> Any:
        if self._groq_client is None:
            from groq import Groq
            self._groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        return self._groq_client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        fast: bool = False,
    ) -> str:
        """
        Call the configured LLM and return the text response.

        Args:
            system_prompt: Persona / context for the model.
            user_prompt: The actual question or task.
            max_tokens: Max response tokens.
            temperature: Sampling temperature (lower = more deterministic).
            fast: If True, route to Groq for sub-5s latency.
        """
        if fast and os.getenv("GROQ_API_KEY"):
            return self._groq_complete(system_prompt, user_prompt, max_tokens, temperature)

        if self.provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            return self._anthropic_complete(system_prompt, user_prompt, max_tokens, temperature)

        if os.getenv("OPENAI_API_KEY"):
            return self._openai_complete(system_prompt, user_prompt, max_tokens, temperature)

        raise RuntimeError("No LLM API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")

    def _anthropic_complete(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        client = self._get_anthropic()
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = response.content[0].text
        log.info("anthropic_response", tokens=response.usage.output_tokens)
        return text

    def _openai_complete(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        client = self._get_openai()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def _groq_complete(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        client = self._get_groq()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


# Singleton for shared use across agents
_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
