"""LLM provider implementations.

Supports:
- anthropic (Claude)
- openai (GPT)
- local (mock for testing)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""
        ...
    
    @abstractmethod
    def extract_topics(self, text: str) -> list[str]:
        """Extract topic keywords from a query."""
        ...
    
    @abstractmethod
    def summarize_community(self, entities: list[str], relations: list[str]) -> str:
        """Summarize a community given its entities and relations."""
        ...


class AnthropicLLM(LLMProvider):
    """LLM provider using Anthropic Claude API."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", temperature: float = 0.1, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                raise ImportError("anthropic package not installed")
        return self._client
    
    def generate(self, prompt: str) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    
    def extract_topics(self, text: str) -> list[str]:
        prompt = f"""Extract 3-5 topic keywords from this query. Return only a comma-separated list of keywords, nothing else.

Query: {text}

Keywords:"""
        response = self.generate(prompt)
        return [t.strip() for t in response.split(",") if t.strip()]
    
    def summarize_community(self, entities: list[str], relations: list[str]) -> str:
        prompt = f"""Summarize this knowledge cluster in 2-3 sentences that capture the key themes and relationships.

Entities:
{chr(10).join('- ' + e for e in entities[:20])}

Relationships:
{chr(10).join('- ' + r for r in relations[:20])}

Summary:"""
        return self.generate(prompt)


class OpenAILLM(LLMProvider):
    """LLM provider using OpenAI API."""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI()
            except ImportError:
                raise ImportError("openai package not installed")
        return self._client
    
    def generate(self, prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
    
    def extract_topics(self, text: str) -> list[str]:
        prompt = f"""Extract 3-5 topic keywords from this query. Return only a comma-separated list of keywords, nothing else.

Query: {text}

Keywords:"""
        response = self.generate(prompt)
        return [t.strip() for t in response.split(",") if t.strip()]
    
    def summarize_community(self, entities: list[str], relations: list[str]) -> str:
        prompt = f"""Summarize this knowledge cluster in 2-3 sentences that capture the key themes and relationships.

Entities:
{chr(10).join('- ' + e for e in entities[:20])}

Relationships:
{chr(10).join('- ' + r for r in relations[:20])}

Summary:"""
        return self.generate(prompt)


class MockLLM(LLMProvider):
    """Mock LLM for testing."""
    
    def generate(self, prompt: str) -> str:
        return f"Mock response for: {prompt[:50]}..."
    
    def extract_topics(self, text: str) -> list[str]:
        # Extract simple keywords from text
        words = text.lower().split()
        return [w for w in words if len(w) > 4][:5]
    
    def summarize_community(self, entities: list[str], relations: list[str]) -> str:
        entity_names = [e.split(":")[0] if ":" in e else e for e in entities[:3]]
        return f"A cluster about {', '.join(entity_names)}."


def create_llm_provider(config: "LLMConfig") -> LLMProvider:
    """Factory function to create the appropriate LLM provider."""
    if config.provider == "anthropic":
        return AnthropicLLM(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif config.provider == "openai":
        return OpenAILLM(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif config.provider in ("local", "mock"):
        return MockLLM()
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")
