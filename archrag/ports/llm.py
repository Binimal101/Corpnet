"""Port: large language model inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool call from the LLM."""
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class ChatResponse:
    """Response from chat_with_tools."""
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    
    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMPort(ABC):
    """Generate text completions from prompts."""

    @abstractmethod
    def generate(self, prompt: str, *, system: str = "") -> str:
        """Return a free-form text completion."""

    @abstractmethod
    def generate_json(self, prompt: str, *, system: str = "") -> dict[str, Any]:
        """Return a structured JSON completion (parsed)."""
    
    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatResponse:
        """Chat with tool calling support.
        
        Args:
            messages: List of messages in OpenAI format.
            tools: List of tool definitions in OpenAI format.
            
        Returns:
            ChatResponse with content and optional tool calls.
        """
