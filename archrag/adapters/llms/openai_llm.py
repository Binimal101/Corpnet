"""LLM adapter: OpenAI API."""

from __future__ import annotations

import json
from typing import Any

from archrag.ports.llm import LLMPort, ChatResponse


class OpenAILLM(LLMPort):
    """Uses the OpenAI chat completions endpoint."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        from openai import OpenAI  # lazy

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def generate(self, prompt: str, *, system: str = "") -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
        )
        return resp.choices[0].message.content or ""

    def generate_json(self, prompt: str, *, system: str = "") -> dict[str, Any]:
        raw = self.generate(prompt, system=system)
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start : end + 1]
        return json.loads(raw)
    
    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatResponse:
        """Chat with tool calling support."""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            temperature=self._temperature,
        )
        
        message = resp.choices[0].message
        content = message.content or ""
        
        # Extract tool calls if any
        tool_calls: list[dict[str, Any]] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
        
        return ChatResponse(content=content, tool_calls=tool_calls)
