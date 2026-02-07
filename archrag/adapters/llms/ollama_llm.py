"""LLM adapter: Ollama local server."""

from __future__ import annotations

import json
from typing import Any

from archrag.ports.llm import LLMPort, ChatResponse


class OllamaLLM(LLMPort):
    """Calls the Ollama /api/chat endpoint."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        from ollama import Client  # lazy

        self._client = Client(host=base_url)
        self._model = model
        self._temperature = temperature

    def generate(self, prompt: str, *, system: str = "") -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat(
            model=self._model,
            messages=messages,
            options={"temperature": self._temperature},
        )
        return resp["message"]["content"]

    def generate_json(self, prompt: str, *, system: str = "") -> dict[str, Any]:
        raw = self.generate(prompt, system=system)
        # Try to extract JSON from the response
        raw = raw.strip()
        # Find first { and last }
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
        """Chat with tool calling support.
        
        Note: Ollama tool calling support varies by model.
        For models without native tool support, we simulate it
        by including tool definitions in the system prompt.
        """
        # Try native tool calling first (supported in newer Ollama versions)
        try:
            resp = self._client.chat(
                model=self._model,
                messages=messages,
                tools=tools,
                options={"temperature": self._temperature},
            )
            
            content = resp["message"].get("content", "")
            
            # Check for tool calls
            tool_calls: list[dict[str, Any]] = []
            if "tool_calls" in resp["message"]:
                for i, tc in enumerate(resp["message"]["tool_calls"]):
                    tool_calls.append({
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": json.dumps(tc["function"]["arguments"]),
                        },
                    })
            
            return ChatResponse(content=content, tool_calls=tool_calls)
        
        except Exception:
            # Fallback: simulate tool calling via prompt engineering
            return self._simulate_tool_calling(messages, tools)
    
    def _simulate_tool_calling(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatResponse:
        """Simulate tool calling for models without native support."""
        # Build a prompt that describes the tools
        tool_desc = "You have access to these tools:\n\n"
        for tool in tools:
            func = tool.get("function", {})
            tool_desc += f"- {func.get('name')}: {func.get('description', '')}\n"
            params = func.get("parameters", {}).get("properties", {})
            if params:
                tool_desc += f"  Parameters: {json.dumps(params)}\n"
        
        tool_desc += """
To use a tool, respond with a JSON object like:
{"tool": "tool_name", "arguments": {"arg1": "value1"}}

If you don't need to use a tool, just respond normally.
"""
        
        # Add tool description to system message
        enhanced_messages = []
        system_added = False
        for msg in messages:
            if msg.get("role") == "system":
                enhanced_messages.append({
                    "role": "system",
                    "content": msg["content"] + "\n\n" + tool_desc,
                })
                system_added = True
            else:
                enhanced_messages.append(msg)
        
        if not system_added:
            enhanced_messages.insert(0, {"role": "system", "content": tool_desc})
        
        # Get response
        resp = self._client.chat(
            model=self._model,
            messages=enhanced_messages,
            options={"temperature": self._temperature},
        )
        
        content = resp["message"]["content"]
        
        # Try to parse tool call from response
        tool_calls: list[dict[str, Any]] = []
        try:
            # Look for JSON in response
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                json_str = content[start:end + 1]
                parsed = json.loads(json_str)
                
                if "tool" in parsed:
                    tool_calls.append({
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": parsed["tool"],
                            "arguments": json.dumps(parsed.get("arguments", {})),
                        },
                    })
                    # Remove JSON from content
                    content = content[:start].strip()
        except (json.JSONDecodeError, KeyError):
            pass
        
        return ChatResponse(content=content, tool_calls=tool_calls)
