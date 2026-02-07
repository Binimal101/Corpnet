"""Mock LLM for testing without API keys.

Provides basic intent classification using keyword matching.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class MockLLM:
    """Simple keyword-based LLM for testing.
    
    Classifies intents without requiring an actual LLM API.
    """
    
    READ_KEYWORDS = [
        "search", "find", "query", "what", "how", "why", "who", "when", "where",
        "look up", "lookup", "stats", "status", "info", "statistics", "show",
        "get", "list", "?",
    ]
    
    WRITE_KEYWORDS = [
        "add", "index", "remove", "delete", "create", "insert", "sync",
        "connect", "database", "import", "upload", "reindex", "rebuild",
        "flush", "drop", "update", "modify", "change",
    ]
    
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a response using keyword matching.
        
        Args:
            prompt: The prompt text.
            system: Optional system prompt.
            
        Returns:
            Intent classification: READ, WRITE, or UNKNOWN.
        """
        prompt_lower = prompt.lower()
        
        # Count keyword matches
        read_score = sum(1 for kw in self.READ_KEYWORDS if kw in prompt_lower)
        write_score = sum(1 for kw in self.WRITE_KEYWORDS if kw in prompt_lower)
        
        log.debug("MockLLM scores - READ: %d, WRITE: %d", read_score, write_score)
        
        if write_score > read_score:
            return "WRITE"
        elif read_score > 0:
            return "READ"
        else:
            return "UNKNOWN"
    
    def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        """Generate a JSON response (not implemented for mock)."""
        return {"intent": self.generate(prompt, system)}
    
    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Chat with tools (returns empty for mock)."""
        from dataclasses import dataclass, field
        
        @dataclass
        class MockResponse:
            content: str = "I'm a mock LLM. Please configure a real LLM for full functionality."
            tool_calls: list[dict] = field(default_factory=list)
        
        return MockResponse()
    
    def model_name(self) -> str:
        return "mock-llm"
