"""Intent-based router for agentic API.

Determines user intent (READ/WRITE) and routes to appropriate backend:
- READ (Consumer): query, search, info
- WRITE (Producer): index, add, remove, reindex, sync

Uses LLM for intent classification and maintains conversation context.
"""

from __future__ import annotations

import json
import logging
import httpx
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from archrag.api.session import Session, Intent, SessionManager, Message

if TYPE_CHECKING:
    from archrag.ports.llm import LLMPort

log = logging.getLogger(__name__)


# Intent classification prompt
INTENT_SYSTEM_PROMPT = """You are an intent classifier for a data management system.
Classify user messages into one of these intents:

READ - User wants to:
  - Query or search for information
  - Ask questions about indexed data
  - Get statistics or info about the system
  - Look up entities or chunks

WRITE - User wants to:
  - Index new data from files
  - Add documents to the system
  - Remove entities or data
  - Connect to databases and sync data
  - Configure sync settings
  - Re-index or rebuild the index

UNKNOWN - Intent is unclear, need more information

Respond with ONLY the intent label: READ, WRITE, or UNKNOWN"""

INTENT_USER_PROMPT = """Classify the intent of this message:

"{message}"

Previous context: {context}

Respond with ONLY: READ, WRITE, or UNKNOWN"""


# Conversation prompts
GREETING_PROMPT = """Welcome! I'm your ArchRAG assistant.

I can help you:
- **Search** your indexed data (queries, entity lookup)
- **Add** new data (from files, databases, or direct input)
- **Manage** your knowledge graph (remove entities, re-index)

What would you like to do?"""

READ_CLARIFY_PROMPT = """I understand you want to search or query your data.

What would you like to know? You can:
- Ask a question in natural language
- Search for specific entities
- Get system statistics

Just tell me what you're looking for."""

WRITE_CLARIFY_PROMPT = """I understand you want to add or modify data.

What would you like to do?
- **Index a file**: Provide a path to a JSONL or JSON file
- **Add documents**: Give me the content directly
- **Connect a database**: Sync data from SQL/NoSQL
- **Remove data**: Delete specific entities

Tell me more about what you want to add or change."""

NO_PERMISSION_READ = """I'm sorry, but you don't have read access to any data scopes.
Your organization hasn't granted read permissions for this user.
Please contact your administrator if you need read access."""

NO_PERMISSION_WRITE = """I'm sorry, but you don't have permission to modify data.
Your organization hasn't granted write permissions for this user.
Please contact your administrator if you need write access."""


@dataclass
class RouterResponse:
    """Response from the router."""
    message: str
    intent: Intent
    action: str | None = None  # "route_producer", "route_consumer", "clarify"
    route_to: str | None = None  # "producer" or "consumer"
    api_endpoint: str | None = None  # e.g., "/query", "/add"
    api_payload: dict[str, Any] | None = None  # Payload for the API


class AgenticRouter:
    """Routes user messages to appropriate backend based on intent.
    
    This router:
    1. Classifies user intent using LLM
    2. Checks permissions
    3. Routes to Producer or Consumer API
    4. Handles multi-turn conversations for clarification
    
    Usage:
        router = AgenticRouter(llm, producer_url, consumer_url)
        response = router.process_message(session, "I want to search for Einstein")
        
        if response.route_to == "consumer":
            # Forward to consumer API
            result = httpx.post(f"{consumer_url}{response.api_endpoint}", json=response.api_payload)
    """
    
    def __init__(
        self,
        llm: "LLMPort",
        producer_url: str = "http://localhost:8000",
        consumer_url: str = "http://localhost:8001",
        session_manager: SessionManager | None = None,
    ):
        """Initialize the router.
        
        Args:
            llm: LLM for intent classification and conversation.
            producer_url: Base URL for producer API.
            consumer_url: Base URL for consumer API.
            session_manager: Optional session manager.
        """
        self._llm = llm
        self._producer_url = producer_url
        self._consumer_url = consumer_url
        self._session_manager = session_manager or SessionManager()
    
    def start_session(self, session: Session) -> RouterResponse:
        """Start a new conversation session.
        
        Args:
            session: The session object.
            
        Returns:
            Greeting response.
        """
        # Add system message
        self._session_manager.add_message(
            session,
            role="system",
            content=self._get_system_prompt(session),
        )
        
        # Add greeting
        self._session_manager.add_message(
            session,
            role="assistant",
            content=GREETING_PROMPT,
        )
        
        return RouterResponse(
            message=GREETING_PROMPT,
            intent=Intent.UNKNOWN,
            action="greet",
        )
    
    def process_message(
        self,
        session: Session,
        user_message: str,
    ) -> RouterResponse:
        """Process a user message and determine routing.
        
        Args:
            session: Current session.
            user_message: User's message.
            
        Returns:
            RouterResponse with intent, routing info, and response message.
        """
        # Add user message
        self._session_manager.add_message(
            session,
            role="user",
            content=user_message,
        )
        
        # Step 1: Classify intent
        intent = self._classify_intent(session, user_message)
        log.info("Classified intent: %s", intent.value)
        
        # Step 2: Check permissions (using new org-based permission model)
        if intent == Intent.READ and not session.get_accessible_ids():
            response = RouterResponse(
                message=NO_PERMISSION_READ,
                intent=intent,
                action="permission_denied",
            )
            self._session_manager.add_message(session, "assistant", response.message, intent)
            return response
        
        if intent == Intent.WRITE and not session.can_write():
            response = RouterResponse(
                message=NO_PERMISSION_WRITE,
                intent=intent,
                action="permission_denied",
            )
            self._session_manager.add_message(session, "assistant", response.message, intent)
            return response
        
        # Step 3: Update session intent
        session.intent = intent
        self._session_manager.save_session(session)
        
        # Step 4: Route based on intent
        if intent == Intent.READ:
            response = self._handle_read_intent(session, user_message)
        elif intent == Intent.WRITE:
            response = self._handle_write_intent(session, user_message)
        else:
            response = self._handle_unknown_intent(session, user_message)
        
        # Add assistant response
        self._session_manager.add_message(session, "assistant", response.message, intent)
        
        return response
    
    def _classify_intent(self, session: Session, message: str) -> Intent:
        """Classify user intent using LLM.
        
        Args:
            session: Current session.
            message: User message to classify.
            
        Returns:
            Classified Intent.
        """
        # Build context from recent messages
        recent_messages = session.messages[-5:]  # Last 5 messages
        context = ""
        if recent_messages:
            context = "; ".join([
                f"{m.role}: {m.content[:50]}..."
                for m in recent_messages
                if m.role in ("user", "assistant")
            ])
        
        prompt = INTENT_USER_PROMPT.format(
            message=message,
            context=context or "No previous context",
        )
        
        try:
            result = self._llm.generate(prompt, system=INTENT_SYSTEM_PROMPT)
            result = result.strip().upper()
            
            if "READ" in result:
                return Intent.READ
            elif "WRITE" in result:
                return Intent.WRITE
            else:
                return Intent.UNKNOWN
        except Exception as e:
            log.warning("Intent classification failed: %s", e)
            return Intent.UNKNOWN
    
    def _handle_read_intent(
        self,
        session: Session,
        message: str,
    ) -> RouterResponse:
        """Handle READ intent - route to Consumer API.
        
        Args:
            session: Current session.
            message: User message.
            
        Returns:
            RouterResponse with Consumer API routing.
        """
        # Determine specific read operation
        message_lower = message.lower()
        
        # Check for query patterns
        if any(kw in message_lower for kw in ["what", "how", "why", "who", "when", "where", "?"]):
            # Natural language query
            return RouterResponse(
                message=f"Let me search for that...",
                intent=Intent.READ,
                action="route_consumer",
                route_to="consumer",
                api_endpoint="/query",
                api_payload={"question": message},
            )
        
        # Check for search patterns
        if any(kw in message_lower for kw in ["search", "find", "look up", "lookup"]):
            # Extract search term (simple heuristic)
            search_term = message
            for prefix in ["search for", "find", "look up", "lookup", "search"]:
                if prefix in message_lower:
                    idx = message_lower.index(prefix) + len(prefix)
                    search_term = message[idx:].strip()
                    break
            
            return RouterResponse(
                message=f"Searching for '{search_term}'...",
                intent=Intent.READ,
                action="route_consumer",
                route_to="consumer",
                api_endpoint="/search",
                api_payload={"query_str": search_term, "search_type": "all"},
            )
        
        # Check for info/stats
        if any(kw in message_lower for kw in ["stats", "status", "info", "statistics", "how many"]):
            return RouterResponse(
                message="Getting system information...",
                intent=Intent.READ,
                action="route_consumer",
                route_to="consumer",
                api_endpoint="/info",
                api_payload=None,  # GET request
            )
        
        # Default: treat as a query
        return RouterResponse(
            message="Let me answer that for you...",
            intent=Intent.READ,
            action="route_consumer",
            route_to="consumer",
            api_endpoint="/query",
            api_payload={"question": message},
        )
    
    def _handle_write_intent(
        self,
        session: Session,
        message: str,
    ) -> RouterResponse:
        """Handle WRITE intent - route to Producer API.
        
        Args:
            session: Current session.
            message: User message.
            
        Returns:
            RouterResponse with Producer API routing or clarification.
        """
        message_lower = message.lower()
        
        # Check for index patterns
        if any(kw in message_lower for kw in ["index", "build index", "create index"]):
            # Check if path is provided
            # Look for file path patterns
            import re
            path_match = re.search(r'[\w./\\-]+\.(jsonl?|txt)', message, re.IGNORECASE)
            
            if path_match:
                return RouterResponse(
                    message=f"Indexing file: {path_match.group()}...",
                    intent=Intent.WRITE,
                    action="route_producer",
                    route_to="producer",
                    api_endpoint="/index",
                    api_payload={"corpus_path": path_match.group()},
                )
            else:
                return RouterResponse(
                    message="I can help you index a file. Please provide the path to your corpus file (JSONL or JSON format).",
                    intent=Intent.WRITE,
                    action="clarify",
                )
        
        # Check for add patterns
        if any(kw in message_lower for kw in ["add", "insert", "create", "new document"]):
            # Check if content is provided directly
            if len(message) > 50:  # Assume content is provided
                content = message
                for prefix in ["add", "add this", "add document", "add:"]:
                    if message_lower.startswith(prefix):
                        content = message[len(prefix):].strip()
                        break
                
                return RouterResponse(
                    message="Adding document...",
                    intent=Intent.WRITE,
                    action="route_producer",
                    route_to="producer",
                    api_endpoint="/add",
                    api_payload={"documents": [{"content": content}]},
                )
            else:
                return RouterResponse(
                    message="I can help you add documents. Please provide the content you want to add, or tell me about your data source (file path, database connection, etc.).",
                    intent=Intent.WRITE,
                    action="clarify",
                )
        
        # Check for remove patterns
        if any(kw in message_lower for kw in ["remove", "delete", "drop"]):
            # Extract entity name
            import re
            # Look for quoted strings or "named X"
            quote_match = re.search(r'["\']([^"\']+)["\']', message)
            if quote_match:
                entity_name = quote_match.group(1)
                return RouterResponse(
                    message=f"Removing entity '{entity_name}'...",
                    intent=Intent.WRITE,
                    action="route_producer",
                    route_to="producer",
                    api_endpoint="/remove",
                    api_payload={"entity_name": entity_name},
                )
            else:
                return RouterResponse(
                    message="Which entity would you like to remove? Please provide the exact name.",
                    intent=Intent.WRITE,
                    action="clarify",
                )
        
        # Check for reindex patterns
        if any(kw in message_lower for kw in ["reindex", "re-index", "rebuild", "flush"]):
            return RouterResponse(
                message="Starting reindex...",
                intent=Intent.WRITE,
                action="route_producer",
                route_to="producer",
                api_endpoint="/reindex",
                api_payload=None,
            )
        
        # Check for database sync patterns
        if any(kw in message_lower for kw in ["sync", "database", "connect", "import"]):
            # This requires the ingestion agent flow
            return RouterResponse(
                message=WRITE_CLARIFY_PROMPT,
                intent=Intent.WRITE,
                action="clarify",
            )
        
        # Default: need clarification
        return RouterResponse(
            message=WRITE_CLARIFY_PROMPT,
            intent=Intent.WRITE,
            action="clarify",
        )
    
    def _handle_unknown_intent(
        self,
        session: Session,
        message: str,
    ) -> RouterResponse:
        """Handle unknown intent - ask for clarification.
        
        Args:
            session: Current session.
            message: User message.
            
        Returns:
            RouterResponse asking for clarification.
        """
        return RouterResponse(
            message="I'm not sure what you'd like to do. Could you tell me if you want to:\n\n"
                    "- **Search or query** existing data (READ)\n"
                    "- **Add or modify** data (WRITE)\n\n"
                    "Please clarify your request.",
            intent=Intent.UNKNOWN,
            action="clarify",
        )
    
    def _get_system_prompt(self, session: Session) -> str:
        """Generate system prompt based on session permissions."""
        perms = []
        accessible_ids = session.get_accessible_ids()
        if accessible_ids:
            perms.append(f"READ (search, query) - scopes: {', '.join(accessible_ids[:5])}")
        if session.can_write():
            perms.append(f"WRITE (add, index, remove) - using access_id: {session.current_access_id}")
        
        return f"""You are the ArchRAG assistant helping a user with data management.
User: {session.user_id} | Organization: {session.org_id}
Permissions: {', '.join(perms) if perms else 'None'}

Help them accomplish their task by routing to the appropriate API:
- READ operations go to the Consumer API (query, search, info)
- WRITE operations go to the Producer API (index, add, remove, reindex)

All data written will be tagged with access_id: {session.current_access_id}"""
    
    async def execute_route(
        self,
        response: RouterResponse,
    ) -> dict[str, Any]:
        """Execute the API call based on the router response.
        
        Args:
            response: RouterResponse with routing info.
            
        Returns:
            API response as dict.
        """
        if not response.route_to or not response.api_endpoint:
            return {"error": "No route specified", "message": response.message}
        
        base_url = self._producer_url if response.route_to == "producer" else self._consumer_url
        url = f"{base_url}{response.api_endpoint}"
        
        async with httpx.AsyncClient() as client:
            try:
                if response.api_payload is None:
                    # GET request
                    resp = await client.get(url, timeout=30.0)
                else:
                    # POST or DELETE
                    if response.api_endpoint == "/remove":
                        resp = await client.request(
                            "DELETE",
                            url,
                            json=response.api_payload,
                            timeout=30.0,
                        )
                    else:
                        resp = await client.post(
                            url,
                            json=response.api_payload,
                            timeout=30.0,
                        )
                
                return resp.json()
            
            except httpx.RequestError as e:
                log.error("API request failed: %s", e)
                return {"error": f"Failed to reach {response.route_to}", "detail": str(e)}
            except Exception as e:
                log.error("Unexpected error: %s", e)
                return {"error": "Unexpected error", "detail": str(e)}
