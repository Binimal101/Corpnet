"""FastAPI server for the agentic interface.

Provides a stateful HTTP API for frontend integration:
1. Session creation with organization-based auth
2. Message processing with intent routing
3. API result forwarding with access control

Endpoints:
    POST /session        - Create a new session (user_id + org_id)
    POST /chat           - Send a message (routes to Producer/Consumer)
    GET  /session/{id}   - Get session state
    DELETE /session/{id} - End a session
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from archrag.api.session import SessionManager, Session, Intent
from archrag.api.router import AgenticRouter, RouterResponse
from archrag.api.permissions import SupabasePermissionStore

log = logging.getLogger(__name__)

# ── Pydantic Models ──────────────────────────────────────────────────────────


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    user_id: str = Field(..., description="User identifier from frontend")
    organisation_id: str = Field(..., description="Organization identifier", alias="org_id")
    
    class Config:
        populate_by_name = True


class CreateSessionResponse(BaseModel):
    """Response from session creation."""
    session_id: str
    greeting: str
    user_id: str
    org_id: str
    write_perm: bool
    accessible_ids: list[str]
    current_access_id: str


class ChatRequest(BaseModel):
    """Request to send a chat message."""
    session_id: str = Field(..., description="Session identifier")
    message: str = Field(..., description="User message")


class ChatResponse(BaseModel):
    """Response from chat processing."""
    session_id: str
    message: str
    intent: str
    action: str
    access_id: str | None = None  # For writes, the access_id used
    api_result: dict[str, Any] | None = None


class SessionResponse(BaseModel):
    """Session state response."""
    session_id: str
    user_id: str
    org_id: str
    intent: str
    current_access_id: str
    accessible_ids: list[str]
    write_perm: bool
    created_at: str
    last_activity: str
    message_count: int


class SetAccessIdRequest(BaseModel):
    """Request to change the current access scope."""
    access_id: str = Field(..., description="New access scope to use for writes")


# ── Global State ─────────────────────────────────────────────────────────────

# These are initialized in create_app()
_session_manager: SessionManager | None = None
_router: AgenticRouter | None = None
_permission_store: SupabasePermissionStore | None = None


def get_session_manager() -> SessionManager:
    """Dependency to get session manager."""
    if _session_manager is None:
        raise RuntimeError("Session manager not initialized")
    return _session_manager


def get_router() -> AgenticRouter:
    """Dependency to get router."""
    if _router is None:
        raise RuntimeError("Router not initialized")
    return _router


# ── FastAPI App ──────────────────────────────────────────────────────────────


def create_app(
    producer_url: str = "http://localhost:8000",
    consumer_url: str = "http://localhost:8001",
    llm: Any = None,
    session_db_path: str = "~/.archrag/sessions.db",
    supabase_url: str | None = None,
    supabase_key: str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        producer_url: URL of the Producer API.
        consumer_url: URL of the Consumer API.
        llm: LLM port for intent classification. If None, uses config.
        session_db_path: Path to session database.
        supabase_url: Supabase project URL (or SUPABASE_URL env var).
        supabase_key: Supabase API key (or SUPABASE_KEY env var).
        
    Returns:
        Configured FastAPI app.
    """
    global _session_manager, _router, _permission_store
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize resources on startup."""
        global _session_manager, _router, _permission_store
        
        # Initialize permission store
        _permission_store = SupabasePermissionStore(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
        )
        log.info("Permission store initialized")
        
        # Initialize session manager with permission store
        _session_manager = SessionManager(
            db_path=session_db_path,
            permission_store=_permission_store,
        )
        log.info("Session manager initialized: %s", session_db_path)
        
        # Initialize LLM if not provided
        nonlocal llm
        if llm is None:
            try:
                from archrag.config import build_llm, load_config
                cfg = load_config("config.yaml")
                llm = build_llm(cfg.get("llm", {}))
                log.info("LLM initialized from config")
            except Exception as e:
                log.warning("Could not load LLM from config: %s", e)
                # Use a mock LLM for testing
                from archrag.api.mock_llm import MockLLM
                llm = MockLLM()
                log.info("Using mock LLM")
        
        # Initialize router
        _router = AgenticRouter(
            llm=llm,
            producer_url=producer_url,
            consumer_url=consumer_url,
            session_manager=_session_manager,
        )
        log.info("Router initialized: producer=%s, consumer=%s", producer_url, consumer_url)
        
        yield
        
        # Cleanup
        log.info("Shutting down agentic API")
    
    app = FastAPI(
        title="ArchRAG Agentic API",
        description="Stateful conversational API for data management with hierarchical access control",
        version="2.0.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ── Endpoints ────────────────────────────────────────────────────────────
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "archrag-agent"}
    
    @app.post("/session", response_model=CreateSessionResponse)
    async def create_session(
        request: CreateSessionRequest,
        session_mgr: SessionManager = Depends(get_session_manager),
        router: AgenticRouter = Depends(get_router),
    ):
        """Create a new conversation session.
        
        The frontend sends user_id and org_id. Permissions are fetched from
        Supabase based on the organization's configured access hierarchy.
        
        Returns session info including:
        - accessible_ids: All access scopes this user can read
        - current_access_id: Default scope for write operations
        - write_perm: Whether user can perform write operations
        """
        try:
            # Create session - permissions are fetched from Supabase
            session = session_mgr.create_session(
                user_id=request.user_id,
                org_id=request.organisation_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=403, detail=str(e))
        
        # Start conversation
        response = router.start_session(session)
        
        return CreateSessionResponse(
            session_id=session.session_id,
            greeting=response.message,
            user_id=session.user_id,
            org_id=session.org_id,
            write_perm=session.can_write(),
            accessible_ids=session.get_accessible_ids(),
            current_access_id=session.current_access_id,
        )
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        request: ChatRequest,
        session_mgr: SessionManager = Depends(get_session_manager),
        router: AgenticRouter = Depends(get_router),
    ):
        """Process a user message.
        
        This endpoint:
        1. Classifies the intent (READ/WRITE)
        2. Checks permissions (including hierarchical access)
        3. Routes to the appropriate backend API
        4. Attaches access_id to write operations
        5. Returns the result
        """
        # Get session
        session = session_mgr.get_session(request.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Process message
        response = router.process_message(session, request.message)
        
        # If we need to route to an API, execute it
        api_result = None
        if response.action in ("route_producer", "route_consumer"):
            # For write operations, inject the access_id
            if response.route_to == "producer" and response.api_payload:
                # Add access_id to the payload for producer operations
                if response.api_endpoint == "/add":
                    # Inject access_id into each document
                    docs = response.api_payload.get("documents", [])
                    for doc in docs:
                        if "access_id" not in doc:
                            doc["access_id"] = session.current_access_id
                elif response.api_endpoint == "/index":
                    # For index operations, add access_id as a parameter
                    response.api_payload["access_id"] = session.current_access_id
            
            api_result = await router.execute_route(response)
            
            # Format the final message with the result
            if "error" not in api_result:
                if response.api_endpoint == "/query":
                    answer = api_result.get("answer", "No answer found.")
                    response.message = answer
                elif response.api_endpoint == "/search":
                    entities = api_result.get("entities", [])
                    chunks = api_result.get("chunks", [])
                    response.message = _format_search_results(entities, chunks)
                elif response.api_endpoint == "/info":
                    response.message = _format_stats(api_result)
                elif response.api_endpoint == "/index":
                    response.message = api_result.get("message", "Indexing complete.")
                elif response.api_endpoint == "/add":
                    response.message = api_result.get("message", "Documents added.")
                elif response.api_endpoint == "/remove":
                    response.message = api_result.get("message", "Entity removed.")
                elif response.api_endpoint == "/reindex":
                    response.message = api_result.get("message", "Reindex started.")
            else:
                response.message = f"Error: {api_result.get('error', 'Unknown error')}"
        
        return ChatResponse(
            session_id=session.session_id,
            message=response.message,
            intent=response.intent.value,
            action=response.action or "none",
            access_id=session.current_access_id if response.intent == Intent.WRITE else None,
            api_result=api_result,
        )
    
    @app.get("/session/{session_id}", response_model=SessionResponse)
    async def get_session(
        session_id: str,
        session_mgr: SessionManager = Depends(get_session_manager),
    ):
        """Get session state including access permissions."""
        session = session_mgr.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionResponse(
            session_id=session.session_id,
            user_id=session.user_id,
            org_id=session.org_id,
            intent=session.intent.value,
            current_access_id=session.current_access_id,
            accessible_ids=session.get_accessible_ids(),
            write_perm=session.can_write(),
            created_at=session.created_at,
            last_activity=session.last_activity,
            message_count=len(session.messages),
        )
    
    @app.put("/session/{session_id}/access")
    async def set_access_id(
        session_id: str,
        request: SetAccessIdRequest,
        session_mgr: SessionManager = Depends(get_session_manager),
    ):
        """Change the current access scope for write operations.
        
        The new access_id must be in the user's accessible scopes.
        """
        session = session_mgr.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if not session_mgr.set_access_id(session, request.access_id):
            raise HTTPException(
                status_code=403,
                detail=f"Access scope '{request.access_id}' not permitted. "
                       f"Allowed: {session.get_accessible_ids()}"
            )
        
        return {
            "status": "updated",
            "current_access_id": session.current_access_id,
        }
    
    @app.delete("/session/{session_id}")
    async def delete_session(
        session_id: str,
        session_mgr: SessionManager = Depends(get_session_manager),
    ):
        """End and delete a session."""
        deleted = session_mgr.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"status": "deleted", "session_id": session_id}
    
    @app.get("/sessions")
    async def list_sessions(
        user_id: str | None = None,
        org_id: str | None = None,
        session_mgr: SessionManager = Depends(get_session_manager),
    ):
        """List sessions by user or organization."""
        if user_id:
            sessions = session_mgr.get_user_sessions(user_id)
        elif org_id:
            sessions = session_mgr.get_org_sessions(org_id)
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide user_id or org_id query parameter"
            )
        
        return {
            "user_id": user_id,
            "org_id": org_id,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "user_id": s.user_id,
                    "org_id": s.org_id,
                    "intent": s.intent.value,
                    "current_access_id": s.current_access_id,
                    "created_at": s.created_at,
                    "last_activity": s.last_activity,
                    "message_count": len(s.messages),
                }
                for s in sessions
            ],
        }
    
    @app.get("/permissions/{org_id}")
    async def get_org_permissions(org_id: str):
        """Get the permission hierarchy for an organization.
        
        Useful for frontend to display available access scopes.
        """
        if _permission_store is None:
            raise HTTPException(status_code=500, detail="Permission store not initialized")
        
        perms = _permission_store.get_org_permissions(org_id)
        if perms is None:
            raise HTTPException(status_code=404, detail=f"Organization '{org_id}' not found")
        
        return perms.to_dict()
    
    return app


def _format_search_results(entities: list, chunks: list) -> str:
    """Format search results for display."""
    lines = []
    
    if entities:
        lines.append(f"**Found {len(entities)} entities:**")
        for e in entities[:5]:  # Limit to 5
            lines.append(f"- {e.get('name', 'Unknown')} ({e.get('type', 'Unknown')})")
        if len(entities) > 5:
            lines.append(f"  ...and {len(entities) - 5} more")
    
    if chunks:
        lines.append(f"\n**Found {len(chunks)} text chunks:**")
        for c in chunks[:3]:  # Limit to 3
            content = c.get('content', '')[:100]
            lines.append(f"- {content}...")
        if len(chunks) > 3:
            lines.append(f"  ...and {len(chunks) - 3} more")
    
    if not entities and not chunks:
        lines.append("No results found.")
    
    return "\n".join(lines)


def _format_stats(stats: dict) -> str:
    """Format system stats for display."""
    return (
        f"**System Statistics:**\n"
        f"- Entities: {stats.get('entities', 0)}\n"
        f"- Relations: {stats.get('relations', 0)}\n"
        f"- Text Chunks: {stats.get('chunks', 0)}\n"
        f"- Hierarchy Levels: {stats.get('hierarchy_levels', 0)}\n"
        f"- Pending Documents: {stats.get('pending', 0)}\n"
        f"- Reindex Status: {stats.get('reindex_status', 'unknown')}"
    )


# Default app instance
app = create_app()


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    """Run the agentic API server."""
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="ArchRAG Agentic API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--producer-url", default="http://localhost:8000", help="Producer API URL")
    parser.add_argument("--consumer-url", default="http://localhost:8001", help="Consumer API URL")
    parser.add_argument("--supabase-url", help="Supabase project URL")
    parser.add_argument("--supabase-key", help="Supabase API key")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Create app with custom URLs
    app = create_app(
        producer_url=args.producer_url,
        consumer_url=args.consumer_url,
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
    )
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
