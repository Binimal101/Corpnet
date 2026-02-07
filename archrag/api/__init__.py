"""ArchRAG Agentic API.

Provides a stateful conversational API that:
1. Authenticates users with organization-based permissions
2. Determines intent (READ/WRITE)
3. Routes to appropriate backend (Producer/Consumer)
4. Manages hierarchical access control via Supabase
"""

from archrag.api.session import SessionManager, Session, Intent
from archrag.api.router import AgenticRouter, RouterResponse
from archrag.api.permissions import (
    SupabasePermissionStore,
    OrganizationPermissions,
    UserPermissions,
    ReadPermission,
    AccessLevel,
    HierarchyNode,
)
from archrag.api.server import app, create_app

__all__ = [
    # Session management
    "SessionManager",
    "Session", 
    "Intent",
    # Routing
    "AgenticRouter",
    "RouterResponse",
    # Permissions
    "SupabasePermissionStore",
    "OrganizationPermissions",
    "UserPermissions",
    "ReadPermission",
    "AccessLevel",
    "HierarchyNode",
    # Server
    "app",
    "create_app",
]
