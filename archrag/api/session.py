"""Session management for stateful agentic API.

Handles:
- User authentication with organization-based permissions
- Session state persistence
- Hierarchical access control via Supabase
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from archrag.api.permissions import UserPermissions, SupabasePermissionStore

log = logging.getLogger(__name__)


class Intent(Enum):
    """User intent classification."""
    UNKNOWN = "unknown"
    READ = "read"      # Query, search, info - routes to Consumer
    WRITE = "write"    # Index, add, remove, sync - routes to Producer


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: str = ""
    intent: Intent | None = None
    tool_calls: list[dict[str, Any]] | None = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "intent": self.intent.value if self.intent else None,
        }


@dataclass
class Session:
    """A user session with conversation history and state.
    
    Attributes:
        session_id: Unique session identifier
        user_id: User identifier from frontend
        org_id: Organization identifier
        permissions: User permissions with hierarchical access
        messages: Conversation history
        intent: Detected user intent (READ/WRITE)
        current_access_id: Active access scope for write operations
        created_at: Session creation timestamp
        last_activity: Last activity timestamp
        metadata: Additional session data
    """
    session_id: str
    user_id: str
    org_id: str
    permissions: "UserPermissions"
    messages: list[Message] = field(default_factory=list)
    intent: Intent = Intent.UNKNOWN
    current_access_id: str = ""  # Active access scope for writes
    created_at: str = ""
    last_activity: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_activity:
            self.last_activity = now
        # Set default access_id if not provided
        if not self.current_access_id and self.permissions:
            self.current_access_id = self.permissions.get_default_access_id()
    
    def can_write(self) -> bool:
        """Check if user can perform write operations."""
        return self.permissions.can_write()
    
    def can_read(self, access_id: str | None = None) -> bool:
        """Check if user can read content with given access_id."""
        if access_id is None:
            return True  # Public content
        return self.permissions.can_read(access_id)
    
    def get_accessible_ids(self) -> list[str]:
        """Get all access_ids this user can read."""
        return self.permissions.get_accessible_ids()
    
    def add_message(self, role: str, content: str, intent: Intent | None = None) -> Message:
        """Add a message to the conversation."""
        msg = Message(role=role, content=content, intent=intent)
        self.messages.append(msg)
        self.last_activity = datetime.now().isoformat()
        return msg
    
    def get_messages_for_llm(self) -> list[dict[str, str]]:
        """Get messages formatted for LLM API."""
        return [{"role": m.role, "content": m.content} for m in self.messages]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "permissions": self.permissions.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
            "intent": self.intent.value,
            "current_access_id": self.current_access_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "metadata": self.metadata,
        }


class SessionManager:
    """Manages user sessions with SQLite persistence and Supabase permissions.
    
    Stores session state so conversations can continue across requests.
    Fetches permissions from Supabase based on organization.
    """
    
    def __init__(
        self,
        db_path: str = "~/.archrag/sessions.db",
        permission_store: "SupabasePermissionStore | None" = None,
    ) -> None:
        """Initialize the session manager.
        
        Args:
            db_path: Path to SQLite database for session storage.
            permission_store: Supabase permission store. Created if not provided.
        """
        db_path_obj = Path(db_path).expanduser()
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = sqlite3.connect(str(db_path_obj), check_same_thread=False)
        self._create_tables()
        
        # Initialize permission store
        if permission_store is None:
            from archrag.api.permissions import SupabasePermissionStore
            permission_store = SupabasePermissionStore()
        self._permission_store = permission_store
        
        # In-memory cache of active sessions
        self._sessions: dict[str, Session] = {}
    
    def _create_tables(self) -> None:
        """Create database tables."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id        TEXT PRIMARY KEY,
                user_id           TEXT NOT NULL,
                org_id            TEXT NOT NULL,
                permissions_json  TEXT NOT NULL,
                intent            TEXT NOT NULL DEFAULT 'unknown',
                current_access_id TEXT NOT NULL DEFAULT '',
                created_at        TEXT NOT NULL,
                last_activity     TEXT NOT NULL,
                metadata          TEXT NOT NULL DEFAULT '{}'
            );
            
            CREATE TABLE IF NOT EXISTS messages (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL,
                role          TEXT NOT NULL,
                content       TEXT NOT NULL,
                timestamp     TEXT NOT NULL,
                intent        TEXT,
                tool_calls    TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_org ON sessions(org_id);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        """)
        self._conn.commit()
    
    def create_session(
        self,
        user_id: str,
        org_id: str,
    ) -> Session:
        """Create a new session with permissions fetched from Supabase.
        
        Args:
            user_id: User identifier from frontend.
            org_id: Organization identifier from frontend.
            
        Returns:
            New Session object.
            
        Raises:
            ValueError: If organization not found or user has no permissions.
        """
        # Fetch permissions from Supabase
        permissions = self._permission_store.get_user_permissions(user_id, org_id)
        
        if permissions is None:
            raise ValueError(f"Organization '{org_id}' not found or user '{user_id}' has no access")
        
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
            permissions=permissions,
        )
        
        # Serialize permissions for storage
        permissions_json = json.dumps(self._serialize_permissions(permissions))
        
        # Save to database
        self._conn.execute(
            """INSERT INTO sessions 
               (session_id, user_id, org_id, permissions_json, intent, 
                current_access_id, created_at, last_activity, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session.session_id,
                session.user_id,
                session.org_id,
                permissions_json,
                session.intent.value,
                session.current_access_id,
                session.created_at,
                session.last_activity,
                json.dumps(session.metadata),
            ),
        )
        self._conn.commit()
        
        # Cache
        self._sessions[session_id] = session
        
        log.info(
            "Created session %s for user %s (org: %s, access_id: %s)",
            session_id, user_id, org_id, session.current_access_id
        )
        return session
    
    def _serialize_permissions(self, permissions: "UserPermissions") -> dict[str, Any]:
        """Serialize permissions to JSON-compatible dict."""
        return {
            "user_id": permissions.user_id,
            "org_id": permissions.org_id,
            "write_perm": permissions.can_write(),
            "accessible_ids": permissions.get_accessible_ids(),
            "default_access_id": permissions.get_default_access_id(),
            "read_perms": [p.to_dict() for p in (permissions.user_read_perms or permissions.org_permissions.read_perms)],
        }
    
    def _deserialize_permissions(self, data: dict[str, Any]) -> "UserPermissions":
        """Deserialize permissions from JSON dict."""
        from archrag.api.permissions import (
            UserPermissions, OrganizationPermissions, ReadPermission, AccessLevel
        )
        
        # Reconstruct minimal permission objects for session use
        read_perms = [
            ReadPermission(
                access_id=p["access_id"],
                level=AccessLevel(p["level"]),
                name=p["name"],
                includes_children=p.get("includes_children", True),
            )
            for p in data.get("read_perms", [])
        ]
        
        # Create a minimal org permissions object
        org_perms = OrganizationPermissions(
            org_id=data["org_id"],
            write_perm=data["write_perm"],
            read_perms=read_perms,
            hierarchy={},  # Not needed for basic permission checks
        )
        
        return UserPermissions(
            user_id=data["user_id"],
            org_id=data["org_id"],
            org_permissions=org_perms,
            user_write_perm=data["write_perm"],
        )
    
    def get_session(self, session_id: str) -> Session | None:
        """Retrieve a session by ID.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Session object or None if not found.
        """
        # Check cache first
        if session_id in self._sessions:
            return self._sessions[session_id]
        
        # Load from database
        cur = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        
        # Parse permissions
        permissions_data = json.loads(row[3])
        permissions = self._deserialize_permissions(permissions_data)
        
        session = Session(
            session_id=row[0],
            user_id=row[1],
            org_id=row[2],
            permissions=permissions,
            intent=Intent(row[4]),
            current_access_id=row[5],
            created_at=row[6],
            last_activity=row[7],
            metadata=json.loads(row[8]),
        )
        
        # Load messages
        cur = self._conn.execute(
            "SELECT role, content, timestamp, intent, tool_calls FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        for msg_row in cur.fetchall():
            session.messages.append(Message(
                role=msg_row[0],
                content=msg_row[1],
                timestamp=msg_row[2],
                intent=Intent(msg_row[3]) if msg_row[3] else None,
                tool_calls=json.loads(msg_row[4]) if msg_row[4] else None,
            ))
        
        # Cache
        self._sessions[session_id] = session
        
        return session
    
    def save_session(self, session: Session) -> None:
        """Persist session state.
        
        Args:
            session: Session to save.
        """
        # Update session record
        self._conn.execute(
            """UPDATE sessions SET
               intent = ?,
               current_access_id = ?,
               last_activity = ?,
               metadata = ?
               WHERE session_id = ?""",
            (
                session.intent.value,
                session.current_access_id,
                session.last_activity,
                json.dumps(session.metadata),
                session.session_id,
            ),
        )
        self._conn.commit()
    
    def add_message(
        self,
        session: Session,
        role: str,
        content: str,
        intent: Intent | None = None,
        tool_calls: list[dict] | None = None,
    ) -> Message:
        """Add a message to a session and persist.
        
        Args:
            session: Target session.
            role: Message role.
            content: Message content.
            intent: Optional intent classification.
            tool_calls: Optional tool calls.
            
        Returns:
            The new Message.
        """
        msg = session.add_message(role, content, intent)
        msg.tool_calls = tool_calls
        
        # Persist message
        self._conn.execute(
            """INSERT INTO messages 
               (session_id, role, content, timestamp, intent, tool_calls)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                session.session_id,
                msg.role,
                msg.content,
                msg.timestamp,
                msg.intent.value if msg.intent else None,
                json.dumps(msg.tool_calls) if msg.tool_calls else None,
            ),
        )
        self._conn.commit()
        
        return msg
    
    def get_user_sessions(self, user_id: str) -> list[Session]:
        """Get all sessions for a user.
        
        Args:
            user_id: User identifier.
            
        Returns:
            List of sessions.
        """
        cur = self._conn.execute(
            "SELECT session_id FROM sessions WHERE user_id = ? ORDER BY last_activity DESC",
            (user_id,),
        )
        sessions = []
        for row in cur.fetchall():
            session = self.get_session(row[0])
            if session:
                sessions.append(session)
        return sessions
    
    def get_org_sessions(self, org_id: str) -> list[Session]:
        """Get all sessions for an organization.
        
        Args:
            org_id: Organization identifier.
            
        Returns:
            List of sessions.
        """
        cur = self._conn.execute(
            "SELECT session_id FROM sessions WHERE org_id = ? ORDER BY last_activity DESC",
            (org_id,),
        )
        sessions = []
        for row in cur.fetchall():
            session = self.get_session(row[0])
            if session:
                sessions.append(session)
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (session_id,),
        )
        result = self._conn.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()
        
        if session_id in self._sessions:
            del self._sessions[session_id]
        
        return result.rowcount > 0
    
    def set_access_id(self, session: Session, access_id: str) -> bool:
        """Set the current access_id for a session.
        
        The access_id must be in the user's accessible scopes.
        
        Args:
            session: Target session.
            access_id: New access scope.
            
        Returns:
            True if set, False if access_id not permitted.
        """
        if access_id not in session.get_accessible_ids():
            log.warning(
                "User %s tried to set access_id '%s' but doesn't have permission",
                session.user_id, access_id
            )
            return False
        
        session.current_access_id = access_id
        self.save_session(session)
        return True


# Keep AuthToken for backwards compatibility but it's deprecated
@dataclass
class AuthToken:
    """DEPRECATED: Use organization-based permissions instead.
    
    Kept for backwards compatibility.
    """
    write_perm: bool = False
    read_perm: bool = True
    
    def can_write(self) -> bool:
        return self.write_perm
    
    def can_read(self) -> bool:
        return self.read_perm
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthToken":
        return cls(
            write_perm=data.get("write_perm", False),
            read_perm=data.get("read_perm", True),
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {"write_perm": self.write_perm, "read_perm": self.read_perm}
