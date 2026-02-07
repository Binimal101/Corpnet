"""Persistent storage for database connection metadata.

Stores connection configurations so producers don't need to re-enter
credentials and settings on subsequent sessions.

Storage includes:
- Connection name (user-friendly alias like "people", "sales")
- Connector type (sql, nosql)
- Connection config (encrypted credentials)
- Table preferences (which tables to sync)
- Text column mappings
- Sync history and statistics
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class SavedConnection:
    """A saved database connection with all metadata."""
    
    name: str  # User-friendly alias (e.g., "people", "sales_db")
    connector_type: str  # "sql" or "nosql"
    connection_config: dict[str, Any]  # Connection parameters
    
    # Optional metadata
    description: str = ""
    tables: list[str] = field(default_factory=list)  # Tables to sync
    text_columns_map: dict[str, list[str]] = field(default_factory=dict)
    
    # Sync preferences
    auto_sync_enabled: bool = False
    auto_sync_interval: int = 300  # seconds
    enable_linking: bool = True
    enable_evolution: bool = False
    
    # Timestamps
    created_at: str = ""
    last_used_at: str = ""
    last_sync_at: str = ""
    
    # Statistics
    total_syncs: int = 0
    total_records_synced: int = 0
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SavedConnection":
        return cls(**data)
    
    def update_last_used(self) -> None:
        self.last_used_at = datetime.now().isoformat()
    
    def update_sync_stats(self, records_synced: int) -> None:
        self.last_sync_at = datetime.now().isoformat()
        self.total_syncs += 1
        self.total_records_synced += records_synced


class ConnectionStore:
    """SQLite-backed persistent storage for connection metadata.
    
    Stores connection configs securely so producers can reference
    connections by name without re-entering credentials.
    
    Usage:
        store = ConnectionStore()
        
        # Save a new connection
        store.save_connection(SavedConnection(
            name="people",
            connector_type="sql",
            connection_config={"connection_string": "postgresql://..."},
            tables=["users", "profiles"],
        ))
        
        # Retrieve by name
        conn = store.get_connection("people")
        
        # List all saved connections
        all_conns = store.list_connections()
    """
    
    def __init__(self, db_path: str | None = None):
        """Initialize the connection store.
        
        Args:
            db_path: Path to SQLite database. Defaults to ~/.archrag/connections.db
        """
        if db_path is None:
            home = Path.home()
            archrag_dir = home / ".archrag"
            archrag_dir.mkdir(exist_ok=True)
            db_path = str(archrag_dir / "connections.db")
        
        self._db_path = db_path
        self._init_db()
        log.debug("ConnectionStore initialized at %s", db_path)
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS connections (
                    name TEXT PRIMARY KEY,
                    connector_type TEXT NOT NULL,
                    connection_config TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    tables TEXT DEFAULT '[]',
                    text_columns_map TEXT DEFAULT '{}',
                    auto_sync_enabled INTEGER DEFAULT 0,
                    auto_sync_interval INTEGER DEFAULT 300,
                    enable_linking INTEGER DEFAULT 1,
                    enable_evolution INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    last_sync_at TEXT,
                    total_syncs INTEGER DEFAULT 0,
                    total_records_synced INTEGER DEFAULT 0
                )
            """)
            
            # Sync history table for detailed tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    connection_name TEXT NOT NULL,
                    sync_type TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    tables_synced TEXT,
                    records_added INTEGER DEFAULT 0,
                    records_failed INTEGER DEFAULT 0,
                    errors TEXT,
                    FOREIGN KEY (connection_name) REFERENCES connections(name)
                )
            """)
            
            # Agent conversation history for context
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    connection_name TEXT,
                    summary TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tool_calls TEXT,
                    FOREIGN KEY (session_id) REFERENCES agent_sessions(id)
                )
            """)
            
            conn.commit()
    
    # ── Connection CRUD ──────────────────────────────────────────────────────
    
    def save_connection(self, connection: SavedConnection) -> None:
        """Save or update a connection configuration."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO connections (
                    name, connector_type, connection_config, description,
                    tables, text_columns_map, auto_sync_enabled, auto_sync_interval,
                    enable_linking, enable_evolution, created_at, last_used_at,
                    last_sync_at, total_syncs, total_records_synced
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                connection.name,
                connection.connector_type,
                json.dumps(connection.connection_config),
                connection.description,
                json.dumps(connection.tables),
                json.dumps(connection.text_columns_map),
                1 if connection.auto_sync_enabled else 0,
                connection.auto_sync_interval,
                1 if connection.enable_linking else 0,
                1 if connection.enable_evolution else 0,
                connection.created_at,
                connection.last_used_at,
                connection.last_sync_at,
                connection.total_syncs,
                connection.total_records_synced,
            ))
            conn.commit()
        log.info("Saved connection: %s", connection.name)
    
    def get_connection(self, name: str) -> SavedConnection | None:
        """Retrieve a connection by name."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM connections WHERE name = ?", (name,)
            ).fetchone()
            
            if row is None:
                return None
            
            return SavedConnection(
                name=row["name"],
                connector_type=row["connector_type"],
                connection_config=json.loads(row["connection_config"]),
                description=row["description"] or "",
                tables=json.loads(row["tables"]),
                text_columns_map=json.loads(row["text_columns_map"]),
                auto_sync_enabled=bool(row["auto_sync_enabled"]),
                auto_sync_interval=row["auto_sync_interval"],
                enable_linking=bool(row["enable_linking"]),
                enable_evolution=bool(row["enable_evolution"]),
                created_at=row["created_at"],
                last_used_at=row["last_used_at"] or "",
                last_sync_at=row["last_sync_at"] or "",
                total_syncs=row["total_syncs"],
                total_records_synced=row["total_records_synced"],
            )
    
    def list_connections(self) -> list[SavedConnection]:
        """List all saved connections."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM connections ORDER BY last_used_at DESC"
            ).fetchall()
            
            return [
                SavedConnection(
                    name=row["name"],
                    connector_type=row["connector_type"],
                    connection_config=json.loads(row["connection_config"]),
                    description=row["description"] or "",
                    tables=json.loads(row["tables"]),
                    text_columns_map=json.loads(row["text_columns_map"]),
                    auto_sync_enabled=bool(row["auto_sync_enabled"]),
                    auto_sync_interval=row["auto_sync_interval"],
                    enable_linking=bool(row["enable_linking"]),
                    enable_evolution=bool(row["enable_evolution"]),
                    created_at=row["created_at"],
                    last_used_at=row["last_used_at"] or "",
                    last_sync_at=row["last_sync_at"] or "",
                    total_syncs=row["total_syncs"],
                    total_records_synced=row["total_records_synced"],
                )
                for row in rows
            ]
    
    def delete_connection(self, name: str) -> bool:
        """Delete a connection by name."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM connections WHERE name = ?", (name,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
        
        if deleted:
            log.info("Deleted connection: %s", name)
        return deleted
    
    def connection_exists(self, name: str) -> bool:
        """Check if a connection with this name exists."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM connections WHERE name = ?", (name,)
            ).fetchone()
            return row is not None
    
    def update_connection_usage(self, name: str) -> None:
        """Update the last_used_at timestamp."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE connections SET last_used_at = ? WHERE name = ?",
                (datetime.now().isoformat(), name)
            )
            conn.commit()
    
    def update_sync_stats(
        self,
        name: str,
        records_synced: int,
    ) -> None:
        """Update sync statistics for a connection."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                UPDATE connections SET
                    last_sync_at = ?,
                    total_syncs = total_syncs + 1,
                    total_records_synced = total_records_synced + ?
                WHERE name = ?
            """, (datetime.now().isoformat(), records_synced, name))
            conn.commit()
    
    # ── Sync History ─────────────────────────────────────────────────────────
    
    def record_sync(
        self,
        connection_name: str,
        sync_type: str,
        tables_synced: list[str],
        records_added: int,
        records_failed: int,
        errors: list[str] | None = None,
    ) -> int:
        """Record a sync operation in history."""
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO sync_history (
                    connection_name, sync_type, started_at, completed_at,
                    tables_synced, records_added, records_failed, errors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                connection_name,
                sync_type,
                now,
                now,
                json.dumps(tables_synced),
                records_added,
                records_failed,
                json.dumps(errors) if errors else None,
            ))
            conn.commit()
            return cursor.lastrowid or 0
    
    def get_sync_history(
        self,
        connection_name: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get sync history for a connection."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM sync_history
                WHERE connection_name = ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (connection_name, limit)).fetchall()
            
            return [
                {
                    "id": row["id"],
                    "sync_type": row["sync_type"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "tables_synced": json.loads(row["tables_synced"]) if row["tables_synced"] else [],
                    "records_added": row["records_added"],
                    "records_failed": row["records_failed"],
                    "errors": json.loads(row["errors"]) if row["errors"] else [],
                }
                for row in rows
            ]
    
    # ── Agent Sessions ───────────────────────────────────────────────────────
    
    def start_session(self, connection_name: str | None = None) -> int:
        """Start a new agent session."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO agent_sessions (started_at, connection_name)
                VALUES (?, ?)
            """, (datetime.now().isoformat(), connection_name))
            conn.commit()
            return cursor.lastrowid or 0
    
    def end_session(self, session_id: int, summary: str = "") -> None:
        """End an agent session."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                UPDATE agent_sessions SET ended_at = ?, summary = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), summary, session_id))
            conn.commit()
    
    def save_message(
        self,
        session_id: int,
        role: str,
        content: str,
        tool_calls: list[dict] | None = None,
    ) -> None:
        """Save a message to session history."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT INTO agent_messages (session_id, role, content, timestamp, tool_calls)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                role,
                content,
                datetime.now().isoformat(),
                json.dumps(tool_calls) if tool_calls else None,
            ))
            conn.commit()
    
    def get_session_messages(self, session_id: int) -> list[dict[str, Any]]:
        """Get all messages for a session."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM agent_messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,)).fetchall()
            
            return [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "tool_calls": json.loads(row["tool_calls"]) if row["tool_calls"] else None,
                }
                for row in rows
            ]
    
    def get_recent_sessions(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get recent agent sessions."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM agent_sessions
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [
                {
                    "id": row["id"],
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                    "connection_name": row["connection_name"],
                    "summary": row["summary"],
                }
                for row in rows
            ]
