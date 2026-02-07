"""Agent tools for the ingestion agent.

These tools wrap the orchestrator functionality and integrate with
the persistent connection store for stateful operation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from archrag.agent.connection_store import ConnectionStore, SavedConnection

log = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    data: Any
    message: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
        }


@dataclass
class ToolDefinition:
    """Definition of an agent tool."""
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., ToolResult]
    
    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class AgentTools:
    """Collection of tools available to the ingestion agent.
    
    These tools provide:
    - Connection management (save, load, list, delete)
    - Database operations (connect, list tables, get schema)
    - Sync operations (full sync, incremental sync)
    - Status and info queries
    
    All operations integrate with the ConnectionStore for persistence.
    """
    
    def __init__(
        self,
        orchestrator: Any,  # ArchRAGOrchestrator
        connection_store: ConnectionStore,
    ):
        self._orch = orchestrator
        self._store = connection_store
        self._current_connection: SavedConnection | None = None
        
        # Build tool definitions
        self._tools = self._build_tools()
    
    @property
    def tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())
    
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool schemas."""
        return [t.to_openai_schema() for t in self._tools.values()]
    
    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given arguments."""
        if name not in self._tools:
            return ToolResult(
                success=False,
                data=None,
                message=f"Unknown tool: {name}",
            )
        
        tool = self._tools[name]
        try:
            return tool.function(**arguments)
        except Exception as e:
            log.exception("Tool execution failed: %s", name)
            return ToolResult(
                success=False,
                data=None,
                message=f"Tool execution failed: {str(e)}",
            )
    
    def _build_tools(self) -> dict[str, ToolDefinition]:
        """Build all tool definitions."""
        return {
            # Connection management
            "list_saved_connections": ToolDefinition(
                name="list_saved_connections",
                description="List all saved database connections. Use this to see what databases the producer has connected before.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                function=self._list_saved_connections,
            ),
            "get_saved_connection": ToolDefinition(
                name="get_saved_connection",
                description="Get details of a saved connection by name. Use this to check if a connection exists and get its configuration.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name/alias of the saved connection (e.g., 'people', 'sales_db').",
                        },
                    },
                    "required": ["name"],
                },
                function=self._get_saved_connection,
            ),
            "save_connection": ToolDefinition(
                name="save_connection",
                description="Save a new database connection with a friendly name. This stores all connection details for future use.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "A friendly name/alias for this connection (e.g., 'people', 'customers_db').",
                        },
                        "connector_type": {
                            "type": "string",
                            "enum": ["sql", "nosql"],
                            "description": "Type of database: 'sql' for PostgreSQL/MySQL/SQLite, 'nosql' for MongoDB.",
                        },
                        "connection_string": {
                            "type": "string",
                            "description": "Connection string/URI (e.g., 'postgresql://user:pass@host/db').",
                        },
                        "host": {
                            "type": "string",
                            "description": "Database host (for NoSQL without connection string).",
                        },
                        "port": {
                            "type": "integer",
                            "description": "Database port (for NoSQL without connection string).",
                        },
                        "database": {
                            "type": "string",
                            "description": "Database name (for NoSQL without connection string).",
                        },
                        "username": {
                            "type": "string",
                            "description": "Username (for NoSQL without connection string).",
                        },
                        "password": {
                            "type": "string",
                            "description": "Password (for NoSQL without connection string).",
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of what this database contains.",
                        },
                    },
                    "required": ["name", "connector_type"],
                },
                function=self._save_connection,
            ),
            "delete_saved_connection": ToolDefinition(
                name="delete_saved_connection",
                description="Delete a saved connection by name.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the connection to delete.",
                        },
                    },
                    "required": ["name"],
                },
                function=self._delete_saved_connection,
            ),
            
            # Database operations
            "connect_database": ToolDefinition(
                name="connect_database",
                description="Connect to a database. Can use a saved connection name or provide new connection details.",
                parameters={
                    "type": "object",
                    "properties": {
                        "saved_name": {
                            "type": "string",
                            "description": "Name of a saved connection to use. If provided, no other connection params needed.",
                        },
                        "connector_type": {
                            "type": "string",
                            "enum": ["sql", "nosql"],
                            "description": "Type of database (only if not using saved_name).",
                        },
                        "connection_string": {
                            "type": "string",
                            "description": "Connection string (only if not using saved_name).",
                        },
                        "host": {
                            "type": "string",
                            "description": "Database host (for NoSQL, only if not using saved_name).",
                        },
                        "port": {
                            "type": "integer",
                            "description": "Database port (for NoSQL, only if not using saved_name).",
                        },
                        "database": {
                            "type": "string",
                            "description": "Database name (for NoSQL, only if not using saved_name).",
                        },
                    },
                    "required": [],
                },
                function=self._connect_database,
            ),
            "disconnect_database": ToolDefinition(
                name="disconnect_database",
                description="Disconnect from the currently connected database.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                function=self._disconnect_database,
            ),
            "list_tables": ToolDefinition(
                name="list_tables",
                description="List all tables/collections in the connected database.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                function=self._list_tables,
            ),
            "get_table_schema": ToolDefinition(
                name="get_table_schema",
                description="Get detailed schema information for a specific table, including columns and their types.",
                parameters={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "The name of the table to inspect.",
                        },
                    },
                    "required": ["table_name"],
                },
                function=self._get_table_schema,
            ),
            
            # Sync operations
            "sync_database": ToolDefinition(
                name="sync_database",
                description="Sync records from the connected database into ArchRAG. Creates MemoryNotes, extracts entities, and builds the knowledge graph.",
                parameters={
                    "type": "object",
                    "properties": {
                        "tables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific tables to sync. If not provided, syncs all tables.",
                        },
                        "text_columns": {
                            "type": "object",
                            "description": "Map of table name -> list of column names to use for text extraction. If not provided, auto-detects.",
                        },
                        "incremental": {
                            "type": "boolean",
                            "description": "If true, only sync new/changed records. Default is true.",
                        },
                        "save_preferences": {
                            "type": "boolean",
                            "description": "Whether to save table and column preferences to the connection. Default is true.",
                        },
                    },
                    "required": [],
                },
                function=self._sync_database,
            ),
            "update_connection_preferences": ToolDefinition(
                name="update_connection_preferences",
                description="Update the sync preferences for a saved connection (tables, columns, linking, evolution settings).",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the saved connection to update.",
                        },
                        "tables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tables to sync.",
                        },
                        "text_columns": {
                            "type": "object",
                            "description": "Map of table name -> columns for text extraction.",
                        },
                        "auto_sync_enabled": {
                            "type": "boolean",
                            "description": "Whether to enable automatic background syncing.",
                        },
                        "auto_sync_interval": {
                            "type": "integer",
                            "description": "Seconds between automatic syncs.",
                        },
                    },
                    "required": ["name"],
                },
                function=self._update_connection_preferences,
            ),
            
            # Info and status
            "get_sync_status": ToolDefinition(
                name="get_sync_status",
                description="Get the current sync status and statistics.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                function=self._get_sync_status,
            ),
            "get_sync_history": ToolDefinition(
                name="get_sync_history",
                description="Get the sync history for a connection.",
                parameters={
                    "type": "object",
                    "properties": {
                        "connection_name": {
                            "type": "string",
                            "description": "Name of the connection. Uses current connection if not provided.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of history entries to return. Default is 10.",
                        },
                    },
                    "required": [],
                },
                function=self._get_sync_history,
            ),
            "get_database_stats": ToolDefinition(
                name="get_database_stats",
                description="Get statistics about the ArchRAG database (entities, relations, chunks, notes).",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                function=self._get_database_stats,
            ),
            
            # Data ingestion
            "add_note": ToolDefinition(
                name="add_note",
                description="Add a single memory note with content. The note will be enriched with LLM-generated metadata.",
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The main text content of the note.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category for the note.",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for the note.",
                        },
                    },
                    "required": ["content"],
                },
                function=self._add_note,
            ),
            "search_notes": ToolDefinition(
                name="search_notes",
                description="Search for notes by semantic similarity to a query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Maximum number of results. Default is 10.",
                        },
                    },
                    "required": ["query"],
                },
                function=self._search_notes,
            ),
        }
    
    # ── Tool Implementations ─────────────────────────────────────────────────
    
    def _list_saved_connections(self) -> ToolResult:
        connections = self._store.list_connections()
        
        if not connections:
            return ToolResult(
                success=True,
                data=[],
                message="No saved connections found. Use save_connection to save a new database connection.",
            )
        
        conn_list = [
            {
                "name": c.name,
                "type": c.connector_type,
                "description": c.description,
                "tables": c.tables,
                "last_used": c.last_used_at or "never",
                "total_syncs": c.total_syncs,
                "total_records": c.total_records_synced,
            }
            for c in connections
        ]
        
        return ToolResult(
            success=True,
            data=conn_list,
            message=f"Found {len(connections)} saved connection(s).",
        )
    
    def _get_saved_connection(self, name: str) -> ToolResult:
        conn = self._store.get_connection(name)
        
        if conn is None:
            return ToolResult(
                success=False,
                data=None,
                message=f"No connection found with name '{name}'.",
            )
        
        # Don't expose password in full
        config = conn.connection_config.copy()
        if "password" in config:
            config["password"] = "***"
        if "connection_string" in config:
            # Mask password in connection string
            import re
            config["connection_string"] = re.sub(
                r":([^:@]+)@",
                ":***@",
                config["connection_string"]
            )
        
        return ToolResult(
            success=True,
            data={
                "name": conn.name,
                "type": conn.connector_type,
                "config": config,
                "description": conn.description,
                "tables": conn.tables,
                "text_columns": conn.text_columns_map,
                "auto_sync": conn.auto_sync_enabled,
                "auto_sync_interval": conn.auto_sync_interval,
                "created_at": conn.created_at,
                "last_used": conn.last_used_at or "never",
                "last_sync": conn.last_sync_at or "never",
                "total_syncs": conn.total_syncs,
                "total_records": conn.total_records_synced,
            },
            message=f"Found connection '{name}'.",
        )
    
    def _save_connection(
        self,
        name: str,
        connector_type: str,
        connection_string: str | None = None,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        description: str = "",
    ) -> ToolResult:
        # Build connection config
        config: dict[str, Any] = {}
        
        if connection_string:
            config["connection_string"] = connection_string
        else:
            if host:
                config["host"] = host
            if port:
                config["port"] = port
            if database:
                config["database"] = database
            if username:
                config["username"] = username
            if password:
                config["password"] = password
        
        if not config:
            return ToolResult(
                success=False,
                data=None,
                message="No connection parameters provided. Need connection_string or host/port/database.",
            )
        
        conn = SavedConnection(
            name=name,
            connector_type=connector_type,
            connection_config=config,
            description=description,
        )
        
        self._store.save_connection(conn)
        
        return ToolResult(
            success=True,
            data={"name": name, "type": connector_type},
            message=f"Saved connection '{name}'. You can now use connect_database(saved_name='{name}') to connect.",
        )
    
    def _delete_saved_connection(self, name: str) -> ToolResult:
        if self._store.delete_connection(name):
            return ToolResult(
                success=True,
                data=None,
                message=f"Deleted connection '{name}'.",
            )
        return ToolResult(
            success=False,
            data=None,
            message=f"Connection '{name}' not found.",
        )
    
    def _connect_database(
        self,
        saved_name: str | None = None,
        connector_type: str | None = None,
        connection_string: str | None = None,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
    ) -> ToolResult:
        # Load from saved connection if name provided
        if saved_name:
            conn = self._store.get_connection(saved_name)
            if conn is None:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"No saved connection found with name '{saved_name}'.",
                )
            
            connector_type = conn.connector_type
            config = conn.connection_config
            self._current_connection = conn
            self._store.update_connection_usage(saved_name)
        else:
            if not connector_type:
                return ToolResult(
                    success=False,
                    data=None,
                    message="Must provide either saved_name or connector_type.",
                )
            
            config = {}
            if connection_string:
                config["connection_string"] = connection_string
            if host:
                config["host"] = host
            if port:
                config["port"] = port
            if database:
                config["database"] = database
        
        try:
            result = self._orch.connect_database(connector_type, config)
            
            tables = result.get("tables", [])
            info = result.get("connection_info", {})
            
            # Update current connection with discovered info
            if self._current_connection:
                self._current_connection.update_last_used()
            
            return ToolResult(
                success=True,
                data={
                    "connected": True,
                    "type": connector_type,
                    "database": info.get("database", ""),
                    "tables": tables,
                    "using_saved": saved_name or None,
                },
                message=f"Connected to {connector_type} database. Found {len(tables)} tables.",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Connection failed: {str(e)}",
            )
    
    def _disconnect_database(self) -> ToolResult:
        if self._orch.disconnect_database():
            self._current_connection = None
            return ToolResult(
                success=True,
                data=None,
                message="Disconnected from database.",
            )
        return ToolResult(
            success=False,
            data=None,
            message="No database connected.",
        )
    
    def _list_tables(self) -> ToolResult:
        try:
            schema = self._orch.get_database_schema()
            tables = schema.get("tables", [])
            
            return ToolResult(
                success=True,
                data=tables,
                message=f"Found {len(tables)} tables: {', '.join(tables[:10])}{'...' if len(tables) > 10 else ''}",
            )
        except RuntimeError as e:
            return ToolResult(
                success=False,
                data=None,
                message=str(e),
            )
    
    def _get_table_schema(self, table_name: str) -> ToolResult:
        try:
            schema = self._orch.get_database_schema(table=table_name)
            
            return ToolResult(
                success=True,
                data=schema,
                message=f"Table '{table_name}' has {len(schema['columns'])} columns.",
            )
        except RuntimeError as e:
            return ToolResult(
                success=False,
                data=None,
                message=str(e),
            )
    
    def _sync_database(
        self,
        tables: list[str] | None = None,
        text_columns: dict[str, list[str]] | None = None,
        incremental: bool = True,
        save_preferences: bool = True,
    ) -> ToolResult:
        try:
            result = self._orch.sync_from_database(
                tables=tables,
                text_columns_map=text_columns,
                incremental=incremental,
            )
            
            records_added = result.get("records_added", 0)
            
            # Update connection stats if using saved connection
            if self._current_connection and save_preferences:
                self._store.update_sync_stats(
                    self._current_connection.name,
                    records_added,
                )
                
                # Save table preferences
                if tables:
                    self._current_connection.tables = tables
                if text_columns:
                    self._current_connection.text_columns_map = text_columns
                self._store.save_connection(self._current_connection)
                
                # Record in sync history
                self._store.record_sync(
                    connection_name=self._current_connection.name,
                    sync_type="incremental" if incremental else "full",
                    tables_synced=result.get("tables_synced", []),
                    records_added=records_added,
                    records_failed=result.get("records_failed", 0),
                    errors=result.get("errors"),
                )
            
            return ToolResult(
                success=True,
                data=result,
                message=f"Sync complete. Added {records_added} records.",
            )
        except RuntimeError as e:
            return ToolResult(
                success=False,
                data=None,
                message=str(e),
            )
    
    def _update_connection_preferences(
        self,
        name: str,
        tables: list[str] | None = None,
        text_columns: dict[str, list[str]] | None = None,
        auto_sync_enabled: bool | None = None,
        auto_sync_interval: int | None = None,
    ) -> ToolResult:
        conn = self._store.get_connection(name)
        if conn is None:
            return ToolResult(
                success=False,
                data=None,
                message=f"Connection '{name}' not found.",
            )
        
        if tables is not None:
            conn.tables = tables
        if text_columns is not None:
            conn.text_columns_map = text_columns
        if auto_sync_enabled is not None:
            conn.auto_sync_enabled = auto_sync_enabled
        if auto_sync_interval is not None:
            conn.auto_sync_interval = auto_sync_interval
        
        self._store.save_connection(conn)
        
        return ToolResult(
            success=True,
            data=None,
            message=f"Updated preferences for connection '{name}'.",
        )
    
    def _get_sync_status(self) -> ToolResult:
        status = self._orch.get_sync_status()
        
        if not status.get("connected"):
            return ToolResult(
                success=True,
                data={"connected": False},
                message="No database connected.",
            )
        
        return ToolResult(
            success=True,
            data=status,
            message=f"Connected. {len(status.get('tables', {}))} tables have been synced.",
        )
    
    def _get_sync_history(
        self,
        connection_name: str | None = None,
        limit: int = 10,
    ) -> ToolResult:
        name = connection_name
        if name is None and self._current_connection:
            name = self._current_connection.name
        
        if name is None:
            return ToolResult(
                success=False,
                data=None,
                message="No connection specified and no current connection.",
            )
        
        history = self._store.get_sync_history(name, limit)
        
        return ToolResult(
            success=True,
            data=history,
            message=f"Found {len(history)} sync history entries for '{name}'.",
        )
    
    def _get_database_stats(self) -> ToolResult:
        stats = self._orch.stats()
        
        return ToolResult(
            success=True,
            data=stats,
            message=f"Database has {stats['entities']} entities, {stats['relations']} relations, {stats.get('memory_notes', 0)} notes.",
        )
    
    def _add_note(
        self,
        content: str,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> ToolResult:
        input_data: dict[str, Any] = {"content": content}
        if category:
            input_data["category"] = category
        if tags:
            input_data["tags"] = tags
        
        try:
            result = self._orch.add_memory_note(input_data)
            return ToolResult(
                success=True,
                data=result,
                message=f"Created note {result['id']} with {len(result.get('keywords', []))} keywords.",
            )
        except RuntimeError as e:
            return ToolResult(
                success=False,
                data=None,
                message=str(e),
            )
    
    def _search_notes(self, query: str, k: int = 10) -> ToolResult:
        results = self._orch.search_notes_by_content(query, k=k)
        
        return ToolResult(
            success=True,
            data=results,
            message=f"Found {len(results)} notes matching '{query}'.",
        )
