"""Generic SQL connector using SQLAlchemy for database-agnostic support."""

from __future__ import annotations

import hashlib
import logging
from typing import Any
from urllib.parse import urlparse

from archrag.domain.models import (
    ColumnInfo,
    ExternalRecord,
    RelationshipInfo,
    TableSchema,
)
from archrag.ports.external_database import ExternalDatabaseConnectorPort

log = logging.getLogger(__name__)

# SQL type mappings to identify text-like columns
TEXT_TYPES = {
    "text",
    "varchar",
    "char",
    "character varying",
    "nvarchar",
    "nchar",
    "clob",
    "longtext",
    "mediumtext",
    "tinytext",
    "string",
}


class GenericSQLConnector(ExternalDatabaseConnectorPort):
    """SQLAlchemy-based connector for SQL databases.

    Supports PostgreSQL, MySQL, SQLite, SQL Server, and other SQLAlchemy-compatible databases.

    Usage:
        connector = GenericSQLConnector()
        connector.connect({"connection_string": "postgresql://user:pass@host/db"})
        tables = connector.list_tables()
        records = connector.fetch_records("users", ["name", "bio"])
        connector.disconnect()
    """

    def __init__(self) -> None:
        self._engine: Any | None = None
        self._inspector: Any | None = None
        self._connection_string: str = ""
        self._database_name: str = ""
        self._connector_id: str = ""

    # ── Connection ──────────────────────────────────────────────────────────

    def connect(self, connection_config: dict[str, Any]) -> None:
        """Connect to a SQL database using SQLAlchemy.

        Args:
            connection_config: Must contain "connection_string" key.
                Example: {"connection_string": "postgresql://user:pass@localhost/mydb"}
        """
        try:
            from sqlalchemy import create_engine, inspect
        except ImportError as e:
            raise ImportError(
                "SQLAlchemy is required for SQL connector. "
                "Install with: pip install sqlalchemy"
            ) from e

        connection_string = connection_config.get("connection_string", "")
        if not connection_string:
            raise ValueError("connection_config must contain 'connection_string'")

        self._connection_string = connection_string
        self._engine = create_engine(connection_string)
        self._inspector = inspect(self._engine)

        # Extract database name from connection string
        parsed = urlparse(connection_string)
        self._database_name = parsed.path.lstrip("/") or "default"

        # Generate connector ID
        self._connector_id = f"sql-{hashlib.md5(connection_string.encode()).hexdigest()[:8]}"

        log.info("Connected to SQL database: %s", self._database_name)

    def disconnect(self) -> None:
        """Close the database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._inspector = None
            log.info("Disconnected from SQL database")

    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._engine is not None

    # ── Schema Discovery ────────────────────────────────────────────────────

    def list_databases(self) -> list[str]:
        """List accessible databases.

        For most SQL connections, this returns the currently connected database.
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to database")
        return [self._database_name]

    def list_tables(self, database: str | None = None) -> list[str]:
        """List all tables in the database."""
        if not self.is_connected():
            raise RuntimeError("Not connected to database")
        return self._inspector.get_table_names()

    def get_table_schema(self, table: str, database: str | None = None) -> TableSchema:
        """Get schema information for a table."""
        if not self.is_connected():
            raise RuntimeError("Not connected to database")

        columns = []
        for col in self._inspector.get_columns(table):
            col_type = str(col["type"]).lower()
            is_text = any(t in col_type for t in TEXT_TYPES)
            columns.append(
                ColumnInfo(
                    name=col["name"],
                    data_type=str(col["type"]),
                    nullable=col.get("nullable", True),
                    is_text=is_text,
                )
            )

        # Get primary key
        pk_constraint = self._inspector.get_pk_constraint(table)
        primary_key = pk_constraint["constrained_columns"][0] if pk_constraint.get("constrained_columns") else None

        # Get foreign keys as relationships
        relationships = []
        for fk in self._inspector.get_foreign_keys(table):
            if fk.get("constrained_columns") and fk.get("referred_columns"):
                relationships.append(
                    RelationshipInfo(
                        from_column=fk["constrained_columns"][0],
                        to_table=fk["referred_table"],
                        to_column=fk["referred_columns"][0],
                        relationship_type="foreign_key",
                    )
                )

        return TableSchema(
            name=table,
            database=self._database_name,
            columns=columns,
            primary_key=primary_key,
            relationships=relationships,
        )

    def discover_relationships(self, database: str | None = None) -> list[RelationshipInfo]:
        """Discover all foreign key relationships in the database."""
        if not self.is_connected():
            raise RuntimeError("Not connected to database")

        relationships = []
        for table in self.list_tables(database):
            for fk in self._inspector.get_foreign_keys(table):
                if fk.get("constrained_columns") and fk.get("referred_columns"):
                    relationships.append(
                        RelationshipInfo(
                            from_column=f"{table}.{fk['constrained_columns'][0]}",
                            to_table=fk["referred_table"],
                            to_column=fk["referred_columns"][0],
                            relationship_type="foreign_key",
                        )
                    )
        return relationships

    # ── Data Extraction ─────────────────────────────────────────────────────

    def fetch_records(
        self,
        table: str,
        text_columns: list[str],
        *,
        database: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        since_timestamp: str | None = None,
        since_id: str | None = None,
        timestamp_column: str | None = None,
        id_column: str | None = None,
        order_by: str | None = None,
    ) -> list[ExternalRecord]:
        """Fetch records from a SQL table."""
        if not self.is_connected():
            raise RuntimeError("Not connected to database")

        from sqlalchemy import text

        # Build query
        query_parts = [f"SELECT * FROM {table}"]
        params: dict[str, Any] = {}

        # Add WHERE clause for incremental sync
        where_clauses = []
        if since_timestamp and timestamp_column:
            where_clauses.append(f"{timestamp_column} > :since_timestamp")
            params["since_timestamp"] = since_timestamp
        if since_id and id_column:
            where_clauses.append(f"{id_column} > :since_id")
            params["since_id"] = since_id

        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))

        # Add ORDER BY
        if order_by:
            query_parts.append(f"ORDER BY {order_by}")
        elif id_column:
            query_parts.append(f"ORDER BY {id_column}")

        # Add LIMIT and OFFSET
        if limit is not None:
            query_parts.append(f"LIMIT {limit}")
        if offset > 0:
            query_parts.append(f"OFFSET {offset}")

        query = " ".join(query_parts)
        log.debug("Executing SQL: %s", query)

        # Execute query
        with self._engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()
            columns = result.keys()

        # Convert to ExternalRecords
        records = []
        schema = self.get_table_schema(table)
        pk = schema.primary_key or "id"

        for row in rows:
            row_dict = dict(zip(columns, row))

            # Extract text content from specified columns
            text_parts = []
            for col in text_columns:
                if col in row_dict and row_dict[col]:
                    text_parts.append(str(row_dict[col]))
            text_content = " ".join(text_parts)

            # Get ID
            record_id = str(row_dict.get(pk, row_dict.get("id", "")))

            # Get timestamps if available
            created_at = None
            updated_at = None
            for ts_col in ["created_at", "created", "createdat", "creation_date"]:
                if ts_col in row_dict and row_dict[ts_col]:
                    created_at = str(row_dict[ts_col])
                    break
            for ts_col in ["updated_at", "updated", "updatedat", "modified_at", "modification_date"]:
                if ts_col in row_dict and row_dict[ts_col]:
                    updated_at = str(row_dict[ts_col])
                    break

            records.append(
                ExternalRecord(
                    id=record_id,
                    source_table=table,
                    source_database=self._database_name,
                    content=row_dict,
                    text_content=text_content,
                    metadata={
                        "columns": list(columns),
                        "primary_key": pk,
                    },
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )

        log.info("Fetched %d records from %s.%s", len(records), self._database_name, table)
        return records

    def count_records(
        self,
        table: str,
        *,
        database: str | None = None,
        since_timestamp: str | None = None,
        timestamp_column: str | None = None,
    ) -> int:
        """Count records in a table."""
        if not self.is_connected():
            raise RuntimeError("Not connected to database")

        from sqlalchemy import text

        query = f"SELECT COUNT(*) FROM {table}"
        params: dict[str, Any] = {}

        if since_timestamp and timestamp_column:
            query += f" WHERE {timestamp_column} > :since_timestamp"
            params["since_timestamp"] = since_timestamp

        with self._engine.connect() as conn:
            result = conn.execute(text(query), params)
            count = result.scalar()

        return count or 0

    # ── Metadata ────────────────────────────────────────────────────────────

    def get_connector_type(self) -> str:
        """Return 'sql' for SQL databases."""
        return "sql"

    def get_connector_id(self) -> str:
        """Return unique connector identifier."""
        return self._connector_id

    def get_connection_info(self) -> dict[str, Any]:
        """Return sanitized connection information."""
        if not self._connection_string:
            return {}

        parsed = urlparse(self._connection_string)
        return {
            "type": "sql",
            "dialect": parsed.scheme.split("+")[0] if parsed.scheme else "unknown",
            "host": parsed.hostname or "localhost",
            "port": parsed.port,
            "database": self._database_name,
            "connected": self.is_connected(),
        }
