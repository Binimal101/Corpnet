"""Port: external database connector for importing user data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from archrag.domain.models import (
    ExternalRecord,
    RelationshipInfo,
    TableSchema,
)


class ExternalDatabaseConnectorPort(ABC):
    """Abstract interface for connecting to external user databases.

    This port enables ArchRAG to connect to various database types (SQL, NoSQL)
    and extract records for indexing into the knowledge graph and memory system.

    Implementations should handle:
    - Connection management
    - Schema discovery (tables, columns, relationships)
    - Data extraction with pagination and incremental sync support
    """

    # ── Connection ──────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self, connection_config: dict[str, Any]) -> None:
        """Establish connection to the external database.

        Args:
            connection_config: Database-specific connection parameters.
                For SQL: {"connection_string": "postgresql://..."}
                For NoSQL: {"host": "...", "port": 27017, "database": "..."}
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close the database connection and release resources."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the connector has an active connection."""
        ...

    # ── Schema Discovery ────────────────────────────────────────────────────

    @abstractmethod
    def list_databases(self) -> list[str]:
        """List all accessible databases.

        Returns:
            List of database names. For single-database systems,
            returns a list with one element.
        """
        ...

    @abstractmethod
    def list_tables(self, database: str | None = None) -> list[str]:
        """List all tables/collections in a database.

        Args:
            database: Target database name. If None, uses the connected database.

        Returns:
            List of table/collection names.
        """
        ...

    @abstractmethod
    def get_table_schema(self, table: str, database: str | None = None) -> TableSchema:
        """Get schema information for a specific table.

        Args:
            table: Table/collection name.
            database: Target database name. If None, uses the connected database.

        Returns:
            TableSchema with column info, primary key, and relationships.
        """
        ...

    @abstractmethod
    def discover_relationships(self, database: str | None = None) -> list[RelationshipInfo]:
        """Discover all foreign key relationships in the database.

        Args:
            database: Target database name. If None, uses the connected database.

        Returns:
            List of RelationshipInfo describing table relationships.
        """
        ...

    # ── Data Extraction ─────────────────────────────────────────────────────

    @abstractmethod
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
        """Fetch records from a table with optional filtering.

        Args:
            table: Table/collection name.
            text_columns: Column names to concatenate for text_content.
            database: Target database. If None, uses the connected database.
            limit: Maximum records to return.
            offset: Number of records to skip (for pagination).
            since_timestamp: Only fetch records updated after this timestamp.
            since_id: Only fetch records with ID greater than this (for incremental).
            timestamp_column: Column name containing update timestamp.
            id_column: Column name for primary key (for ordering).
            order_by: Column to order results by.

        Returns:
            List of ExternalRecord with extracted content.
        """
        ...

    @abstractmethod
    def count_records(
        self,
        table: str,
        *,
        database: str | None = None,
        since_timestamp: str | None = None,
        timestamp_column: str | None = None,
    ) -> int:
        """Count records in a table, optionally with filters.

        Args:
            table: Table/collection name.
            database: Target database. If None, uses the connected database.
            since_timestamp: Only count records updated after this timestamp.
            timestamp_column: Column name containing update timestamp.

        Returns:
            Number of matching records.
        """
        ...

    # ── Metadata ────────────────────────────────────────────────────────────

    @abstractmethod
    def get_connector_type(self) -> str:
        """Return the connector type identifier.

        Returns:
            "sql" for SQL databases, "nosql" for document databases.
        """
        ...

    @abstractmethod
    def get_connector_id(self) -> str:
        """Return a unique identifier for this connector instance.

        Returns:
            A string combining connector type and connection info.
        """
        ...

    @abstractmethod
    def get_connection_info(self) -> dict[str, Any]:
        """Return sanitized connection information (no passwords).

        Returns:
            Dictionary with connection details for display/logging.
        """
        ...
