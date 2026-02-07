"""Generic NoSQL connector for MongoDB-style document databases."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from archrag.domain.models import (
    ColumnInfo,
    ExternalRecord,
    RelationshipInfo,
    TableSchema,
)
from archrag.ports.external_database import ExternalDatabaseConnectorPort

log = logging.getLogger(__name__)


def _infer_type(value: Any) -> str:
    """Infer data type from a Python value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _is_text_type(value: Any) -> bool:
    """Check if a value is text-like."""
    return isinstance(value, str) and len(value) > 0


def _flatten_document(doc: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested document for text extraction."""
    result = {}
    for key, value in doc.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_document(value, full_key))
        elif isinstance(value, list):
            # Flatten lists of primitives
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    result.update(_flatten_document(item, f"{full_key}[{i}]"))
                else:
                    result[f"{full_key}[{i}]"] = item
        else:
            result[full_key] = value
    return result


class GenericNoSQLConnector(ExternalDatabaseConnectorPort):
    """MongoDB-style connector for document databases.

    Supports MongoDB via pymongo. Can be extended for other document databases.

    Usage:
        connector = GenericNoSQLConnector()
        connector.connect({
            "host": "localhost",
            "port": 27017,
            "database": "mydb",
            "username": "user",  # optional
            "password": "pass",  # optional
        })
        collections = connector.list_tables()
        records = connector.fetch_records("users", ["name", "bio"])
        connector.disconnect()
    """

    def __init__(self) -> None:
        self._client: Any | None = None
        self._db: Any | None = None
        self._database_name: str = ""
        self._host: str = ""
        self._port: int = 27017
        self._connector_id: str = ""

    # ── Connection ──────────────────────────────────────────────────────────

    def connect(self, connection_config: dict[str, Any]) -> None:
        """Connect to a MongoDB database.

        Args:
            connection_config: Dictionary with connection parameters:
                - host: MongoDB host (default: localhost)
                - port: MongoDB port (default: 27017)
                - database: Database name (required)
                - username: Optional username
                - password: Optional password
                - connection_string: Alternative full MongoDB URI
        """
        try:
            from pymongo import MongoClient
        except ImportError as e:
            raise ImportError(
                "pymongo is required for NoSQL connector. "
                "Install with: pip install pymongo"
            ) from e

        # Check for connection string first
        if "connection_string" in connection_config:
            connection_string = connection_config["connection_string"]
            self._client = MongoClient(connection_string)
            # Extract database from config or connection string
            self._database_name = connection_config.get("database", "")
            if not self._database_name:
                # Try to extract from connection string
                if "/" in connection_string:
                    self._database_name = connection_string.rsplit("/", 1)[-1].split("?")[0]
        else:
            # Build connection from individual parameters
            self._host = connection_config.get("host", "localhost")
            self._port = connection_config.get("port", 27017)
            self._database_name = connection_config.get("database", "")

            if not self._database_name:
                raise ValueError("connection_config must contain 'database'")

            username = connection_config.get("username")
            password = connection_config.get("password")

            if username and password:
                self._client = MongoClient(
                    host=self._host,
                    port=self._port,
                    username=username,
                    password=password,
                )
            else:
                self._client = MongoClient(host=self._host, port=self._port)

        self._db = self._client[self._database_name]

        # Generate connector ID
        id_string = f"{self._host}:{self._port}/{self._database_name}"
        self._connector_id = f"nosql-{hashlib.md5(id_string.encode()).hexdigest()[:8]}"

        log.info("Connected to MongoDB database: %s", self._database_name)

    def disconnect(self) -> None:
        """Close the database connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            log.info("Disconnected from MongoDB database")

    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._client is not None and self._db is not None

    # ── Schema Discovery ────────────────────────────────────────────────────

    def list_databases(self) -> list[str]:
        """List all accessible databases."""
        if not self.is_connected():
            raise RuntimeError("Not connected to database")
        return self._client.list_database_names()

    def list_tables(self, database: str | None = None) -> list[str]:
        """List all collections in the database."""
        if not self.is_connected():
            raise RuntimeError("Not connected to database")

        db = self._client[database] if database else self._db
        return db.list_collection_names()

    def get_table_schema(self, table: str, database: str | None = None) -> TableSchema:
        """Infer schema from sample documents.

        Since MongoDB is schema-less, this samples documents to infer structure.
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to database")

        db = self._client[database] if database else self._db
        collection = db[table]

        # Sample documents to infer schema
        sample_size = min(100, collection.count_documents({}))
        samples = list(collection.find().limit(sample_size))

        if not samples:
            return TableSchema(
                name=table,
                database=database or self._database_name,
                columns=[],
                primary_key="_id",
            )

        # Aggregate field information across samples
        field_info: dict[str, dict[str, Any]] = {}
        for doc in samples:
            flat_doc = _flatten_document(doc)
            for key, value in flat_doc.items():
                if key not in field_info:
                    field_info[key] = {
                        "types": set(),
                        "is_text": False,
                        "null_count": 0,
                    }
                field_info[key]["types"].add(_infer_type(value))
                if _is_text_type(value):
                    field_info[key]["is_text"] = True
                if value is None:
                    field_info[key]["null_count"] += 1

        # Build column info
        columns = []
        for field_name, info in field_info.items():
            # Determine primary type (most common non-null)
            types = info["types"] - {"null"}
            primary_type = next(iter(types)) if types else "null"

            columns.append(
                ColumnInfo(
                    name=field_name,
                    data_type=primary_type,
                    nullable=info["null_count"] > 0 or "null" in info["types"],
                    is_text=info["is_text"],
                )
            )

        return TableSchema(
            name=table,
            database=database or self._database_name,
            columns=columns,
            primary_key="_id",
            relationships=[],  # MongoDB doesn't have formal foreign keys
        )

    def discover_relationships(self, database: str | None = None) -> list[RelationshipInfo]:
        """Discover relationships in MongoDB.

        MongoDB doesn't have formal foreign keys, but we can look for
        common patterns like fields ending in '_id' or containing 'ref'.
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to database")

        relationships = []
        collections = self.list_tables(database)

        for collection_name in collections:
            schema = self.get_table_schema(collection_name, database)

            for col in schema.columns:
                # Look for potential reference fields
                if col.name.endswith("_id") and col.name != "_id":
                    # Infer target collection from field name
                    potential_target = col.name.rsplit("_id", 1)[0] + "s"
                    if potential_target in collections:
                        relationships.append(
                            RelationshipInfo(
                                from_column=f"{collection_name}.{col.name}",
                                to_table=potential_target,
                                to_column="_id",
                                relationship_type="reference",
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
        """Fetch documents from a MongoDB collection."""
        if not self.is_connected():
            raise RuntimeError("Not connected to database")

        from bson import ObjectId
        from datetime import datetime

        db = self._client[database] if database else self._db
        collection = db[table]

        # Build query filter
        query: dict[str, Any] = {}

        if since_timestamp and timestamp_column:
            try:
                ts = datetime.fromisoformat(since_timestamp.replace("Z", "+00:00"))
                query[timestamp_column] = {"$gt": ts}
            except ValueError:
                log.warning("Invalid timestamp format: %s", since_timestamp)

        if since_id:
            id_col = id_column or "_id"
            try:
                if id_col == "_id":
                    query["_id"] = {"$gt": ObjectId(since_id)}
                else:
                    query[id_col] = {"$gt": since_id}
            except Exception:
                log.warning("Invalid ID format: %s", since_id)

        # Build cursor
        cursor = collection.find(query)

        # Apply sorting
        if order_by:
            cursor = cursor.sort(order_by, 1)
        else:
            cursor = cursor.sort("_id", 1)

        # Apply pagination
        if offset > 0:
            cursor = cursor.skip(offset)
        if limit is not None:
            cursor = cursor.limit(limit)

        # Convert to ExternalRecords
        records = []
        for doc in cursor:
            # Flatten document for easier text extraction
            flat_doc = _flatten_document(doc)

            # Convert ObjectId to string
            content = {}
            for key, value in doc.items():
                if hasattr(value, "__str__"):
                    content[key] = str(value) if isinstance(value, ObjectId) else value
                else:
                    content[key] = value

            # Extract text content from specified columns
            text_parts = []
            for col in text_columns:
                # Support both flat and nested column names
                if col in flat_doc and flat_doc[col]:
                    text_parts.append(str(flat_doc[col]))
                elif col in doc and doc[col]:
                    if isinstance(doc[col], str):
                        text_parts.append(doc[col])
                    elif isinstance(doc[col], list):
                        text_parts.extend(str(item) for item in doc[col] if item)

            text_content = " ".join(text_parts)

            # Get ID
            record_id = str(doc.get("_id", ""))

            # Get timestamps
            created_at = None
            updated_at = None
            for ts_col in ["created_at", "createdAt", "created", "creation_date"]:
                if ts_col in doc and doc[ts_col]:
                    created_at = str(doc[ts_col])
                    break
            for ts_col in ["updated_at", "updatedAt", "updated", "modified_at", "modifiedAt"]:
                if ts_col in doc and doc[ts_col]:
                    updated_at = str(doc[ts_col])
                    break

            records.append(
                ExternalRecord(
                    id=record_id,
                    source_table=table,
                    source_database=database or self._database_name,
                    content=content,
                    text_content=text_content,
                    metadata={
                        "fields": list(doc.keys()),
                        "primary_key": "_id",
                    },
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )

        log.info(
            "Fetched %d records from %s.%s",
            len(records),
            database or self._database_name,
            table,
        )
        return records

    def count_records(
        self,
        table: str,
        *,
        database: str | None = None,
        since_timestamp: str | None = None,
        timestamp_column: str | None = None,
    ) -> int:
        """Count documents in a collection."""
        if not self.is_connected():
            raise RuntimeError("Not connected to database")

        from datetime import datetime

        db = self._client[database] if database else self._db
        collection = db[table]

        query: dict[str, Any] = {}
        if since_timestamp and timestamp_column:
            try:
                ts = datetime.fromisoformat(since_timestamp.replace("Z", "+00:00"))
                query[timestamp_column] = {"$gt": ts}
            except ValueError:
                pass

        return collection.count_documents(query)

    # ── Metadata ────────────────────────────────────────────────────────────

    def get_connector_type(self) -> str:
        """Return 'nosql' for document databases."""
        return "nosql"

    def get_connector_id(self) -> str:
        """Return unique connector identifier."""
        return self._connector_id

    def get_connection_info(self) -> dict[str, Any]:
        """Return sanitized connection information."""
        return {
            "type": "nosql",
            "dialect": "mongodb",
            "host": self._host or "localhost",
            "port": self._port,
            "database": self._database_name,
            "connected": self.is_connected(),
        }
