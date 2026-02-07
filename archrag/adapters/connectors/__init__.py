"""External database connectors for SQL and NoSQL databases."""

from archrag.adapters.connectors.sql_connector import GenericSQLConnector
from archrag.adapters.connectors.nosql_connector import GenericNoSQLConnector

__all__ = ["GenericSQLConnector", "GenericNoSQLConnector"]
