"""Service for synchronizing external databases with ArchRAG."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, TYPE_CHECKING

from archrag.domain.models import (
    ExternalRecord,
    SyncResult,
    SyncState,
)
from archrag.ports.document_store import DocumentStorePort
from archrag.ports.external_database import ExternalDatabaseConnectorPort
from archrag.ports.memory_note_store import MemoryNoteStorePort

if TYPE_CHECKING:
    from archrag.services.note_construction import NoteConstructionService

log = logging.getLogger(__name__)


class DatabaseSyncService:
    """Orchestrates syncing data from external databases into ArchRAG.

    This service:
    - Manages incremental sync state per table
    - Converts external records to MemoryNotes
    - Tracks sync progress and errors
    - Supports both full and incremental sync modes

    Usage:
        sync_service = DatabaseSyncService(
            connector=sql_connector,
            note_service=note_construction_service,
            doc_store=document_store,
        )
        result = sync_service.incremental_sync(tables=["users", "posts"])
    """

    SYNC_STATE_KEY_PREFIX = "sync_state:"

    def __init__(
        self,
        connector: ExternalDatabaseConnectorPort,
        note_service: "NoteConstructionService",
        doc_store: DocumentStorePort,
        note_store: MemoryNoteStorePort,
        *,
        batch_size: int = 100,
        default_timestamp_column: str = "updated_at",
        default_id_column: str = "id",
    ) -> None:
        """Initialize the sync service.

        Args:
            connector: External database connector.
            note_service: Service for creating MemoryNotes from records.
            doc_store: Document store for persisting sync state.
            note_store: Memory note store for saving notes.
            batch_size: Number of records to process per batch.
            default_timestamp_column: Default column for incremental timestamp sync.
            default_id_column: Default column for incremental ID sync.
        """
        self._connector = connector
        self._note_service = note_service
        self._doc_store = doc_store
        self._note_store = note_store
        self._batch_size = batch_size
        self._default_timestamp_column = default_timestamp_column
        self._default_id_column = default_id_column

    # ── Public API ──────────────────────────────────────────────────────────

    def full_sync(
        self,
        tables: list[str] | None = None,
        text_columns_map: dict[str, list[str]] | None = None,
        *,
        enable_linking: bool = True,
        enable_evolution: bool = True,
    ) -> SyncResult:
        """Perform a full sync of specified tables.

        This clears existing sync state and imports all records.

        Args:
            tables: Tables to sync. If None, syncs all tables.
            text_columns_map: Map of table -> list of text columns.
                If not provided, auto-detects text columns.
            enable_linking: Whether to create links between notes.
            enable_evolution: Whether to evolve existing notes.

        Returns:
            SyncResult with statistics and any errors.
        """
        if not self._connector.is_connected():
            return SyncResult(errors=["Not connected to database"])

        start_time = time.time()
        result = SyncResult()

        # Get tables to sync
        if tables is None:
            tables = self._connector.list_tables()

        result.tables_synced = tables

        for table in tables:
            log.info("Starting full sync for table: %s", table)

            # Clear existing sync state
            self._clear_sync_state(table)

            # Get text columns
            text_columns = self._get_text_columns(table, text_columns_map)
            if not text_columns:
                log.warning("No text columns found for table %s, skipping", table)
                result.errors.append(f"No text columns for {table}")
                continue

            # Sync table
            table_result = self._sync_table(
                table,
                text_columns,
                full_sync=True,
                enable_linking=enable_linking,
                enable_evolution=enable_evolution,
            )

            result.records_added += table_result.records_added
            result.records_updated += table_result.records_updated
            result.records_failed += table_result.records_failed
            result.errors.extend(table_result.errors)

        result.duration_seconds = time.time() - start_time
        log.info(
            "Full sync completed: %d added, %d failed in %.2fs",
            result.records_added,
            result.records_failed,
            result.duration_seconds,
        )
        return result

    def incremental_sync(
        self,
        tables: list[str] | None = None,
        text_columns_map: dict[str, list[str]] | None = None,
        *,
        enable_linking: bool = True,
        enable_evolution: bool = True,
    ) -> SyncResult:
        """Perform incremental sync, only importing new/changed records.

        Uses sync state to track last synced record per table.

        Args:
            tables: Tables to sync. If None, syncs all tables.
            text_columns_map: Map of table -> list of text columns.
            enable_linking: Whether to create links between notes.
            enable_evolution: Whether to evolve existing notes.

        Returns:
            SyncResult with statistics and any errors.
        """
        if not self._connector.is_connected():
            return SyncResult(errors=["Not connected to database"])

        start_time = time.time()
        result = SyncResult()

        # Get tables to sync
        if tables is None:
            tables = self._connector.list_tables()

        result.tables_synced = tables

        for table in tables:
            log.info("Starting incremental sync for table: %s", table)

            # Get text columns
            text_columns = self._get_text_columns(table, text_columns_map)
            if not text_columns:
                log.warning("No text columns found for table %s, skipping", table)
                result.errors.append(f"No text columns for {table}")
                continue

            # Sync table incrementally
            table_result = self._sync_table(
                table,
                text_columns,
                full_sync=False,
                enable_linking=enable_linking,
                enable_evolution=enable_evolution,
            )

            result.records_added += table_result.records_added
            result.records_updated += table_result.records_updated
            result.records_failed += table_result.records_failed
            result.errors.extend(table_result.errors)

        result.duration_seconds = time.time() - start_time
        log.info(
            "Incremental sync completed: %d added, %d updated, %d failed in %.2fs",
            result.records_added,
            result.records_updated,
            result.records_failed,
            result.duration_seconds,
        )
        return result

    def get_sync_state(self, table: str) -> SyncState | None:
        """Get the current sync state for a table."""
        key = f"{self.SYNC_STATE_KEY_PREFIX}{table}"
        data = self._doc_store.get_meta(key)
        if not data:
            return None

        if isinstance(data, str):
            data = json.loads(data)

        return SyncState(
            connector_id=data.get("connector_id", ""),
            database_name=data.get("database_name", ""),
            table_name=data.get("table_name", table),
            last_sync_at=data.get("last_sync_at", ""),
            last_record_id=data.get("last_record_id"),
            last_updated_at=data.get("last_updated_at"),
            record_count=data.get("record_count", 0),
        )

    def get_all_sync_states(self) -> dict[str, SyncState]:
        """Get sync states for all tables."""
        states = {}
        for table in self._connector.list_tables():
            state = self.get_sync_state(table)
            if state:
                states[table] = state
        return states

    # ── Internal Methods ────────────────────────────────────────────────────

    def _sync_table(
        self,
        table: str,
        text_columns: list[str],
        *,
        full_sync: bool,
        enable_linking: bool,
        enable_evolution: bool,
    ) -> SyncResult:
        """Sync a single table."""
        result = SyncResult(tables_synced=[table])

        # Get sync state for incremental
        sync_state = None if full_sync else self.get_sync_state(table)

        # Get schema info
        schema = self._connector.get_table_schema(table)
        id_column = schema.primary_key or self._default_id_column
        timestamp_column = self._find_timestamp_column(schema)

        # Prepare fetch parameters
        fetch_kwargs: dict[str, Any] = {
            "table": table,
            "text_columns": text_columns,
            "id_column": id_column,
            "order_by": id_column,
        }

        if not full_sync and sync_state:
            if sync_state.last_updated_at and timestamp_column:
                fetch_kwargs["since_timestamp"] = sync_state.last_updated_at
                fetch_kwargs["timestamp_column"] = timestamp_column
            elif sync_state.last_record_id:
                fetch_kwargs["since_id"] = sync_state.last_record_id

        # Fetch and process records in batches
        offset = 0
        last_record_id = sync_state.last_record_id if sync_state else None
        last_updated_at = sync_state.last_updated_at if sync_state else None
        total_count = 0

        while True:
            fetch_kwargs["offset"] = offset
            fetch_kwargs["limit"] = self._batch_size

            records = self._connector.fetch_records(**fetch_kwargs)
            if not records:
                break

            # Process batch
            for record in records:
                try:
                    self._process_record(
                        record,
                        enable_linking=enable_linking,
                        enable_evolution=enable_evolution,
                    )
                    result.records_added += 1
                    total_count += 1

                    # Track last record for sync state
                    last_record_id = record.id
                    if record.updated_at:
                        last_updated_at = record.updated_at

                except Exception as e:
                    log.error("Failed to process record %s: %s", record.id, e)
                    result.records_failed += 1
                    result.errors.append(f"Record {record.id}: {str(e)}")

            offset += len(records)

            # If we got fewer records than batch size, we're done
            if len(records) < self._batch_size:
                break

        # Update sync state
        self._save_sync_state(
            table,
            last_record_id=last_record_id,
            last_updated_at=last_updated_at,
            record_count=total_count + (sync_state.record_count if sync_state else 0),
        )

        return result

    def _process_record(
        self,
        record: ExternalRecord,
        *,
        enable_linking: bool,
        enable_evolution: bool,
    ) -> None:
        """Convert an external record to a MemoryNote and save it."""
        if not record.text_content.strip():
            log.debug("Skipping record %s: empty text content", record.id)
            return

        # Convert record to note input
        note_input = record.to_note_input()

        # Build note using the note construction service
        note = self._note_service.build_note(
            note_input,
            enable_linking=enable_linking,
            enable_evolution=enable_evolution,
        )

        # Save the note to the store
        self._note_store.save_note(note)
        log.debug("Saved memory note %s from record %s", note.id, record.id)

    def _get_text_columns(
        self,
        table: str,
        text_columns_map: dict[str, list[str]] | None,
    ) -> list[str]:
        """Get text columns for a table."""
        # Check explicit mapping first
        if text_columns_map and table in text_columns_map:
            return text_columns_map[table]

        # Auto-detect from schema
        schema = self._connector.get_table_schema(table)
        return [col.name for col in schema.columns if col.is_text]

    def _find_timestamp_column(self, schema: Any) -> str | None:
        """Find a timestamp column in the schema."""
        timestamp_patterns = [
            "updated_at",
            "updatedat",
            "modified_at",
            "modifiedat",
            "last_modified",
            "lastmodified",
        ]

        for col in schema.columns:
            col_lower = col.name.lower()
            if col_lower in timestamp_patterns:
                return col.name

        return None

    def _save_sync_state(
        self,
        table: str,
        last_record_id: str | None,
        last_updated_at: str | None,
        record_count: int,
    ) -> None:
        """Save sync state for a table."""
        key = f"{self.SYNC_STATE_KEY_PREFIX}{table}"
        state = {
            "connector_id": self._connector.get_connector_id(),
            "database_name": self._connector.get_connection_info().get("database", ""),
            "table_name": table,
            "last_sync_at": datetime.now().isoformat(),
            "last_record_id": last_record_id,
            "last_updated_at": last_updated_at,
            "record_count": record_count,
        }
        self._doc_store.put_meta(key, json.dumps(state))

    def _clear_sync_state(self, table: str) -> None:
        """Clear sync state for a table."""
        key = f"{self.SYNC_STATE_KEY_PREFIX}{table}"
        self._doc_store.put_meta(key, None)
