"""Background worker for automatic database synchronization.

This worker polls the connected external database at regular intervals
and syncs new/changed records into ArchRAG automatically.

It is connector-agnostic: works with any ExternalDatabaseConnectorPort
implementation (SQL, NoSQL, or future adapters).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from archrag.services.database_sync import DatabaseSyncService

log = logging.getLogger(__name__)


@dataclass
class AutoSyncConfig:
    """Configuration for automatic syncing."""

    enabled: bool = False
    poll_interval: float = 300.0  # 5 minutes default
    tables: list[str] | None = None  # None = all tables
    text_columns_map: dict[str, list[str]] | None = None
    max_records_per_poll: int = 1000  # Safety limit


@dataclass
class AutoSyncStats:
    """Statistics for auto-sync operations."""

    total_polls: int = 0
    total_records_synced: int = 0
    total_records_failed: int = 0
    last_poll_time: str | None = None
    last_poll_records: int = 0
    last_error: str | None = None
    errors: list[str] = field(default_factory=list)


class AutoSyncWorker:
    """Background worker that polls database and syncs new records automatically.

    This worker is connector-agnostic: it uses the abstract DatabaseSyncService
    which works with any ExternalDatabaseConnectorPort implementation.

    Features:
    - Background polling at configurable intervals
    - Incremental sync (only new/changed records)
    - Enable/disable without restarting
    - Statistics tracking
    - Error handling and recovery

    Usage:
        worker = AutoSyncWorker(sync_service)
        worker.configure(poll_interval=300, tables=["users", "posts"])
        worker.enable()
        # ... later ...
        worker.disable()
        worker.shutdown()
    """

    def __init__(self, sync_service: "DatabaseSyncService") -> None:
        """Initialize the worker.

        Args:
            sync_service: The DatabaseSyncService to use for syncing.
                         This is connector-agnostic.
        """
        self._sync_service = sync_service
        self._config = AutoSyncConfig()
        self._stats = AutoSyncStats()

        self._lock = threading.Lock()
        self._shutdown = False
        self._sync_in_progress = False

        # Background polling thread (daemon so it dies with main process)
        self._worker = threading.Thread(target=self._poll_loop, daemon=True)
        self._worker.start()

    # ── Public API ──────────────────────────────────────────────────────────

    def configure(
        self,
        poll_interval: float | None = None,
        tables: list[str] | None = None,
        text_columns_map: dict[str, list[str]] | None = None,
        max_records_per_poll: int | None = None,
    ) -> None:
        """Update auto-sync configuration.

        Args:
            poll_interval: Seconds between polls.
            tables: Specific tables to sync. None = all tables.
            text_columns_map: Map of table -> text columns.
            max_records_per_poll: Maximum records per poll cycle.
        """
        with self._lock:
            if poll_interval is not None:
                self._config.poll_interval = poll_interval
            if tables is not None:
                self._config.tables = tables
            if text_columns_map is not None:
                self._config.text_columns_map = text_columns_map
            if max_records_per_poll is not None:
                self._config.max_records_per_poll = max_records_per_poll

        log.info("Auto-sync configuration updated: poll_interval=%.0fs", self._config.poll_interval)

    def enable(self) -> None:
        """Start automatic syncing."""
        with self._lock:
            self._config.enabled = True
        log.info("Auto-sync ENABLED (poll interval: %.0fs)", self._config.poll_interval)

    def disable(self) -> None:
        """Stop automatic syncing (worker thread continues but doesn't sync)."""
        with self._lock:
            self._config.enabled = False
        log.info("Auto-sync DISABLED")

    def is_enabled(self) -> bool:
        """Check if auto-sync is enabled."""
        with self._lock:
            return self._config.enabled

    def is_syncing(self) -> bool:
        """Check if a sync is currently in progress."""
        with self._lock:
            return self._sync_in_progress

    def get_config(self) -> dict[str, Any]:
        """Get current configuration as dict."""
        with self._lock:
            return {
                "enabled": self._config.enabled,
                "poll_interval": self._config.poll_interval,
                "tables": self._config.tables,
                "max_records_per_poll": self._config.max_records_per_poll,
            }

    def get_stats(self) -> dict[str, Any]:
        """Get sync statistics as dict."""
        with self._lock:
            return {
                "total_polls": self._stats.total_polls,
                "total_records_synced": self._stats.total_records_synced,
                "total_records_failed": self._stats.total_records_failed,
                "last_poll_time": self._stats.last_poll_time,
                "last_poll_records": self._stats.last_poll_records,
                "last_error": self._stats.last_error,
                "recent_errors": self._stats.errors[-5:],  # Last 5 errors
            }

    def trigger_sync_now(self) -> dict[str, Any]:
        """Manually trigger an immediate sync.

        Returns:
            Sync result dictionary.
        """
        return self._do_sync()

    def shutdown(self) -> None:
        """Stop the worker thread."""
        self._shutdown = True
        self.disable()
        log.info("AutoSyncWorker shut down")

    # ── Internal Methods ────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background loop: poll database every poll_interval seconds."""
        last_poll = 0.0

        while not self._shutdown:
            # Sleep in small increments to allow quick shutdown
            time.sleep(min(10.0, self._config.poll_interval / 6))

            if self._shutdown:
                break

            # Check if enabled
            with self._lock:
                if not self._config.enabled:
                    continue
                poll_interval = self._config.poll_interval

            # Check if enough time has passed
            now = time.monotonic()
            if now - last_poll < poll_interval:
                continue

            # Perform sync
            try:
                result = self._do_sync()
                last_poll = time.monotonic()

                if result.get("records_added", 0) > 0:
                    log.info(
                        "Auto-sync completed: %d records from %s",
                        result["records_added"],
                        result.get("tables_synced", []),
                    )
            except Exception as e:
                log.error("Auto-sync poll failed: %s", e)
                with self._lock:
                    self._stats.last_error = str(e)
                    self._stats.errors.append(f"{datetime.now().isoformat()}: {e}")
                    # Keep only last 100 errors
                    if len(self._stats.errors) > 100:
                        self._stats.errors = self._stats.errors[-100:]

    def _do_sync(self) -> dict[str, Any]:
        """Perform the actual sync operation."""
        with self._lock:
            if self._sync_in_progress:
                return {"error": "Sync already in progress", "skipped": True}
            self._sync_in_progress = True
            tables = self._config.tables
            text_columns_map = self._config.text_columns_map

        try:
            result = self._sync_service.incremental_sync(
                tables=tables,
                text_columns_map=text_columns_map,
            )

            # Update stats
            with self._lock:
                self._stats.total_polls += 1
                self._stats.total_records_synced += result.records_added
                self._stats.total_records_failed += result.records_failed
                self._stats.last_poll_time = datetime.now().isoformat()
                self._stats.last_poll_records = result.records_added

                if result.errors:
                    self._stats.last_error = result.errors[-1]
                    self._stats.errors.extend(result.errors)

            return {
                "tables_synced": result.tables_synced,
                "records_added": result.records_added,
                "records_failed": result.records_failed,
                "duration_seconds": result.duration_seconds,
                "errors": result.errors,
            }

        finally:
            with self._lock:
                self._sync_in_progress = False
