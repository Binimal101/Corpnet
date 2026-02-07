"""Debounced queue for database sync requests.

Batches multiple sync requests within a configurable time window
to prevent redundant syncs during high-volume periods.

Connector-agnostic: works with any database adapter.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

log = logging.getLogger(__name__)


class SyncQueue:
    """Debounced queue that batches sync requests within a time window.

    When you call request_sync(), the queue waits for the debounce window
    to elapse before actually performing the sync. Multiple requests within
    the window are merged together.

    This prevents redundant syncs when:
    - Multiple data changes happen in quick succession
    - Multiple MCP agents request syncs simultaneously
    - High-frequency database updates occur

    Usage:
        queue = SyncQueue(sync_fn=orchestrator.sync_from_database)
        queue.request_sync(["users"])
        queue.request_sync(["posts"])  # Within debounce window
        # Both will be synced together after debounce_window elapses
    """

    def __init__(
        self,
        sync_fn: Callable[..., Any],
        *,
        debounce_window: float = 30.0,
        min_interval: float = 60.0,
    ) -> None:
        """Initialize the queue.

        Args:
            sync_fn: Function to call for syncing. Must accept:
                     tables: list[str], incremental: bool
            debounce_window: Seconds to wait after last request before syncing.
            min_interval: Minimum seconds between syncs (prevents rapid-fire).
        """
        self._sync_fn = sync_fn
        self._debounce_window = debounce_window
        self._min_interval = min_interval

        self._lock = threading.Lock()
        self._pending_tables: set[str] = set()
        self._pending_all: bool = False  # If True, sync all tables
        self._last_request_time: float = 0
        self._last_sync_time: float = 0
        self._timer: threading.Timer | None = None
        self._shutdown = False

    # ── Public API ──────────────────────────────────────────────────────────

    def request_sync(self, tables: list[str] | None = None) -> int:
        """Request a sync for tables (debounced).

        Args:
            tables: Tables to sync. None means all tables.

        Returns:
            Number of tables currently pending.
        """
        with self._lock:
            if tables is None:
                self._pending_all = True
            else:
                self._pending_tables.update(tables)

            self._last_request_time = time.monotonic()

            # Cancel existing timer
            if self._timer is not None:
                self._timer.cancel()

            # Start new timer
            self._timer = threading.Timer(
                self._debounce_window,
                self._try_flush,
            )
            self._timer.daemon = True
            self._timer.start()

            pending_count = len(self._pending_tables) if not self._pending_all else -1

        log.debug(
            "Sync requested for %s (debouncing for %.0fs)",
            tables or "all tables",
            self._debounce_window,
        )
        return pending_count

    def force_flush(self) -> dict[str, Any]:
        """Immediately flush pending requests, ignoring debounce and min_interval.

        Returns:
            Sync result dictionary.
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

        return self._do_flush(force=True)

    def pending_count(self) -> int:
        """Number of tables waiting in the queue (-1 if syncing all)."""
        with self._lock:
            if self._pending_all:
                return -1
            return len(self._pending_tables)

    def clear(self) -> None:
        """Clear pending requests without syncing."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending_tables.clear()
            self._pending_all = False

    def shutdown(self) -> None:
        """Flush remaining requests and shut down."""
        self._shutdown = True
        self.force_flush()
        log.info("SyncQueue shut down")

    # ── Internal Methods ────────────────────────────────────────────────────

    def _try_flush(self) -> None:
        """Called by timer: flush if min_interval has passed."""
        with self._lock:
            now = time.monotonic()
            since_last_sync = now - self._last_sync_time

            if since_last_sync < self._min_interval:
                # Too soon since last sync, reschedule
                remaining = self._min_interval - since_last_sync
                self._timer = threading.Timer(remaining, self._try_flush)
                self._timer.daemon = True
                self._timer.start()
                log.debug("Sync delayed %.0fs (min interval)", remaining)
                return

        self._do_flush(force=False)

    def _do_flush(self, force: bool) -> dict[str, Any]:
        """Perform the actual flush."""
        with self._lock:
            if not self._pending_tables and not self._pending_all:
                return {"skipped": True, "reason": "Nothing pending"}

            tables = None if self._pending_all else list(self._pending_tables)
            self._pending_tables.clear()
            self._pending_all = False
            self._timer = None

        log.info("Flushing sync queue: %s", tables or "all tables")

        try:
            result = self._sync_fn(tables=tables, incremental=True)

            with self._lock:
                self._last_sync_time = time.monotonic()

            return result

        except Exception as e:
            log.error("Sync queue flush failed: %s", e)
            return {"error": str(e)}
