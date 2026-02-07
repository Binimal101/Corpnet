"""Permission labeling for document chunks.

NOTE: Labeling is NOT part of MVP. This module provides stub implementations
that assign empty label sets. Post-MVP, this will implement:
- Source-based labeling (e.g., slack://eng-* → {"engineering"})
- Policy-based labeling (keyword/regex matching)
- Label propagation through the hierarchy

See docs/ACCESS_CONTROL.md for the full design.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.types import DocumentChunk

logger = logging.getLogger(__name__)


class Labeler:
    """Base labeler that assigns empty labels (MVP stub)."""
    
    def label(self, chunk: DocumentChunk) -> DocumentChunk:
        """Assign permission labels to a chunk.
        
        MVP: Returns chunk with empty labels set.
        """
        # No-op for MVP - labels stay empty
        return chunk
    
    def label_batch(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Label a batch of chunks."""
        return [self.label(c) for c in chunks]


class SourceLabeler(Labeler):
    """Label chunks based on their source (post-MVP).
    
    Example mappings:
    - "slack://eng-*" → {"engineering"}
    - "github://*/issues/*" → {"engineering", "issues"}
    - "docs/hr/*" → {"hr", "people"}
    """
    
    def __init__(self, source_label_map: dict[str, set[str]] | None = None):
        self.source_label_map = source_label_map or {}
    
    def label(self, chunk: DocumentChunk) -> DocumentChunk:
        """Assign labels based on source pattern matching."""
        # MVP: No-op, just return chunk
        # Post-MVP: Match source against patterns in source_label_map
        return chunk


class PolicyLabeler(Labeler):
    """Label chunks based on content policies (post-MVP).
    
    Scans chunk text for keywords/regex patterns and assigns labels.
    Example: "confidential" keyword → {"confidential"} label
    """
    
    def __init__(self, policies: list[dict[str, Any]] | None = None):
        self.policies = policies or []
    
    def label(self, chunk: DocumentChunk) -> DocumentChunk:
        """Assign labels based on content policy matching."""
        # MVP: No-op, just return chunk
        # Post-MVP: Match text against policy patterns
        return chunk
