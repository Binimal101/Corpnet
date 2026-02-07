"""Ingestion module: chunking, entity extraction, and pipeline."""

from src.ingestion.chunker import chunk_text, RecursiveChunker
from src.ingestion.entity_extractor import EntityExtractor
from src.ingestion.pipeline import IngestionPipeline

__all__ = [
    "chunk_text",
    "RecursiveChunker",
    "EntityExtractor",
    "IngestionPipeline",
]
