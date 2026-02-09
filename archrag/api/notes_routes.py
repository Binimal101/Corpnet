"""Notes API routes for direct Note ingestion.

Provides endpoints for:
1. add_data() - Add a single Note via POST /add on main ArchRAG API
2. initial_upload() - Bulk upload 70% of notes by repeatedly calling /add

These endpoints use the existing /add route format from docs/fastapi.md
where the Note is stringified and passed in the "text" field.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Union
import httpx
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/notes", tags=["notes"])

# Main ArchRAG API URL (from docs/fastapi.md)
ARCHRAG_API_URL = "https://4959-128-2-149-230.ngrok-free.app"


# ── Pydantic Models ──────────────────────────────────────────────────────────


class NoteInput(BaseModel):
    """Input model for a simplified MemoryNote.
    
    This represents the Note format expected by the API.
    Fields removed: keywords, last_updated, embedding_model, embedding, id, access_id
    """
    content: str = Field(..., description="The main text content of the note")
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    category: str = Field("", description="Primary category")
    retrieval_count: int = Field(0, description="Number of times this note has been retrieved")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Newton's three laws of motion form the foundation of classical mechanics.",
                "tags": ["physics", "mechanics"],
                "category": "Physics",
                "retrieval_count": 0
            }
        }


class AddDataResponse(BaseModel):
    """Response from add_data endpoint."""
    status: str
    enqueued: int
    pending: int
    message: str
    notes_count: int = Field(..., description="Number of notes added")
    note_preview: str | None = Field(None, description="Preview of first note (if single note)")


class InitialUploadRequest(BaseModel):
    """Request to upload initial dataset from pre-generated notes."""
    source: str = Field(
        "bulk_upload",
        description="Source file name without extension from data/notes/"
    )
    percentage: float = Field(
        0.7, 
        ge=0.0, 
        le=1.0, 
        description="Percentage of notes to upload (default 70%)"
    )


class InitialUploadResponse(BaseModel):
    """Response from initial upload."""
    status: str
    source: str
    total_available: int
    total_uploaded: int
    successful: int
    failed: int
    message: str
    errors: list[str] = Field(default_factory=list)


# ── Helper Functions ─────────────────────────────────────────────────────────


def load_notes_from_file(filename: str) -> list[dict]:
    """Load notes from a JSONL file in data/notes/."""
    notes_dir = Path("data/notes")
    filepath = notes_dir / f"{filename}.jsonl"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Notes file not found: {filepath}")
    
    notes = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    notes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return notes


def note_to_add_payload(note: dict) -> dict:
    """Convert a Note to the /add route payload format.
    
    The Note is stringified and placed in the 'text' field.
    Format from docs/fastapi.md POST /add
    """
    stringified_note = json.dumps(note)
    return {
        "documents": [
            {"text": stringified_note}
        ]
    }


def notes_to_bulk_payload(notes: list[dict]) -> dict:
    """Convert multiple Notes to bulk /add payload format."""
    documents = []
    for note in notes:
        stringified_note = json.dumps(note)
        documents.append({"text": stringified_note})
    return {"documents": documents}


async def call_archrag_add(payload: dict) -> dict:
    """Call the main ArchRAG /add endpoint.
    
    POST http://localhost:8000/add
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{ARCHRAG_API_URL}/add",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()


# ── Main Endpoints ───────────────────────────────────────────────────────────

@router.post("/add_data", response_model=AddDataResponse)
async def add_data(body: Union[NoteInput, list[NoteInput]] = Body(...)):
    """Add one or more Notes to the ArchRAG system.
    
    This endpoint accepts either:
    1. A single Note object
    2. A list of Note objects
    
    All notes are sent in a single batch POST request to the ArchRAG /add endpoint.
    
    **Single Note Format:**
    ```json
    {
        "content": "The fact or information...",
        "tags": ["tag1", "tag2"],
        "category": "Category Name",
        "retrieval_count": 0
    }
    ```
    
    **List of Notes Format:**
    ```json
    [
        {
            "content": "First note...",
            "tags": ["tag1"],
            "category": "Category",
            "retrieval_count": 0
        },
        {
            "content": "Second note...",
            "tags": ["tag2"],
            "category": "Category",
            "retrieval_count": 0
        }
    ]
    ```
    
    **Postman Examples:**
    
    Single note:
    ```
    POST http://localhost:8080/notes/add_data
    Content-Type: application/json
    
    {
        "content": "The Magna Carta was signed in 1215 by King John of England.",
        "tags": ["law", "history"],
        "category": "Law",
        "retrieval_count": 0
    }
    ```
    
    Multiple notes:
    ```
    POST http://localhost:8080/notes/add_data
    Content-Type: application/json
    
    [
        {
            "content": "First note content...",
            "tags": ["tag1"],
            "category": "Category",
            "retrieval_count": 0
        },
        {
            "content": "Second note content...",
            "tags": ["tag2"],
            "category": "Category",
            "retrieval_count": 0
        }
    ]
    ```
    """
    # Normalize to list of dict notes
    if isinstance(body, list):
        # List of notes
        notes = []
        for note in body:
            if isinstance(note, NoteInput):
                notes.append(note.model_dump())
            elif isinstance(note, dict):
                notes.append(note)
            else:
                raise HTTPException(
                    status_code=422,
                    detail="Each item in the list must be a Note object (dict with 'content' field)"
                )
        is_single = False
    elif isinstance(body, NoteInput):
        # Single NoteInput object
        notes = [body.model_dump()]
        is_single = True
    elif isinstance(body, dict):
        # Single dict (raw JSON)
        notes = [body]
        is_single = True
    else:
        raise HTTPException(
            status_code=422,
            detail="Body must be either a single Note object or a list of Note objects"
        )
    
    # Validate all notes have required fields
    for i, note in enumerate(notes):
        if not isinstance(note, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Note {i} must be a dictionary with 'content' field"
            )
        if "content" not in note:
            raise HTTPException(
                status_code=422,
                detail=f"Note {i} is missing required field 'content'"
            )
    
    # Create bulk payload with all notes
    payload = notes_to_bulk_payload(notes)
    
    log.info(
        "Calling ArchRAG /add with %d note(s)...",
        len(notes)
    )
    
    try:
        result = await call_archrag_add(payload)
        
        # Get preview of first note if single note
        note_preview = None
        if is_single and notes:
            first_content = notes[0].get("content", "")
            note_preview = first_content[:100] + "..." if len(first_content) > 100 else first_content
        
        return AddDataResponse(
            status="success",
            enqueued=result.get("enqueued", len(notes)),
            pending=result.get("pending", len(notes)),
            message=result.get("message", f"Successfully added {len(notes)} note(s)"),
            notes_count=len(notes),
            note_preview=note_preview
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to ArchRAG API at {ARCHRAG_API_URL}. Is the server running? Start with: python -m archrag.api_server"
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"ArchRAG API error: {e.response.text}"
        )
    except Exception as e:
        log.error("Error calling ArchRAG /add: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error adding note(s): {str(e)}"
        )


@router.post("/initial_upload", response_model=InitialUploadResponse)
async def initial_upload(request: InitialUploadRequest):
    """Upload initial dataset in a single batch POST request to ArchRAG API.
    
    This endpoint:
    1. Loads notes from data/notes/{source}.jsonl
    2. Randomly samples the specified percentage (default 70%)
    3. Makes a single POST /add request with all selected notes in one batch
    
    **Available sources:**
    - `bulk_upload` - 155 notes across all topics (recommended)
    - `law_notes` - 32 law facts
    - `computer_science_notes` - 32 CS facts
    - `finance_notes` - 32 finance facts
    - `geology_notes` - 32 geology facts
    - `physics_notes` - 32 physics facts
    
    **Postman Example:**
    ```
    POST http://localhost:8080/notes/initial_upload
    Content-Type: application/json
    
    {
        "source": "bulk_upload",
        "percentage": 0.7
    }
    ```
    
    This will upload 70% of the 155 bulk notes (~108 notes) in a single batch request.
    All notes are sent together in one POST /add call.
    """
    try:
        notes = load_notes_from_file(request.source)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    total_available = len(notes)
    count_to_upload = int(total_available * request.percentage)
    
    # Sample notes if not uploading all
    if request.percentage < 1.0:
        selected_notes = random.sample(notes, count_to_upload)
    else:
        selected_notes = notes
    
    log.info(
        "Starting initial upload: %d of %d notes from %s (single batch)",
        len(selected_notes), total_available, request.source
    )
    
    # Create single batch payload with all notes
    payload = notes_to_bulk_payload(selected_notes)
    
    try:
        # Make single POST request with all notes
        result = await call_archrag_add(payload)
        
        return InitialUploadResponse(
            status="success",
            source=request.source,
            total_available=total_available,
            total_uploaded=len(selected_notes),
            successful=result.get("enqueued", len(selected_notes)),
            failed=0,
            message=f"Successfully uploaded {len(selected_notes)} notes in a single batch. {result.get('message', '')}",
            errors=[],
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to ArchRAG API at {ARCHRAG_API_URL}. Is the server running?"
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"ArchRAG API error: {e.response.text}"
        )
    except Exception as e:
        log.error("Error uploading notes: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading notes: {str(e)}"
        )


# ── Helper Endpoints ─────────────────────────────────────────────────────────


@router.get("/sources")
async def list_sources():
    """List available note source files.
    
    Returns metadata about available JSONL files in data/notes/.
    """
    notes_dir = Path("data/notes")
    
    if not notes_dir.exists():
        return {"sources": [], "message": "Notes directory not found. Run scripts/generate_notes.py first."}
    
    sources = []
    for jsonl_file in sorted(notes_dir.glob("*.jsonl")):
        with open(jsonl_file, "r") as f:
            count = sum(1 for line in f if line.strip())
        
        sources.append({
            "name": jsonl_file.stem,
            "file": str(jsonl_file),
            "note_count": count,
        })
    
    return {
        "sources": sources,
        "total_notes": sum(s["note_count"] for s in sources),
    }


@router.get("/sample/{source}")
async def get_source_sample(source: str, count: int = 3):
    """Get sample notes from a source file.
    
    Useful for previewing data before upload.
    """
    try:
        notes = load_notes_from_file(source)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    samples = notes[:count]
    
    samples_with_payload = []
    for note in samples:
        samples_with_payload.append({
            "note": note,
            "add_payload": note_to_add_payload(note),
        })
    
    return {
        "source": source,
        "total_in_source": len(notes),
        "samples": samples_with_payload,
    }


@router.get("/reserved")
async def get_reserved_notes():
    """Get the 5 reserved notes (1 per topic) for testing add_data."""
    try:
        notes = load_notes_from_file("reserved_for_add")
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail="Reserved notes not found. Run scripts/generate_notes.py first."
        )
    
    reserved = []
    for note in notes:
        reserved.append({
            "note": note,
            "category": note.get("category", "Unknown"),
        })
    
    return {
        "count": len(notes),
        "notes": reserved,
        "usage": "Use these with POST /notes/add_data"
    }


@router.get("/health")
async def check_archrag_connection():
    """Check if the main ArchRAG API is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ARCHRAG_API_URL}/health")
            response.raise_for_status()
            return {
                "status": "connected",
                "archrag_url": ARCHRAG_API_URL,
                "archrag_status": response.json()
            }
    except httpx.ConnectError:
        return {
            "status": "disconnected",
            "archrag_url": ARCHRAG_API_URL,
            "error": "Cannot connect. Start ArchRAG with: python -m archrag.api_server"
        }
    except Exception as e:
        return {
            "status": "error",
            "archrag_url": ARCHRAG_API_URL,
            "error": str(e)
        }
