# Notes API - Postman Collection

This document describes the Notes API endpoints for Note ingestion.
These endpoints call the main ArchRAG `/add` endpoint (port 8000).

## Servers

| Server | URL | Purpose |
|--------|-----|---------|
| ArchRAG API | `http://localhost:8000` | Main indexing/query API |
| Notes API | `http://localhost:8080` | Note ingestion wrapper |

## Note Format (Simplified)

```json
{
  "content": "The main text content of the note",
  "tags": ["category_tag"],
  "category": "Category Name",
  "retrieval_count": 0
}
```

**Removed fields:** `id`, `access_id`, `keywords`, `last_updated`, `embedding_model`, `embedding`

---

## Main Endpoints

### 1. `POST /notes/add_data` - Add Note(s)

Add one or more Notes by calling the main ArchRAG `/add` endpoint in a single batch.

**Request:**
```
POST http://localhost:8080/notes/add_data
Content-Type: application/json
```

**Single Note (Request Body):**
```json
{
  "content": "The Magna Carta, signed in 1215 by King John of England, is considered one of the foundational documents of constitutional law.",
  "tags": ["law", "history"],
  "category": "Law",
  "retrieval_count": 0
}
```

**Multiple Notes (Request Body):**
```json
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

**Response (200) - Single Note:**
```json
{
  "status": "success",
  "enqueued": 1,
  "pending": 1,
  "message": "Enqueued 1 document(s). Call /reindex to flush immediately.",
  "notes_count": 1,
  "note_preview": "The Magna Carta, signed in 1215 by King John of England, is considered one of the foundational..."
}
```

**Response (200) - Multiple Notes:**
```json
{
  "status": "success",
  "enqueued": 2,
  "pending": 2,
  "message": "Enqueued 2 document(s). Call /reindex to flush immediately.",
  "notes_count": 2,
  "note_preview": null
}
```

**Error (503) - ArchRAG not running:**
```json
{
  "detail": "Cannot connect to ArchRAG API at http://localhost:8000. Is the server running? Start with: python -m archrag.api_server"
}
```

---

### 2. `POST /notes/initial_upload` - Bulk Upload (70% default)

Upload notes from pre-generated files by repeatedly calling `/add`.

**Request:**
```
POST http://localhost:8080/notes/initial_upload
Content-Type: application/json
```

**Request Body:**
```json
{
  "source": "bulk_upload",
  "percentage": 0.7
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | string | `bulk_upload` | Source file from data/notes/ |
| `percentage` | float | 0.7 | Fraction to upload (0.0-1.0) |

**Response (200):**
```json
{
  "status": "success",
  "source": "bulk_upload",
  "total_available": 155,
  "total_uploaded": 108,
  "successful": 108,
  "failed": 0,
  "message": "Uploaded 108/108 notes. 0 failed.",
  "errors": []
}
```

**Partial Success:**
```json
{
  "status": "partial",
  "source": "bulk_upload",
  "total_available": 155,
  "total_uploaded": 108,
  "successful": 100,
  "failed": 8,
  "message": "Uploaded 100/108 notes. 8 failed.",
  "errors": ["Note 15: HTTP 500", "Note 23: timeout"]
}
```

---

## Helper Endpoints

### 3. `GET /notes/health` - Check ArchRAG Connection

Check if the main ArchRAG API is reachable.

**Request:**
```
GET http://localhost:8080/notes/health
```

**Response (connected):**
```json
{
  "status": "connected",
  "archrag_url": "http://localhost:8000",
  "archrag_status": {"status": "healthy"}
}
```

**Response (disconnected):**
```json
{
  "status": "disconnected",
  "archrag_url": "http://localhost:8000",
  "error": "Cannot connect. Start ArchRAG with: python -m archrag.api_server"
}
```

---

### 4. `GET /notes/sources` - List Available Sources

**Request:**
```
GET http://localhost:8080/notes/sources
```

**Response:**
```json
{
  "sources": [
    {"name": "bulk_upload", "file": "data/notes/bulk_upload.jsonl", "note_count": 155},
    {"name": "reserved_for_add", "file": "data/notes/reserved_for_add.jsonl", "note_count": 5},
    {"name": "law_notes", "file": "data/notes/law_notes.jsonl", "note_count": 32},
    {"name": "computer_science_notes", "file": "data/notes/computer_science_notes.jsonl", "note_count": 32},
    {"name": "finance_notes", "file": "data/notes/finance_notes.jsonl", "note_count": 32},
    {"name": "geology_notes", "file": "data/notes/geology_notes.jsonl", "note_count": 32},
    {"name": "physics_notes", "file": "data/notes/physics_notes.jsonl", "note_count": 32}
  ],
  "total_notes": 315
}
```

---

### 5. `GET /notes/sample/{source}` - Preview Notes

**Request:**
```
GET http://localhost:8080/notes/sample/physics_notes?count=2
```

**Response:**
```json
{
  "source": "physics_notes",
  "total_in_source": 32,
  "samples": [
    {
      "note": {
        "content": "Newton's three laws of motion...",
        "tags": ["physics"],
        "category": "Physics",
        "retrieval_count": 0
      },
      "add_payload": {
        "documents": [
          {"text": "{\"content\": \"Newton's three laws...\", \"tags\": [\"physics\"], \"category\": \"Physics\", \"retrieval_count\": 0}"}
        ]
      }
    }
  ]
}
```

---

### 6. `GET /notes/reserved` - Get Test Notes

Get the 5 reserved notes (1 per topic) for testing `add_data()`.

**Request:**
```
GET http://localhost:8080/notes/reserved
```

**Response:**
```json
{
  "count": 5,
  "notes": [
    {
      "note": {
        "content": "The Universal Declaration of Human Rights...",
        "tags": ["law"],
        "category": "Law",
        "retrieval_count": 0
      },
      "category": "Law"
    }
  ],
  "usage": "Use these with POST /notes/add_data"
}
```

---

## Workflow

### Step 1: Start Servers

```bash
# Terminal 1: Start main ArchRAG API (port 8000)
python -m archrag.api_server

# Terminal 2: Start Notes API (port 8080)
archrag api
```

### Step 2: Check Connection

```
GET http://localhost:8080/notes/health
```

### Step 3: Test Single Add

```
POST http://localhost:8080/notes/add_data
Content-Type: application/json

{
  "content": "E=mc² establishes the equivalence of mass and energy.",
  "tags": ["physics"],
  "category": "Physics",
  "retrieval_count": 0
}
```

### Step 4: Bulk Upload 70%

```
POST http://localhost:8080/notes/initial_upload
Content-Type: application/json

{
  "source": "bulk_upload",
  "percentage": 0.7
}
```

### Step 5: Trigger Reindex (on main API)

```
POST http://localhost:8000/reindex
```

---

## Data Summary

| Source | Notes | For |
|--------|-------|-----|
| `bulk_upload` | 155 | `initial_upload()` with 70% = ~108 notes |
| `reserved_for_add` | 5 | Testing `add_data()` (1 per topic) |
| `law_notes` | 32 | Topic-specific upload |
| `computer_science_notes` | 32 | Topic-specific upload |
| `finance_notes` | 32 | Topic-specific upload |
| `geology_notes` | 32 | Topic-specific upload |
| `physics_notes` | 32 | Topic-specific upload |

---

## Postman Collection

Import into Postman:

### Variables
```
base_url = http://localhost:8080
archrag_url = http://localhost:8000
```

### Requests

1. **Check Health**
   - `GET {{base_url}}/notes/health`

2. **List Sources**
   - `GET {{base_url}}/notes/sources`

3. **Get Reserved Notes**
   - `GET {{base_url}}/notes/reserved`

4. **Add Note(s) - Single**
   - `POST {{base_url}}/notes/add_data`
   - Body:
     ```json
     {
       "content": "Your note content here",
       "tags": ["tag1"],
       "category": "Category",
       "retrieval_count": 0
     }
     ```

5. **Add Note(s) - Multiple**
   - `POST {{base_url}}/notes/add_data`
   - Body:
     ```json
     [
       {
         "content": "First note content",
         "tags": ["tag1"],
         "category": "Category",
         "retrieval_count": 0
       },
       {
         "content": "Second note content",
         "tags": ["tag2"],
         "category": "Category",
         "retrieval_count": 0
       }
     ]
     ```

6. **Initial Upload (70%)**
   - `POST {{base_url}}/notes/initial_upload`
   - Body:
     ```json
     {
       "source": "bulk_upload",
       "percentage": 0.7
     }
     ```

6. **Trigger Reindex**
   - `POST {{archrag_url}}/reindex`
