# Agentic API Reference

The **Agentic API** is a stateful conversational interface that sits between the frontend and the Producer/Consumer backends. It provides:

1. **Organization-based authentication** via Supabase
2. **Hierarchical access control** with inheritance
3. **Session state** across requests
4. **Intent classification** (READ/WRITE)
5. **Automatic routing** to Producer or Consumer API
6. **Access ID propagation** to notes and chunks

## Architecture

```
┌─────────────┐     ┌─────────────────────┐     ┌──────────────┐
│   Frontend  │────▶│   Agentic API       │────▶│  Producer    │
│             │     │   (port 8080)       │     │  (port 8000) │
│             │     │                     │     └──────────────┘
│ user_id     │     │   ┌───────────────┐ │
│ org_id      │     │   │ Supabase      │ │
│             │     │   │ Permissions   │ │
│             │◀────│   └───────────────┘ │────▶┌──────────────┐
│             │     │                     │     │  Consumer    │
└─────────────┘     └─────────────────────┘     │  (port 8001) │
                                                └──────────────┘
```

## Supabase Schema

The permission system requires these Supabase tables:

### `organizations`
```sql
CREATE TABLE organizations (
    org_id      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    write_perm  BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMP DEFAULT NOW()
);
```

### `org_read_permissions`
```sql
CREATE TABLE org_read_permissions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id            TEXT REFERENCES organizations(org_id),
    access_id         TEXT NOT NULL,
    level             TEXT NOT NULL,  -- 'root', 'admin', 'department', 'team', 'project', 'public'
    name              TEXT NOT NULL,
    includes_children BOOLEAN DEFAULT TRUE
);
```

### `access_hierarchy`
```sql
CREATE TABLE access_hierarchy (
    access_id   TEXT PRIMARY KEY,
    org_id      TEXT REFERENCES organizations(org_id),
    level       TEXT NOT NULL,
    name        TEXT NOT NULL,
    parent_id   TEXT REFERENCES access_hierarchy(access_id)
);
```

### `user_permissions` (optional overrides)
```sql
CREATE TABLE user_permissions (
    user_id    TEXT NOT NULL,
    org_id     TEXT NOT NULL,
    write_perm BOOLEAN,  -- NULL = inherit from org
    PRIMARY KEY (user_id, org_id)
);
```

### `user_read_permissions` (optional overrides)
```sql
CREATE TABLE user_read_permissions (
    user_id           TEXT NOT NULL,
    org_id            TEXT NOT NULL,
    access_id         TEXT NOT NULL,
    includes_children BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (user_id, org_id, access_id)
);
```

## Example Hierarchy

```
root (AccessLevel.ROOT)
├── dept_engineering (AccessLevel.DEPARTMENT)
│   ├── team_backend (AccessLevel.TEAM)
│   │   ├── proj_api (AccessLevel.PROJECT)
│   │   └── proj_db (AccessLevel.PROJECT)
│   └── team_frontend (AccessLevel.TEAM)
│       ├── proj_web (AccessLevel.PROJECT)
│       └── proj_mobile (AccessLevel.PROJECT)
└── dept_sales (AccessLevel.DEPARTMENT)
    ├── team_enterprise (AccessLevel.TEAM)
    └── team_smb (AccessLevel.TEAM)
```

If a user has access to `dept_engineering` with `includes_children=true`, they can read:
- `dept_engineering`
- `team_backend`, `team_frontend`
- `proj_api`, `proj_db`, `proj_web`, `proj_mobile`

## Configuration

### Environment Variables

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
```

### Starting the Server

```bash
# Default settings (port 8080)
archrag api

# Custom port
archrag api --port 9000

# With Supabase credentials
archrag api --supabase-url https://xxx.supabase.co --supabase-key xxx

# Custom backend URLs
archrag api --producer-url http://producer:8000 --consumer-url http://consumer:8001
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `8080` | Port to listen on |
| `--producer-url` | `http://localhost:8000` | Producer API base URL |
| `--consumer-url` | `http://localhost:8001` | Consumer API base URL |
| `--supabase-url` | env `SUPABASE_URL` | Supabase project URL |
| `--supabase-key` | env `SUPABASE_KEY` | Supabase service role key |
| `--reload` | `false` | Enable auto-reload for development |

Interactive docs are available at `http://localhost:8080/docs` (Swagger UI).

---

## Endpoints

### `POST /session`

Create a new conversation session with organization-based authentication.

**Request body:**

```json
{
  "user_id": "user_123",
  "org_id": "acme_corp"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | yes | User identifier from frontend |
| `org_id` | string | yes | Organization identifier (looked up in Supabase) |

**Response (200):**

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "greeting": "Welcome! I'm your ArchRAG assistant...",
  "user_id": "user_123",
  "org_id": "acme_corp",
  "write_perm": true,
  "accessible_ids": ["dept_engineering", "team_backend", "team_frontend", "proj_api", "proj_db"],
  "current_access_id": "proj_api"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Unique session identifier |
| `greeting` | string | Initial greeting message |
| `user_id` | string | User identifier |
| `org_id` | string | Organization identifier |
| `write_perm` | boolean | Whether user can write |
| `accessible_ids` | array | All access scopes the user can read |
| `current_access_id` | string | Default scope for write operations |

**Response (403) — Permission denied:**

```json
{
  "detail": "Organization 'unknown_org' not found or user 'user_123' has no access"
}
```

---

### `POST /chat`

Send a message and receive a response. For WRITE operations, the `access_id` is automatically attached to created notes and chunks.

**Request body:**

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Add this document: The API service handles authentication..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | yes | Session identifier from `/session` |
| `message` | string | yes | User message |

**Response (200) — WRITE operation:**

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Documents added successfully.",
  "intent": "write",
  "action": "route_producer",
  "access_id": "proj_api",
  "api_result": {
    "status": "ok",
    "enqueued": 1,
    "pending": 1
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `message` | string | Response message to display |
| `intent` | string | Classified intent: `"read"`, `"write"`, or `"unknown"` |
| `action` | string | Action taken (see below) |
| `access_id` | string\|null | For writes, the access scope used |
| `api_result` | object\|null | Raw API response if routed |

---

### `PUT /session/{session_id}/access`

Change the current access scope for write operations.

**Request body:**

```json
{
  "access_id": "team_backend"
}
```

**Response (200):**

```json
{
  "status": "updated",
  "current_access_id": "team_backend"
}
```

**Response (403) — Access denied:**

```json
{
  "detail": "Access scope 'dept_sales' not permitted. Allowed: ['dept_engineering', 'team_backend', ...]"
}
```

---

### `GET /session/{session_id}`

Get the current state of a session including permissions.

**Response (200):**

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "user_id": "user_123",
  "org_id": "acme_corp",
  "intent": "write",
  "current_access_id": "proj_api",
  "accessible_ids": ["dept_engineering", "team_backend", "proj_api"],
  "write_perm": true,
  "created_at": "2024-02-07T12:00:00.000000",
  "last_activity": "2024-02-07T12:05:30.000000",
  "message_count": 5
}
```

---

### `GET /sessions`

List sessions by user or organization.

**Query parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | one of | User identifier |
| `org_id` | string | one of | Organization identifier |

**Response (200):**

```json
{
  "user_id": "user_123",
  "org_id": null,
  "sessions": [
    {
      "session_id": "a1b2c3d4-...",
      "user_id": "user_123",
      "org_id": "acme_corp",
      "intent": "write",
      "current_access_id": "proj_api",
      "created_at": "2024-02-07T12:00:00",
      "last_activity": "2024-02-07T12:05:30",
      "message_count": 5
    }
  ]
}
```

---

### `GET /permissions/{org_id}`

Get the permission hierarchy for an organization.

**Response (200):**

```json
{
  "org_id": "acme_corp",
  "write_perm": true,
  "read_perms": [
    {
      "access_id": "dept_engineering",
      "level": "department",
      "name": "Engineering",
      "includes_children": true
    }
  ],
  "accessible_ids": ["dept_engineering", "team_backend", "team_frontend", "proj_api", "proj_db"]
}
```

---

### `DELETE /session/{session_id}`

End and delete a session.

**Response (200):**

```json
{
  "status": "deleted",
  "session_id": "a1b2c3d4-..."
}
```

---

### `GET /health`

Health check endpoint.

**Response (200):**

```json
{
  "status": "healthy",
  "service": "archrag-agent"
}
```

---

## Access ID Propagation

When a producer adds data through the API, the `access_id` flows through the entire pipeline:

```
User (org: acme_corp, access_id: proj_api)
  │
  ▼
POST /chat: "Add document: ..."
  │
  ▼
Agentic API injects access_id into payload
  │
  ▼
Producer API creates MemoryNote
  │  access_id: "proj_api"
  ▼
MemoryNote is chunked
  │  Each TextChunk inherits access_id: "proj_api"
  ▼
Knowledge Graph entities extracted
  │  Linked to chunks with access_id
  ▼
C-HNSW Index updated
```

When a consumer queries data, only chunks with matching `access_id` are returned based on their permissions.

---

## Example Frontend Integration

### TypeScript

```typescript
const API_BASE = 'http://localhost:8080';

interface SessionResponse {
  session_id: string;
  greeting: string;
  user_id: string;
  org_id: string;
  write_perm: boolean;
  accessible_ids: string[];
  current_access_id: string;
}

interface ChatResponse {
  message: string;
  intent: 'read' | 'write' | 'unknown';
  action: string;
  access_id: string | null;
  api_result: any;
}

// 1. Create session
async function createSession(userId: string, orgId: string): Promise<SessionResponse> {
  const response = await fetch(`${API_BASE}/session`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, org_id: orgId }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }
  
  return response.json();
}

// 2. Send message
async function sendMessage(sessionId: string, message: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  return response.json();
}

// 3. Change access scope
async function setAccessScope(sessionId: string, accessId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/session/${sessionId}/access`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ access_id: accessId }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }
}

// Usage
async function main() {
  // Create session with org credentials
  const session = await createSession('user_123', 'acme_corp');
  console.log('Session created:', session.session_id);
  console.log('Can write:', session.write_perm);
  console.log('Access scopes:', session.accessible_ids);
  console.log('Current scope:', session.current_access_id);
  
  // Query data (READ)
  const readResponse = await sendMessage(session.session_id, 'What is the API authentication flow?');
  console.log('Answer:', readResponse.message);
  
  // Add data (WRITE) - access_id is automatically attached
  const writeResponse = await sendMessage(
    session.session_id,
    'Add this document: OAuth2 tokens expire after 1 hour...'
  );
  console.log('Added with access_id:', writeResponse.access_id);
  
  // Change access scope for future writes
  await setAccessScope(session.session_id, 'team_backend');
}
```

---

## Mock Mode (No Supabase)

If Supabase credentials are not configured, the API runs in mock mode with sample permissions:

- Organization: Any org_id is accepted
- Read permissions: `dept_engineering` with children
- Write permissions: Enabled
- Hierarchy: Full mock hierarchy (root → departments → teams → projects)

This allows development and testing without a Supabase connection.
