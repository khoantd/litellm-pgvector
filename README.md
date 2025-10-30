# OpenAI Vector Stores API with PGVector

A FastAPI application that provides OpenAI-compatible vector store endpoints using PGVector and LiteLLM proxy for embeddings.

## Features

- 🔌 OpenAI-compatible API endpoints
- 🗄️ PGVector for efficient vector storage and similarity search
- 🎛️ Configurable database field mappings
- 🔄 LiteLLM proxy integration for any embedding model
- 🐳 Docker support
- ⚡ FastAPI with async support

## API Endpoints

Authentication
- All secured endpoints require: `Authorization: Bearer <SERVER_API_KEY>`
- The effective key can be set via `.env` or Admin UI; Admin UI overrides env immediately.

Interactive API docs
- Swagger UI: `http://localhost:8000/docs` (click Authorize to set Bearer key)
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

Health
```
GET /health
200 { "status": "healthy", "timestamp": 1730000000 }
```

### 1. Create Vector Store
```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Support FAQ"
  }'
```

### 2. List Vector Stores
```bash
# List all vector stores
curl -X GET \
  http://localhost:8000/v1/vector_stores \
  -H "Authorization: Bearer your-api-key"

# List with pagination (limit and after parameters)
curl -X GET \
  "http://localhost:8000/v1/vector_stores?limit=10&after=vs_abc123" \
  -H "Authorization: Bearer your-api-key"
```

### 3. Add Single Embedding to Vector Store
```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_abc123/embeddings \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Our return policy allows returns within 30 days of purchase.",
    "embedding": [0.1, 0.2, 0.3, ...],
    "metadata": {
      "category": "returns",
      "source": "faq",
      "id": "return_policy_1"
    }
  }'
```

### 4. Add Multiple Embeddings (Batch)
```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_abc123/embeddings/batch \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings": [
      {
        "content": "Our return policy allows returns within 30 days of purchase.",
        "embedding": [0.1, 0.2, 0.3, ...],
        "metadata": {"category": "returns"}
      },
      {
        "content": "Shipping is free for orders over $50.",
        "embedding": [0.4, 0.5, 0.6, ...],
        "metadata": {"category": "shipping"}
      }
    ]
  }'
```

### 5. Search Vector Store
```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_abc123/search \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the return policy?",
    "limit": 20,
    "filters": {"category": "support"}
  }'
```

### Admin Settings (secured)

Get effective settings (secrets redacted):
```bash
curl -H "Authorization: Bearer $SERVER_API_KEY" http://localhost:8000/v1/admin/settings
```

Update settings (only include fields you want to change; omit redacted/unchanged secrets):
```bash
curl -X PUT \
  http://localhost:8000/v1/admin/settings \
  -H "Authorization: Bearer $SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "server": {"host": "0.0.0.0", "port": 8000},
    "embedding": {"model": "text-embedding-3-small", "base_url": "http://localhost:4000"}
  }'
```

Test connectivity (DB + embeddings):
```bash
curl -X POST -H "Authorization: Bearer $SERVER_API_KEY" http://localhost:8000/v1/admin/settings/test
```

Get editable schema:
```bash
curl -H "Authorization: Bearer $SERVER_API_KEY" http://localhost:8000/v1/admin/settings/schema
```

### Error responses
- 401 {"detail":"Invalid API key"} → wrong/expired key
- 403 → missing Authorization header
- 404 → not found
- 500 → server/database error

## Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# Database Configuration
DATABASE_URL="postgresql://username:password@localhost:5432/vectordb?schema=public"

# API Configuration
OPENAI_API_KEY="your-api-key-here"

# Server Configuration
HOST="0.0.0.0"
PORT=8000

# LiteLLM Proxy Configuration
EMBEDDING__MODEL="text-embedding-ada-002"
EMBEDDING__BASE_URL="http://localhost:4000"
EMBEDDING__API_KEY="sk-1234"
EMBEDDING__DIMENSIONS=1536

# Database Field Configuration (optional)
DB_FIELDS__ID_FIELD="id"
DB_FIELDS__CONTENT_FIELD="content"
DB_FIELDS__METADATA_FIELD="metadata"
DB_FIELDS__EMBEDDING_FIELD="embedding"
DB_FIELDS__VECTOR_STORE_ID_FIELD="vector_store_id"
DB_FIELDS__CREATED_AT_FIELD="created_at"
```

### Database Field Mapping

You can customize the database field names by setting environment variables:

- `DB_FIELDS__ID_FIELD` - Primary key field (default: "id")
- `DB_FIELDS__CONTENT_FIELD` - Text content field (default: "content")
- `DB_FIELDS__METADATA_FIELD` - JSON metadata field (default: "metadata")
- `DB_FIELDS__EMBEDDING_FIELD` - Vector embedding field (default: "embedding")
- `DB_FIELDS__VECTOR_STORE_ID_FIELD` - Foreign key field (default: "vector_store_id")
- `DB_FIELDS__CREATED_AT_FIELD` - Timestamp field (default: "created_at")

### LiteLLM Proxy Configuration

The application uses LiteLLM proxy for embeddings. Configure it with:

- `EMBEDDING__MODEL` - Model name (e.g., "text-embedding-ada-002")
- `EMBEDDING__BASE_URL` - LiteLLM proxy URL (e.g., "http://localhost:4000")
- `EMBEDDING__API_KEY` - LiteLLM proxy API key
- `EMBEDDING__DIMENSIONS` - Embedding dimensions (default: 1536)

## Setup and Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Generate Prisma client
prisma generate

# Run database migrations
prisma db push
```

### 3. Set up LiteLLM Proxy

Start LiteLLM proxy pointing to your preferred embedding model:

```bash
# Example: Start LiteLLM proxy for OpenAI
litellm --model text-embedding-ada-002 --port 4000
```

### 4. Run the Application

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker Deployment

### Build and run with Docker:

```bash
# Build the image
docker build -t vector-store-api .

# Run the container
docker run -p 8000:8000 --env-file .env vector-store-api
```

### Production image build script

Use the helper script to build a production image, including the Admin UI assets.

Requirements
- Docker with Buildx enabled (Docker Desktop includes it)
- Node/npm available locally to build the Admin UI

Build locally (loads into your Docker daemon):
```bash
bash scripts/build-image.sh
# customize
IMAGE_NAME=my-registry/vector-store-api IMAGE_TAG=latest PLATFORM=linux/amd64 bash scripts/build-image.sh
```

Build and push to a registry:
```bash
IMAGE_NAME=ghcr.io/you/vector-store-api \
IMAGE_TAG=prod-$(date +%Y%m%d) \
PLATFORM=linux/amd64 \
PUSH=true \
bash scripts/build-image.sh
```

Run the image (container listens on 8003 by default in Dockerfile):
```bash
docker run --rm -p 8000:8003 --env-file .env my-registry/vector-store-api:latest
```

## Admin UI (Configuration)

An optional admin UI is bundled to manage runtime settings persisted in Postgres.

- URL: `http://localhost:8000/admin`
- Auth: Uses the same `SERVER_API_KEY`. Enter it in the UI once; it is stored in your browser only.
- Backing store: `app_settings` table (created via Prisma). Env values remain defaults; DB rows overlay them.

Admin API:

- `GET /v1/admin/settings` returns effective settings (secrets redacted)
- `PUT /v1/admin/settings` updates groups: `server`, `auth`, `embedding`, `db_fields`, `cors`
- `POST /v1/admin/settings/test` validates DB and embedding connectivity
- `GET /v1/admin/settings/schema` describes editable fields

Notes:

- Changes apply without restart. Embedding settings are hot-applied to new requests.
- Secrets (API keys) are write-only; reads return redacted values.

## Database Schema

The application uses two main tables:

### vector_stores
- `id` (string, primary key)
- `name` (string)
- `file_counts` (json)
- `status` (string)
- `usage_bytes` (integer)
- `created_at` (timestamp)
- `expires_after` (json, optional)
- `expires_at` (timestamp, optional)
- `last_active_at` (timestamp, optional)
- `metadata` (json, optional)

### embeddings
- `id` (string, primary key)
- `vector_store_id` (string, foreign key)
- `content` (string)
- `embedding` (vector(1536))
- `metadata` (json, optional)
- `created_at` (timestamp)

## Supported Models

Any embedding model supported by LiteLLM proxy can be used. Examples:

- OpenAI: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
- Cohere: `embed-english-v3.0`, `embed-multilingual-v3.0`
- Voyage: `voyage-2`, `voyage-large-2`
- And many more...

## API Response Format

### Vector Store Response
```json
{
  "id": "vs_abc123",
  "object": "vector_store",
  "created_at": 1699024800,
  "name": "Support FAQ",
  "usage_bytes": 0,
  "file_counts": {
    "in_progress": 0,
    "completed": 0,
    "failed": 0,
    "cancelled": 0,
    "total": 0
  },
  "status": "completed",
  "metadata": {}
}
```

### Vector Store List Response
```json
{
  "object": "list",
  "data": [
    {
      "id": "vs_abc123",
      "object": "vector_store",
      "created_at": 1699024800,
      "name": "Support FAQ",
      "usage_bytes": 1024,
      "file_counts": {"completed": 5, "total": 5},
      "status": "completed",
      "metadata": {}
    }
  ],
  "first_id": "vs_abc123",
  "last_id": "vs_def456",
  "has_more": false
}
```

### Search Response
```json
{
  "object": "vector_store.search",
  "data": [
    {
      "id": "emb_123",
      "content": "Return policy text...",
      "score": 0.95,
      "metadata": {"category": "support"}
    }
  ],
  "usage": {
    "total_tokens": 1
  }
}
```

## Example Search Request

```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_support_faq/search \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I return an item?",
    "limit": 5,
    "return_metadata": true
  }'
```

## Health Check

```bash
curl http://localhost:8000/health
```

## Migrating Existing Data

If you have an existing database with embeddings and content, you can easily migrate using the embedding APIs:

### 1. Create Vector Store
First, create a vector store for your data:

```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Migrated Data",
    "metadata": {"source": "legacy_system"}
  }'
```

### 2. Batch Insert Embeddings
Use the batch endpoint to efficiently insert multiple embeddings:

```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_your_id/embeddings/batch \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings": [
      {
        "content": "Your text content here",
        "embedding": [0.1, 0.2, 0.3, ...1536 dimensions...],
        "metadata": {"source_id": "doc_123", "category": "support"}
      }
    ]
  }'
```

### 3. Migration Script Example

Here's a Python script example for migrating from an existing database:

```python
import psycopg2
import requests
import json

# Connect to your existing database
conn = psycopg2.connect("your_existing_db_url")
cur = conn.cursor()

# Fetch existing data
cur.execute("SELECT content, embedding, metadata FROM your_table")
rows = cur.fetchall()

# Prepare batch data
embeddings = []
for content, embedding, metadata in rows:
    embeddings.append({
        "content": content,
        "embedding": embedding.tolist(),  # Convert numpy array to list
        "metadata": metadata or {}
    })

# Send batch to API
response = requests.post(
    "http://localhost:8000/v1/vector_stores/your_vector_store_id/embeddings/batch",
    headers={
        "Authorization": "Bearer your-api-key",
        "Content-Type": "application/json"
    },
    json={"embeddings": embeddings}
)

print(f"Migrated {len(embeddings)} embeddings")
```

## License

MIT License
