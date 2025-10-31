# API Usage Examples

This document contains practical, real-world examples for using the OpenAI-compatible Vector Stores API. Examples are organized by feature and include both `curl` and Python code snippets.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Vector Store Search](#vector-store-search)
  - [Hybrid Search (Default)](#hybrid-search-default)
  - [Vector-Only Search](#vector-only-search)
  - [Keyword-Only Search](#keyword-only-search)
  - [Custom Weight Ratios](#custom-weight-ratios)
- [Embedding CRUD Operations](#embedding-crud-operations)
  - [Create Embedding](#create-embedding)
  - [Update Embedding (PUT)](#update-embedding-put)
  - [Patch Embedding (PATCH)](#patch-embedding-patch)
  - [Delete Embedding](#delete-embedding)
  - [Batch Delete Embeddings](#batch-delete-embeddings)
  - [Delete Vector Store](#delete-vector-store)

---

## Prerequisites

**Base Configuration:**
```bash
# Set these environment variables
export BASE_URL="http://localhost:8000"
export SERVER_API_KEY="your-api-key-here"
```

**Variables Used in Examples:**
```bash
VECTOR_STORE_ID="vs_..."      # ID returned from creating a vector store
EMBEDDING_ID="emb_..."        # ID returned from creating an embedding
```

---

## Vector Store Search

### Hybrid Search (Default)

**Hybrid search combines vector similarity and keyword relevance for better results.**

```bash
curl -X POST "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/search" \
  -H "Authorization: Bearer ${SERVER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python web framework",
    "limit": 10,
    "search_mode": "hybrid",
    "vector_weight": 0.7,
    "keyword_weight": 0.3,
    "filters": {
      "language": "en"
    }
  }'
```

**Response:**
```json
{
  "object": "vector_store.search_results.page",
  "search_query": "Python web framework",
  "data": [
    {
      "file_id": "emb_abc123",
      "filename": "fastapi.txt",
      "score": 0.89,
      "attributes": {
        "language": "en",
        "section": "introduction"
      },
      "content": [
        {
          "type": "text",
          "text": "FastAPI is a modern, fast web framework for building APIs with Python."
        }
      ]
    }
  ],
  "has_more": false,
  "next_page": null
}
```

**Python Example:**
```python
import requests

response = requests.post(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/search",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "query": "Python web framework",
        "limit": 10,
        "search_mode": "hybrid",
        "vector_weight": 0.7,
        "keyword_weight": 0.3,
        "filters": {"language": "en"}
    }
)
print(response.json())
```

---

### Vector-Only Search

**Use vector similarity only (original behavior):**

```bash
curl -X POST "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/search" \
  -H "Authorization: Bearer ${SERVER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python web framework",
    "limit": 10,
    "search_mode": "vector_only"
  }'
```

**Python Example:**
```python
response = requests.post(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/search",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "query": "Python web framework",
        "limit": 10,
        "search_mode": "vector_only"
    }
)
print(response.json())
```

**Note:** This is backward compatible with existing code - defaults to hybrid mode, but can use `vector_only` for original behavior.

---

### Keyword-Only Search

**Use full-text keyword search only:**

```bash
curl -X POST "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/search" \
  -H "Authorization: Bearer ${SERVER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "FastAPI web framework Python",
    "limit": 10,
    "search_mode": "keyword_only"
  }'
```

**Python Example:**
```python
response = requests.post(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/search",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "query": "FastAPI web framework Python",
        "limit": 10,
        "search_mode": "keyword_only"
    }
)
print(response.json())
```

**Use Case:** Keyword-only search is useful when you need exact phrase matching or when vector embeddings don't capture specific terminology.

---

### Custom Weight Ratios

**Adjust the balance between vector and keyword relevance:**

```bash
# More weight on keywords (50/50 split)
curl -X POST "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/search" \
  -H "Authorization: Bearer ${SERVER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python API development",
    "limit": 10,
    "search_mode": "hybrid",
    "vector_weight": 0.5,
    "keyword_weight": 0.5
  }'
```

**Heavy vector weight (90/10 split):**
```bash
curl -X POST "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/search" \
  -H "Authorization: Bearer ${SERVER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning models",
    "limit": 10,
    "search_mode": "hybrid",
    "vector_weight": 0.9,
    "keyword_weight": 0.1
  }'
```

**Python Example:**
```python
# Custom weights
response = requests.post(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/search",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "query": "machine learning models",
        "limit": 10,
        "search_mode": "hybrid",
        "vector_weight": 0.9,
        "keyword_weight": 0.1
    }
)
print(response.json())
```

**Note:** Weights are automatically normalized to sum to 1.0. If you provide `vector_weight=0.9` and `keyword_weight=0.3`, they will be normalized to `0.75` and `0.25`.

---

## Embedding CRUD Operations

### Create Embedding

**Single Embedding Creation:**

```bash
curl -X POST "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/embeddings" \
  -H "Authorization: Bearer ${SERVER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "FastAPI is a modern, fast web framework for building APIs with Python.",
    "embedding": [0.01, 0.02, 0.03, /* ... array length must match EMBEDDING__DIMENSIONS (default: 1536) */],
    "metadata": {
      "filename": "fastapi.txt",
      "section": "introduction",
      "language": "en"
    }
  }'
```

**Response:**
```json
{
  "id": "emb_abc123",
  "object": "embedding",
  "vector_store_id": "vs_xyz789",
  "content": "FastAPI is a modern, fast web framework...",
  "metadata": {
    "filename": "fastapi.txt",
    "section": "introduction",
    "language": "en"
  },
  "created_at": 1704067200
}
```

**Python Example:**
```python
import requests
import os

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("SERVER_API_KEY")

vector_store_id = "vs_xyz789"
embedding = [0.01, 0.02, 0.03] * 512  # 1536 dimensions

response = requests.post(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/embeddings",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "content": "FastAPI is a modern, fast web framework for building APIs with Python.",
        "embedding": embedding,
        "metadata": {
            "filename": "fastapi.txt",
            "section": "introduction",
            "language": "en"
        }
    }
)
print(response.json())
```

---

### Update Embedding (PUT)

**Full Update - Replaces all provided fields:**

```bash
curl -X PUT "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/embeddings/${EMBEDDING_ID}" \
  -H "Authorization: Bearer ${SERVER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "FastAPI is a fast web framework for building APIs in Python.",
    "metadata": {
      "filename": "fastapi.txt",
      "section": "overview"
    }
  }'
```

**Update with Embedding Change (validates dimensions):**

```bash
curl -X PUT "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/embeddings/${EMBEDDING_ID}" \
  -H "Authorization: Bearer ${SERVER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Updated content here",
    "embedding": [0.1, 0.2, 0.3, /* ... must be 1536 dimensions */],
    "metadata": {"updated": true}
  }'
```

**Python Example:**
```python
import requests

embedding_id = "emb_abc123"
new_embedding = [0.1, 0.2, 0.3] * 512  # Must match dimensions

response = requests.put(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "content": "Updated content here",
        "embedding": new_embedding,
        "metadata": {"updated": True}
    }
)
print(response.json())
```

**Note:** If embedding dimensions don't match `EMBEDDING__DIMENSIONS`, you'll get a 400 error:
```json
{
  "detail": "Invalid embedding dimensions. Expected 1536, got 768"
}
```

---

### Patch Embedding (PATCH)

**Partial Update - Only updates provided fields:**

```bash
# Update only metadata (merge with existing)
curl -X PATCH "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/embeddings/${EMBEDDING_ID}" \
  -H "Authorization: Bearer ${SERVER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "metadata": {
      "section": "getting-started",
      "updated": true
    }
  }'
```

**Update only content:**

```bash
curl -X PATCH "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/embeddings/${EMBEDDING_ID}" \
  -H "Authorization: Bearer ${SERVER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "New content without changing metadata or embedding"
  }'
```

**Python Example:**
```python
# Partial update - only metadata
response = requests.patch(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "metadata": {
            "section": "getting-started",
            "updated": True
        }
    }
)
print(response.json())
```

**Note:** PATCH merges metadata keys with existing metadata. Other fields replace existing values.

---

### Delete Embedding

**Delete Single Embedding:**

```bash
curl -X DELETE "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/embeddings/${EMBEDDING_ID}" \
  -H "Authorization: Bearer ${SERVER_API_KEY}"
```

**Response:**
```json
{
  "object": "embedding.deleted",
  "id": "emb_abc123",
  "deleted": true
}
```

**Python Example:**
```python
response = requests.delete(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}",
    headers={"Authorization": f"Bearer {API_KEY}"}
)
print(response.json())
```

**Note:** This automatically updates the vector store statistics (`file_counts`, `usage_bytes`).

---

### Batch Delete Embeddings

**Delete by Embedding IDs:**

```bash
curl -X DELETE "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/embeddings/batch" \
  -H "Authorization: Bearer ${SERVER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "embedding_ids": ["emb_abc123", "emb_def456", "emb_ghi789"]
  }'
```

**Response:**
```json
{
  "object": "embedding.batch_deleted",
  "deleted_ids": ["emb_abc123", "emb_def456", "emb_ghi789"],
  "deleted_count": 3
}
```

**Delete by Metadata Filters:**

```bash
curl -X DELETE "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/embeddings/batch" \
  -H "Authorization: Bearer ${SERVER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {
      "language": "en",
      "topic": "vectors"
    }
  }'
```

**Delete all embeddings with specific metadata:**

```bash
# Delete all English language embeddings
curl -X DELETE "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/embeddings/batch" \
  -H "Authorization: Bearer ${SERVER_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {
      "language": "en"
    }
  }'
```

**Python Example:**
```python
# Delete by IDs
response = requests.delete(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/embeddings/batch",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "embedding_ids": ["emb_abc123", "emb_def456"]
    }
)
print(response.json())

# Delete by filters
response = requests.delete(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/embeddings/batch",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "filters": {"language": "en", "topic": "vectors"}
    }
)
print(response.json())
```

**Error Handling:**
If neither `embedding_ids` nor `filters` are provided:
```json
{
  "detail": "Either embedding_ids or filters must be provided"
}
```

---

### Delete Vector Store

**Delete Vector Store with Cascade (default):**

```bash
curl -X DELETE "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}?cascade=true" \
  -H "Authorization: Bearer ${SERVER_API_KEY}"
```

**Delete Vector Store without Cascade:**

```bash
curl -X DELETE "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}?cascade=false" \
  -H "Authorization: Bearer ${SERVER_API_KEY}"
```

**Response (with cascade):**
```json
{
  "object": "vector_store.deleted",
  "id": "vs_xyz789",
  "deleted": true,
  "embeddings_deleted_count": 42
}
```

**Response (without cascade):**
```json
{
  "object": "vector_store.deleted",
  "id": "vs_xyz789",
  "deleted": true,
  "embeddings_deleted_count": null
}
```

**Python Example:**
```python
# With cascade (default)
response = requests.delete(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}?cascade=true",
    headers={"Authorization": f"Bearer {API_KEY}"}
)
print(response.json())

# Without cascade
response = requests.delete(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}?cascade=false",
    headers={"Authorization": f"Bearer {API_KEY}"}
)
print(response.json())
```

**Note:** 
- `cascade=true` (default): Deletes all embeddings first, then the vector store
- `cascade=false`: Only deletes the vector store (may fail if foreign key constraints prevent it)

---

## Complete Workflow Example

Here's a complete workflow demonstrating the CRUD operations:

```python
import requests
import os
import time

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("SERVER_API_KEY")

# 1. Create a vector store
vs_response = requests.post(
    f"{BASE_URL}/v1/vector_stores",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"name": "example-store", "metadata": {"owner": "demo"}}
)
vector_store_id = vs_response.json()["id"]
print(f"Created vector store: {vector_store_id}")

# 2. Add an embedding
emb_response = requests.post(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/embeddings",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "content": "This is example content",
        "embedding": [0.01] * 1536,  # 1536 dimensions
        "metadata": {"source": "example"}
    }
)
embedding_id = emb_response.json()["id"]
print(f"Created embedding: {embedding_id}")

# 3. Update the embedding (PATCH)
requests.patch(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"metadata": {"source": "example", "updated": True}}
)
print("Updated embedding metadata")

# 4. Search (existing endpoint)
search_response = requests.post(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/search",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"query": "example content", "limit": 10}
)
print(f"Found {len(search_response.json()['data'])} results")

# 5. Delete the embedding
requests.delete(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}",
    headers={"Authorization": f"Bearer {API_KEY}"}
)
print("Deleted embedding")

# 6. Delete the vector store
requests.delete(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}?cascade=true",
    headers={"Authorization": f"Bearer {API_KEY}"}
)
print("Deleted vector store")
```

---

## Error Handling

**Common Error Responses:**

**404 - Not Found:**
```json
{
  "detail": "Vector store not found"
}
```

**400 - Bad Request:**
```json
{
  "detail": "Invalid embedding dimensions. Expected 1536, got 768"
}
```

**401 - Unauthorized:**
```json
{
  "detail": "Invalid API key"
}
```

**500 - Server Error:**
```json
{
  "detail": "Failed to delete embedding: <error message>"
}
```

---

## Notes

1. **Embedding Dimensions**: All embeddings must match `EMBEDDING__DIMENSIONS` (default: 1536). Check your configuration.
2. **Statistics Updates**: All CRUD operations automatically update vector store statistics (`file_counts`, `usage_bytes`, `last_active_at`).
3. **Metadata Filtering**: Batch delete filters use exact string matching on metadata JSONB fields.
4. **Cascade Delete**: Vector store deletion with `cascade=true` (default) deletes all associated embeddings first.
5. **API Authentication**: All endpoints require `Authorization: Bearer <SERVER_API_KEY>` header.

---

## Search Mode Comparison

**When to use each search mode:**

1. **Hybrid (default)**: Best for most use cases. Combines semantic understanding (vector) with exact keyword matching.
   - Use when: General search queries, mixed content types
   - Example: `"Python web framework"` - finds semantically similar content AND exact keyword matches

2. **Vector-Only**: Best for semantic similarity and conceptual matching.
   - Use when: Finding conceptually similar content, even without exact keywords
   - Example: `"API framework"` finds "FastAPI", "Flask", "Django" even if they don't contain "API framework"

3. **Keyword-Only**: Best for exact phrase matching and specific terminology.
   - Use when: Searching for specific names, codes, or technical terms
   - Example: `"FastAPI documentation"` finds only documents containing those exact words

---

---

## Usage Analytics & Monitoring

### Get Vector Store Statistics

**Get usage statistics for a specific vector store:**

```bash
# Daily stats (default)
curl -X GET "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/stats?period=daily" \
  -H "Authorization: Bearer ${SERVER_API_KEY}"

# Weekly stats
curl -X GET "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/stats?period=weekly" \
  -H "Authorization: Bearer ${SERVER_API_KEY}"

# Monthly stats
curl -X GET "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/stats?period=monthly" \
  -H "Authorization: Bearer ${SERVER_API_KEY}"

# All-time stats
curl -X GET "${BASE_URL}/v1/vector_stores/${VECTOR_STORE_ID}/stats?period=all" \
  -H "Authorization: Bearer ${SERVER_API_KEY}"
```

**Response:**
```json
{
  "object": "vector_store.stats",
  "vector_store_id": "vs_xyz789",
  "period": "daily",
  "start_time": 1704067200,
  "end_time": 1704153600,
  "total_requests": 150,
  "search_queries": 100,
  "embeddings_created": 30,
  "embeddings_deleted": 5,
  "storage_bytes": 1048576,
  "avg_response_time_ms": 45.5,
  "error_count": 2,
  "error_rate": 0.0133,
  "endpoint_stats": {
    "search": {
      "count": 100,
      "avg_response_time_ms": 50.2,
      "error_count": 1
    },
    "create_embedding": {
      "count": 30,
      "avg_response_time_ms": 25.8,
      "error_count": 0
    },
    "delete_embedding": {
      "count": 5,
      "avg_response_time_ms": 15.3,
      "error_count": 1
    }
  }
}
```

**Python Example:**
```python
# Get daily stats
response = requests.get(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/stats",
    headers={"Authorization": f"Bearer {API_KEY}"},
    params={"period": "daily"}
)
print(response.json())

# Get weekly stats
response = requests.get(
    f"{BASE_URL}/v1/vector_stores/{vector_store_id}/stats",
    headers={"Authorization": f"Bearer {API_KEY}"},
    params={"period": "weekly"}
)
print(response.json())
```

---

### Get Global Statistics

**Get usage statistics across all vector stores:**

```bash
curl -X GET "${BASE_URL}/v1/stats?period=daily" \
  -H "Authorization: Bearer ${SERVER_API_KEY}"
```

**Response:**
```json
{
  "object": "stats.global",
  "period": "daily",
  "start_time": 1704067200,
  "end_time": 1704153600,
  "total_requests": 500,
  "total_vector_stores": 10,
  "total_embeddings": 1500,
  "total_storage_bytes": 52428800,
  "avg_response_time_ms": 42.3,
  "error_count": 5,
  "error_rate": 0.01,
  "endpoint_stats": {
    "search": {"count": 300, "avg_response_time_ms": 48.5, "error_count": 2},
    "create_embedding": {"count": 150, "avg_response_time_ms": 28.2, "error_count": 2},
    "delete_embedding": {"count": 50, "avg_response_time_ms": 18.1, "error_count": 1}
  },
  "top_vector_stores": [
    {
      "vector_store_id": "vs_abc123",
      "request_count": 200,
      "avg_response_time_ms": 45.2
    },
    {
      "vector_store_id": "vs_def456",
      "request_count": 150,
      "avg_response_time_ms": 38.7
    }
  ]
}
```

**Python Example:**
```python
response = requests.get(
    f"{BASE_URL}/v1/stats",
    headers={"Authorization": f"Bearer {API_KEY}"},
    params={"period": "weekly"}
)
stats = response.json()
print(f"Total requests: {stats['total_requests']}")
print(f"Average response time: {stats['avg_response_time_ms']}ms")
print(f"Error rate: {stats['error_rate']:.2%}")
```

---

### Prometheus Metrics Endpoint

**Get metrics in Prometheus format:**

```bash
curl -X GET "${BASE_URL}/metrics"
```

**Response (text/plain):**
```
# HELP vector_store_api_requests_total Total number of requests
# TYPE vector_store_api_requests_total counter
vector_store_api_requests_total 500

# HELP vector_store_api_request_duration_ms Average request duration in milliseconds
# TYPE vector_store_api_request_duration_ms gauge
vector_store_api_request_duration_ms 42.3

# HELP vector_store_api_errors_total Total number of errors
# TYPE vector_store_api_errors_total counter
vector_store_api_errors_total 5

# HELP vector_store_api_success_total Total number of successful requests
# TYPE vector_store_api_success_total counter
vector_store_api_success_total 495

# HELP vector_store_api_endpoint_requests_total Total requests per endpoint
# TYPE vector_store_api_endpoint_requests_total counter
vector_store_api_endpoint_requests_total{endpoint="search"} 300
vector_store_api_endpoint_requests_total{endpoint="create_embedding"} 150
vector_store_api_endpoint_requests_total{endpoint="delete_embedding"} 50

# HELP vector_store_api_endpoint_duration_ms Average duration per endpoint
# TYPE vector_store_api_endpoint_duration_ms gauge
vector_store_api_endpoint_duration_ms{endpoint="search"} 48.5
vector_store_api_endpoint_duration_ms{endpoint="create_embedding"} 28.2
vector_store_api_endpoint_duration_ms{endpoint="delete_embedding"} 18.1

# HELP vector_store_api_endpoint_errors_total Total errors per endpoint
# TYPE vector_store_api_endpoint_errors_total counter
vector_store_api_endpoint_errors_total{endpoint="search"} 2
vector_store_api_endpoint_errors_total{endpoint="create_embedding"} 2
vector_store_api_endpoint_errors_total{endpoint="delete_embedding"} 1
```

**Python Example:**
```python
response = requests.get(f"{BASE_URL}/metrics")
print(response.text)  # Prometheus format
```

**Note:** The `/metrics` endpoint does not require authentication and is designed for Prometheus scraping. It shows metrics from the last hour.

---

**Last Updated:** 2024-01-XX  
**Version:** 1.2.0 (Embedding CRUD Operations + Hybrid Search + Usage Analytics & Monitoring)

