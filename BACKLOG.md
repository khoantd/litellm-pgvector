# Feature Backlog

This document contains a prioritized list of features to enhance the OpenAI-compatible Vector Stores API project.

---

## 🔴 High Priority Features

### 1. Embedding CRUD Operations
**Status:** ✅ Completed  
**Priority:** P0  
**Estimated Effort:** Medium

**Description:**
Add missing CRUD operations for embeddings and vector stores.

**Endpoints:**
- `DELETE /v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}` - Delete individual embedding
- `PUT /v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}` - Update embedding content/metadata
- `PATCH /v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}` - Partial update embedding
- `DELETE /v1/vector_stores/{vector_store_id}` - Delete vector store (with cascade options)
- `DELETE /v1/vector_stores/{vector_store_id}/embeddings/batch` - Batch delete by IDs or filters

**Acceptance Criteria:**
- [x] Delete operations cascade properly
- [x] Update operations validate embedding dimensions if changed
- [x] Batch delete supports filtering by metadata
- [x] All operations update vector store statistics (usage_bytes, file_counts)
- [x] Proper error handling for not found cases

**Related Files:**
- `main.py` - Add new route handlers
- `models.py` - Add request/response models

---

### 2. Hybrid Search (Vector + Keyword)
**Status:** ✅ Completed  
**Priority:** P0  
**Estimated Effort:** High

**Description:**
Combine vector similarity search with PostgreSQL full-text search (tsvector) for better relevance.

**Endpoints:**
- `POST /v1/vector_stores/{vector_store_id}/search` - Extend existing endpoint with hybrid mode

**Features:**
- Vector similarity search (existing)
- Keyword full-text search using PostgreSQL `tsvector`
- Weighted scoring: `final_score = (vector_score * vector_weight) + (keyword_score * keyword_weight)`
- Search modes: `vector_only`, `keyword_only`, `hybrid` (default)

**Acceptance Criteria:**
- [x] Add full-text search index on content field (tsvector GENERATED column with GIN index)
- [x] Configurable weight ratios (default: 70% vector, 30% keyword)
- [x] Search mode parameter in request model
- [x] Backward compatible with existing vector-only searches (defaults to hybrid, supports vector_only)
- [x] Performance optimized with proper indexing (GIN index on tsvector)

**Related Files:**
- `main.py` - Update search endpoint with hybrid search logic
- `models.py` - Add search mode and weight parameters to VectorStoreSearchRequest
- Automatic tsvector column creation on startup via `ensure_fulltext_index_exists()`

---

### 3. Usage Analytics & Monitoring
**Status:** ✅ Completed  
**Priority:** P0  
**Estimated Effort:** Medium

**Description:**
Track usage metrics per vector store and provide analytics endpoints.

**Endpoints:**
- `GET /v1/vector_stores/{vector_store_id}/stats` - Get usage statistics
- `GET /v1/stats` - Global usage statistics
- `GET /metrics` - Prometheus metrics endpoint

**Metrics to Track:**
- Total requests per vector store
- Search queries executed
- Embeddings created/deleted
- Storage used (bytes)
- Average response time
- Error rates

**Acceptance Criteria:**
- [x] Track metrics in database (usage_logs table with indexes)
- [x] Provide time-range filtering (daily, weekly, monthly, all)
- [x] Prometheus-compatible metrics format
- [x] Admin UI integration for visualization
- [x] Low-overhead tracking (async logging with asyncio.create_task)

**Related Files:**
- `main.py` - Added stats endpoints, metrics logging middleware
- `models.py` - Added VectorStoreStatsResponse and GlobalStatsResponse models
- `prisma/schema.prisma` - Added UsageLog model with indexes for efficient querying
- `admin-ui/src/components/AnalyticsCharts.tsx` - Chart visualization components (bar, pie, period comparison)
- `admin-ui/src/App.tsx` - Analytics tab with visualizations for endpoints, stores, and trends

---

### 4. Rate Limiting & Quotas
**Status:** ✅ Completed  
**Priority:** P0  
**Estimated Effort:** Medium

**Description:**
Implement rate limiting and quotas to prevent abuse and manage resources.

**Features:**
- Per API key rate limits (requests per second/minute)
- Per vector store quotas (storage bytes, embedding count)
- Configurable limits via admin UI
- Proper HTTP 429 responses with `Retry-After` header

**Endpoints:**
- Rate limiting enforced on all protected endpoints
- `GET /v1/admin/rate_limits` - View current limits
- `PUT /v1/admin/rate_limits` - Configure limits

**Acceptance Criteria:**
- [x] Token bucket or sliding window algorithm
- [x] Per-API-key rate limits
- [x] Per-vector-store quotas
- [x] Configurable via settings
- [x] Graceful 429 responses with retry hints
- [x] Admin UI for configuration

**Related Files:**
- `main.py` - Added rate limiting middleware and quota checking
- `config.py` - Added RateLimitConfig and QuotaConfig with settings overlay support
- `rate_limiter.py` - Token bucket rate limiter implementation with automatic cleanup
- `models.py` - Added rate limit and quota request/response models
- `admin-ui/src/App.tsx` - Added rate limit and quota configuration UI

---

### 5. Multi-Tenant & Access Control
**Status:** Not Started  
**Priority:** P0  
**Estimated Effort:** High

**Description:**
Add multi-tenant isolation and fine-grained access control.

**Features:**
- Organization/tenant isolation per vector store
- Multiple API keys per tenant with role-based permissions (read/write/admin)
- Scoped API keys (restrict to specific vector stores)

**Endpoints:**
- `GET /v1/api_keys` - List API keys
- `POST /v1/api_keys` - Create new API key
- `DELETE /v1/api_keys/{key_id}` - Revoke API key
- `PUT /v1/api_keys/{key_id}` - Update permissions

**Acceptance Criteria:**
- [ ] Vector stores belong to tenants/organizations
- [ ] API keys have roles (read, write, admin)
- [ ] API keys can be scoped to specific vector stores
- [ ] Proper isolation between tenants
- [ ] Admin UI for key management

**Related Files:**
- `main.py` - Add API key management routes
- `models.py` - Add API key models
- `prisma/schema.prisma` - Add organizations, api_keys tables

---

### 6. Advanced Search Capabilities
**Status:** Not Started  
**Priority:** P1  
**Estimated Effort:** Medium

**Description:**
Enhance search functionality with additional capabilities.

**Features:**
- Multi-vector search: combine multiple query embeddings for compound queries
- Faceted search: filter and count by metadata fields
- Search pagination: cursor-based pagination with `next_page` token
- Minimum similarity threshold: filter out low-scoring results
- Time-range filtering: search embeddings created within date range

**Endpoints:**
- `POST /v1/vector_stores/{vector_store_id}/search` - Extend existing endpoint

**Acceptance Criteria:**
- [ ] Support multiple query vectors
- [ ] Faceted search returns counts per metadata value
- [ ] Cursor-based pagination works reliably
- [ ] Similarity threshold filters results
- [ ] Date range filtering on created_at field

**Related Files:**
- `main.py` - Update search endpoint
- `models.py` - Extend search request/response models

---

### 7. Bulk Operations
**Status:** Not Started  
**Priority:** P1  
**Estimated Effort:** Medium

**Description:**
Add bulk operations for efficient data management.

**Endpoints:**
- `POST /v1/vector_stores/{vector_store_id}/embeddings/bulk` - Bulk upsert/delete by filter
- `GET /v1/vector_stores/{vector_store_id}/jobs/{job_id}` - Check background job status
- `POST /v1/vector_stores/{vector_store_id}/export` - Export vector store as JSON
- `POST /v1/vector_stores/{vector_store_id}/import` - Import embeddings from file
- `POST /v1/vector_stores/{vector_store_id}/clone` - Duplicate vector store

**Acceptance Criteria:**
- [ ] Bulk operations process efficiently (batch SQL)
- [ ] Background jobs for large operations
- [ ] Export includes all embeddings and metadata
- [ ] Import validates data format
- [ ] Clone operation preserves structure

**Related Files:**
- `main.py` - Add bulk operation endpoints
- `models.py` - Add bulk operation models
- Optional: Background job processor (Celery/Redis)

---

### 8. Content Deduplication
**Status:** Not Started  
**Priority:** P1  
**Estimated Effort:** Medium

**Description:**
Optional deduplication of embeddings at insert time.

**Features:**
- Hash-based deduplication (exact content match)
- Embedding-based deduplication (similarity threshold)
- Configurable per vector store

**Configuration:**
- Enable/disable deduplication
- Similarity threshold for near-duplicates
- Strategy: hash-based or embedding-based

**Acceptance Criteria:**
- [ ] Hash-based deduplication for exact matches
- [ ] Embedding similarity check for near-duplicates
- [ ] Configurable threshold per vector store
- [ ] Return existing ID if duplicate found
- [ ] Optional: skip or update existing

**Related Files:**
- `main.py` - Add deduplication logic to insert endpoints
- `models.py` - Add deduplication config to vector store

---

### 9. Webhooks & Event System
**Status:** Not Started  
**Priority:** P1  
**Estimated Effort:** Medium

**Description:**
Webhook notifications and event streaming for integration.

**Events:**
- `embedding.created` - New embedding added
- `embedding.deleted` - Embedding removed
- `vector_store.created` - New vector store created
- `vector_store.deleted` - Vector store removed
- `quota.exceeded` - Storage or count quota exceeded

**Endpoints:**
- `GET /v1/admin/webhooks` - List webhooks
- `POST /v1/admin/webhooks` - Create webhook
- `DELETE /v1/admin/webhooks/{id}` - Delete webhook
- `GET /v1/vector_stores/{id}/events` - SSE stream for events

**Acceptance Criteria:**
- [ ] Webhook delivery with retry logic
- [ ] Event payload includes relevant data
- [ ] SSE streaming for real-time updates
- [ ] Webhook signature verification
- [ ] Admin UI for webhook management

**Related Files:**
- `main.py` - Add webhook endpoints and event emitter
- `models.py` - Add webhook models
- `prisma/schema.prisma` - Add webhooks table

---

### 10. Performance Optimizations
**Status:** Not Started  
**Priority:** P1  
**Estimated Effort:** Medium

**Description:**
Caching and performance improvements.

**Features:**
- Response caching: cache search results with TTL
- Embedding cache: cache generated embeddings for identical text
- Connection pooling metrics
- Async batch processing for large embedding jobs

**Acceptance Criteria:**
- [ ] Redis or in-memory cache for search results
- [ ] Embedding cache reduces API calls
- [ ] Pool metrics exposed via admin endpoint
- [ ] Async job queue for batch operations
- [ ] Cache invalidation on updates

**Related Files:**
- `main.py` - Add caching middleware
- Optional: Redis integration
- Optional: Background job queue

---

## 🟡 Medium Priority Features

### 11. Data Validation & Quality
**Status:** Not Started  
**Priority:** P2  
**Estimated Effort:** Low

**Description:**
Validate content and embeddings before insertion.

**Features:**
- Content validation: enforce schema/constraints on metadata
- Embedding validation: verify dimensions before insert
- Quality metrics: track embedding quality scores

**Acceptance Criteria:**
- [ ] Metadata schema validation
- [ ] Dimension validation with clear errors
- [ ] Quality scoring system
- [ ] Configurable validation rules

**Related Files:**
- `models.py` - Add validation logic
- `embedding_service.py` - Enhance validation

---

### 12. Search Enhancements
**Status:** Not Started  
**Priority:** P2  
**Estimated Effort:** Low

**Description:**
Improve search user experience.

**Features:**
- Auto-complete suggestions for search queries
- Search history: track recent searches per vector store
- Saved searches: store and replay common queries

**Acceptance Criteria:**
- [ ] Auto-complete API endpoint
- [ ] Search history stored per vector store
- [ ] Saved search CRUD operations
- [ ] Admin UI integration

**Related Files:**
- `main.py` - Add search enhancement endpoints
- `prisma/schema.prisma` - Add search_history, saved_searches tables

---

### 13. Export & Integration
**Status:** Not Started  
**Priority:** P2  
**Estimated Effort:** Medium

**Description:**
Export data and improve integration capabilities.

**Features:**
- Export to CSV/JSON: download embeddings for analysis
- OpenAI Assistants API compatibility: exact format matching
- GraphQL endpoint: flexible querying
- SDK generation: OpenAPI → client SDKs

**Acceptance Criteria:**
- [ ] CSV export with all fields
- [ ] JSON export matches OpenAI format
- [ ] GraphQL schema for complex queries
- [ ] OpenAPI schema generates valid SDKs

**Related Files:**
- `main.py` - Add export endpoints
- Optional: GraphQL integration (Strawberry)
- OpenAPI schema improvements

---

### 14. Cost Management
**Status:** Not Started  
**Priority:** P2  
**Estimated Effort:** Low

**Description:**
Track and manage costs.

**Features:**
- Cost tracking: estimate costs per embedding operation
- Budget alerts: notify when approaching limits
- Usage reports: daily/weekly/monthly summaries

**Acceptance Criteria:**
- [ ] Cost calculation per operation
- [ ] Budget configuration
- [ ] Email/webhook alerts for budget
- [ ] Usage reports API

**Related Files:**
- `main.py` - Add cost tracking middleware
- `models.py` - Add cost/budget models

---

### 15. Health & Reliability
**Status:** Not Started  
**Priority:** P2  
**Estimated Effort:** Low

**Description:**
Enhanced health checks and reliability features.

**Features:**
- Readiness probe: `/ready` endpoint (checks DB + embedding service)
- Graceful shutdown: finish in-flight requests
- Health checks: DB connection, embedding service availability
- Retry logic: automatic retries for transient failures

**Acceptance Criteria:**
- [ ] `/ready` endpoint for Kubernetes
- [ ] Graceful shutdown with timeout
- [ ] Health check details
- [ ] Automatic retry with exponential backoff

**Related Files:**
- `main.py` - Enhance health checks
- Add retry logic to embedding service

---

## 🟢 Nice-to-Have Features

### 16. Advanced Features
**Status:** Not Started  
**Priority:** P3  
**Estimated Effort:** High

**Description:**
Advanced capabilities for power users.

**Features:**
- Graph relationships: link embeddings with relationships
- Versioning: version embeddings and track changes
- A/B testing: compare different search algorithms
- Multi-language support: detect and handle languages

**Acceptance Criteria:**
- [ ] Relationship graph visualization
- [ ] Version history tracking
- [ ] A/B test configuration
- [ ] Language detection and routing

**Related Files:**
- New models and endpoints
- Optional graph database integration

---

### 17. Admin UI Enhancements
**Status:** Not Started  
**Priority:** P3  
**Estimated Effort:** Medium

**Description:**
Improve admin UI with visualization and tools.

**Features:**
- Vector store visualization: browse and explore stores
- Search testing interface: test queries in UI
- Analytics dashboard: charts for usage trends
- Embedding playground: test different models

**Acceptance Criteria:**
- [ ] Interactive vector store browser
- [ ] Search testing tool
- [ ] Charts and graphs
- [ ] Model comparison tool

**Related Files:**
- `admin-ui/` - Enhance React components

---

### 18. Developer Experience
**Status:** Not Started  
**Priority:** P3  
**Estimated Effort:** Low

**Description:**
Improve developer onboarding and experience.

**Features:**
- Interactive API explorer: try endpoints in docs
- Postman collection: importable collection
- Example clients: Python, JavaScript, Go, Rust
- Migration guides: migrate from Pinecone/Qdrant/Weaviate

**Acceptance Criteria:**
- [ ] Interactive Swagger UI
- [ ] Postman collection JSON
- [ ] SDK examples in multiple languages
- [ ] Migration guides in README

**Related Files:**
- Documentation updates
- Example code in `examples/` directory

---

## Implementation Guidelines

### Priority Levels
- **P0 (Critical):** Must-have features for production readiness
- **P1 (High):** Important features that significantly enhance functionality
- **P2 (Medium):** Useful features that improve user experience
- **P3 (Nice-to-Have):** Features that add polish but aren't essential

### Development Process
1. Create feature branch from `main`
2. Implement feature following existing code patterns
3. Add tests (when test suite exists)
4. Update documentation
5. Submit PR with clear description

### Code Style
- Follow existing patterns in codebase
- Use async/await consistently
- Respect database field mappings from config
- Maintain OpenAI API compatibility
- Add appropriate error handling

### Notes
- All features should maintain backward compatibility
- Breaking changes require version bump and migration guide
- Consider performance impact of new features
- Add proper logging at DEBUG level for debugging

---

## Tracking

Update this backlog as features are:
- ✅ **Completed** - Marked as done and merged to main
- 🚧 **In Progress** - Currently being worked on
- ⏸️ **Paused** - Temporarily halted
- ❌ **Cancelled** - No longer planned

---

**Last Updated:** 2024-01-XX  
**Total Features:** 18  
**Completed:** 4 (Embedding CRUD, Hybrid Search, Usage Analytics & Monitoring, Rate Limiting & Quotas)  
**In Progress:** 0


