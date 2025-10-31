import os
import asyncio
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prisma import Prisma
from dotenv import load_dotenv

from models import (
    VectorStoreCreateRequest,
    VectorStoreResponse,
    VectorStoreSearchRequest,
    VectorStoreSearchResponse,
    SearchResult,
    EmbeddingCreateRequest,
    EmbeddingResponse,
    EmbeddingBatchCreateRequest,
    EmbeddingBatchCreateResponse,
    VectorStoreListResponse,
    ContentChunk,
    EmbeddingUpdateRequest,
    EmbeddingPatchRequest,
    EmbeddingBatchDeleteRequest,
    EmbeddingDeleteResponse,
    EmbeddingBatchDeleteResponse,
    VectorStoreDeleteRequest,
    VectorStoreDeleteResponse,
    VectorStoreStatsResponse,
    GlobalStatsResponse
)
from config import settings, settings_manager
from embedding_service import embedding_service

load_dotenv()

app = FastAPI(
    title="OpenAI Vector Stores API",
    description="OpenAI-compatible Vector Stores API using PGVector",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optionally serve admin UI if built and present
try:
    app.mount("/admin", StaticFiles(directory="static/admin", html=True), name="admin")
except Exception:
    # Static admin not available in dev; ignore
    pass

# Global Prisma client
db = Prisma()

security = HTTPBearer()
def _on_settings_applied(old, new):
    try:
        if (old.embedding.model != new.embedding.model or
            old.embedding.base_url != new.embedding.base_url or
            old.embedding.api_key != new.embedding.api_key or
            old.embedding.dimensions != new.embedding.dimensions):
            # Hot-apply embedding configuration
            embedding_service.update_config(new.embedding)
    except Exception:
        pass

settings_manager.register_on_apply(_on_settings_applied)



async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key from Authorization header"""
    expected_key = settings_manager.effective.server_api_key
    if credentials.credentials != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


@app.on_event("startup")
async def startup():
    """Connect to database on startup"""
    await db.connect()
    
    # Optionally run migrations on startup (set AUTO_MIGRATE=true in env)
    # Prisma db push is incremental: it adds new tables/columns without dropping existing ones
    if os.getenv("AUTO_MIGRATE", "").lower() == "true":
        try:
            # Check if all required tables exist
            check_result = await db.query_raw(
                """
                SELECT table_name
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('app_settings', 'vector_stores', 'embeddings')
                """
            )
            existing_tables = {row["table_name"] for row in check_result} if check_result else set()
            required_tables = {"app_settings", "vector_stores", "embeddings"}
            
            missing_tables = required_tables - existing_tables
            
            if missing_tables:
                print(f"[Startup] Missing tables: {missing_tables}, running migration...")
            else:
                print("[Startup] All required tables exist, checking for schema updates...")
            
            # Always run prisma db push to apply incremental schema changes
            # Prisma will detect if schema matches and do nothing, or add new tables/columns if needed
            # Note: --accept-data-loss allows adding columns with defaults; it won't drop existing data
            proc = await asyncio.create_subprocess_exec(
                "prisma", "db", "push", "--accept-data-loss",
                cwd="/app",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            
            if proc.returncode != 0:
                error_output = stderr.decode() if stderr else 'Unknown error'
                # If error is just "no changes", that's fine - schema is up to date
                if "no changes" in error_output.lower() or "already in sync" in error_output.lower():
                    print("[Startup] Database schema is up to date, no changes needed")
                else:
                    print(f"[Startup] Migration failed: {error_output}")
            else:
                output = stdout.decode() if stdout else ""
                if "already in sync" in output.lower() or "no changes" in output.lower():
                    print("[Startup] Database schema is up to date")
                else:
                    print("[Startup] Database schema updated successfully")
                    
        except asyncio.TimeoutError:
            print("[Startup] Migration timeout (non-fatal)")
        except Exception as e:
            # If check fails, try running migration anyway (might be first run on empty DB)
            try:
                print(f"[Startup] Schema check failed: {e}, attempting migration...")
                proc = await asyncio.create_subprocess_exec(
                    "prisma", "db", "push", "--accept-data-loss",
                    cwd="/app",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                if proc.returncode != 0:
                    print(f"[Startup] Migration failed: {stderr.decode() if stderr else 'Unknown error'}")
                else:
                    print("[Startup] Migrations complete")
            except asyncio.TimeoutError:
                print("[Startup] Migration timeout (non-fatal)")
            except Exception as migration_error:
                print(f"[Startup] Migration failed (non-fatal): {migration_error}")
    
    # Load admin overlays from DB and apply
    try:
        await settings_manager.load_overlay_from_db(db)
    except Exception:
        # Continue with base settings if overlay load fails
        pass
    
    # Ensure full-text search index exists
    try:
        await ensure_fulltext_index_exists()
    except Exception:
        # Continue if index creation fails (may already exist)
        pass


@app.on_event("shutdown")
async def shutdown():
    """Disconnect from database on shutdown"""
    await db.disconnect()


async def generate_query_embedding(query: str) -> List[float]:
    """
    Generate an embedding for the query using LiteLLM
    """
    return await embedding_service.generate_embedding(query)


async def ensure_fulltext_index_exists():
    """
    Ensure tsvector column and GIN index exist for full-text search.
    This is called on startup and can be safely called multiple times.
    """
    try:
        table_name = settings.table_names["embeddings"]
        fields = settings.db_fields
        
        # Check if tsvector column exists
        check_result = await db.query_raw(
            """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = $1 AND column_name = 'content_tsvector'
            """,
            table_name
        )
        
        if not check_result:
            # Add tsvector column
            await db.query_raw(
                f"""
                ALTER TABLE {table_name} 
                ADD COLUMN IF NOT EXISTS content_tsvector tsvector
                GENERATED ALWAYS AS (to_tsvector('english', {fields.content_field})) STORED
                """
            )
        
        # Check if GIN index exists
        index_result = await db.query_raw(
            """
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = $1 AND indexname = $2
            """,
            table_name,
            f"{table_name}_content_tsvector_gin_idx"
        )
        
        if not index_result:
            # Create GIN index for full-text search
            await db.query_raw(
                f"""
                CREATE INDEX IF NOT EXISTS {table_name}_content_tsvector_gin_idx 
                ON {table_name} USING GIN (content_tsvector)
                """
            )
    except Exception as e:
        # Log but don't fail - index creation is optional
        import logging
        logging.debug(f"Full-text index setup note: {e}")


async def update_vector_store_stats_on_delete(
    vector_store_id: str,
    deleted_count: int,
    deleted_content_length: int
) -> None:
    """
    Update vector store statistics after embedding deletion.
    
    Args:
        vector_store_id: ID of the vector store
        deleted_count: Number of embeddings deleted
        deleted_content_length: Total length of deleted content
    """
    vector_store_table = settings.table_names["vector_stores"]
    fields = settings.db_fields
    
    await db.query_raw(
        f"""
        UPDATE {vector_store_table} 
        SET file_counts = jsonb_set(
                COALESCE(file_counts, '{{"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0}}'::jsonb),
                '{{completed}}',
                GREATEST((COALESCE(file_counts->>'completed', '0')::int - $2), 0)::text::jsonb
            ),
            file_counts = jsonb_set(
                file_counts,
                '{{total}}',
                GREATEST((COALESCE(file_counts->>'total', '0')::int - $2), 0)::text::jsonb
            ),
            usage_bytes = GREATEST(COALESCE(usage_bytes, 0) - $3, 0),
            last_active_at = NOW()
        WHERE id = $1
        """,
        vector_store_id,
        deleted_count,
        deleted_content_length
    )


async def update_vector_store_stats_on_update(
    vector_store_id: str,
    old_content_length: int,
    new_content_length: int
) -> None:
    """
    Update vector store statistics after embedding update.
    
    Args:
        vector_store_id: ID of the vector store
        old_content_length: Length of old content
        new_content_length: Length of new content
    """
    vector_store_table = settings.table_names["vector_stores"]
    length_diff = new_content_length - old_content_length
    
    await db.query_raw(
        f"""
        UPDATE {vector_store_table} 
        SET usage_bytes = GREATEST(COALESCE(usage_bytes, 0) + $2, 0),
            last_active_at = NOW()
        WHERE id = $1
        """,
        vector_store_id,
        length_diff
    )


async def log_metrics_async(
    vector_store_id: Optional[str],
    endpoint: str,
    method: str,
    response_time_ms: int,
    status_code: int
) -> None:
    """
    Asynchronously log metrics to usage_logs table.
    This function should be called in a fire-and-forget manner.
    
    Args:
        vector_store_id: ID of the vector store (None for global endpoints)
        endpoint: Endpoint name (e.g., "search", "create_embedding")
        method: HTTP method (e.g., "POST", "GET")
        response_time_ms: Response time in milliseconds
        status_code: HTTP status code
    """
    try:
        await db.query_raw(
            """
            INSERT INTO usage_logs (vector_store_id, endpoint, method, response_time_ms, status_code, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            """,
            vector_store_id,
            endpoint,
            method,
            response_time_ms,
            status_code
        )
    except Exception:
        # Silently fail metrics logging to avoid impacting main request flow
        pass


@app.post("/v1/vector_stores", response_model=VectorStoreResponse)
async def create_vector_store(
    request: VectorStoreCreateRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Create a new vector store.
    """
    try:
        # Use raw SQL to insert the vector store with configurable table/field names
        vector_store_table = settings.table_names["vector_stores"]
        
        result = await db.query_raw(
            f"""
            INSERT INTO {vector_store_table} (id, name, file_counts, status, usage_bytes, expires_after, metadata, created_at)
            VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, NOW())
            RETURNING id, name, file_counts, status, usage_bytes, expires_after, expires_at, last_active_at, metadata, 
                     EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
            """,
            request.name,
            {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            "completed",
            0,
            request.expires_after,
            request.metadata or {}
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create vector store")
            
        vector_store = result[0]
        
        # Convert to response format
        created_at = int(vector_store["created_at_timestamp"])
        expires_at = int(vector_store["expires_at"].timestamp()) if vector_store.get("expires_at") else None
        last_active_at = int(vector_store["last_active_at"].timestamp()) if vector_store.get("last_active_at") else None
        
        return VectorStoreResponse(
            id=vector_store["id"],
            created_at=created_at,
            name=vector_store["name"],
            usage_bytes=vector_store["usage_bytes"] or 0,
            file_counts=vector_store["file_counts"] or {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            status=vector_store["status"],
            expires_after=vector_store["expires_after"],
            expires_at=expires_at,
            last_active_at=last_active_at,
            metadata=vector_store["metadata"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {str(e)}")


@app.get("/v1/vector_stores", response_model=VectorStoreListResponse)
async def list_vector_stores(
    limit: Optional[int] = 20,
    after: Optional[str] = None,
    before: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    List vector stores with optional pagination.
    """
    try:
        limit = min(limit or 20, 100)  # Cap at 100 results
        
        vector_store_table = settings.table_names["vector_stores"]
        
        # Build base query
        base_query = f"""
        SELECT id, name, file_counts, status, usage_bytes, expires_after, expires_at, last_active_at, metadata,
               EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
        FROM {vector_store_table}
        """
        
        # Add pagination conditions
        conditions = []
        params = []
        param_count = 1
        
        if after:
            conditions.append(f"id > ${param_count}")
            params.append(after)
            param_count += 1
            
        if before:
            conditions.append(f"id < ${param_count}")
            params.append(before)
            param_count += 1
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        # Add ordering and limit
        final_query = base_query + f" ORDER BY created_at DESC LIMIT {limit + 1}"
        
        # Execute query
        results = await db.query_raw(final_query, *params)
        
        # Check if there are more results
        has_more = len(results) > limit
        if has_more:
            results = results[:limit]  # Remove extra result
        
        # Convert to response format
        vector_stores = []
        for row in results:
            created_at = int(row["created_at_timestamp"])
            expires_at = int(row["expires_at"].timestamp()) if row.get("expires_at") else None
            last_active_at = int(row["last_active_at"].timestamp()) if row.get("last_active_at") else None
            
            vector_store = VectorStoreResponse(
                id=row["id"],
                created_at=created_at,
                name=row["name"],
                usage_bytes=row["usage_bytes"] or 0,
                file_counts=row["file_counts"] or {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
                status=row["status"],
                expires_after=row["expires_after"],
                expires_at=expires_at,
                last_active_at=last_active_at,
                metadata=row["metadata"]
            )
            vector_stores.append(vector_store)
        
        # Determine first_id and last_id
        first_id = vector_stores[0].id if vector_stores else None
        last_id = vector_stores[-1].id if vector_stores else None
        
        return VectorStoreListResponse(
            data=vector_stores,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to list vector stores: {str(e)}")


@app.post("/v1/vector_stores/{vector_store_id}/search", response_model=VectorStoreSearchResponse)
@app.post("/vector_stores/{vector_store_id}/search", response_model=VectorStoreSearchResponse)
async def search_vector_store(
    vector_store_id: str,
    request: VectorStoreSearchRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Search a vector store for similar content.
    Supports vector-only, keyword-only, and hybrid search modes.
    """
    start_time = time.time()
    status_code = 200
    try:
        # Validate search mode
        search_mode = request.search_mode or "hybrid"
        if search_mode not in ["vector_only", "keyword_only", "hybrid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid search_mode: {search_mode}. Must be 'vector_only', 'keyword_only', or 'hybrid'"
            )
        
        # Validate and normalize weights
        vector_weight = request.vector_weight or 0.7
        keyword_weight = request.keyword_weight or 0.3
        
        # Normalize weights to sum to 1.0 if both are provided
        if search_mode == "hybrid":
            total_weight = vector_weight + keyword_weight
            if total_weight > 0:
                vector_weight = vector_weight / total_weight
                keyword_weight = keyword_weight / total_weight
        elif search_mode == "vector_only":
            vector_weight = 1.0
            keyword_weight = 0.0
        else:  # keyword_only
            vector_weight = 0.0
            keyword_weight = 1.0
        
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        # Build the raw SQL query
        limit = min(request.limit or 20, 100)  # Cap at 100 results
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        param_count = 1
        query_params = []
        select_parts = [
            f"{fields.id_field}",
            f"{fields.content_field}",
            f"{fields.metadata_field}"
        ]
        
        # Build query based on search mode
        if search_mode == "vector_only" or search_mode == "hybrid":
            # Generate embedding for query
            query_embedding = await generate_query_embedding(request.query)
            query_vector_str = "[" + ",".join(map(str, query_embedding)) + "]"
            query_params.append(query_vector_str)
            
            # Vector similarity score (convert distance to similarity)
            # Cosine distance ranges from 0 (identical) to 2 (opposite)
            # We'll normalize to 0-1 scale
            select_parts.append(
                f"CASE WHEN {fields.embedding_field} IS NOT NULL "
                f"THEN GREATEST(0, 1 - (({fields.embedding_field} <=> ${param_count}::vector) / 2.0)) "
                f"ELSE 0.0 END as vector_score"
            )
            param_count += 1
        
        if search_mode == "keyword_only" or search_mode == "hybrid":
            # Prepare query text for full-text search (plainto_tsquery for phrase search)
            query_params.append(request.query)
            
            # Keyword relevance score using ts_rank_cd
            # ts_rank_cd returns values typically in 0-1 range, but can exceed 1.0
            # We cap at 1.0 for consistency with vector scores
            select_parts.append(
                f"CASE WHEN content_tsvector IS NOT NULL "
                f"THEN LEAST(1.0, ts_rank_cd(content_tsvector, plainto_tsquery('english', ${param_count}))) "
                f"ELSE 0.0 END as keyword_score"
            )
            param_count += 1
        
        # Build WHERE clause
        query_params.append(vector_store_id)
        where_clause = f"WHERE {fields.vector_store_id_field} = ${param_count}"
        param_count += 1
        
        # Add metadata filters if provided
        if request.filters:
            filter_conditions = []
            for key, value in request.filters.items():
                filter_conditions.append(f"{fields.metadata_field}->>${param_count} = ${param_count + 1}")
                query_params.extend([key, str(value)])
                param_count += 2
            if filter_conditions:
                where_clause += " AND " + " AND ".join(filter_conditions)
        
        # Build final score calculation for hybrid mode
        if search_mode == "hybrid":
            select_parts.append(
                f"((vector_score * {vector_weight}) + (keyword_score * {keyword_weight})) as final_score"
            )
            order_by = "ORDER BY final_score DESC"
        elif search_mode == "vector_only":
            order_by = "ORDER BY vector_score DESC"
        else:  # keyword_only
            order_by = "ORDER BY keyword_score DESC"
        
        # Build complete query
        base_query = f"""
        SELECT {", ".join(select_parts)}
        FROM {table_name}
        {where_clause}
        {order_by}
        LIMIT {limit}
        """
        
        # Execute the query
        results = await db.query_raw(base_query, *query_params)
        
        # Convert results to SearchResult objects
        search_results = []
        for row in results:
            # Get score based on search mode
            if search_mode == "hybrid":
                score = float(row.get('final_score', 0.0))
            elif search_mode == "vector_only":
                score = float(row.get('vector_score', 0.0))
            else:  # keyword_only
                # Get keyword score (already normalized in SQL)
                score = float(row.get('keyword_score', 0.0))
            
            # Extract filename from metadata or use a default
            metadata = row[fields.metadata_field] or {}
            filename = metadata.get('filename', 'document.txt')
            
            content_chunks = [ContentChunk(type="text", text=row[fields.content_field])]
            
            result = SearchResult(
                file_id=row[fields.id_field],
                filename=filename,
                score=score,
                attributes=metadata if request.return_metadata else None,
                content=content_chunks
            )
            search_results.append(result)
        
        response = VectorStoreSearchResponse(
            search_query=request.query,
            data=search_results,
            has_more=False,  # TODO: Implement pagination
            next_page=None
        )
        
        # Log metrics asynchronously (fire-and-forget)
        response_time_ms = int((time.time() - start_time) * 1000)
        asyncio.create_task(
            log_metrics_async(vector_store_id, "search", "POST", response_time_ms, status_code)
        )
        
        return response
        
    except HTTPException as e:
        status_code = e.status_code
        response_time_ms = int((time.time() - start_time) * 1000)
        asyncio.create_task(
            log_metrics_async(vector_store_id, "search", "POST", response_time_ms, status_code)
        )
        raise
    except Exception as e:
        status_code = 500
        response_time_ms = int((time.time() - start_time) * 1000)
        asyncio.create_task(
            log_metrics_async(vector_store_id, "search", "POST", response_time_ms, status_code)
        )
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/v1/vector_stores/{vector_store_id}/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    vector_store_id: str,
    request: EmbeddingCreateRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Add a single embedding to a vector store.
    """
    start_time = time.time()
    status_code = 200
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        # Convert embedding to vector string format
        embedding_vector_str = "[" + ",".join(map(str, request.embedding)) + "]"
        
        # Insert embedding using configurable field names
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        result = await db.query_raw(
            f"""
            INSERT INTO {table_name} ({fields.id_field}, {fields.vector_store_id_field}, {fields.content_field}, 
                                     {fields.embedding_field}, {fields.metadata_field}, {fields.created_at_field})
            VALUES (gen_random_uuid(), $1, $2, $3::vector, $4, NOW())
            RETURNING {fields.id_field}, {fields.vector_store_id_field}, {fields.content_field}, 
                     {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
            """,
            vector_store_id,
            request.content,
            embedding_vector_str,
            request.metadata or {}
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create embedding")
            
        embedding = result[0]
        
        # Update vector store statistics
        await db.query_raw(
            f"""
            UPDATE {vector_store_table} 
            SET file_counts = jsonb_set(
                    COALESCE(file_counts, '{{"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0}}'::jsonb),
                    '{{completed}}',
                    (COALESCE(file_counts->>'completed', '0')::int + 1)::text::jsonb
                ),
                file_counts = jsonb_set(
                    file_counts,
                    '{{total}}',
                    (COALESCE(file_counts->>'total', '0')::int + 1)::text::jsonb
                ),
                usage_bytes = COALESCE(usage_bytes, 0) + LENGTH($2),
                last_active_at = NOW()
            WHERE id = $1
            """,
            vector_store_id,
            request.content
        )
        
        response = EmbeddingResponse(
            id=embedding[fields.id_field],
            vector_store_id=embedding[fields.vector_store_id_field],
            content=embedding[fields.content_field],
            metadata=embedding[fields.metadata_field],
            created_at=int(embedding["created_at_timestamp"])
        )
        
        # Log metrics asynchronously
        response_time_ms = int((time.time() - start_time) * 1000)
        asyncio.create_task(
            log_metrics_async(vector_store_id, "create_embedding", "POST", response_time_ms, status_code)
        )
        
        return response
        
    except HTTPException as e:
        status_code = e.status_code
        response_time_ms = int((time.time() - start_time) * 1000)
        asyncio.create_task(
            log_metrics_async(vector_store_id, "create_embedding", "POST", response_time_ms, status_code)
        )
        raise
    except Exception as e:
        status_code = 500
        response_time_ms = int((time.time() - start_time) * 1000)
        asyncio.create_task(
            log_metrics_async(vector_store_id, "create_embedding", "POST", response_time_ms, status_code)
        )
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create embedding: {str(e)}")


@app.post("/v1/vector_stores/{vector_store_id}/embeddings/batch", response_model=EmbeddingBatchCreateResponse)
async def create_embeddings_batch(
    vector_store_id: str,
    request: EmbeddingBatchCreateRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Add multiple embeddings to a vector store in batch.
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        if not request.embeddings:
            raise HTTPException(status_code=400, detail="No embeddings provided")
        
        # Prepare batch insert
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        # Build VALUES clause for batch insert
        values_clauses = []
        params = []
        param_count = 1
        
        for embedding_req in request.embeddings:
            embedding_vector_str = "[" + ",".join(map(str, embedding_req.embedding)) + "]"
            values_clauses.append(f"(gen_random_uuid(), ${param_count}, ${param_count + 1}, ${param_count + 2}::vector, ${param_count + 3}, NOW())")
            params.extend([
                vector_store_id,
                embedding_req.content,
                embedding_vector_str,
                embedding_req.metadata or {}
            ])
            param_count += 4
        
        values_clause = ", ".join(values_clauses)
        
        # Execute batch insert
        result = await db.query_raw(
            f"""
            INSERT INTO {table_name} ({fields.id_field}, {fields.vector_store_id_field}, {fields.content_field}, 
                                     {fields.embedding_field}, {fields.metadata_field}, {fields.created_at_field})
            VALUES {values_clause}
            RETURNING {fields.id_field}, {fields.vector_store_id_field}, {fields.content_field}, 
                     {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
            """,
            *params
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create embeddings")
        
        # Calculate total content length for usage bytes update
        total_content_length = sum(len(emb.content) for emb in request.embeddings)
        
        # Update vector store statistics
        await db.query_raw(
            f"""
            UPDATE {vector_store_table} 
            SET file_counts = jsonb_set(
                    COALESCE(file_counts, '{{"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0}}'::jsonb),
                    '{{completed}}',
                    (COALESCE(file_counts->>'completed', '0')::int + $2)::text::jsonb
                ),
                file_counts = jsonb_set(
                    file_counts,
                    '{{total}}',
                    (COALESCE(file_counts->>'total', '0')::int + $2)::text::jsonb
                ),
                usage_bytes = COALESCE(usage_bytes, 0) + $3,
                last_active_at = NOW()
            WHERE id = $1
            """,
            vector_store_id,
            len(request.embeddings),
            total_content_length
        )
        
        # Convert results to response format
        embeddings = []
        for row in result:
            embeddings.append(EmbeddingResponse(
                id=row[fields.id_field],
                vector_store_id=row[fields.vector_store_id_field],
                content=row[fields.content_field],
                metadata=row[fields.metadata_field],
                created_at=int(row["created_at_timestamp"])
            ))
        
        return EmbeddingBatchCreateResponse(
            data=embeddings,
            created=int(time.time())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings batch: {str(e)}")


@app.delete("/v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}", response_model=EmbeddingDeleteResponse)
async def delete_embedding(
    vector_store_id: str,
    embedding_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Delete an individual embedding from a vector store.
    """
    start_time = time.time()
    status_code = 200
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        # Get embedding details before deletion
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        embedding_result = await db.query_raw(
            f"""
            SELECT {fields.id_field}, {fields.content_field}, {fields.vector_store_id_field}
            FROM {table_name}
            WHERE {fields.id_field} = $1 AND {fields.vector_store_id_field} = $2
            """,
            embedding_id,
            vector_store_id
        )
        
        if not embedding_result:
            raise HTTPException(status_code=404, detail="Embedding not found")
        
        embedding = embedding_result[0]
        content_length = len(embedding[fields.content_field])
        
        # Delete the embedding
        delete_result = await db.query_raw(
            f"""
            DELETE FROM {table_name}
            WHERE {fields.id_field} = $1 AND {fields.vector_store_id_field} = $2
            RETURNING {fields.id_field}
            """,
            embedding_id,
            vector_store_id
        )
        
        if not delete_result:
            raise HTTPException(status_code=404, detail="Embedding not found")
        
        # Update vector store statistics
        await update_vector_store_stats_on_delete(vector_store_id, 1, content_length)
        
        response = EmbeddingDeleteResponse(
            id=embedding_id,
            deleted=True
        )
        
        # Log metrics asynchronously
        response_time_ms = int((time.time() - start_time) * 1000)
        asyncio.create_task(
            log_metrics_async(vector_store_id, "delete_embedding", "DELETE", response_time_ms, status_code)
        )
        
        return response
        
    except HTTPException as e:
        status_code = e.status_code
        response_time_ms = int((time.time() - start_time) * 1000)
        asyncio.create_task(
            log_metrics_async(vector_store_id, "delete_embedding", "DELETE", response_time_ms, status_code)
        )
        raise
    except Exception as e:
        status_code = 500
        response_time_ms = int((time.time() - start_time) * 1000)
        asyncio.create_task(
            log_metrics_async(vector_store_id, "delete_embedding", "DELETE", response_time_ms, status_code)
        )
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete embedding: {str(e)}")


@app.put("/v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}", response_model=EmbeddingResponse)
async def update_embedding(
    vector_store_id: str,
    embedding_id: str,
    request: EmbeddingUpdateRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Update an embedding (full update - PUT).
    All fields that are provided will be updated.
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        # Get existing embedding
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        existing_result = await db.query_raw(
            f"""
            SELECT {fields.id_field}, {fields.content_field}, {fields.embedding_field}, 
                   {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
            FROM {table_name}
            WHERE {fields.id_field} = $1 AND {fields.vector_store_id_field} = $2
            """,
            embedding_id,
            vector_store_id
        )
        
        if not existing_result:
            raise HTTPException(status_code=404, detail="Embedding not found")
        
        existing = existing_result[0]
        old_content_length = len(existing[fields.content_field])
        
        # Prepare update values
        new_content = request.content if request.content is not None else existing[fields.content_field]
        new_metadata = request.metadata if request.metadata is not None else (existing[fields.metadata_field] or {})
        
        # Build update query dynamically
        update_parts = []
        params = []
        param_count = 1
        
        # Always update content and metadata
        update_parts.append(f"{fields.content_field} = ${param_count}")
        params.append(new_content)
        param_count += 1
        
        # Handle embedding update
        if request.embedding is not None:
            # Validate embedding dimensions
            if len(request.embedding) != settings.embedding.dimensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid embedding dimensions. Expected {settings.embedding.dimensions}, got {len(request.embedding)}"
                )
            embedding_vector_str = "[" + ",".join(map(str, request.embedding)) + "]"
            update_parts.append(f"{fields.embedding_field} = ${param_count}::vector")
            params.append(embedding_vector_str)
            param_count += 1
        
        update_parts.append(f"{fields.metadata_field} = ${param_count}")
        params.append(new_metadata)
        param_count += 1
        
        # Build and execute update query
        update_query = f"""
        UPDATE {table_name}
        SET {", ".join(update_parts)}
        WHERE {fields.id_field} = ${param_count} AND {fields.vector_store_id_field} = ${param_count + 1}
        RETURNING {fields.id_field}, {fields.vector_store_id_field}, {fields.content_field}, 
                 {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
        """
        params.extend([embedding_id, vector_store_id])
        
        result = await db.query_raw(update_query, *params)
        
        if not result:
            raise HTTPException(status_code=404, detail="Embedding not found or update failed")
        
        updated = result[0]
        new_content_length = len(new_content)
        
        # Update vector store statistics if content length changed
        if old_content_length != new_content_length:
            await update_vector_store_stats_on_update(vector_store_id, old_content_length, new_content_length)
        
        return EmbeddingResponse(
            id=updated[fields.id_field],
            vector_store_id=updated[fields.vector_store_id_field],
            content=updated[fields.content_field],
            metadata=updated[fields.metadata_field],
            created_at=int(updated["created_at_timestamp"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to update embedding: {str(e)}")


@app.patch("/v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}", response_model=EmbeddingResponse)
async def patch_embedding(
    vector_store_id: str,
    embedding_id: str,
    request: EmbeddingPatchRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Partially update an embedding (PATCH).
    Only provided fields will be updated.
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        # Get existing embedding
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        existing_result = await db.query_raw(
            f"""
            SELECT {fields.id_field}, {fields.content_field}, {fields.embedding_field}, 
                   {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
            FROM {table_name}
            WHERE {fields.id_field} = $1 AND {fields.vector_store_id_field} = $2
            """,
            embedding_id,
            vector_store_id
        )
        
        if not existing_result:
            raise HTTPException(status_code=404, detail="Embedding not found")
        
        existing = existing_result[0]
        old_content_length = len(existing[fields.content_field])
        
        # Prepare update values - only update provided fields
        new_content = request.content if request.content is not None else existing[fields.content_field]
        
        # Merge metadata if provided
        existing_metadata = existing[fields.metadata_field] or {}
        if request.metadata is not None:
            new_metadata = {**existing_metadata, **request.metadata}
        else:
            new_metadata = existing_metadata
        
        # Build update query dynamically
        update_parts = []
        params = []
        param_count = 1
        
        if request.content is not None:
            update_parts.append(f"{fields.content_field} = ${param_count}")
            params.append(new_content)
            param_count += 1
        
        if request.embedding is not None:
            # Validate embedding dimensions
            if len(request.embedding) != settings.embedding.dimensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid embedding dimensions. Expected {settings.embedding.dimensions}, got {len(request.embedding)}"
                )
            embedding_vector_str = "[" + ",".join(map(str, request.embedding)) + "]"
            update_parts.append(f"{fields.embedding_field} = ${param_count}::vector")
            params.append(embedding_vector_str)
            param_count += 1
        
        if request.metadata is not None:
            update_parts.append(f"{fields.metadata_field} = ${param_count}")
            params.append(new_metadata)
            param_count += 1
        
        if not update_parts:
            # No fields to update, return existing embedding
            return EmbeddingResponse(
                id=existing[fields.id_field],
                vector_store_id=vector_store_id,
                content=existing[fields.content_field],
                metadata=existing_metadata,
                created_at=int(existing["created_at_timestamp"])
            )
        
        # Build and execute update query
        update_query = f"""
        UPDATE {table_name}
        SET {", ".join(update_parts)}
        WHERE {fields.id_field} = ${param_count} AND {fields.vector_store_id_field} = ${param_count + 1}
        RETURNING {fields.id_field}, {fields.vector_store_id_field}, {fields.content_field}, 
                 {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
        """
        params.extend([embedding_id, vector_store_id])
        
        result = await db.query_raw(update_query, *params)
        
        if not result:
            raise HTTPException(status_code=404, detail="Embedding not found or update failed")
        
        updated = result[0]
        new_content_length = len(new_content)
        
        # Update vector store statistics if content length changed
        if old_content_length != new_content_length:
            await update_vector_store_stats_on_update(vector_store_id, old_content_length, new_content_length)
        
        return EmbeddingResponse(
            id=updated[fields.id_field],
            vector_store_id=updated[fields.vector_store_id_field],
            content=updated[fields.content_field],
            metadata=updated[fields.metadata_field],
            created_at=int(updated["created_at_timestamp"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to patch embedding: {str(e)}")


@app.delete("/v1/vector_stores/{vector_store_id}/embeddings/batch", response_model=EmbeddingBatchDeleteResponse)
async def delete_embeddings_batch(
    vector_store_id: str,
    request: EmbeddingBatchDeleteRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Batch delete embeddings by IDs or filters.
    """
    start_time = time.time()
    status_code = 200
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        # Build WHERE clause
        conditions = [f"{fields.vector_store_id_field} = $1"]
        params = [vector_store_id]
        param_count = 2
        
        if request.embedding_ids:
            # Delete by IDs
            placeholders = [f"${i}" for i in range(param_count, param_count + len(request.embedding_ids))]
            conditions.append(f"{fields.id_field} = ANY(ARRAY[{', '.join(placeholders)}])")
            params.extend(request.embedding_ids)
            param_count += len(request.embedding_ids)
        elif request.filters:
            # Delete by metadata filters
            for key, value in request.filters.items():
                conditions.append(f"{fields.metadata_field}->>${param_count} = ${param_count + 1}")
                params.extend([key, str(value)])
                param_count += 2
        else:
            raise HTTPException(status_code=400, detail="Either embedding_ids or filters must be provided")
        
        # Get embeddings to delete (for stats calculation)
        select_query = f"""
        SELECT {fields.id_field}, {fields.content_field}
        FROM {table_name}
        WHERE {' AND '.join(conditions)}
        """
        embeddings_to_delete = await db.query_raw(select_query, *params)
        
        if not embeddings_to_delete:
            return EmbeddingBatchDeleteResponse(
                deleted_ids=[],
                deleted_count=0
            )
        
        deleted_ids = [emb[fields.id_field] for emb in embeddings_to_delete]
        total_content_length = sum(len(emb[fields.content_field]) for emb in embeddings_to_delete)
        
        # Delete the embeddings
        delete_query = f"""
        DELETE FROM {table_name}
        WHERE {' AND '.join(conditions)}
        RETURNING {fields.id_field}
        """
        deleted_result = await db.query_raw(delete_query, *params)
        
        deleted_count = len(deleted_result)
        
        # Update vector store statistics
        if deleted_count > 0:
            await update_vector_store_stats_on_delete(vector_store_id, deleted_count, total_content_length)
        
        response = EmbeddingBatchDeleteResponse(
            deleted_ids=deleted_ids[:deleted_count],  # Only return IDs that were actually deleted
            deleted_count=deleted_count
        )
        
        # Log metrics asynchronously
        response_time_ms = int((time.time() - start_time) * 1000)
        asyncio.create_task(
            log_metrics_async(vector_store_id, "delete_embeddings_batch", "DELETE", response_time_ms, status_code)
        )
        
        return response
        
    except HTTPException as e:
        status_code = e.status_code
        response_time_ms = int((time.time() - start_time) * 1000)
        asyncio.create_task(
            log_metrics_async(vector_store_id, "delete_embeddings_batch", "DELETE", response_time_ms, status_code)
        )
        raise
    except Exception as e:
        status_code = 500
        response_time_ms = int((time.time() - start_time) * 1000)
        asyncio.create_task(
            log_metrics_async(vector_store_id, "delete_embeddings_batch", "DELETE", response_time_ms, status_code)
        )
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete embeddings batch: {str(e)}")


@app.delete("/v1/vector_stores/{vector_store_id}", response_model=VectorStoreDeleteResponse)
async def delete_vector_store(
    vector_store_id: str,
    api_key: str = Depends(get_api_key),
    cascade: bool = True  # Default to cascade delete
):
    """
    Delete a vector store.
    By default, cascades to delete all embeddings. Set cascade=False to prevent cascade.
    """
    try:
        vector_store_table = settings.table_names["vector_stores"]
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        # Check if vector store exists
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        # cascade parameter already set
        
        embeddings_deleted_count = 0
        
        if cascade:
            # Count embeddings before deletion
            count_result = await db.query_raw(
                f"""
                SELECT COUNT(*) as count
                FROM {table_name}
                WHERE {fields.vector_store_id_field} = $1
                """,
                vector_store_id
            )
            embeddings_deleted_count = count_result[0]["count"] if count_result else 0
            
            # Delete embeddings first (cascade)
            await db.query_raw(
                f"""
                DELETE FROM {table_name}
                WHERE {fields.vector_store_id_field} = $1
                """,
                vector_store_id
            )
        
        # Delete vector store
        delete_result = await db.query_raw(
            f"""
            DELETE FROM {vector_store_table}
            WHERE id = $1
            RETURNING id
            """,
            vector_store_id
        )
        
        if not delete_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        return VectorStoreDeleteResponse(
            id=vector_store_id,
            deleted=True,
            embeddings_deleted_count=embeddings_deleted_count if cascade else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete vector store: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": int(time.time())}


@app.get("/v1/vector_stores/{vector_store_id}/stats", response_model=VectorStoreStatsResponse)
async def get_vector_store_stats(
    vector_store_id: str,
    period: Optional[str] = "daily",  # "daily", "weekly", "monthly", "all"
    api_key: str = Depends(get_api_key)
):
    """
    Get usage statistics for a specific vector store.
    """
    try:
        # Validate period
        valid_periods = ["daily", "weekly", "monthly", "all"]
        if period not in valid_periods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid period: {period}. Must be one of: {', '.join(valid_periods)}"
            )
        
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        # Calculate time range
        now = int(time.time())
        start_time = None
        end_time = now
        
        if period == "daily":
            start_time = now - 86400  # 24 hours
        elif period == "weekly":
            start_time = now - 604800  # 7 days
        elif period == "monthly":
            start_time = now - 2592000  # 30 days
        # "all" doesn't set start_time
        
        # Build time filter
        time_filter = ""
        params = [vector_store_id]
        if start_time:
            time_filter = f"AND created_at >= TO_TIMESTAMP(${len(params) + 1})"
            params.append(start_time)
        
        # Get overall stats from usage_logs
        stats_query = f"""
        SELECT 
            COUNT(*) as total_requests,
            COUNT(*) FILTER (WHERE endpoint = 'search') as search_queries,
            COUNT(*) FILTER (WHERE endpoint = 'create_embedding') as embeddings_created,
            COUNT(*) FILTER (WHERE endpoint = 'delete_embedding' OR endpoint = 'delete_embeddings_batch') as embeddings_deleted,
            AVG(response_time_ms) as avg_response_time_ms,
            COUNT(*) FILTER (WHERE status_code >= 400) as error_count,
            COUNT(*) as total_with_errors
        FROM usage_logs
        WHERE vector_store_id = $1 {time_filter}
        """
        
        stats_result = await db.query_raw(stats_query, *params)
        
        if not stats_result or not stats_result[0]:
            # Return empty stats
            return VectorStoreStatsResponse(
                vector_store_id=vector_store_id,
                period=period,
                start_time=start_time,
                end_time=end_time
            )
        
        stats = stats_result[0]
        total_requests = stats.get("total_requests", 0) or 0
        search_queries = stats.get("search_queries", 0) or 0
        embeddings_created = stats.get("embeddings_created", 0) or 0
        embeddings_deleted = stats.get("embeddings_deleted", 0) or 0
        avg_response_time_ms = float(stats.get("avg_response_time_ms", 0) or 0)
        error_count = stats.get("error_count", 0) or 0
        error_rate = (error_count / total_requests) if total_requests > 0 else 0.0
        
        # Get storage bytes from vector_stores table
        storage_result = await db.query_raw(
            f"SELECT usage_bytes FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        storage_bytes = storage_result[0]["usage_bytes"] if storage_result else 0
        
        # Get endpoint breakdown
        endpoint_query = f"""
        SELECT 
            endpoint,
            COUNT(*) as count,
            AVG(response_time_ms) as avg_response_time_ms,
            COUNT(*) FILTER (WHERE status_code >= 400) as error_count
        FROM usage_logs
        WHERE vector_store_id = $1 {time_filter}
        GROUP BY endpoint
        """
        
        endpoint_result = await db.query_raw(endpoint_query, *params)
        endpoint_stats = {}
        for row in endpoint_result:
            endpoint_stats[row["endpoint"]] = {
                "count": row["count"],
                "avg_response_time_ms": float(row["avg_response_time_ms"] or 0),
                "error_count": row["error_count"] or 0
            }
        
        return VectorStoreStatsResponse(
            vector_store_id=vector_store_id,
            period=period,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            search_queries=search_queries,
            embeddings_created=embeddings_created,
            embeddings_deleted=embeddings_deleted,
            storage_bytes=storage_bytes or 0,
            avg_response_time_ms=avg_response_time_ms,
            error_count=error_count,
            error_rate=error_rate,
            endpoint_stats=endpoint_stats if endpoint_stats else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get vector store stats: {str(e)}")


@app.get("/v1/stats", response_model=GlobalStatsResponse)
async def get_global_stats(
    period: Optional[str] = "daily",  # "daily", "weekly", "monthly", "all"
    api_key: str = Depends(get_api_key)
):
    """
    Get global usage statistics across all vector stores.
    """
    try:
        # Validate period
        valid_periods = ["daily", "weekly", "monthly", "all"]
        if period not in valid_periods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid period: {period}. Must be one of: {', '.join(valid_periods)}"
            )
        
        # Calculate time range
        now = int(time.time())
        start_time = None
        end_time = now
        
        if period == "daily":
            start_time = now - 86400
        elif period == "weekly":
            start_time = now - 604800
        elif period == "monthly":
            start_time = now - 2592000
        
        # Build time filter
        time_filter = ""
        params = []
        if start_time:
            time_filter = f"AND created_at >= TO_TIMESTAMP(${len(params) + 1})"
            params.append(start_time)
        
        # Get overall stats from usage_logs
        stats_query = f"""
        SELECT 
            COUNT(*) as total_requests,
            AVG(response_time_ms) as avg_response_time_ms,
            COUNT(*) FILTER (WHERE status_code >= 400) as error_count,
            COUNT(DISTINCT vector_store_id) as distinct_vector_stores
        FROM usage_logs
        WHERE 1=1 {time_filter}
        """
        
        stats_result = await db.query_raw(stats_query, *params)
        
        # Get vector store counts
        vector_store_table = settings.table_names["vector_stores"]
        embeddings_table = settings.table_names["embeddings"]
        
        vs_result = await db.query_raw(f"SELECT COUNT(*) as count FROM {vector_store_table}")
        total_vector_stores = vs_result[0]["count"] if vs_result else 0
        
        emb_result = await db.query_raw(f"SELECT COUNT(*) as count FROM {embeddings_table}")
        total_embeddings = emb_result[0]["count"] if emb_result else 0
        
        # Get total storage
        storage_result = await db.query_raw(
            f"SELECT COALESCE(SUM(usage_bytes), 0) as total FROM {vector_store_table}"
        )
        total_storage_bytes = storage_result[0]["total"] if storage_result else 0
        
        if not stats_result or not stats_result[0]:
            return GlobalStatsResponse(
                period=period,
                start_time=start_time,
                end_time=end_time,
                total_vector_stores=total_vector_stores,
                total_embeddings=total_embeddings,
                total_storage_bytes=total_storage_bytes or 0
            )
        
        stats = stats_result[0]
        total_requests = stats.get("total_requests", 0) or 0
        avg_response_time_ms = float(stats.get("avg_response_time_ms", 0) or 0)
        error_count = stats.get("error_count", 0) or 0
        error_rate = (error_count / total_requests) if total_requests > 0 else 0.0
        
        # Get endpoint breakdown
        endpoint_query = f"""
        SELECT 
            endpoint,
            COUNT(*) as count,
            AVG(response_time_ms) as avg_response_time_ms,
            COUNT(*) FILTER (WHERE status_code >= 400) as error_count
        FROM usage_logs
        WHERE 1=1 {time_filter}
        GROUP BY endpoint
        """
        
        endpoint_result = await db.query_raw(endpoint_query, *params)
        endpoint_stats = {}
        for row in endpoint_result:
            endpoint_stats[row["endpoint"]] = {
                "count": row["count"],
                "avg_response_time_ms": float(row["avg_response_time_ms"] or 0),
                "error_count": row["error_count"] or 0
            }
        
        # Get top vector stores by request count
        top_vs_query = f"""
        SELECT 
            vector_store_id,
            COUNT(*) as request_count,
            AVG(response_time_ms) as avg_response_time_ms
        FROM usage_logs
        WHERE vector_store_id IS NOT NULL {time_filter}
        GROUP BY vector_store_id
        ORDER BY request_count DESC
        LIMIT 10
        """
        
        top_vs_result = await db.query_raw(top_vs_query, *params)
        top_vector_stores = []
        for row in top_vs_result:
            top_vector_stores.append({
                "vector_store_id": row["vector_store_id"],
                "request_count": row["request_count"],
                "avg_response_time_ms": float(row["avg_response_time_ms"] or 0)
            })
        
        return GlobalStatsResponse(
            period=period,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            total_vector_stores=total_vector_stores,
            total_embeddings=total_embeddings,
            total_storage_bytes=total_storage_bytes or 0,
            avg_response_time_ms=avg_response_time_ms,
            error_count=error_count,
            error_rate=error_rate,
            endpoint_stats=endpoint_stats if endpoint_stats else None,
            top_vector_stores=top_vector_stores if top_vector_stores else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get global stats: {str(e)}")


@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus-compatible metrics endpoint.
    Returns metrics in Prometheus text format.
    """
    try:
        # Get global stats for the last hour
        now = int(time.time())
        start_time = now - 3600  # Last hour
        
        # Get basic metrics
        stats_query = """
        SELECT 
            COUNT(*) as total_requests,
            AVG(response_time_ms) as avg_response_time_ms,
            COUNT(*) FILTER (WHERE status_code >= 400) as error_count,
            COUNT(*) FILTER (WHERE status_code < 400) as success_count
        FROM usage_logs
        WHERE created_at >= TO_TIMESTAMP($1)
        """
        
        stats_result = await db.query_raw(stats_query, start_time)
        
        # Get endpoint-specific metrics
        endpoint_query = """
        SELECT 
            endpoint,
            COUNT(*) as count,
            AVG(response_time_ms) as avg_response_time_ms,
            COUNT(*) FILTER (WHERE status_code >= 400) as error_count
        FROM usage_logs
        WHERE created_at >= TO_TIMESTAMP($1)
        GROUP BY endpoint
        """
        
        endpoint_result = await db.query_raw(endpoint_query, start_time)
        
        # Build Prometheus format
        lines = [
            "# HELP vector_store_api_requests_total Total number of requests",
            "# TYPE vector_store_api_requests_total counter",
            f"vector_store_api_requests_total {stats_result[0]['total_requests'] if stats_result and stats_result[0] else 0}",
            "",
            "# HELP vector_store_api_request_duration_ms Average request duration in milliseconds",
            "# TYPE vector_store_api_request_duration_ms gauge",
            f"vector_store_api_request_duration_ms {float(stats_result[0]['avg_response_time_ms'] or 0) if stats_result and stats_result[0] else 0}",
            "",
            "# HELP vector_store_api_errors_total Total number of errors",
            "# TYPE vector_store_api_errors_total counter",
            f"vector_store_api_errors_total {stats_result[0]['error_count'] if stats_result and stats_result[0] else 0}",
            "",
            "# HELP vector_store_api_success_total Total number of successful requests",
            "# TYPE vector_store_api_success_total counter",
            f"vector_store_api_success_total {stats_result[0]['success_count'] if stats_result and stats_result[0] else 0}",
            ""
        ]
        
        # Add endpoint-specific metrics
        if endpoint_result:
            lines.append("# HELP vector_store_api_endpoint_requests_total Total requests per endpoint")
            lines.append("# TYPE vector_store_api_endpoint_requests_total counter")
            for row in endpoint_result:
                endpoint = row["endpoint"]
                count = row["count"]
                lines.append(f'vector_store_api_endpoint_requests_total{{endpoint="{endpoint}"}} {count}')
            
            lines.append("")
            lines.append("# HELP vector_store_api_endpoint_duration_ms Average duration per endpoint")
            lines.append("# TYPE vector_store_api_endpoint_duration_ms gauge")
            for row in endpoint_result:
                endpoint = row["endpoint"]
                avg_time = float(row["avg_response_time_ms"] or 0)
                lines.append(f'vector_store_api_endpoint_duration_ms{{endpoint="{endpoint}"}} {avg_time}')
            
            lines.append("")
            lines.append("# HELP vector_store_api_endpoint_errors_total Total errors per endpoint")
            lines.append("# TYPE vector_store_api_endpoint_errors_total counter")
            for row in endpoint_result:
                endpoint = row["endpoint"]
                error_count = row["error_count"] or 0
                lines.append(f'vector_store_api_endpoint_errors_total{{endpoint="{endpoint}"}} {error_count}')
        
        from fastapi.responses import Response
        return Response(content="\n".join(lines), media_type="text/plain")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings_manager.effective.host, port=settings_manager.effective.port, reload=True)


# -----------------------------
# Admin settings API (secured)
# -----------------------------

from fastapi import APIRouter

admin_router = APIRouter(prefix="/v1/admin", dependencies=[Depends(get_api_key)])


def _redact_secrets(val: Optional[str]) -> Optional[str]:
    if not val:
        return val
    return "***" if len(val) > 0 else val


@admin_router.get("/settings")
async def get_settings():
    eff = settings_manager.effective
    return {
        "server": {
            "host": eff.host,
            "port": eff.port,
        },
        "auth": {
            "server_api_key": _redact_secrets(eff.server_api_key),
        },
        "embedding": {
            "model": eff.embedding.model,
            "base_url": eff.embedding.base_url,
            "api_key": _redact_secrets(eff.embedding.api_key),
            "dimensions": eff.embedding.dimensions,
        },
        "db_fields": eff.db_fields.model_dump(),
    }


@admin_router.get("/settings/schema")
async def get_settings_schema():
    return {
        "groups": {
            "server": {"fields": {"host": {"type": "string"}, "port": {"type": "integer", "min": 1, "max": 65535}}},
            "auth": {"fields": {"server_api_key": {"type": "secret"}}},
            "embedding": {
                "fields": {
                    "model": {"type": "string"},
                    "base_url": {"type": "string"},
                    "api_key": {"type": "secret"},
                    "dimensions": {"type": "integer", "min": 1}
                }
            },
            "db_fields": {
                "fields": {
                    "id_field": {"type": "string"},
                    "content_field": {"type": "string"},
                    "metadata_field": {"type": "string"},
                    "embedding_field": {"type": "string"},
                    "vector_store_id_field": {"type": "string"},
                    "created_at_field": {"type": "string"}
                }
            }
        }
    }


@admin_router.put("/settings")
async def update_settings(payload: dict):
    # Split payload into groups and upsert each as JSON into app_settings
    groups = {k: v for k, v in payload.items() if k in {"server", "auth", "embedding", "db_fields", "cors"}}

    if not groups:
        raise HTTPException(status_code=400, detail="No valid setting groups provided")

    # Build upsert queries
    # INSERT ... ON CONFLICT (key) DO UPDATE SET value=$2, updated_at=NOW()
    for group_key, group_val in groups.items():
        await db.query_raw(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = NOW()
            """,
            group_key,
            group_val,
        )

    # Reload overlay and apply
    await settings_manager.load_overlay_from_db(db)

    return {"status": "ok"}


@admin_router.post("/settings/test")
async def test_connectivity():
    # Test DB
    try:
        _ = await db.query_raw("SELECT 1")
        db_ok = True
    except Exception as e:
        db_ok = False
        db_err = str(e)

    # Test embedding provider with a tiny request
    emb_ok = True
    emb_err = None
    try:
        _ = await embedding_service.generate_embedding("ping")
    except Exception as e:
        emb_ok = False
        emb_err = str(e)

    result = {"database": {"ok": db_ok}, "embedding": {"ok": emb_ok}}
    if not db_ok:
        result["database"]["error"] = db_err
    if not emb_ok:
        result["embedding"]["error"] = emb_err
    return result


app.include_router(admin_router)