from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel
from datetime import datetime


class VectorStoreCreateRequest(BaseModel):
    name: str
    file_ids: Optional[List[str]] = None
    expires_after: Optional[Dict[str, Any]] = None
    chunking_strategy: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorStoreResponse(BaseModel):
    id: str
    object: str = "vector_store"
    created_at: int
    name: str
    usage_bytes: int
    file_counts: Dict[str, int]
    status: str
    expires_after: Optional[Dict[str, Any]] = None
    expires_at: Optional[int] = None
    last_active_at: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorStoreSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 20
    filters: Optional[Dict[str, Any]] = None
    return_metadata: Optional[bool] = True
    search_mode: Optional[str] = "hybrid"  # Options: "vector_only", "keyword_only", "hybrid"
    vector_weight: Optional[float] = 0.7  # Weight for vector similarity (0.0-1.0)
    keyword_weight: Optional[float] = 0.3  # Weight for keyword relevance (0.0-1.0)


class ContentChunk(BaseModel):
    type: str = "text"
    text: str


class SearchResult(BaseModel):
    file_id: str
    filename: str
    score: float
    attributes: Optional[Dict[str, Any]] = None
    content: List[ContentChunk]


class VectorStoreSearchResponse(BaseModel):
    object: str = "vector_store.search_results.page"
    search_query: str
    data: List[SearchResult]
    has_more: bool = False
    next_page: Optional[str] = None


class EmbeddingCreateRequest(BaseModel):
    content: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingResponse(BaseModel):
    id: str
    object: str = "embedding"
    vector_store_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: int


class EmbeddingListItem(BaseModel):
    """List item model for embeddings listing"""
    id: str
    vector_store_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: int


class EmbeddingListResponse(BaseModel):
    """Response model for listing embeddings"""
    object: str = "list"
    data: List[EmbeddingListItem]
    last_id: Optional[str] = None
    has_more: bool = False


class EmbeddingBatchCreateRequest(BaseModel):
    embeddings: List[EmbeddingCreateRequest]


class EmbeddingBatchCreateResponse(BaseModel):
    object: str = "embedding.batch"
    data: List[EmbeddingResponse]
    created: int


class VectorStoreListResponse(BaseModel):
    object: str = "list"
    data: List[VectorStoreResponse]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class EmbeddingUpdateRequest(BaseModel):
    """Request model for PUT (full update) of an embedding"""
    content: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingPatchRequest(BaseModel):
    """Request model for PATCH (partial update) of an embedding"""
    content: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingBatchDeleteRequest(BaseModel):
    """Request model for batch deletion of embeddings"""
    embedding_ids: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None


class EmbeddingDeleteResponse(BaseModel):
    """Response model for embedding deletion"""
    object: str = "embedding.deleted"
    id: str
    deleted: bool = True


class EmbeddingBatchDeleteResponse(BaseModel):
    """Response model for batch deletion"""
    object: str = "embedding.batch_deleted"
    deleted_ids: List[str]
    deleted_count: int


class VectorStoreDeleteRequest(BaseModel):
    """Request model for vector store deletion"""
    cascade: bool = True  # Whether to cascade delete embeddings


class VectorStoreDeleteResponse(BaseModel):
    """Response model for vector store deletion"""
    object: str = "vector_store.deleted"
    id: str
    deleted: bool = True
    embeddings_deleted_count: Optional[int] = None


class VectorStoreStatsResponse(BaseModel):
    """Response model for vector store statistics"""
    object: str = "vector_store.stats"
    vector_store_id: str
    period: str  # "daily", "weekly", "monthly", "all"
    start_time: Optional[int] = None  # Unix timestamp
    end_time: Optional[int] = None  # Unix timestamp
    
    # Metrics
    total_requests: int = 0
    search_queries: int = 0
    embeddings_created: int = 0
    embeddings_deleted: int = 0
    storage_bytes: int = 0
    avg_response_time_ms: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    
    # Breakdown by endpoint
    endpoint_stats: Optional[Dict[str, Dict[str, Any]]] = None


class GlobalStatsResponse(BaseModel):
    """Response model for global statistics"""
    object: str = "stats.global"
    period: str  # "daily", "weekly", "monthly", "all"
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    
    # Global metrics
    total_requests: int = 0
    total_vector_stores: int = 0
    total_embeddings: int = 0
    total_storage_bytes: int = 0
    avg_response_time_ms: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    
    # Breakdown by endpoint
    endpoint_stats: Optional[Dict[str, Dict[str, Any]]] = None
    
    # Breakdown by vector store (top N)
    top_vector_stores: Optional[List[Dict[str, Any]]] = None


class FileIngestOptions(BaseModel):
    """Options for file ingestion and chunking"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    splitter: Literal["tokens", "chars", "lines", "paragraphs"] = "chars"
    max_chunks: Optional[int] = None
    delimiter: str = "\n"  # For CSV/XLSX row concatenation
    sheet: Optional[str] = None  # For XLSX sheet selection
    normalize_whitespace: bool = True
    lowercase: bool = False
    metadata: Optional[Dict[str, Any]] = None


class FileIngestResult(BaseModel):
    """Result for a single file ingestion"""
    file_name: str
    num_chunks: int
    num_embeddings: int
    metadata: Optional[Dict[str, Any]] = None
    warnings: List[str] = []


class FileIngestResponse(BaseModel):
    """Response model for file ingestion"""
    object: str = "file.ingest"
    results: List[FileIngestResult]
    total_files: int
    total_chunks: int
    total_embeddings: int