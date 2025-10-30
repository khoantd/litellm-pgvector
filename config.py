from typing import Dict, Optional, Any, Callable
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class DatabaseFieldConfig(BaseModel):
    """Configuration for database field mappings"""
    id_field: str = "id"
    content_field: str = "content"
    metadata_field: str = "metadata"
    embedding_field: str = "embedding"
    vector_store_id_field: str = "vector_store_id"
    created_at_field: str = "created_at"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation via LiteLLM proxy"""
    model: str = "text-embedding-ada-002"
    base_url: str = "http://localhost:4000"  # LiteLLM proxy URL
    api_key: str = "sk-1234"  # LiteLLM proxy API key
    dimensions: int = 1536


class Settings(BaseSettings):
    """Application settings"""
    # Database configuration
    database_url: str = "postgresql://username:password@localhost:5432/vectordb?schema=public"
    
    # API configuration
    # Optional compatibility with environments that set OPENAI_API_KEY directly
    openai_api_key: Optional[str] = None
    server_api_key: str = "your-api-key-here"
    port: int = 8000
    host: str = "0.0.0.0"
    
    # Database field mappings
    db_fields: DatabaseFieldConfig = DatabaseFieldConfig()
    
    # Embedding configuration
    embedding: EmbeddingConfig = EmbeddingConfig()
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False
        
        # Allow environment variables like:
        # DB_FIELDS__ID_FIELD=custom_id
        # EMBEDDING__MODEL=text-embedding-3-small
        # EMBEDDING__API_BASE=https://api.openai.com/v1
        
    @property
    def table_names(self) -> Dict[str, str]:
        """Get table names"""
        return {
            "vector_stores": "vector_stores",
            "embeddings": "embeddings"
        }


# Global settings instance
settings = Settings() 


# ------------------------
# Admin overlay management
# ------------------------

class AppSettingsOverlay(BaseModel):
    """Optional settings overlay loaded from DB and merged over env defaults."""
    # Grouped overlays by logical domains
    server: Optional[Dict[str, Any]] = None  # keys: host, port, cors_origins
    auth: Optional[Dict[str, Any]] = None    # keys: server_api_key (write-only in UI)
    embedding: Optional[EmbeddingConfig] = None
    db_fields: Optional[DatabaseFieldConfig] = None
    cors: Optional[Dict[str, Any]] = None    # keys: allow_origins (list[str])


def merge_settings_with_overlay(base: Settings, overlay: Optional[AppSettingsOverlay]) -> Settings:
    """Return a new Settings merged with the provided overlay (without mutating base)."""
    if not overlay:
        return base

    merged = base.model_copy(deep=True)

    # Server
    if overlay.server:
        if "host" in overlay.server:
            merged.host = overlay.server["host"]
        if "port" in overlay.server:
            merged.port = overlay.server["port"]

    # Auth
    if overlay.auth and "server_api_key" in overlay.auth and overlay.auth["server_api_key"]:
        merged.server_api_key = overlay.auth["server_api_key"]

    # Embedding
    if overlay.embedding:
        emb = merged.embedding.model_copy(deep=True)
        if overlay.embedding.model is not None:
            emb.model = overlay.embedding.model
        if overlay.embedding.base_url is not None:
            emb.base_url = overlay.embedding.base_url
        if overlay.embedding.api_key is not None:
            emb.api_key = overlay.embedding.api_key
        if overlay.embedding.dimensions is not None:
            emb.dimensions = overlay.embedding.dimensions
        merged.embedding = emb

    # DB fields
    if overlay.db_fields:
        dbf = merged.db_fields.model_copy(deep=True)
        for field_name, value in overlay.db_fields.model_dump(exclude_none=True).items():
            setattr(dbf, field_name, value)
        merged.db_fields = dbf

    # CORS (if added later to Settings)
    # Placeholder: merged does not currently track CORS; handled in app wiring.

    return merged


class SettingsManager:
    """Holds effective settings with DB overlays and supports hot-reload."""

    def __init__(self, base_settings: Settings) -> None:
        self._base = base_settings
        self._overlay: Optional[AppSettingsOverlay] = None
        self._effective: Settings = base_settings
        self._on_apply_callbacks: list[Callable[[Settings, Settings], None]] = []

    @property
    def effective(self) -> Settings:
        return self._effective

    def register_on_apply(self, callback: Callable[[Settings, Settings], None]) -> None:
        """Register a callback invoked after applying a new effective settings.

        Callback signature: (old_settings, new_settings) -> None
        """
        self._on_apply_callbacks.append(callback)

    async def load_overlay_from_db(self, db: Any) -> None:
        """Load overlay rows from app_settings and recompute effective settings.

        Expects `db` to expose `query_raw(sql, *params)` like Prisma.
        """
        rows = await db.query_raw("SELECT key, value FROM app_settings")
        overlay = AppSettingsOverlay()
        for row in rows:
            key = row.get("key")
            val = row.get("value") or {}
            if key == "server":
                overlay.server = val
            elif key == "auth":
                # Never echo back secrets; we accept setting them
                overlay.auth = {k: v for k, v in val.items() if v}
            elif key == "embedding":
                try:
                    overlay.embedding = EmbeddingConfig(**val)
                except Exception:
                    # Invalid overlay row; skip and keep safe base
                    continue
            elif key == "db_fields":
                try:
                    overlay.db_fields = DatabaseFieldConfig(**val)
                except Exception:
                    continue
            elif key == "cors":
                overlay.cors = val

        await self.apply_overlay(overlay)

    async def apply_overlay(self, overlay: Optional[AppSettingsOverlay]) -> None:
        old_effective = self._effective
        self._overlay = overlay
        self._effective = merge_settings_with_overlay(self._base, overlay)
        # Invoke callbacks to hot-apply changes
        for cb in self._on_apply_callbacks:
            try:
                cb(old_effective, self._effective)
            except Exception:
                # Callbacks must not break app; swallow and continue
                continue


# Global settings manager instance used by the app
settings_manager = SettingsManager(settings)