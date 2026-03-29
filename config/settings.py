"""
Enterprise Text2SQL PRISM - Application Settings
Centralized configuration with Pydantic Settings for validation and type safety.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM model configuration for each agent in the PRISM swarm."""

    orchestrator_model: str = Field(default="gemini-2.0-flash")
    schema_agent_model: str = Field(default="gemini-2.0-flash")
    query_analyzer_model: str = Field(default="gemini-2.5-pro", description="Deep Think uses Pro for reasoning")
    sql_generator_model: str = Field(default="gemini-2.5-pro")
    sql_validator_model: str = Field(default="gemini-2.0-flash")
    optimizer_model: str = Field(default="gemini-2.0-flash")
    formatter_model: str = Field(default="gemini-2.0-flash")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class DatabaseSettings(BaseSettings):
    """SQLAlchemy database connection (non-Databricks)."""

    database_url: str = Field(default="sqlite+aiosqlite:///./data/test.db")
    database_pool_size: int = Field(default=10, ge=1, le=100)
    database_max_overflow: int = Field(default=20, ge=0, le=100)
    database_echo: bool = Field(default=False)
    max_rows_return: int = Field(default=1000)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class DatabricksSettings(BaseSettings):
    """
    Databricks Unity Catalog connection.
    When DATABRICKS_HOST is set, the system uses the native Databricks
    connector instead of SQLAlchemy for all database operations.
    """

    host: str = Field(default="", description="Workspace hostname: adb-xxx.azuredatabricks.net")
    http_path: str = Field(default="", description="SQL warehouse HTTP path: /sql/1.0/warehouses/xxx")
    token: SecretStr | None = Field(default=None, description="Personal access token")
    client_id: str = Field(default="", description="OAuth M2M client ID")
    client_secret: SecretStr | None = Field(default=None, description="OAuth M2M client secret")
    default_catalog: str = Field(default="main", description="Default Unity Catalog catalog")
    default_schema: str = Field(default="default", description="Default schema")
    connect_timeout: int = Field(default=30)
    query_timeout: int = Field(default=600)

    @property
    def is_configured(self) -> bool:
        return bool(self.host and self.http_path)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DATABRICKS_",
        extra="ignore",
    )


class DeepThinkSettings(BaseSettings):
    deep_think_max_iterations: int = Field(default=3, ge=1, le=10)
    deep_think_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    enable_self_reflection: bool = Field(default=True)
    enable_chain_of_thought: bool = Field(default=True)
    block_advanced_queries: bool = Field(
        default=False,
        description="Reject ADVANCED complexity queries (score > 15) before SQL generation.",
    )
    max_execution_retries: int = Field(
        default=2, ge=0, le=5,
        description="Max auto-retry attempts when SQL execution fails.",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class CacheSettings(BaseSettings):
    redis_url: str = Field(default="redis://localhost:6379/0")
    cache_ttl_schema: int = Field(default=3600)
    cache_ttl_query: int = Field(default=300)
    enable_query_caching: bool = Field(default=True)
    enable_schema_caching: bool = Field(default=True)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class VectorStoreSettings(BaseSettings):
    """Vector store for few-shot SQL examples (separate from schema index)."""

    chroma_persist_dir: str = Field(default="./data/chroma/examples")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    few_shot_top_k: int = Field(default=5, ge=1, le=20)
    enable_few_shot_examples: bool = Field(default=True)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class VectorIndexSettings(BaseSettings):
    """
    Vector index for schema metadata (tables / columns from Unity Catalog).
    Used by the incremental agent-driven schema indexer.

    Backends:
      chroma   — ChromaDB local persistent store (default, zero infra)
      pgvector — PostgreSQL with pgvector extension (production recommended)
      in_memory — ephemeral, for testing only
    """

    backend: str = Field(
        default="chroma",
        description="Index backend: chroma | pgvector | in_memory",
    )
    chroma_persist_dir: str = Field(default="./data/chroma/schema_index")
    pg_dsn: str = Field(
        default="",
        description="PostgreSQL DSN for pgvector backend: postgresql://user:pass@host/db",
    )
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    search_top_k: int = Field(default=15, ge=1, le=50)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SCHEMA_INDEX_",
        extra="ignore",
    )


class SessionSettings(BaseSettings):
    """ADK DatabaseSessionService configuration (4 tables: sessions, events, app/user state)."""

    session_db_url: str = Field(
        default="sqlite+aiosqlite:///./data/sessions.db",
        description="SQLAlchemy async URL for ADK session persistence. "
                    "Use postgresql+asyncpg://... for production.",
    )
    # When set, overrides session_db_url for get_fast_api_app
    session_service_uri: str = Field(
        default="",
        description="If non-empty, passed directly to get_fast_api_app as session_service_uri. "
                    "Overrides session_db_url. Use 'sqlite+aiosqlite:///./data/sessions.db' etc.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SESSION_",
        extra="ignore",
    )


class MemorySettings(BaseSettings):
    """ADK MemoryService configuration — semantic vector store for agent self-improvement."""

    backend: str = Field(
        default="in_memory",
        description="Memory backend: in_memory | vertex_ai. "
                    "vertex_ai requires MEMORY_AGENT_ENGINE_ID to be set.",
    )
    agent_engine_id: str = Field(
        default="",
        description="Vertex AI Agent Engine ID for VertexAiMemoryBankService.",
    )
    # Controls how many past sessions the memory tools return
    memory_top_k: int = Field(default=5, ge=1, le=20)
    # Minimum confidence for memory-retrieved examples to be used
    memory_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MEMORY_",
        extra="ignore",
    )


class SemanticCacheSettings(BaseSettings):
    """Two-tier semantic query result cache: Redis (L1 exact) + ChromaDB (L2 semantic)."""

    enable_l1_cache: bool = Field(default=True, description="Redis exact-match cache")
    enable_l2_cache: bool = Field(default=True, description="ChromaDB semantic similarity cache")
    l1_ttl_seconds: int = Field(default=300, description="Redis TTL in seconds (5 minutes)")
    l2_ttl_seconds: int = Field(default=3600, description="ChromaDB result TTL in seconds (1 hour)")
    l2_similarity_threshold: float = Field(
        default=0.92,
        ge=0.0, le=1.0,
        description="Min cosine similarity to use L2 cache hit",
    )
    chroma_persist_dir: str = Field(default="./data/chroma/semantic_cache")
    max_cached_results: int = Field(default=10000, description="Max cached query results")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SEMANTIC_CACHE_",
        extra="ignore",
    )


class FeedbackSettings(BaseSettings):
    """Feedback loop + active learning configuration."""

    auto_add_to_fewshot: bool = Field(
        default=True,
        description="Automatically add highly-rated queries to the few-shot store.",
    )
    auto_add_to_memory: bool = Field(
        default=True,
        description="Automatically add highly-rated queries to the ADK memory service.",
    )
    min_rating_for_fewshot: float = Field(
        default=4.0, ge=1.0, le=5.0,
        description="Minimum rating (1-5) to add to few-shot store.",
    )
    cost_warn_gb: float = Field(
        default=10.0,
        description="Warn if query scans more than this many GB.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FEEDBACK_",
        extra="ignore",
    )


class APISettings(BaseSettings):
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080, ge=1, le=65535)
    api_workers: int = Field(default=4, ge=1)
    api_reload: bool = Field(default=False)
    api_log_level: str = Field(default="info")
    api_key_header: str = Field(default="X-API-Key")
    allowed_origins: list[str] = Field(default=["http://localhost:3000", "*"])
    max_query_length: int = Field(default=2000)
    enable_query_sanitization: bool = Field(default=True)
    enable_sql_allowlist: bool = Field(default=False)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class RateLimitSettings(BaseSettings):
    """Per-user / per-API-key request throttling."""
    enabled: bool = Field(default=False, description="Enable rate limiting (disabled by default)")
    requests_per_minute: int = Field(default=60, ge=1)
    requests_per_hour: int = Field(default=500, ge=1)
    burst_limit: int = Field(
        default=10, ge=1,
        description="Max concurrent sliding-window requests in a 10-second burst",
    )

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RATE_LIMIT_", extra="ignore")


class AllowlistSettings(BaseSettings):
    """SQL and database access control."""
    enabled: bool = Field(default=False, description="Enable allowlist/blocklist checks")
    # Regex patterns against the NL query or generated SQL
    blocked_table_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns; if SQL matches any, the query is rejected",
    )
    blocked_sql_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns checked against raw NL query before pipeline",
    )
    allowed_databases: list[str] = Field(
        default_factory=list,
        description="If non-empty, only these database names are accepted",
    )

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ALLOWLIST_", extra="ignore")


class TenantSettings(BaseSettings):
    """Multi-tenant isolation for glossary and few-shot examples."""
    enabled: bool = Field(default=False, description="Enable tenant-scoped isolation")
    tenant_id_header: str = Field(
        default="X-Tenant-ID",
        description="HTTP header name to extract tenant ID from",
    )
    default_tenant: str = Field(default="default")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="TENANT_", extra="ignore")


class ObservabilitySettings(BaseSettings):
    enable_tracing: bool = Field(default=False)
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Settings(BaseSettings):
    """
    Master settings. All sections are composed here.
    Priority: env vars > .env file > defaults.
    """

    # Google AI / ADK
    google_api_key: SecretStr = Field(default=SecretStr(""))
    google_cloud_project: str = Field(default="")
    google_cloud_location: str = Field(default="us-central1")
    google_genai_use_vertexai: bool = Field(default=False)

    # Feature flags
    enable_query_optimization: bool = Field(default=True)
    enable_result_explanation: bool = Field(default=True)

    # Sections
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    databricks: DatabricksSettings = Field(default_factory=DatabricksSettings)
    deep_think: DeepThinkSettings = Field(default_factory=DeepThinkSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    vector_index: VectorIndexSettings = Field(default_factory=VectorIndexSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    semantic_cache: SemanticCacheSettings = Field(default_factory=SemanticCacheSettings)
    feedback: FeedbackSettings = Field(default_factory=FeedbackSettings)
    api: APISettings = Field(default_factory=APISettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    allowlist: AllowlistSettings = Field(default_factory=AllowlistSettings)
    tenant: TenantSettings = Field(default_factory=TenantSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("google_api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        return v or ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()
