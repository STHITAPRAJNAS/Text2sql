"""
Enterprise Text2SQL PRISM - Application Settings
Centralized configuration with Pydantic Settings for validation and type safety.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM model configuration for each agent in the PRISM swarm."""

    orchestrator_model: str = Field(default="gemini-2.0-flash", description="Orchestrator agent model")
    schema_agent_model: str = Field(default="gemini-2.0-flash", description="Schema discovery agent model")
    query_analyzer_model: str = Field(default="gemini-2.5-pro", description="Deep Think query analyzer model")
    sql_generator_model: str = Field(default="gemini-2.5-pro", description="SQL generator model")
    sql_validator_model: str = Field(default="gemini-2.0-flash", description="SQL validator model")
    optimizer_model: str = Field(default="gemini-2.0-flash", description="Query optimizer model")
    formatter_model: str = Field(default="gemini-2.0-flash", description="Response formatter model")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class DatabaseSettings(BaseSettings):
    """Database connection and pool configuration."""

    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/test.db",
        description="Primary database connection URL",
    )
    database_pool_size: int = Field(default=10, ge=1, le=100)
    database_max_overflow: int = Field(default=20, ge=0, le=100)
    database_echo: bool = Field(default=False)
    max_rows_return: int = Field(default=1000, description="Maximum rows to return per query")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class DeepThinkSettings(BaseSettings):
    """Configuration for the Deep Think reasoning pipeline."""

    deep_think_max_iterations: int = Field(default=3, ge=1, le=10, description="Max refinement iterations")
    deep_think_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    enable_self_reflection: bool = Field(default=True, description="Enable self-reflection in Deep Think")
    enable_chain_of_thought: bool = Field(default=True, description="Enable CoT reasoning")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class CacheSettings(BaseSettings):
    """Redis cache configuration."""

    redis_url: str = Field(default="redis://localhost:6379/0")
    cache_ttl_schema: int = Field(default=3600, description="Schema cache TTL in seconds")
    cache_ttl_query: int = Field(default=300, description="Query cache TTL in seconds")
    enable_query_caching: bool = Field(default=True)
    enable_schema_caching: bool = Field(default=True)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class VectorStoreSettings(BaseSettings):
    """Vector store configuration for few-shot examples."""

    chroma_persist_dir: str = Field(default="./data/chroma")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    few_shot_top_k: int = Field(default=5, ge=1, le=20)
    enable_few_shot_examples: bool = Field(default=True)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class APISettings(BaseSettings):
    """FastAPI server configuration."""

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080, ge=1, le=65535)
    api_workers: int = Field(default=4, ge=1)
    api_reload: bool = Field(default=False)
    api_log_level: str = Field(default="info")
    api_key_header: str = Field(default="X-API-Key")
    allowed_origins: list[str] = Field(default=["http://localhost:3000"])
    max_query_length: int = Field(default=2000)
    enable_query_sanitization: bool = Field(default=True)
    enable_sql_allowlist: bool = Field(default=False)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class ObservabilitySettings(BaseSettings):
    """Tracing, logging and metrics configuration."""

    enable_tracing: bool = Field(default=False)
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Settings(BaseSettings):
    """
    Master settings object composing all configuration sections.
    Loaded from environment variables with .env file fallback.
    """

    # Google AI / ADK
    google_api_key: SecretStr = Field(default=SecretStr(""), description="Google AI API Key")
    google_cloud_project: str = Field(default="", description="GCP Project ID")
    google_cloud_location: str = Field(default="us-central1")
    google_genai_use_vertexai: bool = Field(default=False)

    # Feature flags
    enable_query_optimization: bool = Field(default=True)
    enable_result_explanation: bool = Field(default=True)

    # Composed settings sections
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    deep_think: DeepThinkSettings = Field(default_factory=DeepThinkSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    api: APISettings = Field(default_factory=APISettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

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
