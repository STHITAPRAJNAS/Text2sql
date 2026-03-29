"""
Two-Tier Semantic Query Result Cache
=====================================
L1 — Redis exact hash cache   (sub-millisecond, TTL 5 min by default)
L2 — ChromaDB semantic cache  (cosine similarity, TTL configurable)

Why two tiers:
  - L1 catches repeated identical queries (copy-paste, scheduled reports)
    with virtually zero overhead.
  - L2 catches semantically equivalent queries
    ("top 10 customers by revenue" ≈ "10 largest customers by total sales")
    so the expensive agent pipeline runs only for genuinely new queries.

Cache key design:
  L1 key:  SHA-256( normalize(query) + "|" + database_name )
  L2 key:  semantic embedding of the normalized query; stored in ChromaDB
           collection "text2sql_semantic_cache"

Cache invalidation:
  - L1 TTL: configurable (default 5 min)
  - L2 TTL: configurable (default 1 hour), stored as metadata field `expires_at`
  - Manual flush: call clear_cache(database_name=...) to invalidate a DB's cache
  - Schema change: schema_change_detector calls clear_cache for affected tables

Thread safety: Redis client is thread-safe; ChromaDB client is thread-safe.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_redis_client = None
_chroma_cache_collection = None
_schema_embedder = None

CACHE_COLLECTION = "text2sql_semantic_cache"


# ------------------------------------------------------------------ #
# Initialization                                                       #
# ------------------------------------------------------------------ #

def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    from config.settings import get_settings
    settings = get_settings()
    if not settings.semantic_cache.enable_l1_cache:
        return None
    try:
        import redis.asyncio as aioredis
        _redis_client = aioredis.from_url(
            settings.cache.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info("Semantic cache L1 (Redis) initialized", url=settings.cache.redis_url)
        return _redis_client
    except ImportError:
        logger.warning("redis not installed — L1 cache disabled")
        return None


def _get_chroma_collection():
    global _chroma_cache_collection
    if _chroma_cache_collection is not None:
        return _chroma_cache_collection
    from config.settings import get_settings
    settings = get_settings()
    if not settings.semantic_cache.enable_l2_cache:
        return None
    try:
        import chromadb
        from pathlib import Path
        Path(settings.semantic_cache.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=settings.semantic_cache.chroma_persist_dir)
        _chroma_cache_collection = client.get_or_create_collection(
            name=CACHE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Semantic cache L2 (ChromaDB) initialized",
            size=_chroma_cache_collection.count(),
        )
        return _chroma_cache_collection
    except ImportError:
        logger.warning("chromadb not installed — L2 semantic cache disabled")
        return None


def _get_embedder():
    global _schema_embedder
    if _schema_embedder is not None:
        return _schema_embedder
    try:
        from sentence_transformers import SentenceTransformer
        _schema_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        pass
    return _schema_embedder


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _normalize_query(query: str) -> str:
    """Normalize whitespace and case for consistent hashing."""
    return " ".join(query.lower().split())


def _cache_key(query: str, database_name: str) -> str:
    normalized = _normalize_query(query) + "|" + database_name
    return "t2sql:cache:" + hashlib.sha256(normalized.encode()).hexdigest()


def _embed_query(query: str) -> list[float] | None:
    embedder = _get_embedder()
    if embedder is None:
        return None
    return embedder.encode(query, show_progress_bar=False).tolist()


# ------------------------------------------------------------------ #
# L1 — Redis Exact Cache                                               #
# ------------------------------------------------------------------ #

async def get_l1_cache(query: str, database_name: str) -> dict[str, Any] | None:
    """Check Redis for exact cache hit. Returns cached result or None."""
    r = _get_redis()
    if r is None:
        return None
    try:
        key = _cache_key(query, database_name)
        raw = await r.get(key)
        if raw:
            result = json.loads(raw)
            logger.debug("L1 cache hit", key=key[:16])
            return result
    except Exception as exc:
        logger.warning("L1 cache get error", error=str(exc))
    return None


async def set_l1_cache(
    query: str,
    database_name: str,
    result: dict[str, Any],
) -> None:
    """Store result in Redis with TTL."""
    r = _get_redis()
    if r is None:
        return
    from config.settings import get_settings
    ttl = get_settings().semantic_cache.l1_ttl_seconds
    try:
        key = _cache_key(query, database_name)
        await r.setex(key, ttl, json.dumps(result, default=str))
        logger.debug("L1 cache set", key=key[:16], ttl=ttl)
    except Exception as exc:
        logger.warning("L1 cache set error", error=str(exc))


# ------------------------------------------------------------------ #
# L2 — ChromaDB Semantic Cache                                         #
# ------------------------------------------------------------------ #

def get_l2_cache(query: str, database_name: str) -> dict[str, Any] | None:
    """
    Search ChromaDB for semantically similar cached result.
    Returns cached result if similarity ≥ threshold and not expired.
    """
    col = _get_chroma_collection()
    if col is None or col.count() == 0:
        return None

    from config.settings import get_settings
    settings = get_settings()
    threshold = settings.semantic_cache.l2_similarity_threshold

    embedding = _embed_query(_normalize_query(query))
    if embedding is None:
        return None

    try:
        results = col.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"database_name": database_name},
            include=["documents", "metadatas", "distances"],
        )
        if not results["ids"] or not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        similarity = 1 - distance  # cosine distance → similarity

        if similarity < threshold:
            return None

        meta = results["metadatas"][0][0]
        # Check TTL
        expires_at = float(meta.get("expires_at", 0))
        if expires_at and time.time() > expires_at:
            logger.debug("L2 cache expired", similarity=round(similarity, 3))
            return None

        doc = results["documents"][0][0]
        cached = json.loads(doc)
        cached["_cache_source"] = "l2_semantic"
        cached["_cache_similarity"] = round(similarity, 4)
        logger.debug(
            "L2 cache hit",
            similarity=round(similarity, 3),
            database=database_name,
        )
        return cached

    except Exception as exc:
        logger.warning("L2 cache query error", error=str(exc))
        return None


def set_l2_cache(
    query: str,
    database_name: str,
    result: dict[str, Any],
) -> None:
    """Store result in ChromaDB semantic cache."""
    col = _get_chroma_collection()
    if col is None:
        return

    from config.settings import get_settings
    settings = get_settings()
    ttl = settings.semantic_cache.l2_ttl_seconds

    embedding = _embed_query(_normalize_query(query))
    if embedding is None:
        return

    try:
        doc_id = _cache_key(query, database_name)
        expires_at = time.time() + ttl
        col.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[json.dumps(result, default=str)],
            metadatas=[{
                "database_name": database_name,
                "query": query[:200],
                "expires_at": str(expires_at),
                "cached_at": str(time.time()),
            }],
        )
        logger.debug("L2 cache set", database=database_name, ttl=ttl)
    except Exception as exc:
        logger.warning("L2 cache set error", error=str(exc))


# ------------------------------------------------------------------ #
# Combined lookup + store                                              #
# ------------------------------------------------------------------ #

async def get_cached_result(
    query: str,
    database_name: str,
) -> tuple[dict[str, Any] | None, str]:
    """
    Check both cache tiers. Returns (result, cache_source) where
    cache_source is 'l1_exact', 'l2_semantic', or 'miss'.
    """
    # L1 first (fastest)
    l1 = await get_l1_cache(query, database_name)
    if l1 is not None:
        l1["_cache_source"] = "l1_exact"
        return l1, "l1_exact"

    # L2 semantic
    l2 = get_l2_cache(query, database_name)
    if l2 is not None:
        return l2, "l2_semantic"

    return None, "miss"


async def cache_result(
    query: str,
    database_name: str,
    result: dict[str, Any],
) -> None:
    """Store result in both cache tiers (only when success=True)."""
    if not result.get("success"):
        return
    # Strip cache metadata before storing
    clean = {k: v for k, v in result.items() if not k.startswith("_cache")}
    await set_l1_cache(query, database_name, clean)
    set_l2_cache(query, database_name, clean)


async def clear_cache(database_name: str | None = None) -> int:
    """
    Invalidate cache entries.
    If database_name is given, only clear that database's entries.
    Returns number of L1 keys deleted (L2 cleanup is best-effort).
    """
    deleted = 0

    # L1 Redis: scan and delete matching keys
    r = _get_redis()
    if r:
        try:
            pattern = "t2sql:cache:*"
            async for key in r.scan_iter(pattern, count=100):
                await r.delete(key)
                deleted += 1
        except Exception as exc:
            logger.warning("L1 cache clear error", error=str(exc))

    # L2 ChromaDB: delete by database_name filter
    col = _get_chroma_collection()
    if col and col.count() > 0:
        try:
            where = {"database_name": database_name} if database_name else None
            if where:
                existing = col.get(where=where, include=[])
                if existing["ids"]:
                    col.delete(ids=existing["ids"])
        except Exception as exc:
            logger.warning("L2 cache clear error", error=str(exc))

    logger.info("Cache cleared", database=database_name, l1_deleted=deleted)
    return deleted


def get_cache_stats() -> dict[str, Any]:
    """Return cache statistics for the /api/v1/metrics endpoint."""
    col = _get_chroma_collection()
    l2_size = col.count() if col else 0
    return {
        "l1_backend": "redis",
        "l2_backend": "chromadb",
        "l2_size": l2_size,
    }
