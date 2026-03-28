"""
Schema Index — Agent-Driven Incremental Vector Store
=====================================================
Stores table/column metadata as embeddings so agents can:
  1. Semantically search across 10,000+ Unity Catalog tables
  2. Auto-index any table they encounter that isn't yet stored
  3. Retrieve rich metadata without fetching the full catalog

Storage backends (chosen at startup based on config):
  - ChromaDB   (default, local, no extra infra)
  - pgvector   (production, needs PostgreSQL + pgvector extension)

Index document per table:
  id:        "catalog.schema.table"
  text:      "<table_name>: <comment>. Columns: col1 (type, comment), ..."
  embedding: sentence_transformer(text)
  metadata:  {catalog, schema, table, columns_json, row_count,
               partition_cols, clustering_cols, indexed_at}

Agents call this via the indexing tools in agents/tools/indexing_tools.py.
The index grows organically — every newly discovered table is indexed on the
first encounter, without a separate batch pipeline.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class IndexBackend(str, Enum):
    CHROMA = "chroma"
    PGVECTOR = "pgvector"
    IN_MEMORY = "in_memory"


@dataclass
class TableDocument:
    """A single table's indexable representation."""
    id: str                          # "catalog.schema.table"
    text: str                        # concatenated description for embedding
    metadata: dict[str, Any]         # rich metadata stored alongside embedding
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """A single vector search result."""
    table_id: str                    # "catalog.schema.table"
    score: float                     # cosine similarity 0–1
    metadata: dict[str, Any]
    text: str = ""


class SchemaIndex:
    """
    Vector index over database/Unity Catalog schema metadata.

    Responsibilities:
    - Embed table descriptions using a sentence transformer
    - Store/retrieve embeddings in ChromaDB or pgvector
    - Expose semantic search: "find tables related to 'revenue'"
    - Support incremental upsert (called by agents on discovery)
    - Track which tables are indexed and when

    Thread-safety: ChromaDB and pgvector clients are both thread-safe.
    """

    COLLECTION_NAME = "text2sql_schema_index"

    def __init__(
        self,
        backend: IndexBackend = IndexBackend.CHROMA,
        chroma_persist_dir: str = "./data/chroma",
        pg_dsn: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.backend = backend
        self.chroma_persist_dir = chroma_persist_dir
        self.pg_dsn = pg_dsn
        self.embedding_model_name = embedding_model

        self._chroma_collection = None
        self._pg_pool = None
        self._embedder = None
        self._in_memory: dict[str, TableDocument] = {}  # fallback store

        logger.info("SchemaIndex initialized", backend=backend.value)

    # ------------------------------------------------------------------ #
    # Embedding                                                            #
    # ------------------------------------------------------------------ #

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model loaded", model=self.embedding_model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not available — using hash-based mock embeddings"
                )
        return self._embedder

    def _embed(self, text: str) -> list[float]:
        """Embed text into a float vector."""
        embedder = self._get_embedder()
        if embedder is not None:
            return embedder.encode(text, show_progress_bar=False).tolist()
        # Deterministic mock embedding for testing without sentence-transformers
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        return [(b - 128) / 128.0 for b in h] * 24  # 384-dim mock

    # ------------------------------------------------------------------ #
    # ChromaDB backend                                                     #
    # ------------------------------------------------------------------ #

    def _get_chroma_collection(self):
        if self._chroma_collection is not None:
            return self._chroma_collection
        try:
            import chromadb
            from pathlib import Path
            Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=self.chroma_persist_dir)
            self._chroma_collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB schema index ready",
                size=self._chroma_collection.count(),
            )
            return self._chroma_collection
        except ImportError:
            logger.warning("chromadb not installed — falling back to in-memory index")
            self.backend = IndexBackend.IN_MEMORY
            return None

    # ------------------------------------------------------------------ #
    # pgvector backend                                                     #
    # ------------------------------------------------------------------ #

    async def _ensure_pg_table(self):
        """Create the pgvector table if it doesn't exist."""
        import asyncpg
        conn = await asyncpg.connect(self.pg_dsn)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.COLLECTION_NAME} (
                id          TEXT PRIMARY KEY,
                text        TEXT,
                embedding   vector(384),
                metadata    JSONB,
                indexed_at  TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.COLLECTION_NAME}_emb_idx
            ON {self.COLLECTION_NAME}
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
        """)
        await conn.close()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build_table_text(self, metadata: dict[str, Any]) -> str:
        """
        Build the natural-language text representation of a table for embedding.
        Richer text → better semantic search quality.
        """
        parts = []

        # Table name and business description
        full_name = metadata.get("full_name", metadata.get("id", ""))
        table = metadata.get("table", "")
        comment = metadata.get("comment", "")

        parts.append(f"Table: {full_name}")
        if comment:
            parts.append(f"Description: {comment}")
        else:
            # Humanize the table name as a proxy description
            human_name = table.replace("_", " ").replace("-", " ")
            parts.append(f"Description: {human_name}")

        # Column summaries — most important for semantic matching
        columns = metadata.get("columns", [])
        if columns:
            col_summaries = []
            for col in columns[:40]:  # cap at 40 columns per doc
                name = col.get("name", "")
                dtype = col.get("data_type", col.get("type_text", ""))
                col_comment = col.get("comment", "")
                if col_comment:
                    col_summaries.append(f"{name} ({dtype}): {col_comment}")
                else:
                    col_summaries.append(f"{name} ({dtype})")
            parts.append("Columns: " + ", ".join(col_summaries))

        # Partition / clustering info
        partitions = metadata.get("partitioning", metadata.get("partition_cols", []))
        if partitions:
            parts.append(f"Partitioned by: {', '.join(partitions)}")

        clustering = metadata.get("clustering_columns", [])
        if clustering:
            parts.append(f"Clustered by: {', '.join(clustering)}")

        # Row count context
        row_count = metadata.get("row_count", -1)
        if row_count and row_count > 0:
            parts.append(f"Approximate rows: {row_count:,}")

        return "\n".join(parts)

    def upsert(self, metadata: dict[str, Any]) -> str:
        """
        Index or update a table's metadata in the vector store.
        Called by agents whenever they fetch fresh table metadata.

        Returns the table ID that was indexed.
        """
        table_id = metadata.get("full_name") or metadata.get("id", "")
        if not table_id:
            raise ValueError("metadata must contain 'full_name' or 'id'")

        text = self.build_table_text(metadata)
        embedding = self._embed(text)

        # Serialize metadata (JSON-safe)
        meta_to_store = {
            "full_name": table_id,
            "catalog": metadata.get("catalog", ""),
            "schema": metadata.get("schema", ""),
            "table": metadata.get("table", ""),
            "comment": metadata.get("comment", ""),
            "table_type": metadata.get("table_type", ""),
            "columns_json": json.dumps(metadata.get("columns", []), default=str),
            "row_count": str(metadata.get("row_count", -1)),
            "partitioning": json.dumps(metadata.get("partitioning", []), default=str),
            "clustering_columns": json.dumps(
                metadata.get("clustering_columns", []), default=str
            ),
            "indexed_at": str(int(time.time())),
        }

        # --- ChromaDB path ---
        if self.backend == IndexBackend.CHROMA:
            col = self._get_chroma_collection()
            if col is not None:
                col.upsert(
                    ids=[table_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[meta_to_store],
                )
                return table_id

        # --- In-memory fallback ---
        self._in_memory[table_id] = TableDocument(
            id=table_id,
            text=text,
            metadata=meta_to_store,
            embedding=embedding,
        )
        return table_id

    def search(
        self,
        query: str,
        top_k: int = 15,
        catalog_filter: str | None = None,
        schema_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Semantic search: find the most relevant tables for a natural language query.
        Returns top_k results ordered by cosine similarity.
        """
        query_embedding = self._embed(query)

        # --- ChromaDB path ---
        if self.backend == IndexBackend.CHROMA:
            col = self._get_chroma_collection()
            if col is not None and col.count() > 0:
                where: dict[str, Any] | None = None
                if catalog_filter and schema_filter:
                    where = {"$and": [
                        {"catalog": catalog_filter},
                        {"schema": schema_filter},
                    ]}
                elif catalog_filter:
                    where = {"catalog": catalog_filter}
                elif schema_filter:
                    where = {"schema": schema_filter}

                results = col.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, col.count()),
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
                out = []
                if results["ids"] and results["ids"][0]:
                    for i, tid in enumerate(results["ids"][0]):
                        dist = results["distances"][0][i]
                        meta = results["metadatas"][0][i]
                        text = results["documents"][0][i] if results["documents"] else ""
                        out.append(SearchResult(
                            table_id=tid,
                            score=round(1 - dist, 4),
                            metadata=meta,
                            text=text,
                        ))
                return out

        # --- In-memory cosine similarity fallback ---
        return self._in_memory_search(query_embedding, top_k, catalog_filter, schema_filter)

    def _in_memory_search(
        self,
        query_embedding: list[float],
        top_k: int,
        catalog_filter: str | None,
        schema_filter: str | None,
    ) -> list[SearchResult]:
        """Pure Python cosine similarity search over in-memory store."""
        import math

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb + 1e-9)

        results = []
        for doc_id, doc in self._in_memory.items():
            if catalog_filter and doc.metadata.get("catalog") != catalog_filter:
                continue
            if schema_filter and doc.metadata.get("schema") != schema_filter:
                continue
            if doc.embedding:
                score = cosine(query_embedding, doc.embedding)
                results.append(SearchResult(
                    table_id=doc_id,
                    score=round(score, 4),
                    metadata=doc.metadata,
                    text=doc.text,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def is_indexed(self, table_id: str) -> bool:
        """Check if a table is already in the index."""
        if self.backend == IndexBackend.CHROMA:
            col = self._get_chroma_collection()
            if col is not None:
                result = col.get(ids=[table_id], include=[])
                return len(result["ids"]) > 0
        return table_id in self._in_memory

    def get_indexed_count(self) -> int:
        """Return total number of indexed tables."""
        if self.backend == IndexBackend.CHROMA:
            col = self._get_chroma_collection()
            if col is not None:
                return col.count()
        return len(self._in_memory)

    def get_by_id(self, table_id: str) -> dict[str, Any] | None:
        """Retrieve stored metadata for a specific table ID."""
        if self.backend == IndexBackend.CHROMA:
            col = self._get_chroma_collection()
            if col is not None:
                result = col.get(
                    ids=[table_id],
                    include=["documents", "metadatas"],
                )
                if result["ids"]:
                    meta = result["metadatas"][0]
                    # Deserialize columns_json back
                    if "columns_json" in meta:
                        try:
                            meta["columns"] = json.loads(meta["columns_json"])
                        except Exception:
                            meta["columns"] = []
                    return meta
        doc = self._in_memory.get(table_id)
        return doc.metadata if doc else None

    def list_indexed_tables(
        self,
        catalog_filter: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, str]]:
        """List all indexed tables with basic metadata."""
        if self.backend == IndexBackend.CHROMA:
            col = self._get_chroma_collection()
            if col is not None:
                where = {"catalog": catalog_filter} if catalog_filter else None
                result = col.get(
                    where=where,
                    limit=limit,
                    include=["metadatas"],
                )
                return [
                    {
                        "id": tid,
                        "catalog": m.get("catalog", ""),
                        "schema": m.get("schema", ""),
                        "table": m.get("table", ""),
                        "indexed_at": m.get("indexed_at", ""),
                    }
                    for tid, m in zip(result["ids"], result["metadatas"])
                ]
        items = self._in_memory.items()
        if catalog_filter:
            items = {k: v for k, v in items if v.metadata.get("catalog") == catalog_filter}.items()
        return [
            {
                "id": k,
                "catalog": v.metadata.get("catalog", ""),
                "schema": v.metadata.get("schema", ""),
                "table": v.metadata.get("table", ""),
                "indexed_at": v.metadata.get("indexed_at", ""),
            }
            for k, v in list(items)[:limit]
        ]


# ------------------------------------------------------------------ #
# Singleton                                                            #
# ------------------------------------------------------------------ #

_schema_index: SchemaIndex | None = None


def get_schema_index() -> SchemaIndex:
    """Return the singleton SchemaIndex, creating it on first call."""
    global _schema_index
    if _schema_index is None:
        from config.settings import get_settings
        s = get_settings()
        vi = s.vector_index

        backend = IndexBackend(vi.backend)
        _schema_index = SchemaIndex(
            backend=backend,
            chroma_persist_dir=vi.chroma_persist_dir,
            pg_dsn=vi.pg_dsn or None,
            embedding_model=vi.embedding_model,
        )
    return _schema_index
