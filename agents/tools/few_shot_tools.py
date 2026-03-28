"""
Few-Shot Example Tools for PRISM Agents
Vector-store based semantic search for similar SQL examples.
Used by the SQL Generator agent to improve generation quality.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# In-memory example store (for environments without ChromaDB)
_in_memory_examples: list[dict[str, Any]] = []
_chroma_collection = None
_embedder = None


def _get_chroma_collection():
    """Lazily initialize ChromaDB collection."""
    global _chroma_collection, _embedder
    if _chroma_collection is not None:
        return _chroma_collection

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from config.settings import get_settings

        settings = get_settings()
        persist_dir = settings.vector_store.chroma_persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=persist_dir)
        _chroma_collection = client.get_or_create_collection(
            name="text2sql_examples",
            metadata={"hnsw:space": "cosine"},
        )

        model_name = settings.vector_store.embedding_model
        _embedder = SentenceTransformer(model_name)

        logger.info("ChromaDB initialized", collection="text2sql_examples")
        return _chroma_collection
    except ImportError:
        logger.warning("ChromaDB/sentence-transformers not available, using in-memory store")
        return None


def get_similar_examples(
    query: str,
    top_k: int = 5,
    database_name: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve similar text-to-SQL examples from the few-shot store.

    Returns the most semantically similar query-SQL pairs from the
    example database. These examples guide the SQL generator to produce
    consistent, high-quality SQL following established patterns.

    Args:
        query: The natural language query to find similar examples for
        top_k: Number of similar examples to return (default: 5)
        database_name: Optional filter for database-specific examples

    Returns:
        dict with:
            - examples (list): Similar examples with question, sql, and similarity
            - formatted_examples (str): Examples formatted for LLM prompt insertion
            - total_in_store (int): Total examples in the store
    """
    collection = _get_chroma_collection()

    if collection is None:
        # Fallback to in-memory examples
        examples = _in_memory_examples[:top_k]
        formatted = _format_examples(examples)
        return {
            "examples": examples,
            "formatted_examples": formatted,
            "total_in_store": len(_in_memory_examples),
            "source": "in_memory",
        }

    try:
        global _embedder
        query_embedding = _embedder.encode(query).tolist()

        where_filter = {"database_name": database_name} if database_name else None

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        examples = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Convert cosine distance to similarity

                examples.append({
                    "question": metadata.get("question", results["documents"][0][i]),
                    "sql": metadata.get("sql", ""),
                    "database": metadata.get("database_name", ""),
                    "similarity": round(similarity, 3),
                    "tags": json.loads(metadata.get("tags", "[]")),
                })

        formatted = _format_examples(examples)
        return {
            "examples": examples,
            "formatted_examples": formatted,
            "total_in_store": collection.count(),
            "source": "chromadb",
        }
    except Exception as e:
        logger.error("Failed to retrieve examples", error=str(e))
        return {
            "examples": [],
            "formatted_examples": "",
            "total_in_store": 0,
            "error": str(e),
        }


def add_example_to_store(
    question: str,
    sql: str,
    database_name: str = "default",
    tags: list[str] | None = None,
    feedback_score: float = 1.0,
) -> dict[str, Any]:
    """
    Add a new text-to-SQL example to the few-shot example store.

    Use this to continuously improve the system with verified, high-quality
    examples from production queries or expert annotations.

    Args:
        question: The natural language question
        sql: The correct SQL query
        database_name: The database this example is for
        tags: Optional tags for categorization (e.g., ["aggregation", "join"])
        feedback_score: Quality score 0.0-1.0 (default: 1.0)

    Returns:
        dict with:
            - success (bool): Whether the example was added
            - example_id (str): The ID assigned to the example
            - total_examples (int): Total examples now in store
    """
    tags = tags or []
    collection = _get_chroma_collection()

    example = {
        "question": question,
        "sql": sql,
        "database_name": database_name,
        "tags": tags,
        "feedback_score": feedback_score,
    }

    if collection is None:
        # Fallback: in-memory store
        _in_memory_examples.append(example)
        return {
            "success": True,
            "example_id": f"mem_{len(_in_memory_examples)}",
            "total_examples": len(_in_memory_examples),
            "source": "in_memory",
        }

    try:
        global _embedder
        embedding = _embedder.encode(question).tolist()
        example_id = f"ex_{hash(question + sql) & 0xFFFFFF:06x}"

        collection.add(
            ids=[example_id],
            embeddings=[embedding],
            documents=[question],
            metadatas=[{
                "question": question,
                "sql": sql,
                "database_name": database_name,
                "tags": json.dumps(tags),
                "feedback_score": str(feedback_score),
            }],
        )

        return {
            "success": True,
            "example_id": example_id,
            "total_examples": collection.count(),
            "source": "chromadb",
        }
    except Exception as e:
        logger.error("Failed to add example", error=str(e))
        return {"success": False, "error": str(e)}


def _format_examples(examples: list[dict[str, Any]]) -> str:
    """Format examples as a string for LLM prompt injection."""
    if not examples:
        return "No similar examples found in the store."

    lines = ["## Similar Query Examples\n"]
    for i, ex in enumerate(examples, 1):
        similarity = ex.get("similarity", "N/A")
        lines.append(f"### Example {i} (similarity: {similarity})")
        lines.append(f"**Question:** {ex.get('question', '')}")
        lines.append(f"```sql\n{ex.get('sql', '')}\n```")
        lines.append("")

    return "\n".join(lines)


def load_examples_from_file(file_path: str) -> dict[str, Any]:
    """
    Bulk-load examples from a JSON file into the few-shot store.

    The JSON file should contain a list of objects with 'question' and 'sql' fields.

    Args:
        file_path: Path to the JSON file with examples

    Returns:
        dict with load results
    """
    try:
        with open(file_path) as f:
            examples = json.load(f)

        loaded = 0
        failed = 0
        for ex in examples:
            result = add_example_to_store(
                question=ex.get("question", ""),
                sql=ex.get("sql", ""),
                database_name=ex.get("database", "default"),
                tags=ex.get("tags", []),
            )
            if result.get("success"):
                loaded += 1
            else:
                failed += 1

        return {
            "success": True,
            "loaded": loaded,
            "failed": failed,
            "total": len(examples),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
