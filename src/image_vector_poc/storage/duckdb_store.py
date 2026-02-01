"""DuckDB-based vector storage with HNSW index."""

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from numpy.typing import NDArray

from ..core.types import SearchResult


class DuckDBVectorStore:
    """Vector storage using DuckDB with VSS extension.

    This store uses DuckDB's vector similarity search (VSS) extension
    with HNSW indexing for efficient approximate nearest neighbor search.

    Example:
        >>> store = DuckDBVectorStore("vectors.duckdb", embedding_dim=768)
        >>> store.add(["img1"], embeddings, ["path/to/img1.jpg"])
        >>> store.create_index()
        >>> results = store.search(query_embedding, k=10)
    """

    def __init__(
        self,
        db_path: Path | str = ":memory:",
        embedding_dim: int = 768,
        table_name: str = "images",
    ):
        """Initialize the DuckDB vector store.

        Args:
            db_path: Path to the DuckDB database file, or ":memory:" for in-memory.
            embedding_dim: Dimension of the embedding vectors.
            table_name: Name of the table to store vectors.
        """
        self.db_path = Path(db_path) if db_path != ":memory:" else db_path
        self.embedding_dim = embedding_dim
        self.table_name = table_name

        self.conn = duckdb.connect(str(self.db_path))
        self._setup()

    def _setup(self) -> None:
        """Initialize VSS extension and create table."""
        self.conn.execute("INSTALL vss; LOAD vss;")

        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id VARCHAR PRIMARY KEY,
                embedding FLOAT[{self.embedding_dim}],
                file_path VARCHAR,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def create_index(self, metric: str = "cosine") -> None:
        """Create HNSW index for fast similarity search.

        Args:
            metric: Distance metric ("cosine", "l2sq", or "ip").
        """
        index_name = f"{self.table_name}_embedding_idx"

        # Drop existing index if exists
        try:
            self.conn.execute(f"DROP INDEX IF EXISTS {index_name}")
        except duckdb.CatalogException:
            pass

        self.conn.execute(f"""
            CREATE INDEX {index_name}
            ON {self.table_name}
            USING HNSW (embedding)
            WITH (metric = '{metric}')
        """)

    def add(
        self,
        ids: list[str],
        embeddings: NDArray[np.float32],
        file_paths: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add vectors to storage.

        Args:
            ids: Unique identifiers for each vector.
            embeddings: 2D array of shape (n, embedding_dim).
            file_paths: Optional file paths for each vector.
            metadata: Optional metadata dictionaries for each vector.
        """
        n = len(ids)
        file_paths = file_paths or [""] * n
        metadata = metadata or [{}] * n

        for id_, emb, path, meta in zip(ids, embeddings, file_paths, metadata):
            emb_list = emb.tolist() if isinstance(emb, np.ndarray) else emb
            meta_json = json.dumps(meta)

            self.conn.execute(
                f"""
                INSERT OR REPLACE INTO {self.table_name}
                (id, embedding, file_path, metadata)
                VALUES (?, ?::FLOAT[{self.embedding_dim}], ?, ?)
                """,
                [id_, emb_list, path, meta_json],
            )

    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors using cosine distance.

        Args:
            query_embedding: Query vector of shape (embedding_dim,).
            k: Number of results to return.
            filters: Optional metadata filters (key-value pairs).

        Returns:
            List of SearchResult objects sorted by similarity.
        """
        query_list = query_embedding.tolist()

        # Build WHERE clause from filters
        where_clause = ""
        if filters:
            conditions = []
            for key, value in filters.items():
                escaped_value = str(value).replace("'", "''")
                conditions.append(
                    f"json_extract_string(metadata, '$.{key}') = '{escaped_value}'"
                )
            where_clause = "WHERE " + " AND ".join(conditions)

        results = self.conn.execute(
            f"""
            SELECT
                id,
                file_path,
                metadata,
                array_cosine_distance(embedding, ?::FLOAT[{self.embedding_dim}]) as distance
            FROM {self.table_name}
            {where_clause}
            ORDER BY distance
            LIMIT ?
            """,
            [query_list, k],
        ).fetchall()

        return [
            SearchResult(
                id=row[0],
                score=1.0 - row[3],  # Convert distance to similarity
                distance=row[3],
                metadata={
                    "file_path": row[1],
                    **(json.loads(row[2]) if row[2] else {}),
                },
            )
            for row in results
        ]

    def get(self, id: str) -> SearchResult | None:
        """Get a single record by ID.

        Args:
            id: The record ID.

        Returns:
            SearchResult if found, None otherwise.
        """
        result = self.conn.execute(
            f"""
            SELECT id, file_path, metadata
            FROM {self.table_name}
            WHERE id = ?
            """,
            [id],
        ).fetchone()

        if result is None:
            return None

        return SearchResult(
            id=result[0],
            score=1.0,
            distance=0.0,
            metadata={
                "file_path": result[1],
                **(json.loads(result[2]) if result[2] else {}),
            },
        )

    def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID.

        Args:
            ids: List of IDs to delete.
        """
        if not ids:
            return

        placeholders = ", ".join(["?"] * len(ids))
        self.conn.execute(
            f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})",
            ids,
        )

    def count(self) -> int:
        """Return number of vectors in storage."""
        result = self.conn.execute(
            f"SELECT COUNT(*) FROM {self.table_name}"
        ).fetchone()
        return result[0] if result else 0

    def clear(self) -> None:
        """Delete all vectors from storage."""
        self.conn.execute(f"DELETE FROM {self.table_name}")

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> "DuckDBVectorStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
