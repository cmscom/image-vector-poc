"""検索ロジック"""

import sys
from pathlib import Path

# appディレクトリの親をパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import numpy as np
from PIL import Image

from app.config import DB_PATH, MODEL_CONFIGS


def get_connection() -> duckdb.DuckDBPyConnection:
    """データベース接続を取得"""
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    conn.execute("LOAD vss;")
    return conn


def load_embeddings(
    conn: duckdb.DuckDBPyConnection, model_key: str
) -> tuple[np.ndarray, list[dict]]:
    """データベースから埋め込みとメタデータを読み込み

    Returns:
        embeddings: (N, dim) の埋め込み行列
        metadata: 各画像のメタデータ（id, file_path, file_name, category）のリスト
    """
    model_name = MODEL_CONFIGS[model_key]["model_name"]
    embedding_dim = MODEL_CONFIGS[model_key]["embedding_dim"]

    results = conn.execute(
        f"""
        SELECT
            c.id,
            c.file_path,
            c.file_name,
            c.category,
            e.embedding
        FROM image_catalog c
        JOIN image_embeddings e ON c.id = e.id
        WHERE e.model_name = ?
        ORDER BY c.id
    """,
        [model_name],
    ).fetchall()

    if not results:
        return np.array([]), []

    embeddings = np.array([r[4] for r in results], dtype=np.float32)
    metadata = [
        {
            "id": r[0],
            "file_path": r[1],
            "file_name": r[2],
            "category": r[3],
        }
        for r in results
    ]

    return embeddings, metadata


def text_search(
    query: str,
    embedder,
    conn: duckdb.DuckDBPyConnection,
    model_key: str,
    k: int = 10,
) -> list[dict]:
    """テキストクエリで類似画像を検索

    Args:
        query: 検索クエリ（自然言語）
        embedder: 埋め込みモデル
        conn: データベース接続
        model_key: モデルキー（"CLIP-L" or "SigLIP"）
        k: 取得件数

    Returns:
        検索結果のリスト（score, metadata）
    """
    model_name = MODEL_CONFIGS[model_key]["model_name"]
    embedding_dim = MODEL_CONFIGS[model_key]["embedding_dim"]

    # テキストを埋め込み
    query_embedding = embedder.embed_text(query)

    # 類似検索
    results = conn.execute(
        f"""
        SELECT
            c.id,
            c.file_path,
            c.file_name,
            c.category,
            1.0 - array_cosine_distance(e.embedding, ?::FLOAT[{embedding_dim}]) as score
        FROM image_embeddings e
        JOIN image_catalog c ON e.id = c.id
        WHERE e.model_name = ?
        ORDER BY array_cosine_distance(e.embedding, ?::FLOAT[{embedding_dim}])
        LIMIT ?
    """,
        [query_embedding.tolist(), model_name, query_embedding.tolist(), k],
    ).fetchall()

    return [
        {
            "id": r[0],
            "file_path": r[1],
            "file_name": r[2],
            "category": r[3],
            "score": r[4],
        }
        for r in results
    ]


def image_search(
    image: Image.Image,
    embedder,
    conn: duckdb.DuckDBPyConnection,
    model_key: str,
    k: int = 10,
) -> list[dict]:
    """画像から類似画像を検索

    Args:
        image: クエリ画像
        embedder: 埋め込みモデル
        conn: データベース接続
        model_key: モデルキー（"CLIP-L" or "SigLIP"）
        k: 取得件数

    Returns:
        検索結果のリスト（score, metadata）
    """
    model_name = MODEL_CONFIGS[model_key]["model_name"]
    embedding_dim = MODEL_CONFIGS[model_key]["embedding_dim"]

    # 画像を埋め込み
    query_embedding = embedder.embed_image(image)

    # 類似検索
    results = conn.execute(
        f"""
        SELECT
            c.id,
            c.file_path,
            c.file_name,
            c.category,
            1.0 - array_cosine_distance(e.embedding, ?::FLOAT[{embedding_dim}]) as score
        FROM image_embeddings e
        JOIN image_catalog c ON e.id = c.id
        WHERE e.model_name = ?
        ORDER BY array_cosine_distance(e.embedding, ?::FLOAT[{embedding_dim}])
        LIMIT ?
    """,
        [query_embedding.tolist(), model_name, query_embedding.tolist(), k],
    ).fetchall()

    return [
        {
            "id": r[0],
            "file_path": r[1],
            "file_name": r[2],
            "category": r[3],
            "score": r[4],
        }
        for r in results
    ]


def get_image_count(conn: duckdb.DuckDBPyConnection, model_key: str) -> int:
    """指定モデルの埋め込み済み画像数を取得"""
    model_name = MODEL_CONFIGS[model_key]["model_name"]
    result = conn.execute(
        "SELECT COUNT(*) FROM image_embeddings WHERE model_name = ?",
        [model_name],
    ).fetchone()
    return result[0] if result else 0
