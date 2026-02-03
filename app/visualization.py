"""3D可視化ロジック"""

import sys
from pathlib import Path

# appディレクトリの親をパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from app.config import CATEGORY_COLORS, DEFAULT_COLOR


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "pca",
    n_components: int = 3,
    perplexity: int = 30,
) -> np.ndarray:
    """次元削減を実行

    Args:
        embeddings: (N, dim) の埋め込み行列
        method: "pca" または "tsne"
        n_components: 出力次元数（デフォルト: 3）
        perplexity: t-SNEのperplexityパラメータ

    Returns:
        (N, n_components) の座標行列
    """
    if method == "pca":
        reducer = PCA(n_components=n_components)
        coords = reducer.fit_transform(embeddings)
    elif method == "tsne":
        # t-SNEは直接3次元に削減するか、PCAで前処理してから適用
        if embeddings.shape[1] > 50:
            # 高次元の場合はPCAで前処理
            pca = PCA(n_components=50)
            embeddings_reduced = pca.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings

        reducer = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(embeddings) - 1),
            random_state=42,
            max_iter=1000,
        )
        coords = reducer.fit_transform(embeddings_reduced)
    else:
        raise ValueError(f"Unknown method: {method}")

    return coords


def create_3d_plot(
    coords: np.ndarray,
    metadata: list[dict],
    selected_idx: int | None = None,
) -> go.Figure:
    """Plotly 3Dプロットを作成

    Args:
        coords: (N, 3) の座標行列
        metadata: 各点のメタデータ
        selected_idx: 選択されている点のインデックス（ハイライト用）

    Returns:
        Plotly Figure
    """
    # カテゴリごとに色を割り当て（ソートして順序を固定）
    categories = sorted(set(m["category"] for m in metadata))

    # 色リスト生成
    colors = [
        CATEGORY_COLORS.get(m["category"], DEFAULT_COLOR) for m in metadata
    ]

    # ホバーテキスト生成
    hover_texts = [
        f"ID: {m['id']}<br>"
        f"Category: {m['category']}<br>"
        f"File: {m['file_name']}"
        for m in metadata
    ]

    # サイズ（選択されている点は大きく）
    sizes = [8] * len(metadata)
    if selected_idx is not None and 0 <= selected_idx < len(metadata):
        sizes[selected_idx] = 20

    fig = go.Figure()

    # カテゴリごとにトレースを追加（凡例用）
    for category in categories:
        cat_indices = [i for i, m in enumerate(metadata) if m["category"] == category]
        if not cat_indices:
            continue

        cat_coords = coords[cat_indices]
        cat_colors = [colors[i] for i in cat_indices]
        cat_hover = [hover_texts[i] for i in cat_indices]
        cat_sizes = [sizes[i] for i in cat_indices]
        # customdataにメタデータのインデックスを保存（選択時に使用）
        cat_metadata_indices = cat_indices

        fig.add_trace(
            go.Scatter3d(
                x=cat_coords[:, 0],
                y=cat_coords[:, 1],
                z=cat_coords[:, 2],
                mode="markers",
                name=category,
                marker=dict(
                    size=cat_sizes,
                    color=CATEGORY_COLORS.get(category, DEFAULT_COLOR),
                    opacity=0.8,
                    line=dict(width=0.5, color="white"),
                ),
                text=cat_hover,
                hoverinfo="text",
                customdata=cat_metadata_indices,
            )
        )

    # レイアウト設定
    fig.update_layout(
        title="Image Embeddings 3D Visualization",
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )

    return fig


def get_point_info(
    click_data: dict | None,
    metadata: list[dict],
) -> dict | None:
    """クリックされた点の情報を取得

    Args:
        click_data: Plotlyのクリックデータ
        metadata: メタデータリスト

    Returns:
        クリックされた点のメタデータ、またはNone
    """
    if click_data is None:
        return None

    try:
        # Plotlyのクリックデータから点のIDを取得
        point_data = click_data["points"][0]
        point_id = point_data.get("customdata")

        if point_id is not None:
            # IDからメタデータを検索
            for m in metadata:
                if m["id"] == point_id:
                    return m
    except (KeyError, IndexError):
        pass

    return None
