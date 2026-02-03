"""Streamlit 画像検索アプリ"""

import sys
from pathlib import Path

# appディレクトリの親をパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image

from image_vector_poc import CLIPEmbedder, SigLIPEmbedder

from app.config import DEFAULT_TOP_K, MODEL_CONFIGS
from app.search import (
    get_connection,
    get_image_count,
    image_search,
    load_embeddings,
    text_search,
)
from app.visualization import create_3d_plot, get_point_info, reduce_dimensions

# ページ設定
st.set_page_config(
    page_title="Image Vector Search",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def load_embedder(model_key: str):
    """モデルをロード（キャッシュ）"""
    if model_key == "CLIP-L":
        return CLIPEmbedder(device="cuda")
    else:
        return SigLIPEmbedder(device="cuda")


@st.cache_data
def get_cached_embeddings(_conn, model_key: str):
    """埋め込みデータをロード（キャッシュ）"""
    return load_embeddings(_conn, model_key)


@st.cache_data
def get_cached_3d_coords(_embeddings, method: str, perplexity: int = 30):
    """3D座標を計算（キャッシュ）"""
    import numpy as np
    # tupleからnumpy配列に変換
    embeddings_array = np.array(_embeddings, dtype=np.float32)
    return reduce_dimensions(embeddings_array, method=method, perplexity=perplexity)


def display_search_results(results: list[dict], cols: int = 5):
    """検索結果をグリッド表示"""
    if not results:
        st.warning("検索結果がありません")
        return

    rows = (len(results) + cols - 1) // cols

    for row in range(rows):
        columns = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx < len(results):
                result = results[idx]
                with columns[col]:
                    try:
                        img = Image.open(result["file_path"])
                        st.image(img, use_container_width=True)
                        st.caption(
                            f"Score: {result['score']:.3f}\n{result['category']}"
                        )
                    except Exception as e:
                        st.error(f"画像読み込みエラー: {e}")


@st.dialog("画像プレビュー", width="large")
def show_image_dialog(metadata: dict):
    """画像をダイアログで表示"""
    try:
        img = Image.open(metadata["file_path"])
        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"画像読み込みエラー: {e}")

    st.divider()
    st.write(f"**ID:** {metadata['id']}")
    st.write(f"**カテゴリ:** {metadata['category']}")
    st.write(f"**ファイル名:** {metadata['file_name']}")
    st.write(f"**パス:** {metadata['file_path']}")


def main():
    st.title("Image Vector Search")

    # サイドバー
    with st.sidebar:
        st.header("設定")

        # モデル選択
        model_key = st.selectbox(
            "モデル選択",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda x: MODEL_CONFIGS[x]["display_name"],
        )

        # 機能選択
        feature = st.radio(
            "機能",
            options=["テキスト検索", "画像検索", "3D可視化"],
        )

        st.divider()

        # DB接続とモデルロード状態
        with st.spinner("データベース接続中..."):
            conn = get_connection()
            image_count = get_image_count(conn, model_key)

        st.info(f"検索対象画像数: {image_count}")

        if image_count == 0:
            st.error(f"{model_key}の埋め込みデータがありません")
            return

    # モデルロード
    with st.spinner(f"{model_key}モデルを読み込み中..."):
        embedder = load_embedder(model_key)

    # メインエリア
    if feature == "テキスト検索":
        st.header("テキスト検索")
        st.write("自然言語クエリから類似画像を検索します")

        query = st.text_input("検索クエリ", placeholder="例: 青い空、猫、食べ物")

        if st.button("検索", type="primary") and query:
            with st.spinner("検索中..."):
                results = text_search(
                    query=query,
                    embedder=embedder,
                    conn=conn,
                    model_key=model_key,
                    k=DEFAULT_TOP_K,
                )

            st.subheader(f'"{query}" の検索結果')
            display_search_results(results)

    elif feature == "画像検索":
        st.header("画像検索")
        st.write("アップロード画像から類似画像を検索します")

        uploaded_file = st.file_uploader(
            "画像をアップロード",
            type=["jpg", "jpeg", "png", "webp"],
        )

        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("クエリ画像")
                st.image(query_image, use_container_width=True)

            with col2:
                if st.button("検索", type="primary"):
                    with st.spinner("検索中..."):
                        results = image_search(
                            image=query_image,
                            embedder=embedder,
                            conn=conn,
                            model_key=model_key,
                            k=DEFAULT_TOP_K,
                        )

                    st.subheader("類似画像")
                    display_search_results(results)

    elif feature == "3D可視化":
        st.header("3D可視化")
        st.write("画像埋め込みを3次元空間に可視化します")

        # 次元削減手法選択
        method = st.selectbox(
            "次元削減手法",
            options=["pca", "tsne"],
            format_func=lambda x: "PCA" if x == "pca" else "t-SNE",
        )

        perplexity = 30
        if method == "tsne":
            perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30)

        # データロード
        with st.spinner("埋め込みデータを読み込み中..."):
            embeddings, metadata = get_cached_embeddings(conn, model_key)

        if len(embeddings) == 0:
            st.error("埋め込みデータがありません")
            return

        st.info(f"データ数: {len(embeddings)}")

        # 次元削減
        with st.spinner(f"{method.upper()}で次元削減中..."):
            coords = get_cached_3d_coords(
                tuple(map(tuple, embeddings)),  # hashableにするためtupleに変換
                method=method,
                perplexity=perplexity,
            )

        # 3Dプロット
        fig = create_3d_plot(coords, metadata)
        st.plotly_chart(fig, use_container_width=True, key="3d_plot")

        st.divider()
        st.caption("※ グラフ上でマウスオーバーすると詳細が表示されます。下のセレクトボックスで画像を選択できます。")

        # 画像選択UI
        st.subheader("画像を選択してプレビュー")

        col1, col2 = st.columns(2)
        with col1:
            # カテゴリでフィルタリング
            categories = sorted(set(m["category"] for m in metadata))
            selected_category = st.selectbox(
                "カテゴリ",
                options=["すべて"] + categories,
                key="viz_category",
            )

        # フィルタリングされた画像リスト
        if selected_category == "すべて":
            filtered_metadata = metadata
        else:
            filtered_metadata = [m for m in metadata if m["category"] == selected_category]

        with col2:
            # 画像選択
            image_options = [f"{m['file_name']}" for m in filtered_metadata]
            selected_idx = st.selectbox(
                f"画像 ({len(filtered_metadata)}件)",
                options=range(len(image_options)),
                format_func=lambda i: image_options[i],
                key="viz_image",
            )

        if selected_idx is not None and filtered_metadata:
            selected_metadata = filtered_metadata[selected_idx]

            col1, col2 = st.columns([1, 2])
            with col1:
                try:
                    img = Image.open(selected_metadata["file_path"])
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"画像読み込みエラー: {e}")
            with col2:
                st.write(f"**ID:** {selected_metadata['id']}")
                st.write(f"**カテゴリ:** {selected_metadata['category']}")
                st.write(f"**ファイル名:** {selected_metadata['file_name']}")
                st.write(f"**パス:** {selected_metadata['file_path']}")


if __name__ == "__main__":
    main()
