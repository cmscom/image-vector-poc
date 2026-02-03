# Streamlit画像検索アプリ実装計画

## 概要
DuckDBに保存された画像埋め込みを使用して、テキスト検索・画像検索・3D可視化を行うStreamlitアプリを作成する。

## 機能要件
1. **モデル切り替え**: CLIP-L / SigLIP を選択可能
2. **テキスト検索**: 自然言語クエリから類似画像を10件抽出
3. **画像検索**: アップロード画像から類似画像を10件抽出
4. **3Dグラフ**: PCA/t-SNEで次元削減し、画像をプレビュー可能な3Dプロット

## 技術スタック
- **UI**: Streamlit
- **グラフ**: Plotly (3Dインタラクティブ)
- **次元削減**: scikit-learn (PCA, t-SNE)
- **DB**: DuckDB (既存のdata/images.duckdb)
- **埋め込み**: image_vector_poc.CLIPEmbedder / SigLIPEmbedder

## ファイル構成
```
app/
├── __init__.py
├── main.py              # Streamlitエントリーポイント
├── search.py            # 検索ロジック
├── visualization.py     # 3D可視化ロジック
└── config.py            # 設定・定数
```

## 実装詳細

### 1. 依存関係の追加 (pyproject.toml)
```toml
[dependency-groups]
dev = [
    ...
    "streamlit>=1.45.0",
]
```

### 2. config.py
- DB_PATH, MODEL_CONFIGS (モデル名と表示名のマッピング)
- カテゴリ色設定

### 3. search.py
- `load_embeddings(model_name)`: DBから埋め込みとメタデータを読み込み
- `text_search(query, embedder, embeddings, metadata, k=10)`: テキスト検索
- `image_search(image, embedder, embeddings, metadata, k=10)`: 画像検索

### 4. visualization.py
- `reduce_dimensions(embeddings, method="pca", n_components=3)`: 次元削減
- `create_3d_plot(coords, metadata, selected_idx=None)`: Plotly 3Dプロット

### 5. main.py (Streamlit)
```python
# サイドバー
- モデル選択 (CLIP-L / SigLIP)
- 機能選択 (テキスト検索 / 画像検索 / 3D可視化)

# メインエリア
- テキスト検索: テキスト入力 → 結果グリッド表示
- 画像検索: 画像アップロード → 結果グリッド表示
- 3D可視化: 次元削減手法選択 → インタラクティブ3Dプロット
```

## UI設計

### サイドバー
- モデル選択ドロップダウン
- 機能タブ切り替え

### テキスト検索画面
- テキスト入力ボックス
- 検索ボタン
- 結果: 2行×5列のグリッド (類似度スコア、カテゴリ表示)

### 画像検索画面
- 画像アップローダー
- クエリ画像プレビュー
- 結果: 2行×5列のグリッド

### 3D可視化画面
- 次元削減手法選択 (PCA / t-SNE)
- モデル選択反映
- Plotly 3Dプロット (カテゴリで色分け)
- 点クリックで画像プレビュー

## 実装手順

1. **Step 1**: pyproject.tomlにstreamlitを追加
2. **Step 2**: app/config.py 作成
3. **Step 3**: app/search.py 作成
4. **Step 4**: app/visualization.py 作成
5. **Step 5**: app/main.py 作成
6. **Step 6**: 動作確認・調整

## 実行方法
```bash
uv run streamlit run app/main.py
```

## 注意点
- モデル読み込みは重いので、`@st.cache_resource`でキャッシュ
- 埋め込みデータも`@st.cache_data`でキャッシュ
- t-SNEは計算に時間がかかるのでプログレス表示
- 画像パスはDBに保存されている相対パスを使用
