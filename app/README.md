# Image Vector Search - Streamlitアプリ

DuckDBに保存された画像埋め込みを使用して、テキスト検索・画像検索・3D可視化を行うStreamlitアプリ。

## 実行方法

```bash
uv run streamlit run app/main.py
```

ブラウザで http://localhost:8501 にアクセス。

## 機能

### 1. モデル切り替え
サイドバーで以下のモデルを選択可能:
- **CLIP-L** (OpenAI): `openai/clip-vit-large-patch14`
- **SigLIP** (Google): `google/siglip-base-patch16-224`

### 2. テキスト検索
自然言語クエリから類似画像を10件抽出。

例: 「青い空」「猫」「食べ物」

### 3. 画像検索
アップロードした画像から類似画像を10件抽出。

対応形式: JPG, JPEG, PNG, WEBP

### 4. 3D可視化
画像埋め込みを3次元空間にマッピング。

- **次元削減手法**: PCA / t-SNE
- **カテゴリ別色分け**: 凡例で確認可能
- **マウスオーバー**: 画像の詳細情報を表示
- **画像プレビュー**: セレクトボックスで選択

## ファイル構成

```
app/
├── __init__.py          # パッケージ初期化
├── config.py            # 設定（DB_PATH, モデル設定, カテゴリ色）
├── search.py            # 検索ロジック
├── visualization.py     # 3D可視化ロジック
├── main.py              # Streamlitエントリーポイント
└── README.md            # このファイル
```

## 依存関係

- streamlit
- plotly
- scikit-learn (PCA, t-SNE)
- duckdb (VSS拡張)
- image_vector_poc (CLIPEmbedder, SigLIPEmbedder)

## データベース

`data/images.duckdb`を使用。以下のテーブルが必要:

- `image_catalog`: 画像メタデータ (id, file_path, file_name, category)
- `image_embeddings`: 埋め込みベクトル (id, model_name, embedding)

## 注意事項

- モデル読み込みは`@st.cache_resource`でキャッシュ
- 埋め込みデータは`@st.cache_data`でキャッシュ
- t-SNEは計算に時間がかかるためプログレス表示あり
- 3Dプロットのクリック選択はStreamlit制限により未対応
