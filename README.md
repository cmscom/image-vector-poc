# Image Vector PoC

画像のベクトル化と意味検索の実験プロジェクト。

## 目的

- 画像の意味検索（自然言語から検索）
- 類似画像検索
- 画像の自動タグ付け
- （将来）顔や物体の抽出、同一人物の認識

## 環境構築

```bash
# Python 3.13 + 依存関係のインストール
uv sync

# テスト実行
uv run pytest
```

## プロジェクト構成

```
image-vector-poc/
├── src/image_vector_poc/    # メインパッケージ
│   ├── core/                # 型定義・プロトコル
│   ├── embeddings/          # 埋め込みモデル (SigLIP等)
│   ├── storage/             # ベクトルストレージ (DuckDB)
│   ├── search/              # 検索機能
│   └── tagging/             # 自動タグ付け
├── tests/                   # テストコード
├── notebooks/               # 実験用Notebook
└── data/                    # データ保存 (gitignore)
```

## 主要コンポーネント

| クラス | 説明 |
|-------|------|
| `SigLIPEmbedder` | 画像・テキストをベクトル化 |
| `DuckDBVectorStore` | ベクトルの保存・HNSW検索 |
| `SemanticSearch` | テキスト/画像による検索 |
| `ZeroShotTagger` | ゼロショット画像分類 |

## 実験の進め方

### 1. notebooksフォルダの作成

```bash
mkdir notebooks
```

### 2. Jupyter Labの起動

```bash
uv run jupyter lab
```

### 3. 基本的な実験フロー

```python
from image_vector_poc import SigLIPEmbedder, DuckDBVectorStore, SemanticSearch
from pathlib import Path

# 1. モデルとストレージの初期化
embedder = SigLIPEmbedder(device="cuda")
store = DuckDBVectorStore("data/images.duckdb", embedder.embedding_dim)
search = SemanticSearch(embedder, store)

# 2. 画像のインデックス作成
image_dir = Path("your_images/")
for i, path in enumerate(image_dir.glob("*.jpg")):
    search.index_image(f"img_{i}", path)

# HNSWインデックス作成（高速検索用）
store.create_index()

# 3. テキストで検索
results = search.search_by_text("青い空と海", k=10)
for r in results:
    print(f"{r.metadata['file_path']}: {r.score:.3f}")

# 4. 類似画像検索
similar = search.search_by_image("query.jpg", k=5)
```

### 4. タグ付け実験

```python
from image_vector_poc import ZeroShotTagger
from PIL import Image

categories = ["風景", "人物", "動物", "食べ物", "建物"]
tagger = ZeroShotTagger(embedder, categories)

image = Image.open("photo.jpg")
tags = tagger.tag(image, top_k=3)
# [('風景', 0.85), ('建物', 0.42), ...]
```

## 技術スタック

- **SigLIP**: Google の Vision-Language モデル（ベースライン）
- **DuckDB VSS**: ベクトル類似検索（HNSW インデックス）
- **PyTorch**: GPU推論
- **transformers**: モデル読み込み
