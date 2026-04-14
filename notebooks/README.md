# Experiment Notebooks

## 01-04: 基礎構築・データ準備

| # | Notebook | 概要 |
|---|----------|------|
| 01 | [build-image-catalog](01-build-image-catalog.ipynb) | 画像カタログの作成 |
| 02 | [vectorize-images](02-vectorize-images.ipynb) | 画像のベクトル化 |
| 03 | [search-visualization](03-search-visualization.ipynb) | 検索結果の可視化 |
| 04 | [dimensionality-reduction-visualization](04-dimensionality-reduction-visualization.ipynb) | 次元削減による画像ベクトル可視化 |

## 11-19: Embeddingモデル評価

| # | Notebook | 概要 |
|---|----------|------|
| 11 | [eval-siglip](11-eval-siglip.ipynb) | SigLIP モデル評価（ベースライン） |
| 12 | [eval-clip](12-eval-clip.ipynb) | CLIP ViT-L/14 モデル評価 |
| 13 | [eval-dinov2](13-eval-dinov2.ipynb) | DINOv2 ViT-L モデル評価 |
| 14 | [eval-openclip](14-eval-openclip.ipynb) | OpenCLIP ViT-H/14 モデル評価 |
| 15 | [eval-japanese-clip](15-eval-japanese-clip.ipynb) | Japanese CLIP (Rinna) モデル評価 ⚠️スキップ |
| 16 | [eval-jina-clip](16-eval-jina-clip.ipynb) | Jina CLIP v2 モデル評価 ⚠️スキップ |
| 17 | [clip-performance-benchmark](17-clip-performance-benchmark.ipynb) | CLIP-L パフォーマンスベンチマーク |
| 18 | [clip-cpu-inference](18-clip-cpu-inference.ipynb) | CLIP-L CPU推論の検証 |
| 19 | [model-comparison](19-model-comparison.ipynb) | 複数モデル比較分析 |

## 21-25: 検索評価・ベクトル空間分析

| # | Notebook | 概要 |
|---|----------|------|
| 21 | [similarity-search-evaluation](21-similarity-search-evaluation.ipynb) | 類似画像検索の評価 |
| 22 | [text-search-evaluation](22-text-search-evaluation.ipynb) | テキスト検索の評価 |
| 23 | [text-search-siglip](23-text-search-siglip.ipynb) | テキスト検索の評価（SigLIP） |
| 24 | [multilingual-text-search](24-multilingual-text-search.ipynb) | 多言語テキスト検索の評価 |
| 25 | [vector-space-analysis](25-vector-space-analysis.ipynb) | ベクトル空間分析 |

## 31-41: JavaScript (Transformers.js) 互換性検証

| # | Notebook | 概要 |
|---|----------|------|
| 31 | [js-setup-and-embedding](31-js-setup-and-embedding.ipynb) | JS環境構築と埋め込み生成 |
| 32 | [embedding-similarity-analysis](32-embedding-similarity-analysis.ipynb) | 埋め込みベクトル定量比較分析 |
| 33 | [search-quality-comparison](33-search-quality-comparison.ipynb) | 検索品質への影響評価 |
| 34 | [chromium-browser-evaluation](34-chromium-browser-evaluation.ipynb) | Chromium ブラウザ評価 |
| 35 | [comprehensive-summary](35-comprehensive-summary.ipynb) | Python vs JS SigLIP 総合サマリー |
| 36 | [store-js-embeddings-in-duckdb](36-store-js-embeddings-in-duckdb.ipynb) | JS埋め込みを DuckDB に保存 |
| 37 | [tokenizer-debug](37-tokenizer-debug.ipynb) | トークナイザデバッグ: Python vs JS 日本語処理の差異分析 |
| 38 | [text-preprocessing](38-text-preprocessing.ipynb) | テキスト前処理戦略とtokenizer修正の評価 |
| 39 | [comprehensive-evaluation](39-comprehensive-evaluation.ipynb) | 総合評価: JS環境でのSigLIPテキスト→画像検索の課題と対策 |
| 40 | [byte-fallback-fix](40-byte-fallback-fix.ipynb) | byte_fallback 修正による日本語テキスト検索の改善実験 |
| 41 | [clip-l-js-evaluation](41-clip-l-js-evaluation.ipynb) | CLIP-L (JS) での日本語テキスト→画像検索実験 |

## 51-59: 顔認識・顔検索

| # | Notebook | 概要 |
|---|----------|------|
| 51 | [face-detection-insightface](51-face-detection-insightface.ipynb) | 顔検出の基礎: InsightFace セットアップと動作確認 |
| 52 | [face-embedding-storage](52-face-embedding-storage.ipynb) | 顔Embedding抽出とDuckDB保存 |
| 53 | [face-similarity-search](53-face-similarity-search.ipynb) | 顔類似検索: 同一人物の発見 |
| 54 | [face-clustering](54-face-clustering.ipynb) | 顔クラスタリング: 自動人物グループ化 |
| 55 | [face-search-evaluation](55-face-search-evaluation.ipynb) | Ground Truth と顔検索精度の定量評価 |
| 56 | [detection-model-comparison](56-detection-model-comparison.ipynb) | 検出モデル比較: InsightFace vs MediaPipe vs DeepFace |
| 57 | [face-search-at-scale](57-face-search-at-scale.ipynb) | 大規模顔検索: pyconjp 23,628枚での人物検索 |
| 58 | [face-recognition-summary](58-face-recognition-summary.ipynb) | 顔認識実験シリーズ: 総合まとめ |
| 59 | [face-embedding-js](59-face-embedding-js.ipynb) | 顔Embedding: Python vs JS 互換性検証 |

## 61-68: SigLIP 2 評価

| # | Notebook | 概要 |
|---|----------|------|
| 61 | [eval-siglip2](61-eval-siglip2.ipynb) | SigLIP 2 モデル評価（ベースライン） |
| 62 | [text-search-siglip2](62-text-search-siglip2.ipynb) | SigLIP 2 テキスト検索の評価 |
| 63 | [siglip2-variants-comparison](63-siglip2-variants-comparison.ipynb) | SigLIP 2 モデルバリエーション比較 |
| 64 | [siglip2-js-embedding](64-siglip2-js-embedding.ipynb) | SigLIP 2 JS 埋め込み（Python vs JS 比較） |
| 65 | [siglip2-js-tokenizer-japanese](65-siglip2-js-tokenizer-japanese.ipynb) | SigLIP 2 JS トークナイザー＋日本語テキスト評価 |
| 66 | [comprehensive-model-comparison](66-comprehensive-model-comparison.ipynb) | SigLIP 2 総合比較サマリー |
| 67 | [siglip2-large-onnx](67-siglip2-large-onnx.ipynb) | SigLIP 2 Large ONNX 評価（Python vs JS 総合比較） |
| 68 | [siglip2-large-384-onnx](68-siglip2-large-384-onnx.ipynb) | SigLIP 2 Large 384 ONNX 評価（Python vs JS） |

## 71-78: 画像キャプション・画像→テキスト

| # | Notebook | 概要 |
|---|----------|------|
| 71 | [retrieval-based-captioning](71-retrieval-based-captioning.ipynb) | 既存Embeddingモデルによる画像→テキスト近似 |
| 72 | [blip2-captioning](72-blip2-captioning.ipynb) | BLIP-2 画像キャプション生成 |
| 73 | [florence2-captioning](73-florence2-captioning.ipynb) | Florence-2 マルチタスク画像理解 |
| 74 | [qwen25vl-captioning](74-qwen25vl-captioning.ipynb) | Qwen2.5-VL-7B 多言語キャプション生成 |
| 75 | [internvl-and-model-comparison](75-internvl-and-model-comparison.ipynb) | InternVL2.5-8B とモデル比較総括 |
| 76 | [caption-enhanced-search](76-caption-enhanced-search.ipynb) | キャプション→Embedding 検索パイプライン |
| 77 | [colpali-late-interaction](77-colpali-late-interaction.ipynb) | ColPali 遅延相互作用モデル |
| 78 | [captioning-summary](78-captioning-summary.ipynb) | 画像→テキスト実験総括 |

## 81-88: 物体検出

| # | Notebook | 概要 |
|---|----------|------|
| 81 | [yolo11-object-detection](81-yolo11-object-detection.ipynb) | YOLO11 固定クラス物体検出 |
| 82 | [yolo-world-open-vocab](82-yolo-world-open-vocab.ipynb) | YOLO-World オープン語彙検出 |
| 83 | [grounding-dino](83-grounding-dino.ipynb) | Grounding DINO テキストプロンプト検出 |
| 84 | [owlv2-zero-one-shot](84-owlv2-zero-one-shot.ipynb) | OWLv2 ゼロショット + ワンショット検出 |
| 85 | [omdet-rtdetr-speed](85-omdet-rtdetr-speed.ipynb) | OmDet-Turbo + RT-DETRv2 速度・精度比較 |
| 86 | [detection-statistics](86-detection-statistics.ipynb) | 物体検出統計の総合分析 |
| 87 | [object-search-evaluation](87-object-search-evaluation.ipynb) | 物体ベース画像検索評価 |
| 88 | [object-detection-summary](88-object-detection-summary.ipynb) | 物体検出実験総括 |

## 99: 大規模スケール評価

| # | Notebook | 概要 |
|---|----------|------|
| 99 | [pyconjp-scale-evaluation](99-pyconjp-scale-evaluation.ipynb) | PyCon JP 大規模データでのスケール検証 |

## 100-103: Voronoi パーティション (Firestore 向け近似検索)

| # | Notebook | 概要 |
|---|----------|------|
| 100 | [voronoi-partition-firestore](100-voronoi-partition-firestore.ipynb) | Voronoi 分割による Firestore 向け近似最近傍検索 |
| 101 | [voronoi-scale-analysis](101-voronoi-scale-analysis.ipynb) | Voronoi 検索のスケール分析（ピボット数・割り当て数・top-k の最適化） |
| 102 | [voronoi-generalization](102-voronoi-generalization.ipynb) | 汎化性能実験（Train/Test 分割 + 増分追加シミュレーション） |
| 103 | [voronoi-js-search](103-voronoi-js-search.ipynb) | JS 互換性検証: Python vs JS Voronoi 検索の完全一致確認 |

## 104-107: Voronoi テキスト→画像検索への拡張

| # | Notebook | 概要 |
|---|----------|------|
| 104 | [voronoi-text-to-image](104-voronoi-text-to-image.ipynb) | Voronoi 分割のテキスト→画像検索への適用可能性（SigLIP 2 Large, 378 件） |
| 105 | [voronoi-text-image-scale](105-voronoi-text-image-scale.ipynb) | 大規模評価: 23,464 件での Voronoi テキスト→画像パラメータスイープ |
| 106 | [voronoi-text-image-generalization](106-voronoi-text-image-generalization.ipynb) | ピボットの汎化性能検証（Train/Test 分割 + 増分追加） |
| 107 | [voronoi-pivot-export](107-voronoi-pivot-export.ipynb) | 3 モデル（SigLIP 2 Base/Large, CLIP-L）のピボット学習・エクスポート |

## 111-114: Voronoi vs 厳密 kNN 比較

| # | Notebook | 概要 |
|---|----------|------|
| 111 | [voronoi-vs-exact-knn-text-to-image](111-voronoi-vs-exact-knn-text-to-image.ipynb) | テキスト→画像: 厳密 kNN vs Voronoi（SigLIP 2 Large, 23K 件） |
| 112 | [voronoi-vs-exact-knn-image-to-image](112-voronoi-vs-exact-knn-image-to-image.ipynb) | 画像→画像: 厳密 kNN vs Voronoi（SigLIP 2 Large, 23K 件） |
| 113 | [voronoi-vs-exact-knn-fair-eval](113-voronoi-vs-exact-knn-fair-eval.ipynb) | 公正評価: カタログ 378 件のみ（ラベル欠損排除）→ MRR_ratio = 1.000 |
| 114 | [voronoi-vs-exact-knn-face](114-voronoi-vs-exact-knn-face.ipynb) | 顔認識: 厳密 kNN vs Voronoi（ArcFace 512-D, 75K 顔） |

## 115-118: HNSW 比較と埋め込み特性レポート

| # | Notebook | 概要 |
|---|----------|------|
| 115 | [hnsw-vs-knn-voronoi-text-to-image](115-hnsw-vs-knn-voronoi-text-to-image.ipynb) | 3-way 比較: 厳密 kNN / Voronoi / HNSW（テキスト→画像, DuckDB VSS） |
| 116 | [hnsw-vs-knn-voronoi-image-to-image](116-hnsw-vs-knn-voronoi-image-to-image.ipynb) | 3-way 比較: 画像→画像（同一モダリティで HNSW ef=128 → BF-R@10=1.000） |
| 117 | [hnsw-vs-knn-voronoi-face](117-hnsw-vs-knn-voronoi-face.ipynb) | 3-way 比較: 顔認識（ArcFace, 75K 顔, HNSW ef=16 でも BF-R@10=0.967） |
| 118 | [embedding-characteristics-report](118-embedding-characteristics-report.ipynb) | 埋め込み空間特性レポート: 異方性・Two-tower・モダリティギャップ × ANN 性能 |

## 119-122: HNSW 最適化とライブラリ比較

| # | Notebook | 概要 |
|---|----------|------|
| 119 | [hnsw-extended-sweep-text-to-image](119-hnsw-extended-sweep-text-to-image.ipynb) | HNSW 拡張ハイパラスイープ（ef_search 2048 で BF-R@10=1.000, M/ef_c 拡張） |
| 120 | [modality-gap-correction](120-modality-gap-correction.ipynb) | モダリティギャップ補正実験 → シフト補正は失敗、HNSW+Rerank k=100 が有効 |
| 121 | [voronoi-hnsw-hybrid](121-voronoi-hnsw-hybrid.ipynb) | Voronoi + HNSW ハイブリッド → ハイブリッドは Voronoi 単独と同等で不要と結論 |
| 122 | [hnswlib-comparison](122-hnswlib-comparison.ipynb) | hnswlib vs DuckDB VSS: hnswlib が 10-70x 高速、構築も 4x 高速 |

## 131-138: Jina CLIP v2 調査シリーズ

SigLIP 2 Large 256 との直接対決。`transformers>=5.0` で PyTorch 経路が壊れているため **公式 ONNX + onnxruntime-gpu** で迂回（`JinaCLIPONNXEmbedder` を実装）。

| # | Notebook | 概要 |
|---|----------|------|
| 131 | [eval-jina-clip-v2](131-eval-jina-clip-v2.ipynb) | 画像埋め込み品質（ONNX fp32, 1024-D, 378 枚）→ Trustworthiness 2D=0.974, 3D 距離比 0.550 で全モデル中最高 |
| 132 | [text-search-jina-clip-v2](132-text-search-jina-clip-v2.ipynb) | Text→Image MRR 評価（18 クエリ）: Overall 0.881 / EN 0.896 / JA 0.852（SigLIP 2 L256 の JA 1.000 に負け） |
| 133 | [vector-space-jina-clip-v2](133-vector-space-jina-clip-v2.ipynb) | ベクトル空間特性: モダリティギャップ・異方性・Matryoshka → **32-D truncate で MRR 0.972**（SigLIP 2 L256 は 0.475 で崩壊） |
| 134 | [query-engineering-jina-clip-v2](134-query-engineering-jina-clip-v2.ipynb) | プロンプトプレフィクス実験 → `Query: ` prefix で **EN MRR 0.896→1.000** に跳ね上がる |
| 135 | [onnx-quantization-jina-clip-v2](135-onnx-quantization-jina-clip-v2.ipynb) | fp32/fp16/int8/q4/q4f16 比較: **fp16 が fp32 と完全一致で最速**、**q4f16 は 861MB で MRR 0.905** 維持（int8 は GPU 上で 2000x 遅く実用不可） |
| 136 | [js-compatibility-jina-clip-v2](136-js-compatibility-jina-clip-v2.ipynb) | Transformers.js (`JinaCLIPModel`) で **Python vs JS テキスト cos=0.999998**（完全再現）・画像 cos=0.979（PIL vs Sharp 差） |
| 138 | [summary-jina-clip-v2](138-summary-jina-clip-v2.ipynb) | 総合まとめ：**置き換え見送り推奨**。ただし多言語化・ブラウザ配布・100 万枚超スケール時は Jina v2 + 32-D Matryoshka + q4f16 が有力候補 |
