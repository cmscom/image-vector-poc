"""アプリケーション設定"""

from pathlib import Path

# データベースパス
DB_PATH = Path(__file__).parent.parent / "data" / "images.duckdb"

# モデル設定
MODEL_CONFIGS = {
    "CLIP-L": {
        "model_name": "openai/clip-vit-large-patch14",
        "display_name": "CLIP-L (OpenAI)",
        "embedding_dim": 768,
    },
    "SigLIP": {
        "model_name": "google/siglip-base-patch16-224",
        "display_name": "SigLIP (Google)",
        "embedding_dim": 768,
    },
}

# 検索設定
DEFAULT_TOP_K = 10

# カテゴリ色設定（Plotly用）
CATEGORY_COLORS = {
    "EuroPython2025": "#1f77b4",
    "PyConJP2025": "#ff7f0e",
    "PyConJP2025-PreCampHiroshima": "#2ca02c",
    "KashiwaVillagePark2026": "#d62728",
    "TokyoNight202505": "#9467bd",
    "terada": "#8c564b",
}

# デフォルト色（未定義カテゴリ用）
DEFAULT_COLOR = "#7f7f7f"
