"""Jina CLIP v2 ONNX embedder (PyTorch bypass via onnxruntime).

Why ONNX: Jina CLIP v2's remote `trust_remote_code` PyTorch path is incompatible
with transformers>=5.0 (meta-tensor init collides with `.item()` in eva_model.py,
and state_dict loader chokes on mixed-type keys). The official ONNX export side-steps
all of that and works identically on Python and Transformers.js.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


class JinaCLIPONNXEmbedder:
    DEFAULT_MODEL = "jinaai/jina-clip-v2"
    DTYPE_FILES = {
        "fp32": "model.onnx",
        "fp16": "model_fp16.onnx",
        "int8": "model_int8.onnx",
        "q4": "model_q4.onnx",
        "q4f16": "model_q4f16.onnx",
        "quantized": "model_quantized.onnx",
        "uint8": "model_uint8.onnx",
        "bnb4": "model_bnb4.onnx",
    }

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dtype: str = "fp32",
        device: str = "cuda",
        cache_dir: str | Path = "data/jina_clip_v2_onnx",
        max_text_length: int = 77,
    ):
        if dtype not in self.DTYPE_FILES:
            raise ValueError(f"dtype must be one of {list(self.DTYPE_FILES)}")
        self._model_name = model_name
        self.dtype = dtype
        self.device = device
        self.max_text_length = max_text_length

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = self._ensure_onnx_files(cache_dir)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._output_names = [o.name for o in self.session.get_outputs()]

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # 全 ONNX variant とも入力は fp32 を期待（内部演算のみ低精度）。
        self._pixel_dtype = np.float32
        self._dummy_ids = np.zeros((1, 2), dtype=np.int64)

    def _ensure_onnx_files(self, cache_dir: Path) -> Path:
        onnx_file = self.DTYPE_FILES[self.dtype]
        local_onnx = cache_dir / onnx_file
        # fp32 のみ external data (.onnx_data) あり
        needs_external = self.dtype == "fp32"
        local_data = cache_dir / f"{onnx_file}_data" if needs_external else None

        if local_onnx.exists() and (not needs_external or (local_data and local_data.exists())):
            return local_onnx

        src = Path(hf_hub_download(self._model_name, f"onnx/{onnx_file}")).resolve()
        shutil.copy(src, local_onnx)
        if needs_external:
            src_data = Path(hf_hub_download(self._model_name, f"onnx/{onnx_file}_data")).resolve()
            shutil.copy(src_data, local_data)
        return local_onnx

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return 1024

    def embed_images(self, images: list[Image.Image]) -> NDArray[np.float32]:
        pv = self.processor(images=images, return_tensors="pt")["pixel_values"]
        pv = pv.numpy().astype(self._pixel_dtype)
        dummy_ids = np.broadcast_to(self._dummy_ids, (len(images), 2)).astype(np.int64, copy=True)
        out = self.session.run(None, {"pixel_values": pv, "input_ids": dummy_ids})
        embs = out[self._output_names.index("l2norm_image_embeddings")]
        return embs.astype(np.float32)

    def embed_image(self, image: Image.Image) -> NDArray[np.float32]:
        return self.embed_images([image])[0]

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        ids = tok["input_ids"].numpy().astype(np.int64)
        dummy_pv = np.zeros((len(texts), 3, 512, 512), dtype=self._pixel_dtype)
        out = self.session.run(None, {"pixel_values": dummy_pv, "input_ids": ids})
        embs = out[self._output_names.index("l2norm_text_embeddings")]
        return embs.astype(np.float32)

    def embed_text(self, text: str) -> NDArray[np.float32]:
        return self.embed_texts([text])[0]
