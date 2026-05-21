"""bge-large-zh-v1.5 向量编码（Sentence-Transformers）。"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from rag.device_utils import is_cuda_runtime_error, resolve_device

# 避免多进程/部分环境下线程问题
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class BgeEmbedder:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        from rag import config

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = resolve_device(
            device, "RAG_EMBEDDING_DEVICE", config.EMBEDDING_DEVICE
        )
        self._model = SentenceTransformer(model_name, device=self.device)

    def _reload(self, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = device
        self._model = SentenceTransformer(self.model_name, device=device)

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros(
                (0, self._model.get_sentence_embedding_dimension()), dtype=np.float32
            )
        try:
            return self._encode_once(texts, normalize)
        except Exception as e:
            if self.device != "cpu" and is_cuda_runtime_error(e):
                print(
                    "[WARN] BGE embedding 在 CUDA 上失败，自动切换到 CPU 重试。",
                    flush=True,
                )
                self._reload("cpu")
                return self._encode_once(texts, normalize)
            raise

    def _encode_once(self, texts: Sequence[str], normalize: bool) -> np.ndarray:
        emb = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 32,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return np.asarray(emb, dtype=np.float32)
