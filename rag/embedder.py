"""bge-large-zh-v1.5 向量编码（Sentence-Transformers）。"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

# 避免多进程/部分环境下线程问题
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class BgeEmbedder:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        emb = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 32,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return np.asarray(emb, dtype=np.float32)
