"""推理设备解析（embedding / rerank）。"""

from __future__ import annotations

import os


def resolve_device(
    explicit: str | None,
    env_var: str,
    default: str = "cpu",
) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    env = os.environ.get(env_var, "").strip()
    if env:
        return env
    return default


def is_cuda_runtime_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "cuda" in msg or "cublas" in msg or "acceleratorerror" in msg
